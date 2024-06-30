import numpy as np
from PIL import Image
from einops import repeat
import torch
import torch.nn.functional as F
from torch import autocast
from contextlib import nullcontext
from omegaconf import OmegaConf

from . import ddim
from . import sdInpaint

def make_batch_sd(img_path, mask_uint8, txt, device, num_samples=1):
    img_array = np.array(Image.open(img_path).resize((512, 512)).convert("RGB"))
    img_array = img_array.astype(np.float32)/255.0
    img_array = img_array[None].transpose(0,3,1,2)
    img_array = torch.from_numpy(img_array)
    image = img_array*2.0 - 1.0 # range of [-1, 1]

    resized_mask = np.array(Image.fromarray(mask_uint8).resize((512, 512), Image.NEAREST))
    resized_mask = resized_mask[None,None]
    mask_float = resized_mask.astype(np.float32)
    mask = torch.from_numpy(mask_float)

    masked_image = (1 - mask) * image # same as image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

class SdInpainting:
    def __init__(self, config_yaml, ckpt, device='cpu'):
        self.device = device
        config = OmegaConf.load(config_yaml)
        # print(f"Loading model from {ckpt}")
        config.model.cond_stage_config.device = device
        
        inpaint_sd = torch.load(ckpt, map_location="cpu")
        self.ld_model = sdInpaint.LatentInpaintDiffusion(**config.model)
        self.ld_model.load_state_dict(inpaint_sd["state_dict"], strict=False)

        del inpaint_sd

        device = torch.device(device)
        self.ld_model = self.ld_model.to(device)

        self.sampler = ddim.DDIMSampler(self.ld_model)

        self.uc_cross = self.ld_model.get_unconditional_conditioning(1, "")
        self.start_code = torch.randn([1, 4, 64, 64], dtype=torch.float32, device=device)

        precision = 'autocast' if device == 'cuda' else 'full'
        self.precision_scope = autocast if precision=="autocast" else nullcontext

    def main(self, img_path, mask_uint8, prompt, ddim_steps=50, ddim_eta=0.0, scale=7.5, batch_size=1):

        with torch.no_grad():
            with self.precision_scope("cuda"):
                origImg = Image.open(img_path)
                width, height = origImg.size # [W, H, C]
                img_array = np.array(origImg.convert("RGB")) # [H, W, C]
                img_array = img_array.astype(np.float32)/255.0
                img_array = img_array[None].transpose(0,3,1,2) # [1, C, H, W]
                img_tensor = torch.from_numpy(img_array).to(self.device)
                
                mask_array = np.array(Image.fromarray(mask_uint8)) # [n, H, W]
                mask_array = mask_array[None,None]
                mask_float = mask_array.astype(np.float32)
                origMask = torch.from_numpy(mask_float).to(self.device)

                batch = make_batch_sd(img_path, mask_uint8, prompt, device=self.device, num_samples=batch_size)

                c = self.ld_model.cond_stage_model.encode(batch["txt"])

                c_cat = list()
                for ck in self.ld_model.concat_keys: # "mask", "masked_image"
                    cc = batch[ck].float()
                    if ck != self.ld_model.masked_image_key: # masked_image
                        bchw = [batch_size, 4, 64, 64]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:]) # [1, 1, 64, 64]
                    else:
                        cc = self.ld_model.get_first_stage_encoding(self.ld_model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                # uc_cross = ld_model.get_unconditional_conditioning(batch_size, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [self.uc_cross]}

                shape = [4, 64, 64]
                samples_ddim, _ = self.sampler.sample(
                                    S=ddim_steps,
                                    batch_size=batch_size,
                                    shape=shape,
                                    conditioning=cond,
                                    verbose=False,
                                    eta=ddim_eta, # corresponds to deterministic sampling
                                    unconditional_conditioning=uc_full,
                                    unconditional_guidance_scale=scale,
                                    x_T=self.start_code
                                )

                x_samples_ddim = self.ld_model.decode_first_stage(samples_ddim)
                predicted_img = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                resized_img_tensor = F.interpolate(predicted_img, size=(height, width), mode='bilinear')
                
                inpainted_img = (1-origMask)*img_tensor + origMask*resized_img_tensor
                inpainted_img = inpainted_img.cpu().numpy().transpose(0,2,3,1)[0] * 255

        return inpainted_img.astype(np.uint8)