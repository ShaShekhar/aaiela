from PIL import Image
import numpy as np
from models.sd_inpaint import text2image_inpaint
# import matplotlib.pyplot as plt

img_path = 'tests/images/dog.png'
mask_path = 'tests/images/dog_mask.png'

mask = np.array(Image.open(mask_path).convert("L"))
mask_float = mask.astype(np.float32)/255.0
mask_float[mask_float < 0.5] = 0
mask_float[mask_float >= 0.5] = 1

mask_uint8 = mask_float.astype(np.uint8)

prompt = "a white tiger sitting on the bench."
config_path = 'configs/v1-inpainting-inference.yaml'
weight_path = 'weights/sd-v1-5-inpainting.ckpt'

try:
    sd_model = text2image_inpaint.SdInpainting(config_path, weight_path, device='cuda')
    img_inpaint = sd_model.main(img_path, mask_float, prompt, ddim_steps=5, ddim_eta=0.0, scale=12.0, batch_size=1)
    # plt.imshow(img_inpaint)
    # plt.show()
    print('Test Passed!')
except Exception as e:
    print('Test Failed!')
    print(e)