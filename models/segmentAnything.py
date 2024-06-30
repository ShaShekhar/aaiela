import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import colorsys
from segment_anything import (
    sam_model_registry, 
    SamPredictor, 
    SamAutomaticMaskGenerator
)

# Initialize sam model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

predictor = SamPredictor(sam)

def predict(img_array, bbox=None):

    if bbox is None:
        masks = mask_generator.generate(img_array)
        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
        # if len(anns) == 0:
        #     return
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for mask in sorted_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        # TODO: add the mask and image
        # merged_img = cv2.addWeighted(dimmed_img.astype(np.uint8), 1, colored_mask, 0.6, 0)
        
        return img, sorted_masks # aggregate all the masks
    
    else:
        predictor.set_image(img_array)
        input_boxes = torch.tensor(bbox, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_array.shape[:2])
        
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None, 
            boxes=transformed_boxes, multimask_output=False
        )
        masks_numpy = masks.permute(0, 2, 3, 1).cpu().numpy() # [n,1,512,512] -> [n,512,512,1]
        num_masks = masks_numpy.shape[0]
        # print(num_masks)

        # Generate distinct colors in HSV space
        hsv_colors = [(i / num_masks, 1, 1) for i in range(num_masks)]  # Evenly spaced hues
        rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]  # Convert to RGB
        colors = np.array(rgb_colors) * 255
        
        # colors = np.random.random((num_masks, 3))
        masks_numpy = masks_numpy.astype(np.float32)
        colored_mask = (masks_numpy * colors[:, np.newaxis, np.newaxis, :]).sum(axis=0)
        colored_mask = colored_mask.astype(np.uint8)

        # Create a single dimming mask by averaging the masks along the first axis
        dimming_mask = masks_numpy.max(axis=0)
        dimming_mask = np.repeat(dimming_mask, 3, axis=2)  # Repeat for each color channel

        dimming_mask *= 0.6 # alpha

        dimmed_img = cv2.multiply(img_array.astype(np.float32), 1 - dimming_mask)
        # Blend with the colored mask (increase colored_mask weight for brighter colors)
        merged_img = cv2.addWeighted(dimmed_img.astype(np.uint8), 1, colored_mask, 0.6, 0)

        return merged_img, masks_numpy.squeeze(-1).astype(np.uint8)

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import detectron

    plt.figure(figsize=(10, 10))

    pil_image = Image.open('tests/images/input.jpg').convert("RGB")
    img_array = np.array(pil_image)
    model_name = 'faster_rcnn_R_101_FPN_3x.yaml'
    model_type = "OD"
    detectron = detectron.Detectron(
        model_name=model_name, model_type=model_type,
        device="cuda", thresh=0.7
    )

    automatic_mask_generator = False

    if automatic_mask_generator:
        masked_img, _ = predict(img_array)
        plt.imshow(masked_img)
        # show_anns(masks)
        plt.axis('off')
        plt.show()
    else:
        img_bbox, bbox, class_name = detectron.predict(img_array)

        merged_mask, _ = predict(img_array, bbox)

        plt.imshow(merged_mask)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in input_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.show()