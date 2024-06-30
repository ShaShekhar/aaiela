from models import detectron
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

pil_image = Image.open('tests/images/input.jpg').convert("RGB")
img_array = np.array(pil_image)[:, :, ::-1] # detectron input: BGR
model_name, model_type = "panoptic_fpn_R_101_3x.yaml", "PS"
# model_name, model_type = faster_rcnn_R_101_FPN_3x.yaml, "OD"
# model_name, model_type = keypoint_rcnn_R_50_FPN_3x.yaml, "KP"
# model_name, model_type = mask_rcnn_R_50_FPN_3x.yaml, "IS"
try:
    model = detectron.Detectron(model_name=model_name, model_type=model_type, device="cuda", thresh=0.7)
    proc_img, _, _ = model.predict(img_array)
    print('Test Passed!')
    # plt.imshow(proc_img)
    # plt.show()
except Exception as e:
    print('Test Failed!')