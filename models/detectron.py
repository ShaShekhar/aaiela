from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import torch
import numpy as np

# environment variable that specifies the directory where model weights will be stored.
os.environ["FVCORE_CACHE"] = "weights"

class Detectron:
    def __init__(self, model_name="panoptic_fpn_R_101_3x.yaml", model_type="PS", device="cuda", thresh=0.7):
        self.cfg = get_cfg()
        self.model_type = model_type
        # Load model config and pretrained model
        if model_type == "OD":
            model_name = os.path.join("COCO-Detection", model_name)
            self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        # elif model_type == "IS":
        #     model_name = os.path.join("COCO-InstanceSegmentation", model_name)
        #     self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        #     self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        elif model_type == "KP":
            model_name = os.path.join("COCO-Keypoints", model_name)
            self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        elif model_type == "PS":
            model_name = os.path.join("COCO-PanopticSegmentation", model_name)
            self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.cfg.MODEL.DEVICE = device
        
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        if self.model_type == "OD":
            predictions = self.predictor(img)["instances"]

            boxes = predictions.pred_boxes.tensor.detach().cpu().numpy().astype(np.int32)
            # scores = predictions.scores
            classes = predictions.pred_classes.tolist()
            class_names = metadata.get("thing_classes", None)
            labels = [class_names[i] for i in classes]

            viz = Visualizer(img[:, :, ::-1], metadata) # img BGR to RGB
            out = viz.draw_instance_predictions(predictions.to("cpu"))

            return out.get_image(), boxes, labels # output image in BGR format
        
        elif self.model_type == "PS": # PANOPTIC SEG
            predictions, segmentInfo = self.predictor(img)["panoptic_seg"]
            
            viz = Visualizer(img[:, :, ::-1], metadata) # img BGR to RGB
            out = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            # extract mask and class label
            segment_ids = torch.unique(predictions, sorted=True)
            seg_ids = segment_ids.tolist()[1:]
            segment_info = {s["id"]: s for s in segmentInfo}
            masks, labels = [], []
            for sid in seg_ids:
                sinfo = segment_info.get(sid)
                masks.append((predictions == sid).cpu().numpy().astype(np.uint8))
                if sinfo["isthing"]:
                    labels.append(metadata.thing_classes[sinfo["category_id"]])
                else:
                    labels.append(metadata.stuff_classes[sinfo["category_id"]])
            
            masks = np.array(masks) # [n, h, w]

            return out.get_image(), masks, labels # output image in BGR format

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    pil_image = Image.open('../tests/images/input.jpg').convert("RGB")
    img_array = np.array(pil_image)[:, :, ::-1] # detectron input: BGR
    model_name = "panoptic_fpn_R_101_3x.yaml"
    # model_name = faster_rcnn_R_101_FPN_3x.yaml, type = "OD"
    # model_name = keypoint_rcnn_R_50_FPN_3x.yaml, type = "KP"
    # model_name = mask_rcnn_R_50_FPN_3x.yaml, type = "IS"
    detectron = Detectron(model_name=model_name, model_type="PS", device="cuda", thresh=0.7)
    proc_img, _, _ = detectron.predict(img_array)

    plt.imshow(proc_img)
    plt.show()