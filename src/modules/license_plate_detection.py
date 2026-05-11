from ..config import cfg
from ..utils.helper import read_license_plate
from ultralytics import YOLO
import torch
import cv2

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.config = cfg.LicensePlateDetector

    def get_highest_confidence_plate(self, results):
        if not results:
            return None
        
        best_plate = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy          # (N, 4)
            confs = result.boxes.conf          # (N,)

            if len(confs) == 0:
                continue

            best_idx = torch.argmax(confs).item()

            if confs[best_idx].item() > best_conf:
                best_plate = {
                    "boxes": boxes[best_idx].unsqueeze(0),   # keep batch shape
                    "confidence": confs[best_idx].item()
                }
                best_conf = confs[best_idx].item()
            
        return best_plate
    
    def predict(self, image):
        results = self.model.predict(
            source=image,
            conf=self.config["conf_threshold"],
            iou=self.config["iou_threshold"],
            verbose=False
        )

        return self.get_highest_confidence_plate(results)


    def get_license_plate(self, image):
        plate = self.predict(image)
        # keep highest confidence plate only
        if not plate:
            return None, None
        
        lp_box = plate.get('boxes', None)
        lp_box_crop = image[int(lp_box[0][1]):int(lp_box[0][3]), int(lp_box[0][0]):int(lp_box[0][2])]
        lp_box_crop = cv2.cvtColor(lp_box_crop, cv2.COLOR_BGR2GRAY)
        _, lp_thresh = cv2.threshold(lp_box_crop, 64, 255, cv2.THRESH_BINARY)
      
        lp_text, lp_confidence = read_license_plate(lp_thresh)

        return lp_text, lp_confidence
    
        