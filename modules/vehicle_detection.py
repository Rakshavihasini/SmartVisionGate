from ultralytics import YOLO
from ..config import cfg
import torch

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.config = cfg.VehicleDetector

    def get_highest_confidence_detection(self, results):
        if not results:
            return None
        
        best_vehicle = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None or result.masks is None:
                continue

            boxes = result.boxes.xyxy          # (N, 4)
            confs = result.boxes.conf          # (N,)
            masks = result.masks.data          # (N, H, W)

            if len(confs) == 0:
                continue

            best_idx = torch.argmax(confs).item()

            if confs[best_idx].item() > best_conf:
                best_vehicle = {
                    "boxes": boxes[best_idx].unsqueeze(0),   # keep batch shape
                    "masks": masks[best_idx].unsqueeze(0),
                    "confidence": confs[best_idx].item()
                }
                best_conf = confs[best_idx].item()
            
        return best_vehicle

    def predict(self, image):
        results = self.model.predict(
            source=image,
            conf=self.config["conf_threshold"],
            iou=self.config["iou_threshold"],
            verbose=False
        )
        return self.get_highest_confidence_detection(results)

      
    def get_vehicle_body(self, image):
        predictions = self.predict(image)
        return predictions
        