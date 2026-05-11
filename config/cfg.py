import os

# Get base path dynamically (works from anywhere)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Store masked vehicle images by license plate under database/vehicle_mask
mask_output_path = os.path.join(base_path, "database", "vehicle_mask")

VehicleDetector = {
    "conf_threshold": 0.5,
    "iou_threshold": 0.4,
    "model_path": os.path.join(base_path, "models", "yolo11m-seg.pt")
}

LicensePlateDetector = {
    "conf_threshold": 0.1,
    "iou_threshold": 0.3,
    "model_path": os.path.join(base_path, "models", "license_plate_detector.pt")
}


FacialRecognition = {
    "detector_backend": "retinaface",
    "database_path" : os.path.join(base_path, "database", "faces"),
    "model_name" : "ArcFace",
    # DeepFace distance threshold for accepting a match.
    # Tune this per your dataset/model; lower is stricter.
    "max_distance": 0.65
}

