import cv2
import os
from deepface import DeepFace
from src.config import cfg

# Disable GPU to avoid CUDA/libdevice issues - use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow to use CPU only
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception as e:
    print(f"[WARNING] Could not disable GPU: {e}")

class FaceRecognition:
    def __init__(self):
        """
        detector_backend options:
        - retinaface (best)
        - mtcnn
        - opencv
        - ssd
        """
        
        self.detector_backend = cfg.FacialRecognition["detector_backend"]
        self.database_path = cfg.FacialRecognition["database_path"]
        self.model_name = cfg.FacialRecognition["model_name"]

    def extract_face(self, image_path, user_name=None):
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            align=True,
        )

        if not faces:
            return None

        # DeepFace returns a list of detected faces; pick the first one.
        face_img = faces[0].get("face")
        return face_img


    def recognize_faces(self, image):
        """
        Returns list of matches (one DataFrame per detected face)
        """
        results = DeepFace.find(
            img_path=image,
            db_path=self.database_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=True,
        )
        # Depending on DeepFace version/config, this can be a DataFrame or a list of DataFrames.
        if results is None:
            return []
        if isinstance(results, list):
            print(f"Detected {len(results)} face(s); returning candidate matches from database.")
            return results
        print("Detected 1 face; returning candidate matches from database.")
        return [results]

    def identify_faces(self, image, top_k=1, distance_col="distance"):
        """
        Returns list of top matches per detected face with names.
        """
        results = self.recognize_faces(image)
        identified = []
        for match_df in results:
            if match_df is None or match_df.empty:
                identified.append([])
                continue

            if distance_col in match_df.columns:
                match_df = match_df.sort_values(by=distance_col, ascending=True)

            top_matches = []
            for _, row in match_df.head(top_k).iterrows():
                identity_path = row.get("identity")
                name = (
                    os.path.basename(os.path.dirname(identity_path))
                    if identity_path
                    else "unknown"
                )
                distance = row.get(distance_col)
                top_matches.append(
                    {
                        "name": name,
                        "distance": float(distance) if distance is not None else None,
                        "identity": identity_path,
                    }
                )

            identified.append(top_matches)

        return identified
    
# main
if __name__ == "__main__":
    fr = FaceRecognition()
    test_image_path = "/home/hitech/Downloads/SmartVision/test_chandana.jpeg"
    face = fr.extract_face(test_image_path)
    print("Extracted face shape:", face.shape)
    top_matches = fr.identify_faces(test_image_path, top_k=1)
    print("Top matches:", top_matches)