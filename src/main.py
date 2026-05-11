from src.modules.vehicle_detection import VehicleDetector
from src.modules.license_plate_detection import LicensePlateDetector
import cv2
from cv2 import resize
from src.config import cfg
import os
import re
import sqlite3
from typing import Dict, Optional
from src.modules.vehicle_verification import VehicleVerificationEngine


DB_PATH = os.path.abspath(os.path.join(cfg.base_path, "database", "vehicles.db"))
IMAGES_DIR = os.path.abspath(os.path.join(cfg.base_path, "vehicle_images"))

class SmartVision:
    def __init__(self):
        self.vehicle_detector = VehicleDetector(cfg.VehicleDetector["model_path"])
        self.license_plate_detector = LicensePlateDetector(cfg.LicensePlateDetector["model_path"])


    def _print(self, msg, level="info"):
        levels = {
            "info": "[INFO]",
            "warn": "[WARN]",
            "error": "[ERROR]",
            "success": "[SUCCESS]"
        }
        print(f"{levels.get(level, '[INFO]')} {msg}")

    def _save_mask_image(self, mask_img, lp_text):
        os.makedirs(cfg.mask_output_path, exist_ok=True)
        out_path = os.path.join(cfg.mask_output_path, self._normalize_plate(lp_text) + ".jpg")
        cv2.imwrite(out_path, mask_img)
        self._print(f"Saved mask image: {out_path}", "success")

    def analyze_image(self, image):
        vehicle_data = self.vehicle_detector.get_vehicle_body(image)
        if not vehicle_data:
            self._print("No vehicles detected.", "warn")
            return None

        vehicle_mask = vehicle_data.get('masks', None)
        if vehicle_mask is None:
            self._print("No vehicle mask found.", "warn")
            return None
        vehicle_mask = vehicle_mask[0].cpu().numpy()
        vehicle_mask = (vehicle_mask > 0).astype("uint8")

        if vehicle_mask.shape != image.shape[:2]:
            vehicle_mask = resize(vehicle_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        vehicle_mask_img = image * vehicle_mask[:, :, None]
        if vehicle_mask_img is None:
            self._print("No vehicle mask image found.", "warn")
            return None

        lp_text, lp_confidence = self.license_plate_detector.get_license_plate(vehicle_mask_img)
        if lp_text is None:
            self._print("No license plate found.", "warn")
            return None

        self._print(f"License Plate: {lp_text}, Confidence: {lp_confidence:.2f}")
        self._save_mask_image(vehicle_mask_img, lp_text)
        return vehicle_data

    

    def _image_to_binary_mask(self, masked_img):
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        return ((gray > 0).astype("uint8") * 255)

    def _normalize_plate(self, plate_text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(plate_text).upper())

    def _lookup_vehicle_owner(self, plate_text: str) -> Optional[Dict[str, Optional[str]]]:
        if not os.path.exists(DB_PATH):
            return None

        normalized_plate = self._normalize_plate(plate_text)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT owner_name, owner_phone
                FROM vehicles
                WHERE REPLACE(REPLACE(REPLACE(UPPER(license_plate), ' ', ''), '-', ''), '.', '') = ?
                LIMIT 1
                """,
                (normalized_plate,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "owner_name": row["owner_name"],
                "owner_phone": row["owner_phone"],
            }
        except Exception as e:
            self._print(f"DB lookup error: {e}", "error")
            return None
        finally:
            conn.close()

    def _build_vehicle_mask_from_image(self, image):
        vehicle_data = self.vehicle_detector.get_vehicle_body(image)
        if not vehicle_data:
            return None

        vehicle_mask = vehicle_data.get('masks', None)
        if vehicle_mask is None:
            return None

        vehicle_mask = vehicle_mask[0].cpu().numpy()
        vehicle_mask = (vehicle_mask > 0).astype("uint8")

        if vehicle_mask.shape != image.shape[:2]:
            vehicle_mask = resize(vehicle_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return image * vehicle_mask[:, :, None]

    def _find_registered_image_path(self, plate_text: str, db_filename: Optional[str]) -> Optional[str]:
        if db_filename:
            candidate = os.path.join(IMAGES_DIR, db_filename)
            if os.path.exists(candidate):
                return candidate

        # Fallback: match by normalized plate prefix in vehicle_images
        normalized = self._normalize_plate(plate_text).lower()
        if not os.path.isdir(IMAGES_DIR):
            return None

        for fname in os.listdir(IMAGES_DIR):
            fpath = os.path.join(IMAGES_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            stem, _ = os.path.splitext(fname)
            if stem.lower().startswith(normalized):
                return fpath

        return None

    def generate_reference_masks_for_registered_vehicles(self, force: bool = False):
        """
        Build and save masked vehicle reference images for all registered vehicles.
        Output path: database/vehicle_mask/<PLATE>.jpg
        """
        os.makedirs(cfg.mask_output_path, exist_ok=True)

        if not os.path.exists(DB_PATH):
            self._print(f"DB not found at {DB_PATH}. Skipping mask generation.", "warn")
            return

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        created = 0
        skipped_existing = 0
        skipped_missing_image = 0
        failed_detection = 0

        try:
            cur = conn.cursor()
            cur.execute("SELECT license_plate, image_filename FROM vehicles ORDER BY id ASC")
            rows = cur.fetchall()

            if not rows:
                self._print("No registered vehicles found in DB for mask generation.", "warn")
                return

            self._print(f"Auto-generating reference masks for {len(rows)} registered vehicles...", "info")

            for row in rows:
                plate = self._normalize_plate(row["license_plate"])
                image_filename = row["image_filename"]
                out_path = os.path.join(cfg.mask_output_path, f"{plate}.jpg")

                if os.path.exists(out_path) and not force:
                    skipped_existing += 1
                    continue

                source_path = self._find_registered_image_path(plate, image_filename)
                if not source_path:
                    skipped_missing_image += 1
                    self._print(f"Mask skipped for {plate}: source registration image not found.", "warn")
                    continue

                image = cv2.imread(source_path)
                if image is None:
                    skipped_missing_image += 1
                    self._print(f"Mask skipped for {plate}: failed to read image {source_path}.", "warn")
                    continue

                masked = self._build_vehicle_mask_from_image(image)
                if masked is None:
                    failed_detection += 1
                    self._print(f"Mask skipped for {plate}: no vehicle mask detected in {source_path}.", "warn")
                    continue

                cv2.imwrite(out_path, masked)
                created += 1
                self._print(f"Mask saved for {plate}: {out_path}", "success")

            self._print(
                f"Mask generation completed. created={created}, "
                f"skipped_existing={skipped_existing}, "
                f"missing_source={skipped_missing_image}, "
                f"failed_detection={failed_detection}",
                "info",
            )
        except Exception as e:
            self._print(f"Mask generation failed: {e}", "error")
        finally:
            conn.close()

    def _find_db_image(self, plate_text):
        os.makedirs(cfg.mask_output_path, exist_ok=True)
        normalized_plate = self._normalize_plate(plate_text)

        # Common candidate filenames / extensions
        possible_basenames = [plate_text, normalized_plate, normalized_plate.lower()]
        possible_exts = [".jpg", ".jpeg", ".png", ".webp"]

        search_folders = [cfg.mask_output_path]
        if hasattr(cfg, 'base_path'):
            search_folders.extend([
                os.path.abspath(os.path.join(cfg.base_path, 'database', 'vehicle_masks')),
                os.path.abspath(os.path.join(cfg.base_path, 'database', 'vehicle_mask')),
            ])

        for folder in search_folders:
            if not folder or not os.path.isdir(folder):
                continue
            for base in possible_basenames:
                for ext in possible_exts:
                    candidate = os.path.join(folder, f"{base}{ext}")
                    if os.path.exists(candidate):
                        return candidate

            # Fallback: case-insensitive basename match
            for fname in os.listdir(folder):
                stem, ext = os.path.splitext(fname)
                if ext.lower() in possible_exts and self._normalize_plate(stem) == normalized_plate:
                    return os.path.join(folder, fname)

        return None

    def _build_invalid_reason(self, result: Dict, threshold: float) -> str:
        """
        Build a common-man-friendly reason when verification fails.
        """
        ssim_score = float(result.get("ssim_score", 0.0) or 0.0)
        color_score = float(result.get("color_score", 0.0) or 0.0)
        shape_score = float(result.get("shape_score", 0.0) or 0.0)
        phash_score = float(result.get("phash_score", 0.0) or 0.0)
        composite = float(result.get("composite_score", 0.0) or 0.0)

        reasons = []

        # Shape + structure mismatch are most intuitive for users
        if ssim_score < 0.55 or shape_score < 0.55:
            reasons.append("Vehicle structure doesn't match the registered vehicle")

        if color_score < 0.55:
            reasons.append("Vehicle colour doesn't match the registered vehicle")

        if phash_score < 0.50:
            reasons.append("Overall vehicle appearance doesn't match")

        if not reasons:
            reasons.append("Vehicle similarity is below the required level")

        reasons_text = "; ".join(reasons)
        return f"{reasons_text} (score {composite:.2f} < required {threshold:.2f})"

    def process_video(self, video_path):
        """
        Process a video stream to:
        - Detect the first visible license plate
        - Match it against the DB (by filename in `database/vehicle_mask/<PLATE>.jpg`)
        - For subsequent frames, extract the vehicle mask and compare similarity
          to the DB mask using the verification engine. Print VALID/FRAUD status.
        Args:
            video_path: path to a video file or camera index (e.g., 0)
        """
        cap = None
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self._print(f"Could not open video source: {video_path}", "error")
            return

        verifier = VehicleVerificationEngine()
        plate_text = None
        db_img_path = None
        db_img = None
        db_mask = None
        is_registered = False
        frame_idx = 0
        valid_count = 0
        fraud_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 30  # fallback if FPS not available
        frame_interval = int(fps)

        self._print(f"Processing video at {fps:.2f} FPS, analyzing 1 frame per second.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Only process one frame per second
            if (frame_idx - 1) % frame_interval != 0:
                continue

            # Detect vehicle
            vehicle_data = self.vehicle_detector.get_vehicle_body(frame)
            if not vehicle_data:
                self._print(f"Frame {frame_idx}: No vehicle detected.", "warn")
                continue

            vmask = vehicle_data.get('masks', None)
            if vmask is None:
                self._print(f"Frame {frame_idx}: Vehicle mask not available.", "warn")
                continue

            vmask = vmask[0].cpu().numpy()
            vmask = (vmask > 0).astype("uint8")
            if vmask.shape != frame.shape[:2]:
                vmask = resize(vmask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            vehicle_mask_img = frame * vmask[:, :, None]

            # If we don't have a plate yet, try to read it asap
            if plate_text is None:
                lp_text, lp_conf = self.license_plate_detector.get_license_plate(vehicle_mask_img)
                if lp_text:
                    plate_text = lp_text
                    self._print(f"Detected plate: {plate_text} (conf: {lp_conf:.2f})", "success")
                    owner = self._lookup_vehicle_owner(plate_text)
                    if owner:
                        is_registered = True
                        owner_name = owner.get("owner_name") or "Unknown"
                        owner_phone = owner.get("owner_phone") or "N/A"
                        self._print(
                            f"DB: Registration found for plate {self._normalize_plate(plate_text)} | "
                            f"Owner: {owner_name} | Phone: {owner_phone}",
                            "success",
                        )

                        found_path = self._find_db_image(plate_text)
                        if not found_path:
                            self._print(
                                f"DB: Registered plate {self._normalize_plate(plate_text)}, but no reference mask image found "
                                f"in {cfg.mask_output_path}.",
                                "warn",
                            )
                            self._print("Status: Registered in DB. Verification skipped (missing reference mask).", "warn")
                        else:
                            db_img_path = found_path
                            db_img = cv2.imread(db_img_path)
                            if db_img is None:
                                self._print(f"DB: Failed to load reference image {db_img_path}", "error")
                                db_img_path = None
                            else:
                                db_mask = self._image_to_binary_mask(db_img)
                                self._print(f"DB: Found reference for {self._normalize_plate(plate_text)}: {db_img_path}", "info")
                    else:
                        is_registered = False
                        self._print(f"DB: No registration record for plate {self._normalize_plate(plate_text)}.", "warn")
                        self._print("Status: Not registered in DB.", "warn")
                continue

            # If we have a plate and a DB reference, verify per frame
            if plate_text and db_img is not None and db_mask is not None:
                cam_mask = self._image_to_binary_mask(vehicle_mask_img)
                try:
                    verify_threshold = 0.65
                    result = verifier.verify_vehicle(
                        camera_mask=cam_mask,
                        camera_image=vehicle_mask_img,
                        db_mask=db_mask,
                        db_image=db_img,
                        threshold=verify_threshold
                    )
                    status = "VALID" if result.get('verified') else "FRAUD"
                    score = result.get('composite_score', 0.0)
                    if result.get('verified'):
                        valid_count += 1
                    else:
                        fraud_count += 1
                    self._print(f"Frame {frame_idx}: Plate {plate_text} → {status} (score={score:.3f})", "info")
                    if not result.get('verified'):
                        reason = self._build_invalid_reason(result, verify_threshold)
                        self._print(f"Reason: {reason}", "warn")
                except Exception as e:
                    self._print(f"Verification error on frame {frame_idx}: {e}", "error")

        cap.release()
        self._print("\nVideo processing completed.", "info")
        if plate_text is None:
            self._print("No license plate was detected in the stream.", "warn")
        elif not is_registered:
            self._print(f"Plate {self._normalize_plate(plate_text)} detected, but it is not registered in DB.", "warn")
        elif db_img_path is None:
            self._print(
                f"Plate {self._normalize_plate(plate_text)} is registered in DB, but no reference mask image was found.",
                "warn",
            )
        else:
            self._print(f"Summary for {plate_text}: VALID frames={valid_count}, FRAUD frames={fraud_count}", "success")
            
            # Final verdict based on frame counts
            if valid_count >= fraud_count:
                self._print(f"Final Verdict: VALID VEHICLE - {plate_text}", "success")
            else:
                self._print(f"Final Verdict: FRAUD VEHICLE - {plate_text}", "error")

if __name__ == "__main__":

    # # Get test image path relative to base_path
    # test_image_path = ("/home/hitech/Downloads/SmartVision/test_data/images/image.png")
    
    # sv = SmartVision()
    # test_image = cv2.imread(test_image_path)
    
    # if test_image is None:
    #     print(f"Error: Could not load test image from {test_image_path}")
    #     print("Make sure test.jpg exists in data/ directory")
    # else:
    #     sv.analyze_image(test_image)

    sv = SmartVision()
    sv.generate_reference_masks_for_registered_vehicles(force=False)
    sv.process_video("video.mp4")