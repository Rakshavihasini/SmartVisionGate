from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import sqlite3
from datetime import datetime
import re
import cv2
import numpy as np

# Add src to path for SmartVision imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'vehicles.db')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'vehicle_images')
MASKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'database', 'vehicle_mask')

# Initialize detectors (lazy load to handle missing dependencies gracefully)
vehicle_detector = None
license_plate_detector = None

# Face recognition (lazy init)
face_recognizer = None

def init_detectors():
    """Initialize SmartVision detectors"""
    global vehicle_detector, license_plate_detector
    try:
        from src.modules.vehicle_detection import VehicleDetector
        from src.modules.license_plate_detection import LicensePlateDetector
        
        # Get actual project paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vehicle_model_path = os.path.join(project_root, "models", "yolo11m-seg.pt")
        license_model_path = os.path.join(project_root, "models", "license_plate_detector.pt")
        
        print("[*] Initializing SmartVision detectors...")
        vehicle_detector = VehicleDetector(vehicle_model_path)
        license_plate_detector = LicensePlateDetector(license_model_path)
        print("[OK] Detectors initialized successfully!")
        return True
    except Exception as e:
        print(f"[WARNING] Could not initialize detectors: {str(e)}")
        print("[WARNING] License plate detection will be disabled")
        return False


def init_face_recognizer():
    """Initialize face recognizer (DeepFace-backed)"""
    global face_recognizer
    try:
        from src.modules.facial_recognition import FaceRecognition

        print("[*] Initializing face recognizer...")
        face_recognizer = FaceRecognition()
        print("[OK] Face recognizer initialized successfully!")
        return True
    except Exception as e:
        print(f"[WARNING] Could not initialize face recognizer: {str(e)}")
        return False

def detect_plate_from_image(image):
    """
    Detect license plate text from vehicle image
    Returns: (plate_text, confidence) or (None, None) if not detected
    """
    global vehicle_detector, license_plate_detector
    
    if vehicle_detector is None or license_plate_detector is None:
        return None, None
    
    try:
        # Detect vehicle body
        vehicle_data = vehicle_detector.get_vehicle_body(image)
        if not vehicle_data:
            return None, None
        
        # Get vehicle mask
        vehicle_mask = vehicle_data.get('masks', None)
        if vehicle_mask is None:
            return None, None
        
        vehicle_mask = vehicle_mask[0].cpu().numpy()
        vehicle_mask = (vehicle_mask > 0).astype("uint8")
        
        # Resize mask to match image
        if vehicle_mask.shape != image.shape[:2]:
            vehicle_mask = cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask to image
        vehicle_mask_img = image * vehicle_mask[:, :, None]
        
        # Detect license plate
        plate_text, plate_confidence = license_plate_detector.get_license_plate(vehicle_mask_img)
        
        return plate_text, plate_confidence
        
    except Exception as e:
        print(f"[ERROR] Error during plate detection: {str(e)}")
        return None, None


def extract_vehicle_mask_image(image):
    """
    Extract masked vehicle image from input frame.
    Returns masked image or None if vehicle/mask is unavailable.
    """
    global vehicle_detector

    if vehicle_detector is None:
        return None

    try:
        vehicle_data = vehicle_detector.get_vehicle_body(image)
        if not vehicle_data:
            return None

        vehicle_mask = vehicle_data.get('masks', None)
        if vehicle_mask is None:
            return None

        vehicle_mask = vehicle_mask[0].cpu().numpy()
        vehicle_mask = (vehicle_mask > 0).astype("uint8")

        if vehicle_mask.shape != image.shape[:2]:
            vehicle_mask = cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return image * vehicle_mask[:, :, None]
    except Exception as e:
        print(f"[WARNING] Failed to extract vehicle mask: {str(e)}")
        return None


def save_reference_vehicle_mask(image, license_plate):
    """
    Save canonical reference mask as database/vehicle_mask/<PLATE>.jpg.
    Best-effort; returns True when saved.
    """
    try:
        os.makedirs(MASKS_DIR, exist_ok=True)
        normalized_plate = re.sub(r'[^A-Z0-9]', '', str(license_plate).upper())
        if not normalized_plate:
            return False

        masked_image = extract_vehicle_mask_image(image)
        if masked_image is None:
            return False

        out_path = os.path.join(MASKS_DIR, f"{normalized_plate}.jpg")
        ok = cv2.imwrite(out_path, masked_image)
        if ok:
            print(f"[OK] Saved vehicle reference mask: {out_path}")
        return bool(ok)
    except Exception as e:
        print(f"[WARNING] Could not save reference mask for {license_plate}: {str(e)}")
        return False


def init_db():
    """Initialize simple SQLite database"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT NOT NULL,
            owner_name TEXT,
            owner_phone TEXT,
            owner_email TEXT,
            owner_address TEXT,
            image_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Enforce one-time registration per license plate.
    # If legacy duplicate rows exist, keep app running and rely on API-level checks.
    try:
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicles_license_plate_unique
            ON vehicles(license_plate)
        """)
    except (sqlite3.IntegrityError, sqlite3.OperationalError):
        # OperationalError is raised when data already has duplicates ("indexed columns are not unique")
        pass

    conn.commit()
    conn.close()

def sanitize_filename(text):
    """Clean text for use in filename"""
    if not text:
        return "unknown"
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Remove multiple underscores
    text = re.sub(r'_+', '_', text)
    return text.lower()[:30]  # Limit length

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database on startup
init_db()


# ============== ROUTES ==============

@app.route('/')
def index():
    """API Info"""
    return jsonify({
        "app": "SmartVision Vehicle Registration",
        "version": "1.0.0",
        "status": "running",
        "database": f"SQLite Local: {DB_PATH}",
        "endpoints": {
            "health": "GET /health",
            "register": "POST /api/register",
            "get_vehicle": "GET /api/vehicle/<license_plate>",
            "list_all": "GET /api/vehicles",
            "search": "GET /api/vehicles?search=term"
        }
    })


@app.route('/health')
def health_check():
    """API health check"""
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })


@app.route('/api/register', methods=['POST'])
def register_vehicle():
    """Register vehicle - detects & validates license plate from image"""
    temp_path = None
    try:
        # Get image
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "message": "No image selected"}), 400
        
        # Get form data
        # Normalize plate: uppercase + remove all spaces, dashes, dots and non-alphanumeric chars
        # so 'MH 20 EE 7602', 'MH-20-EE-7602', 'MH20EE7602' all resolve to the same key.
        raw_plate = request.form.get('license_plate', '').strip().upper()
        license_plate_input = re.sub(r'[^A-Z0-9]', '', raw_plate)
        owner_name = request.form.get('owner_name', '').strip()
        owner_phone = request.form.get('owner_phone', '').strip()
        owner_email = request.form.get('owner_email', '').strip()
        owner_address = request.form.get('owner_address', '').strip()
        
        # Validate
        if not license_plate_input:
            return jsonify({"success": False, "message": "License plate is required"}), 400

        # Prevent duplicate registration by license plate
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM vehicles WHERE license_plate = ?", (license_plate_input,))
        existing_vehicle = cursor.fetchone()
        conn.close()

        if existing_vehicle:
            return jsonify({
                "success": False,
                "message": f"Vehicle with license plate '{license_plate_input}' is already registered.",
                "error_code": "DUPLICATE_VEHICLE"
            }), 409
        
        # Save image temporarily for processing
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_plate = sanitize_filename(license_plate_input)
        temp_filename = f"{safe_plate}_{timestamp}_temp.jpg"
        temp_path = os.path.join(IMAGES_DIR, temp_filename)
        image_file.save(temp_path)
        
        # Read image for detection
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({"success": False, "message": "Failed to read image"}), 400
        
        # DETECT LICENSE PLATE FROM IMAGE
        detected_plate_text, detected_confidence = detect_plate_from_image(image)
        
        # Check 1: Was a plate detected?
        if detected_plate_text is None:
            os.remove(temp_path)  # Clean up
            return jsonify({
                "success": False, 
                "message": "License plate not detected in image. Please upload a clearer image with visible license plate.",
                "error_code": "NO_PLATE_DETECTED"
            }), 400
        
        # Check 2: Does detected plate match input plate?
        if detected_plate_text != license_plate_input:
            os.remove(temp_path)  # Clean up
            return jsonify({
                "success": False,
                "message": f"License plate mismatch! You entered '{license_plate_input}' but detected '{detected_plate_text}' in the image. Please verify and try again.",
                "entered_plate": license_plate_input,
                "detected_plate": detected_plate_text,
                "detected_confidence": float(detected_confidence) if detected_confidence else 0,
                "error_code": "PLATE_MISMATCH"
            }), 400
        
        # All validations passed! Now save the vehicle
        safe_owner = sanitize_filename(owner_name)
        
        # Save to database
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO vehicles (license_plate, owner_name, owner_phone, owner_email, owner_address, image_filename)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (license_plate_input, owner_name, owner_phone, owner_email, owner_address, temp_filename))
        except sqlite3.IntegrityError:
            conn.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                "success": False,
                "message": f"Vehicle with license plate '{license_plate_input}' is already registered.",
                "error_code": "DUPLICATE_VEHICLE"
            }), 409
        
        vehicle_id = cursor.lastrowid
        conn.commit()
        
        # Rename file with proper ID
        final_filename = f"{safe_plate}_{safe_owner}_{timestamp}_{vehicle_id}.jpg"
        final_path = os.path.join(IMAGES_DIR, final_filename)
        os.rename(temp_path, final_path)
        temp_path = None  # Don't try to delete it again
        
        # Update database with final filename
        cursor.execute("UPDATE vehicles SET image_filename = ? WHERE id = ?", (final_filename, vehicle_id))
        conn.commit()
        conn.close()

        # Auto-create/update reference mask for immediate verification use
        if not save_reference_vehicle_mask(image, license_plate_input):
            print(f"[WARNING] Vehicle registered, but reference mask was not created for {license_plate_input}")
        
        return jsonify({
            "success": True,
            "message": "Vehicle registered successfully!",
            "id": vehicle_id,
            "license_plate": license_plate_input,
            "detected_confidence": float(detected_confidence) if detected_confidence else 0,
            "image_filename": final_filename
        }), 201
        
    except Exception as e:
        # Clean up temp file if exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    """Verify face against database/faces and return access decision."""
    temp_path = None
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "authorized": False,
                "message": "No image provided"
            }), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "authorized": False,
                "message": "No image selected"
            }), 400

        global face_recognizer
        if face_recognizer is None:
            if not init_face_recognizer():
                return jsonify({
                    "success": False,
                    "authorized": False,
                    "message": "Face verification is not available on the server"
                }), 500

        # Save selfie temporarily
        os.makedirs(IMAGES_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"face_verify_{timestamp}.jpg"
        temp_path = os.path.join(IMAGES_DIR, temp_filename)
        image_file.save(temp_path)

        from src.config import cfg
        default_max_distance = float(cfg.FacialRecognition.get("max_distance", 0.65))

        # Optional override sent by client (lets you tune threshold without editing code)
        # multipart/form-data field name: max_distance (or threshold)
        max_distance_raw = request.form.get('max_distance') or request.form.get('threshold')
        if max_distance_raw is None or str(max_distance_raw).strip() == '':
            max_distance = default_max_distance
        else:
            try:
                max_distance = float(max_distance_raw)
            except Exception:
                return jsonify({
                    "success": False,
                    "authorized": False,
                    "message": "Invalid max_distance; must be a number"
                }), 400

        if max_distance <= 0:
            return jsonify({
                "success": False,
                "authorized": False,
                "message": "Invalid max_distance; must be > 0"
            }), 400

        identified = face_recognizer.identify_faces(temp_path, top_k=1)
        best = None
        if identified and len(identified) > 0 and identified[0] and len(identified[0]) > 0:
            best = identified[0][0]

        authorized = bool(
            best
            and best.get("distance") is not None
            and float(best["distance"]) <= max_distance
            and best.get("name") not in (None, "", "unknown")
        )

        if authorized:
            return jsonify({
                "success": True,
                "authorized": True,
                "message": "Access granted",
                "match": best,
                "max_distance": max_distance,
                "default_max_distance": default_max_distance,
            })

        return jsonify({
            "success": True,
            "authorized": False,
            "message": "You can't access the app. Please contact the administration team.",
            "match": best,
            "max_distance": max_distance,
            "default_max_distance": default_max_distance,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "authorized": False,
            "message": f"Face verification failed: {str(e)}"
        }), 400

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route('/api/vehicle/<license_plate>')
def get_vehicle(license_plate):
    """Get vehicle by license plate"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, license_plate, owner_name, owner_phone, owner_email, owner_address, image_filename, created_at
            FROM vehicles WHERE license_plate = ?
        """, (license_plate.upper(),))
        
        vehicle = cursor.fetchone()
        conn.close()
        
        if vehicle:
            return jsonify({
                "success": True,
                "vehicle": dict(vehicle)
            })
        else:
            return jsonify({"success": False, "message": "Vehicle not found"}), 404
            
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/vehicles')
def list_vehicles():
    """List all vehicles"""
    try:
        search = request.args.get('search', '').lower()
        
        conn = get_db()
        cursor = conn.cursor()
        
        if search:
            cursor.execute("""
                SELECT id, license_plate, owner_name, owner_phone, owner_email, owner_address, image_filename, created_at
                FROM vehicles 
                WHERE license_plate LIKE ? OR owner_name LIKE ? OR owner_phone LIKE ? OR owner_email LIKE ?
                ORDER BY created_at DESC
            """, (f'%{search}%', f'%{search}%', f'%{search}%', f'%{search}%'))
        else:
            cursor.execute("""
                SELECT id, license_plate, owner_name, owner_phone, owner_email, owner_address, image_filename, created_at
                FROM vehicles ORDER BY created_at DESC
            """)
        
        vehicles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            "success": True,
            "count": len(vehicles),
            "vehicles": vehicles
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get simple stats"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM vehicles")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {"total_vehicles": total}
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("SmartVision - Vehicle Registration")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Images: {IMAGES_DIR}")
    print("API: http://localhost:5000")
    print("Phone: http://YOUR_LOCAL_IP:5000")
    print("=" * 70)
    
    # Initialize detectors
    init_detectors()

    # Initialize face recognition (optional; endpoint lazy-loads anyway)
    init_face_recognizer()
    
    print("\n[OK] Ready to register vehicles!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
