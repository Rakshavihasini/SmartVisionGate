import string
import easyocr
import re
import cv2
import numpy as np

# Initialize the OCR reader with optimized settings for license plates
reader = easyocr.Reader(
    ['en'], 
    gpu=False,
    model_storage_directory='./easyocr_models',
    download_enabled=True,
    verbose=False
)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def preprocess_license_plate(image):
    """
    Apply multiple preprocessing techniques to enhance license plate readability.
    
    Args:
        image: Input image (numpy array or PIL Image)
    
    Returns:
        list: Multiple preprocessed versions of the image
    """
    # Convert PIL Image to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Convert to BGR if needed (for cv2)
    if len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        img = image.copy()
    
    preprocessed_images = []
    
    # 1. Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize for better OCR (upscale if too small)
    height, width = gray.shape
    if height < 100:
        scale_factor = 100 / height
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 3. Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 5. Binary thresholding - Otsu's method
    _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh_otsu)
    
    # 6. Adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(thresh_adaptive)
    
    # 7. Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(morph)
    
    # 8. Sharpening
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    preprocessed_images.append(sharpened)
    
    # 9. Inverted binary (for dark text on light background)
    _, thresh_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh_inv)
    
    # 10. Enhanced grayscale (original)
    preprocessed_images.append(enhanced)
    
    return preprocessed_images


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image with preprocessing.

    Args:
        license_plate_crop: Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    
    # Get multiple preprocessed versions
    preprocessed_images = preprocess_license_plate(license_plate_crop)
    
    all_detections = []
    
    # Try OCR on each preprocessed version with optimized parameters
    for idx, processed_img in enumerate(preprocessed_images):
        detections = reader.readtext(
            processed_img,
            detail=1,
            paragraph=False,
            batch_size=1,
            workers=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Only alphanumeric
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.3,
            canvas_size=2560,
            mag_ratio=1.5,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.5,
            width_ths=0.5,
            add_margin=0.1
        )
        
        for bbox, text, score in detections:
            all_detections.append((bbox, text, score, idx))
    
    # Sort all detections by confidence score
    all_detections = sorted(all_detections, key=lambda x: x[2], reverse=True)
    
    for bbox, text, score, img_idx in all_detections:
        if score < 0.1:
            continue
        
        # Clean up the text
        text = text.upper().replace(" ", "").replace("-", "")
        
        # Remove special characters
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Basic validation - license plate should have reasonable length
        if 6 <= len(text) <= 12:
            print(f"Valid license plate detected: {text} (confidence: {score:.2f}, preprocessing version: {img_idx})")
            return text, score
    
    return None, None


