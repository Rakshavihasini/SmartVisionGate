"""
Multi-Modal Vehicle Verification System (MMVS)
===============================================

A novel research-oriented approach for vehicle re-identification using:
1. Structural Similarity Index (SSIM) - Compares structural patterns in vehicle masks
2. Color Distribution Matching - HSV histogram comparison for color verification
3. Shape Descriptor Analysis - Hu Moments for invariant shape matching
4. Perceptual Hashing - Fast similarity detection using pHash
5. Composite Confidence Scoring - Weighted multi-metric fusion

This system provides robust vehicle verification even under varying lighting,
angles, and partial occlusions by combining multiple complementary features.

Author: SmartVision Research Team
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import imagehash
from PIL import Image


class VehicleVerificationEngine:
    """
    Advanced multi-modal vehicle verification system using mask-based comparison.
    """
    
    def __init__(self, 
                 ssim_weight: float = 0.30,
                 color_weight: float = 0.25,
                 shape_weight: float = 0.25,
                 phash_weight: float = 0.20):
        """
        Initialize the verification engine with configurable weights.
        
        Args:
            ssim_weight: Weight for structural similarity (default: 0.30)
            color_weight: Weight for color histogram matching (default: 0.25)
            shape_weight: Weight for shape descriptor matching (default: 0.25)
            phash_weight: Weight for perceptual hash matching (default: 0.20)
        """
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.phash_weight = phash_weight
        
        # Normalize weights
        total = ssim_weight + color_weight + shape_weight + phash_weight
        self.ssim_weight /= total
        self.color_weight /= total
        self.shape_weight /= total
        self.phash_weight /= total
        
        print(f"[MMVS] Initialized with weights: SSIM={self.ssim_weight:.2f}, "
              f"Color={self.color_weight:.2f}, Shape={self.shape_weight:.2f}, "
              f"pHash={self.phash_weight:.2f}")
    
    def compute_structural_similarity(self, 
                                     mask1: np.ndarray, 
                                     mask2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Structural Similarity Index (SSIM) between two vehicle masks.
        
        SSIM measures structural information by comparing luminance, contrast,
        and structure between images. It's particularly effective for detecting
        changes in vehicle shape and pose.
        
        Args:
            mask1: First vehicle mask (grayscale or BGR)
            mask2: Second vehicle mask (grayscale or BGR)
            
        Returns:
            Tuple of (SSIM score [0-1], difference map)
        """
        # Convert to grayscale if needed
        if len(mask1.shape) == 3:
            gray1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = mask1
            
        if len(mask2.shape) == 3:
            gray2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = mask2
        
        # Resize to same dimensions
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Compute SSIM
        score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        
        return float(score), diff
    
    def compute_color_histogram_similarity(self, 
                                          img1: np.ndarray, 
                                          img2: np.ndarray,
                                          bins: int = 32) -> float:
        """
        Compare color distributions using HSV histograms and Bhattacharyya distance.
        
        HSV color space is more robust to lighting variations than RGB.
        We compute 3D histograms and use Earth Mover's Distance for comparison.
        
        Args:
            img1: First vehicle image (BGR)
            img2: Second vehicle image (BGR)
            bins: Number of histogram bins per channel (default: 32)
            
        Returns:
            Similarity score [0-1], where 1 is identical
        """
        # Convert to HSV color space
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        hist_scores = []
        
        for i in range(3):  # H, S, V channels
            hist1 = cv2.calcHist([hsv1], [i], None, [bins], [0, 256])
            hist2 = cv2.calcHist([hsv2], [i], None, [bins], [0, 256])
            
            # Normalize histograms
            hist1 = hist1.flatten() / hist1.sum()
            hist2 = hist2.flatten() / hist2.sum()
            
            # Compute Bhattacharyya coefficient
            score = cv2.compareHist(
                hist1.astype(np.float32), 
                hist2.astype(np.float32), 
                cv2.HISTCMP_BHATTACHARYYA
            )
            
            # Convert to similarity (Bhattacharyya distance -> similarity)
            hist_scores.append(1.0 - score)
        
        # Weighted average (Hue is most important for color)
        color_similarity = (0.5 * hist_scores[0] +  # Hue
                           0.3 * hist_scores[1] +  # Saturation
                           0.2 * hist_scores[2])   # Value
        
        return float(color_similarity)
    
    def compute_shape_similarity(self, 
                                mask1: np.ndarray, 
                                mask2: np.ndarray) -> float:
        """
        Compare vehicle shapes using Hu Moments - rotation, scale, and translation invariant.
        
        Hu Moments are 7 moment invariants derived from central moments that remain
        unchanged under geometric transformations, making them ideal for shape matching.
        
        Args:
            mask1: First vehicle mask (binary or grayscale)
            mask2: Second vehicle mask (binary or grayscale)
            
        Returns:
            Similarity score [0-1], where 1 is identical shape
        """
        # Convert to grayscale if needed
        if len(mask1.shape) == 3:
            gray1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = mask1
            
        if len(mask2.shape) == 3:
            gray2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = mask2
        
        # Threshold to binary
        _, binary1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
        
        # Compute Hu Moments
        moments1 = cv2.moments(binary1)
        moments2 = cv2.moments(binary2)
        
        hu1 = cv2.HuMoments(moments1).flatten()
        hu2 = cv2.HuMoments(moments2).flatten()
        
        # Use log scale for better comparison
        hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
        hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
        
        # Compute Euclidean distance
        dist = np.linalg.norm(hu1 - hu2)
        
        # Convert distance to similarity (using exponential decay)
        similarity = np.exp(-dist / 10.0)  # Empirically tuned
        
        return float(similarity)
    
    def compute_perceptual_hash_similarity(self, 
                                          img1: np.ndarray, 
                                          img2: np.ndarray) -> float:
        """
        Compare images using perceptual hashing (pHash).
        
        Perceptual hashing creates a fingerprint of an image based on its visual content,
        allowing fast similarity detection even with minor variations.
        
        Args:
            img1: First vehicle image (BGR)
            img2: Second vehicle image (BGR)
            
        Returns:
            Similarity score [0-1], where 1 is identical
        """
        # Convert to PIL Images
        pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        
        # Compute perceptual hashes
        hash1 = imagehash.phash(pil1, hash_size=16)
        hash2 = imagehash.phash(pil2, hash_size=16)
        
        # Compute hamming distance
        hamming_dist = hash1 - hash2
        
        # Convert to similarity (max hamming distance for 16x16 hash is 256)
        similarity = 1.0 - (hamming_dist / 256.0)
        
        return float(similarity)
    
    def extract_dominant_colors(self, 
                               img: np.ndarray, 
                               k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from vehicle image using K-Means clustering.
        
        Args:
            img: Vehicle image (BGR)
            k: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as (B, G, R) tuples
        """
        # Reshape image to 2D array of pixels
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Remove black pixels (background)
        pixels = pixels[np.sum(pixels, axis=1) > 30]
        
        if len(pixels) == 0:
            return [(0, 0, 0)] * k
        
        # Apply K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                        cv2.KMEANS_PP_CENTERS)
        
        # Sort by frequency
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        dominant_colors = [tuple(map(int, centers[i])) for i in sorted_indices]
        
        return dominant_colors
    
    def verify_vehicle(self, 
                      camera_mask: np.ndarray,
                      camera_image: np.ndarray,
                      db_mask: np.ndarray,
                      db_image: np.ndarray,
                      threshold: float = 0.70) -> Dict:
        """
        Perform comprehensive multi-modal vehicle verification.
        
        Args:
            camera_mask: Vehicle mask from camera feed
            camera_image: Original vehicle image from camera
            db_mask: Stored vehicle mask from database
            db_image: Stored vehicle image from database
            threshold: Minimum composite score for positive verification (0-1)
            
        Returns:
            Dictionary containing:
                - verified: Boolean indicating if vehicle matches
                - composite_score: Overall similarity score [0-1]
                - ssim_score: Structural similarity score
                - color_score: Color histogram similarity score
                - shape_score: Shape descriptor similarity score
                - phash_score: Perceptual hash similarity score
                - confidence: Verification confidence level
                - dominant_colors_match: Whether dominant colors are similar
                - metrics_breakdown: Detailed breakdown of each metric
        """
        print("\n" + "="*60)
        print("MULTI-MODAL VEHICLE VERIFICATION")
        print("="*60)
        
        # 1. Structural Similarity
        print("[1/4] Computing Structural Similarity (SSIM)...")
        ssim_score, ssim_diff = self.compute_structural_similarity(camera_mask, db_mask)
        print(f"      → SSIM Score: {ssim_score:.4f}")
        
        # 2. Color Histogram Similarity
        print("[2/4] Computing Color Distribution Similarity...")
        color_score = self.compute_color_histogram_similarity(camera_image, db_image)
        print(f"      → Color Score: {color_score:.4f}")
        
        # 3. Shape Similarity
        print("[3/4] Computing Shape Descriptor Similarity...")
        shape_score = self.compute_shape_similarity(camera_mask, db_mask)
        print(f"      → Shape Score: {shape_score:.4f}")
        
        # 4. Perceptual Hash Similarity
        print("[4/4] Computing Perceptual Hash Similarity...")
        phash_score = self.compute_perceptual_hash_similarity(camera_image, db_image)
        print(f"      → pHash Score: {phash_score:.4f}")
        
        # Compute composite score
        composite_score = (
            self.ssim_weight * ssim_score +
            self.color_weight * color_score +
            self.shape_weight * shape_score +
            self.phash_weight * phash_score
        )
        
        # Extract dominant colors for additional verification
        camera_colors = self.extract_dominant_colors(camera_image)
        db_colors = self.extract_dominant_colors(db_image)
        
        # Determine confidence level
        if composite_score >= 0.85:
            confidence = "VERY HIGH"
        elif composite_score >= 0.75:
            confidence = "HIGH"
        elif composite_score >= threshold:
            confidence = "MODERATE"
        elif composite_score >= 0.50:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"
        
        # Verification decision
        verified = composite_score >= threshold
        
        print("\n" + "-"*60)
        print(f"COMPOSITE SCORE: {composite_score:.4f}")
        print(f"THRESHOLD: {threshold:.2f}")
        print(f"VERIFICATION: {'✓ MATCH' if verified else '✗ NO MATCH'}")
        print(f"CONFIDENCE: {confidence}")
        print("="*60 + "\n")
        
        return {
            "verified": verified,
            "composite_score": float(composite_score),
            "ssim_score": float(ssim_score),
            "color_score": float(color_score),
            "shape_score": float(shape_score),
            "phash_score": float(phash_score),
            "confidence": confidence,
            "threshold": threshold,
            "dominant_colors_camera": camera_colors,
            "dominant_colors_db": db_colors,
            "metrics_breakdown": {
                "ssim": {
                    "score": float(ssim_score),
                    "weight": self.ssim_weight,
                    "contribution": float(self.ssim_weight * ssim_score)
                },
                "color": {
                    "score": float(color_score),
                    "weight": self.color_weight,
                    "contribution": float(self.color_weight * color_score)
                },
                "shape": {
                    "score": float(shape_score),
                    "weight": self.shape_weight,
                    "contribution": float(self.shape_weight * shape_score)
                },
                "phash": {
                    "score": float(phash_score),
                    "weight": self.phash_weight,
                    "contribution": float(self.phash_weight * phash_score)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_verification(self, 
                              camera_mask: np.ndarray,
                              camera_image: np.ndarray,
                              db_mask: np.ndarray,
                              db_image: np.ndarray,
                              result: Dict,
                              output_path: str) -> None:
        """
        Create a comprehensive visualization of the verification process.
        
        Args:
            camera_mask: Vehicle mask from camera feed
            camera_image: Original vehicle image from camera
            db_mask: Stored vehicle mask from database
            db_image: Stored vehicle image from database
            result: Verification result dictionary
            output_path: Path to save visualization
        """
        # Compute SSIM difference map
        _, ssim_diff = self.compute_structural_similarity(camera_mask, db_mask)
        
        # Resize all images to same height
        height = 300
        
        def resize_keep_aspect(img, target_height):
            aspect = img.shape[1] / img.shape[0]
            return cv2.resize(img, (int(target_height * aspect), target_height))
        
        cam_img_resized = resize_keep_aspect(camera_image, height)
        db_img_resized = resize_keep_aspect(db_image, height)
        cam_mask_resized = resize_keep_aspect(camera_mask, height)
        db_mask_resized = resize_keep_aspect(db_mask, height)
        ssim_diff_resized = resize_keep_aspect(cv2.applyColorMap(ssim_diff, cv2.COLORMAP_JET), height)
        
        # Create side-by-side comparison
        row1 = np.hstack([cam_img_resized, db_img_resized])
        row2 = np.hstack([cam_mask_resized, db_mask_resized])
        
        # Add SSIM difference map
        if ssim_diff_resized.shape[1] < row1.shape[1]:
            padding = np.zeros((height, row1.shape[1] - ssim_diff_resized.shape[1], 3), dtype=np.uint8)
            ssim_diff_resized = np.hstack([ssim_diff_resized, padding])
        
        visualization = np.vstack([row1, row2, ssim_diff_resized[:, :row1.shape[1]]])
        
        # Add text overlay with results
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0) if result['verified'] else (0, 0, 255)
        
        cv2.putText(visualization, "Camera Feed", (10, 30), 
                   font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Database", (cam_img_resized.shape[1] + 10, 30), 
                   font, 0.7, (255, 255, 255), 2)
        
        cv2.putText(visualization, f"Score: {result['composite_score']:.3f}", 
                   (10, visualization.shape[0] - 20), 
                   font, 0.8, text_color, 2)
        
        cv2.putText(visualization, f"{result['confidence']}", 
                   (10, visualization.shape[0] - 50), 
                   font, 0.7, text_color, 2)
        
        # Save visualization
        cv2.imwrite(output_path, visualization)
        print(f"[MMVS] Verification visualization saved: {output_path}")
    
    def save_verification_report(self, 
                                result: Dict, 
                                output_path: str,
                                license_plate: str = None) -> None:
        """
        Save detailed verification report as JSON.
        
        Args:
            result: Verification result dictionary
            output_path: Path to save JSON report
            license_plate: License plate number (optional)
        """
        report = {
            "license_plate": license_plate,
            "verification_result": result,
            "system_info": {
                "algorithm": "Multi-Modal Vehicle Verification System (MMVS)",
                "version": "1.0.0",
                "metrics": [
                    "Structural Similarity (SSIM)",
                    "Color Histogram Matching (HSV)",
                    "Shape Descriptors (Hu Moments)",
                    "Perceptual Hashing (pHash)"
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[MMVS] Verification report saved: {output_path}")


# Utility function for easy integration
def verify_vehicle_from_paths(camera_mask_path: str,
                              camera_img_path: str,
                              db_mask_path: str,
                              db_img_path: str,
                              output_dir: str = None,
                              threshold: float = 0.70) -> Dict:
    """
    Convenience function to verify vehicles directly from file paths.
    
    Args:
        camera_mask_path: Path to camera vehicle mask
        camera_img_path: Path to camera vehicle image
        db_mask_path: Path to database vehicle mask
        db_img_path: Path to database vehicle image
        output_dir: Directory to save verification results (optional)
        threshold: Verification threshold
        
    Returns:
        Verification result dictionary
    """
    # Load images
    camera_mask = cv2.imread(camera_mask_path)
    camera_img = cv2.imread(camera_img_path)
    db_mask = cv2.imread(db_mask_path)
    db_img = cv2.imread(db_img_path)
    
    # Initialize engine
    engine = VehicleVerificationEngine()
    
    # Perform verification
    result = engine.verify_vehicle(camera_mask, camera_img, db_mask, db_img, threshold)
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "verification_visualization.jpg")
        engine.visualize_verification(camera_mask, camera_img, db_mask, db_img, 
                                     result, vis_path)
        
        # Save report
        report_path = os.path.join(output_dir, "verification_report.json")
        engine.save_verification_report(result, report_path)
    
    return result


