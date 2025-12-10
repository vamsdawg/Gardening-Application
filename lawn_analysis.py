import cv2
import numpy as np
from PIL import Image

class LawnAnalyzer:
    """
    Computer vision models for lawn analysis.
    
    Techniques implemented or supported:
    1. Color/Texture Analysis (HSV): Used for robust vegetation mapping, distinguishing between healthy/stressed grass.
    2. Object Detection (YOLO, Faster R-CNN): Identifies and locates weeds (e.g., broadleaf) or specific grass types.
    3. Semantic Segmentation: Classifies every pixel (e.g., grass, weed, dirt, bare spot) for detailed coverage analysis.
    4. Deep Convolutional Neural Networks (DCNN): Pattern-based recognition of plant stresses and species identification.
    """
    
    def __init__(self):
        pass

    def segment_green_hsv(self, image: np.ndarray, lower_h=35, upper_h=85, sat_min=40, val_min=40, morph_k=7):
        """
        Color/Texture Analysis: HSV-based green segmentation.
        Returns binary mask (uint8 0/255).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array([lower_h, sat_min, val_min], dtype=np.uint8)
        upper = np.array([upper_h, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def analyze_health(self, image: np.ndarray):
        """
        Analyze lawn health using Color/Texture Analysis (HSV).
        Calculates percentages of healthy green grass, dead grass, and bald spots.
        """
        # Segmentation parameters
        lower_h, upper_h, sat_min, val_min, morph_k = 35, 85, 40, 40, 7
        bald_prob_thresh = 145
        dead_upper_thresh = 165
        
        mask = self.segment_green_hsv(image, lower_h, upper_h, sat_min, val_min, morph_k)
        
        # Calculate raw pixel counts
        green_pixels = np.count_nonzero(mask)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Dead Grass (Light Brown): Not green, and brightness between bald_thresh and dead_upper
        dead_mask = (mask == 0) & (gray >= bald_prob_thresh) & (gray < dead_upper_thresh)
        dead_pixels = np.count_nonzero(dead_mask)
        
        # Bald Spots (Dark Brown/Soil): Not green, and brightness < bald_thresh
        bald_mask = (mask == 0) & (gray < bald_prob_thresh)
        bald_pixels = np.count_nonzero(bald_mask)
        
        # Normalize to lawn area only (Green + Dead + Bald)
        total_lawn_pixels = green_pixels + dead_pixels + bald_pixels
        
        if total_lawn_pixels > 0:
            green_frac = green_pixels / total_lawn_pixels
            dead_frac = dead_pixels / total_lawn_pixels
            bald_frac = bald_pixels / total_lawn_pixels
        else:
            green_frac = 0.0
            dead_frac = 0.0
            bald_frac = 0.0
            
        return {
            "green_coverage_pct": green_frac,
            "dead_grass_pct": dead_frac,
            "bald_spots_pct": bald_frac,
            "masks": {
                "green": mask,
                "dead": dead_mask,
                "bald": bald_mask
            }
        }

    def overlay_masks(self, image: np.ndarray, masks: dict):
        """
        Create visualization overlay for the analyzed lawn.
        """
        # 1. Healthy Green Grass -> Green (0, 200, 0)
        overlay = self._apply_overlay(image, masks['green'], color=(0, 200, 0), alpha=0.4)
        # 2. Dead Grass -> Yellow (255, 255, 0)
        overlay = self._apply_overlay(overlay, masks['dead'], color=(255, 255, 0), alpha=0.4)
        # 3. Bald Spots -> Red (255, 0, 0)
        overlay = self._apply_overlay(overlay, masks['bald'], color=(255, 0, 0), alpha=0.4)
        return overlay

    def _apply_overlay(self, image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5):
        colored = image.copy()
        # Ensure mask is boolean or 0/255
        mask_bool = mask > 0
        if mask_bool.any():
            colored[mask_bool] = (np.array(color) * alpha + colored[mask_bool] * (1 - alpha)).astype(np.uint8)
        return colored

    def detect_weeds_yolo(self, image: np.ndarray):
        """
        Object Detection (YOLO): Identifies and locates weeds (e.g., broadleaf) 
        or specific grass types with bounding boxes.
        
        Note: This requires a trained YOLO model. Currently returns a placeholder.
        """
        # Placeholder for YOLO inference
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/weights.pt')
        # results = model(image)
        return {
            "detected_objects": [],
            "message": "YOLO model not loaded. Placeholder for weed detection."
        }

    def segment_semantic(self, image: np.ndarray):
        """
        Semantic Segmentation: Classifies every pixel (e.g., grass, weed, dirt, bare spot) 
        for detailed coverage analysis.
        
        Note: This requires a trained segmentation model (e.g., U-Net, DeepLab).
        """
        return {
            "segmentation_map": None,
            "message": "Semantic segmentation model not loaded."
        }

    def identify_species_dcnn(self, image: np.ndarray):
        """
        Deep Convolutional Neural Networks (DCNN): Pattern-based recognition 
        of plant stresses and species identification.
        """
        return {
            "species": "Unknown",
            "confidence": 0.0,
            "message": "DCNN model not loaded."
        }
