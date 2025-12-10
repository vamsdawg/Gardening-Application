import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tensorflow as tf

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
        # Initialize YOLOv8 model (using nano version for speed)
        try:
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None
            
        # Initialize U-Net Model for Semantic Segmentation
        self.unet_model = None
        try:
            unet_path = 'models/lawn_segmentation_unet.h5'
            if os.path.exists(unet_path):
                self.unet_model = tf.keras.models.load_model(unet_path)
                print("U-Net model loaded successfully.")
            else:
                print(f"U-Net model not found at {unet_path}. Using heuristic fallback.")
        except Exception as e:
            print(f"Error loading U-Net model: {e}")

    def analyze_health(self, image: np.ndarray):
        """
        Analyze lawn health using Semantic Segmentation (U-Net or Heuristic).
        Calculates percentages of healthy green grass, dead grass, and bald spots.
        Returns the colored segmentation map as 'overlay'.
        """
        # Perform segmentation
        seg_result = self.segment_semantic(image)
        
        # Extract stats
        if "raw_mask" in seg_result:
            # U-Net based stats
            mask = seg_result["raw_mask"]
            # Classes: 1=Green, 2=Stressed, 3=Dormant/Dead, 4=Bald
            green_pixels = np.sum((mask == 1) | (mask == 2))
            dead_pixels = np.sum(mask == 3)
            bald_pixels = np.sum(mask == 4)
        else:
            # Heuristic based stats (passed back from segment_semantic fallback)
            # We need to ensure segment_semantic returns these or we recalculate them here.
            # Since segment_semantic fallback logic is complex, let's rely on what it returns.
            # Update: segment_semantic fallback below now returns 'stats'
            stats = seg_result.get("stats", {})
            green_pixels = stats.get("green_pixels", 0)
            dead_pixels = stats.get("dead_pixels", 0)
            bald_pixels = stats.get("bald_pixels", 0)

        # Normalize
        total_lawn_pixels = green_pixels + dead_pixels + bald_pixels
        
        if total_lawn_pixels > 0:
            green_frac = green_pixels / total_lawn_pixels
            dead_frac = dead_pixels / total_lawn_pixels
            bald_frac = bald_pixels / total_lawn_pixels
        else:
            green_frac, dead_frac, bald_frac = 0.0, 0.0, 0.0
            
        return {
            "green_coverage_pct": green_frac,
            "dead_grass_pct": dead_frac,
            "bald_spots_pct": bald_frac,
            "overlay": seg_result["segmentation_map"],
            "message": seg_result["message"]
        }

    def detect_weeds_yolo(self, image: np.ndarray):
        """
        Object Detection (YOLO): Identifies and locates objects using YOLOv8.
        """
        if self.yolo_model is None:
             return {
                "detected_objects": [],
                "message": "YOLO model failed to load."
            }
        
        try:
            # Run inference
            results = self.yolo_model(image)
            
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.yolo_model.names[cls_id]
                    
                    # Filter for relevant classes if using standard COCO model
                    # COCO classes: 58=potted plant, but standard model isn't great for weeds.
                    # We'll return everything for now so the user sees it works.
                    detected_objects.append({
                        "label": label,
                        "confidence": conf,
                        "box": box.xyxy[0].tolist()
                    })
            
            if not detected_objects:
                 return {
                    "detected_objects": [],
                    "message": "No objects detected by YOLOv8."
                }

            # Summarize detections
            counts = {}
            for obj in detected_objects:
                l = obj['label']
                counts[l] = counts.get(l, 0) + 1
            
            summary_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
            
            return {
                "detected_objects": detected_objects,
                "message": f"YOLOv8 detected: {summary_str}"
            }
            
        except Exception as e:
            return {
                "detected_objects": [],
                "message": f"Error running YOLOv8: {str(e)}"
            }

    def segment_semantic(self, image: np.ndarray):
        """
        Semantic Segmentation: Classifies every pixel using U-Net or fallback to HSV heuristics.
        Classes: 
        0: Background/Other (Black)
        1: Healthy Grass (Green)
        2: Stressed/Diseased Grass (Yellow)
        3: Dormant/Dead Grass (Light Brown)
        4: Dirt/Bald Spot (Dark Brown)
        5: Weed (Red - if detected)
        """
        # Define colors for visualization
        colors = {
            0: (0, 0, 0),       # Unclassified
            1: (0, 255, 0),     # Green Grass
            2: (255, 255, 0),   # Stressed/Diseased (Yellow)
            3: (210, 180, 140), # Dormant/Dead (Tan)
            4: (101, 67, 33),   # Dirt/Bald (Dark Brown)
            5: (255, 0, 0)      # Weed (Red)
        }
        
        # 1. Try U-Net Model Inference
        if self.unet_model:
            try:
                # Preprocess
                input_shape = (256, 256) 
                img_resized = cv2.resize(image, input_shape)
                img_norm = img_resized / 255.0
                img_batch = np.expand_dims(img_norm, axis=0)
                
                # Predict
                pred = self.unet_model.predict(img_batch, verbose=0)
                mask = np.argmax(pred[0], axis=-1)
                
                # Resize mask back to original size
                mask_full = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create colored overlay
                seg_map = np.zeros_like(image)
                for class_id, color in colors.items():
                    seg_map[mask_full == class_id] = color
                    
                return {
                    "segmentation_map": seg_map,
                    "raw_mask": mask_full,
                    "message": "U-Net segmentation applied."
                }
            except Exception as e:
                print(f"U-Net inference failed: {e}")
                # Fallthrough to heuristic
        
        # 2. Fallback: Heuristic Segmentation (HSV + Logic)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # A. Healthy Grass (Green): H=35-85
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # B. Stressed/Diseased (Yellowish): H=20-35
        lower_yellow = np.array([20, 40, 40])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # C. Brown/Dead/Bald logic (using Grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bald_prob_thresh = 100
        dead_upper_thresh = 255
        
        # Combined vegetation mask (Green + Yellow)
        veg_mask = cv2.bitwise_or(mask_green, mask_yellow)
        
        # Dead: Not Veg AND Brightness in range
        mask_dead = (veg_mask == 0) & (gray >= bald_prob_thresh) & (gray <= dead_upper_thresh)
        
        # Bald: Not Veg AND Dark
        mask_bald = (veg_mask == 0) & (gray < bald_prob_thresh)
        
        # Create map
        seg_map = np.zeros_like(image)
        
        # Apply layers (order matters for overlaps, though these should be disjoint)
        seg_map[mask_bald] = colors[4]      # Dirt
        seg_map[mask_dead] = colors[3]      # Dormant
        seg_map[mask_yellow > 0] = colors[2] # Stressed
        seg_map[mask_green > 0] = colors[1]  # Healthy
        
        # Calculate stats for fallback
        green_pixels = np.count_nonzero(mask_green) + np.count_nonzero(mask_yellow)
        dead_pixels = np.count_nonzero(mask_dead)
        bald_pixels = np.count_nonzero(mask_bald)
        
        return {
            "segmentation_map": seg_map,
            "stats": {
                "green_pixels": green_pixels,
                "dead_pixels": dead_pixels,
                "bald_pixels": bald_pixels
            },
            "message": "Heuristic segmentation (U-Net model not found)."
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
