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
        # Initialize YOLOv8 model
        # Prioritize our custom trained model if it exists
        try:
            custom_model_path = 'models/weed_detector.pt'
            if os.path.exists(custom_model_path):
                print(f"Loading custom weed detector from {custom_model_path}...")
                self.yolo_model = YOLO(custom_model_path)
            else:
                print("Custom model not found. Loading standard YOLOv8n model...")
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
        Uses 'models/weed_detector.pt' if available, otherwise falls back to standard YOLOv8n.
        """
        detected_objects = []
        yolo_message = ""

        # 1. Run YOLO if available
        if self.yolo_model:
            try:
                # Run inference with a lower confidence threshold to catch more potential weeds
                # Custom model might need a lower threshold if trained briefly
                print("Running YOLO inference...")
                results = self.yolo_model(image, conf=0.18)
                
                for r in results:
                    boxes = r.boxes
                    print(f"YOLO found {len(boxes)} boxes.")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.yolo_model.names[cls_id]
                        
                        print(f"Detected: {label} ({conf:.2f})")
                        
                        # If using custom model, classes are likely 'crop' and 'weed'
                        # If using standard model, we return all detections
                        detected_objects.append({
                            "label": label,
                            "confidence": conf,
                            "box": box.xyxy[0].tolist()
                        })
                
                if not detected_objects:
                     yolo_message = "No objects detected by YOLOv8."
                else:
                    # Summarize detections
                    counts = {}
                    for obj in detected_objects:
                        l = obj['label']
                        counts[l] = counts.get(l, 0) + 1
                    
                    summary_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    yolo_message = f"YOLOv8 detected: {summary_str}"
            
            except Exception as e:
                yolo_message = f"Error running YOLOv8: {str(e)}"
        else:
             yolo_message = "YOLO model failed to load."

        # 2. Fallback/Augmentation: CV-based Weed Detection
        # If YOLO misses weeds (common with standard models), try heuristic
        try:
            cv_weeds = self._detect_weeds_cv(image)
            if cv_weeds:
                detected_objects.extend(cv_weeds)
                yolo_message += f" | CV detected {len(cv_weeds)} potential weeds."
        except Exception as e:
            print(f"CV weed detection error: {e}")

        return {
            "detected_objects": detected_objects,
            "message": yolo_message
        }

    def _detect_weeds_cv(self, image: np.ndarray):
        """
        Heuristic weed detection using computer vision (Morphological operations on Green Mask).
        Useful when grass is dormant (brown) and weeds are green, or for broadleaf weeds in green grass.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define green range (broad range for vegetation)
        # Adjusting slightly to catch more variety
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological Opening to remove thin grass blades/noise
        # Kernel size 3x3 or 5x5. 
        kernel = np.ones((3,3), np.uint8) 
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilate to merge close components
        dilated = cv2.dilate(opened, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        min_area = 50   # Minimum area (pixels)
        max_area = (image.shape[0] * image.shape[1]) * 0.3 # Max 30% of screen (avoid detecting whole lawn)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detected.append({
                    "label": "Weed (CV)",
                    "confidence": 0.65,
                    "box": [float(x), float(y), float(x+w), float(y+h)]
                })
        
        return detected

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
