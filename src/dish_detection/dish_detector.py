"""
Dish Detection Model for KichiKichi Conveyor Belt System
Detects and classifies dish colors: normal, red, yellow, advertisement
"""

import cv2
import numpy as np
import torch
import sys
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DishDetection:
    """Represents a detected dish"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    dish_type: str  # 'normal_dish', 'red_dish', 'yellow_dish', 'advertisement_dish'
    center_point: Tuple[int, int]
    timestamp: datetime
    counting_point: Optional[Tuple[int, int]] = None  # 1/3 vertical center point for counting
    
    def __post_init__(self):
        """Calculate the counting point (1/3 from bottom center) after initialization"""
        if self.counting_point is None:
            x1, y1, x2, y2 = self.bbox
            center_x = (x1 + x2) // 2
            # Calculate 1/3 point from bottom of bbox (near bottom edge)
            counting_y = y2 - int((y2 - y1) * (1/3))
            self.counting_point = (center_x, counting_y)
    
class DishDetector:
    """
    Main dish detection class using YOLO for object detection
    and color analysis for classification
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        
        # Different confidence thresholds for different dish types
        # Lower threshold for red dishes to catch more red detections
        self.dish_confidence_thresholds = {
            'normal_dish': 0.5,
            'red_dish': 0.4,  # Lower threshold for red dishes
            'yellow_dish': 0.4
        }
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLOv11 model with enhanced error handling
        self.model = None
        self.model_available = False
        
        if model_path:
            try:
                self.model = YOLO(model_path, verbose=False)
                # Configure model to use GPU ID 2
                # self.model.to('cuda:2')
                self.model_available = True
                self.logger.info(f"✅ Loaded YOLOv11 dish detection model from: {model_path}")
                # self.logger.info(f"🔥 Model configured to use GPU ID 2")
                self.logger.info(f"📊 Model classes: {self.model.names}")
            except Exception as e:
                self.logger.error(f"❌ Error loading model from {model_path}: {e}")
                # try:
                #     # Fallback to default model
                #     self.model = YOLO('yolo11n.pt', verbose=False)
                #     # Configure fallback model to use GPU ID 2
                #     self.model.to('cuda:2')
                #     self.model_available = True
                #     self.logger.warning("⚠️ Using fallback YOLOv11 model on GPU ID 2")
                # except Exception as e2:
                # self.logger.error(f"❌ Fallback model also failed: {e2}")
                self.logger.warning("🔄 Running in mock detection mode - video streaming will still work")
                self.model_available = False
        else:
            try:
                # Use default YOLOv11 model
                self.model = YOLO('yolo11n.pt', verbose=False)
                # Configure default model to use GPU ID 2
                # self.model.to('cuda:2')
                self.model_available = True
                # self.logger.info("Using default YOLOv11 model on GPU ID 2")
            except Exception as e:
                # self.logger.error(f"❌ Default model failed: {e}")
                # self.logger.warning("🔄 Running in mock detection mode - video streaming will still work")
                self.model_available = False
        
        # Setup dish class mapping from model
        self._setup_dish_classes()
            
        # Enhanced color ranges for dish classification (HSV) - used as fallback
        # Expanded ranges to catch more red variations
        self.color_ranges = {
            'red_dish': {
                # Lower range for red (0-15 degrees) - expanded range
                'lower1': np.array([0, 30, 40]),     # Lower saturation and value thresholds
                'upper1': np.array([15, 255, 255]), # Expanded hue range
                # Upper range for red (165-180 degrees) - expanded range  
                'lower2': np.array([165, 30, 40]),   # Lower saturation and value thresholds
                'upper2': np.array([180, 255, 255]), # Expanded hue range
                # Middle range for orange-red (15-25 degrees) - catches reddish-orange dishes
                'lower3': np.array([10, 30, 40]),
                'upper3': np.array([25, 255, 255])
            },
            'yellow_dish': {
                'lower': np.array([20, 40, 40]),     # Slightly lower thresholds
                'upper': np.array([35, 255, 255])    # Expanded range
            }
        }
        
        self.logger.info("DishDetector initialized successfully")
    
    def _setup_dish_classes(self):
        """Setup dish class mapping from trained model"""
        # Default dish class mapping
        self.dish_classes = {
            0: 'normal_dish',
            1: 'red_dish', 
            2: 'yellow_dish',
            3: 'advertisement_dish'
        }
        
        # Use model's actual class names if available
        if hasattr(self.model, 'names') and self.model.names:
            self.logger.info(f"🏷️ Model has {len(self.model.names)} classes")
            
            # Try to map model classes to our dish types
            model_classes = {}
            for class_id, class_name in self.model.names.items():
                class_name_lower = class_name.lower().strip()
                
                # Map based on common naming patterns
                if 'normal' in class_name_lower or 'white' in class_name_lower or 'regular' in class_name_lower:
                    model_classes[class_id] = 'normal_dish'
                elif 'red' in class_name_lower:
                    model_classes[class_id] = 'red_dish'
                elif 'yellow' in class_name_lower:
                    model_classes[class_id] = 'yellow_dish'
                elif 'advertisement' in class_name_lower or 'banner' in class_name_lower or 'ad' in class_name_lower:
                    model_classes[class_id] = 'advertisement_dish'
                else:
                    # Try to use the class name directly if it matches our expected format
                    if class_name_lower in ['normal_dish', 'red_dish', 'yellow_dish', 'advertisement_dish']:
                        model_classes[class_id] = class_name_lower
                    else:
                        # Default unknown classes to normal_dish
                        model_classes[class_id] = 'normal_dish'
                        self.logger.warning(f"⚠️ Unknown dish class '{class_name}' mapped to 'normal_dish'")
            
            if model_classes:
                self.dish_classes = model_classes
                self.use_model_classes = True
                self.logger.info(f"🎯 Using model-based dish classification: {self.dish_classes}")
            else:
                self.use_model_classes = False
                self.logger.warning("⚠️ Could not map model classes, using color-based classification")
        else:
            self.use_model_classes = False
            self.logger.info("🎨 Using color-based dish classification (fallback)")
    
    def detect_dishes(self, frame: np.ndarray) -> List[DishDetection]:
        """
        Detect dishes in the given frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of DishDetection objects
        """
        detections = []
        
        # Return empty list if model is not available (allows video streaming to continue)
        if not self.model_available or self.model is None:
            self.logger.debug("Model not available, returning empty detections (video will still stream)")
            return detections
        
        try:
            # Run YOLO detection with lowest threshold to catch all possible detections
            min_threshold = min(self.dish_confidence_thresholds.values())
            results = self.model(frame, conf=min_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates with safe tensor access
                        try:
                            if hasattr(box.xyxy, 'cpu'):
                                xyxy = box.xyxy[0].cpu().numpy() if len(box.xyxy) > 0 else None
                            else:
                                xyxy = box.xyxy[0] if len(box.xyxy) > 0 else None
                                
                            if xyxy is None:
                                continue
                                
                            x1, y1, x2, y2 = xyxy.astype(int)
                        except (IndexError, AttributeError, TypeError):
                            self.logger.warning("⚠️ Failed to extract bounding box coordinates")
                            continue
                        
                        # Get actual confidence from model with safe tensor access
                        try:
                            if hasattr(box.conf, 'cpu'):
                                actual_confidence = float(box.conf[0].cpu().numpy().item()) if len(box.conf) > 0 else 0.0
                            else:
                                actual_confidence = float(box.conf[0].item()) if len(box.conf) > 0 and hasattr(box.conf[0], 'item') else float(box.conf[0]) if len(box.conf) > 0 else 0.0
                        except (IndexError, AttributeError, TypeError):
                            actual_confidence = 0.0
                        
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Classify dish type - use model classes if available, otherwise color analysis
                        if self.use_model_classes and hasattr(box, 'cls'):
                            # Use model's predicted class with safe tensor access
                            try:
                                if hasattr(box.cls, 'cpu'):
                                    cls = box.cls[0].cpu().numpy().item() if len(box.cls) > 0 else 0
                                else:
                                    cls = box.cls[0].item() if len(box.cls) > 0 and hasattr(box.cls[0], 'item') else box.cls[0] if len(box.cls) > 0 else 0
                                class_id = int(cls)
                                model_prediction = self.dish_classes.get(class_id, 'normal_dish')
                                
                                # Also do color-based classification for comparison
                                dish_roi = frame[y1:y2, x1:x2]
                                # color_prediction = self._classify_dish_color(dish_roi)
                                
                                # Use model prediction but log disagreements for debugging
                                dish_type = model_prediction
                                # if model_prediction != color_prediction:
                                #     self.logger.info(f"🔍 CLASSIFICATION DISAGREEMENT: Model={model_prediction} vs Color={color_prediction} (confidence={actual_confidence:.3f})")
                                #     # For red dishes, trust color analysis if model says normal but color says red
                                #     if model_prediction == 'normal_dish' and color_prediction == 'red_dish':
                                #         dish_type = 'red_dish'
                                #         self.logger.info(f"🔴 OVERRIDE: Using color prediction 'red_dish' over model 'normal_dish'")
                                
                                # self.logger.debug(f"🎯 Final classification: {dish_type} (model={model_prediction}, color={color_prediction})")
                            except (IndexError, AttributeError, TypeError):
                                # Fallback to color-based classification
                                dish_roi = frame[y1:y2, x1:x2]
                                # dish_type = self._classify_dish_color(dish_roi)
                                # self.logger.debug(f"🎨 Color-based classification (model failed): {dish_type}")
                        else:
                            # Fallback to color-based classification
                            dish_roi = frame[y1:y2, x1:x2]
                            # dish_type = self._classify_dish_color(dish_roi)
                            # self.logger.debug(f"🎨 Color-based classification: {dish_type}")
                        
                        # Check if detection meets confidence threshold for this dish type
                        required_confidence = self.dish_confidence_thresholds.get(dish_type, 0.3)
                        if actual_confidence >= required_confidence:
                            # Enhanced logging for red dish detection monitoring
                            # if dish_type == 'red_dish':
                            #     self.logger.info(f"🔴 RED DISH DETECTED: at ({center_x},{center_y}) confidence={actual_confidence:.3f} (req: {required_confidence})")
                            # elif dish_type == 'yellow_dish':
                            #     self.logger.info(f"🟡 YELLOW DISH DETECTED: at ({center_x},{center_y}) confidence={actual_confidence:.3f}")
                            
                            detection = DishDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=1.0,  # Display confidence is still 100%
                                dish_type=dish_type,
                                center_point=(center_x, center_y),
                                timestamp=datetime.now()
                            )
                            
                            detections.append(detection)
                        else:
                            # Special logging for rejected red dishes to debug threshold issues
                            if dish_type == 'red_dish':
                                self.logger.warning(f"🚫 REJECTED RED DISH: confidence {actual_confidence:.3f} < required {required_confidence} at ({center_x},{center_y})")
                            else:
                                self.logger.debug(f"🚫 Rejected {dish_type}: confidence {actual_confidence:.3f} < required {required_confidence}")
                        
        except Exception as e:
            self.logger.error(f"Error in dish detection: {e}")
            
        return detections
    
    # def _classify_dish_color(self, dish_roi: np.ndarray) -> str:
    #     """
    #     Enhanced dish color classification with improved red detection
        
    #     Args:
    #         dish_roi: Region of interest containing the dish
            
    #     Returns:
    #         Dish type string
    #     """
    #     if dish_roi.size == 0:
    #         return 'normal_dish'
            
    #     try:
    #         # Convert to HSV for better color detection
    #         hsv = cv2.cvtColor(dish_roi, cv2.COLOR_BGR2HSV)
            
    #         # Enhanced red dish detection with multiple ranges
    #         red_mask1 = cv2.inRange(hsv, self.color_ranges['red_dish']['lower1'], 
    #                                self.color_ranges['red_dish']['upper1'])
    #         red_mask2 = cv2.inRange(hsv, self.color_ranges['red_dish']['lower2'], 
    #                                self.color_ranges['red_dish']['upper2'])
    #         red_mask3 = cv2.inRange(hsv, self.color_ranges['red_dish']['lower3'], 
    #                                self.color_ranges['red_dish']['upper3'])
    #         red_mask = red_mask1 + red_mask2 + red_mask3
            
    #         # Check for yellow dish
    #         yellow_mask = cv2.inRange(hsv, self.color_ranges['yellow_dish']['lower'], 
    #                                  self.color_ranges['yellow_dish']['upper'])
            
    #         # Calculate color percentages
    #         total_pixels = dish_roi.shape[0] * dish_roi.shape[1]
    #         red_percentage = np.sum(red_mask > 0) / total_pixels
    #         yellow_percentage = np.sum(yellow_mask > 0) / total_pixels
            
    #         # Lowered thresholds for better detection
    #         red_threshold = 0.08   # Lowered from 0.15 to 0.08 (8%)
    #         yellow_threshold = 0.12 # Slightly higher threshold for yellow
            
    #         # Additional red detection methods for edge cases
    #         red_detected = False
            
    #         # Method 1: Percentage-based detection (primary)
    #         if red_percentage > red_threshold:
    #             red_detected = True
                
    #         # Method 2: Dominant color analysis (secondary)
    #         if not red_detected:
    #             # Check if red is the dominant color even if below threshold
    #             hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                
    #             # Check for red peaks in histogram
    #             red_peak1 = np.sum(hist_hue[0:15])   # Lower red range
    #             red_peak2 = np.sum(hist_hue[165:180]) # Upper red range
    #             red_peak3 = np.sum(hist_hue[10:25])  # Orange-red range
    #             total_red_peak = red_peak1 + red_peak2 + red_peak3
                
    #             # If red is dominant in histogram
    #             total_hist = np.sum(hist_hue)
    #             if total_hist > 0 and (total_red_peak / total_hist) > 0.25:
    #                 red_detected = True
                    
    #         # Method 3: RGB-based red detection (fallback)
    #         if not red_detected:
    #             # Convert back to BGR for RGB analysis
    #             bgr = dish_roi
    #             b, g, r = cv2.split(bgr)
                
    #             # Calculate average color values
    #             avg_r = np.mean(r)
    #             avg_g = np.mean(g)
    #             avg_b = np.mean(b)
                
    #             # Red dominant condition: R > G and R > B with significant difference
    #             if avg_r > avg_g + 20 and avg_r > avg_b + 20 and avg_r > 80:
    #                 red_detected = True
                    
    #         # Classification decision
    #         if red_detected:
    #             self.logger.debug(f"🔴 RED DISH detected: HSV={red_percentage:.3f}, methods used")
    #             return 'red_dish'
    #         elif yellow_percentage > yellow_threshold:
    #             self.logger.debug(f"🟡 YELLOW DISH detected: {yellow_percentage:.3f}")
    #             return 'yellow_dish'
    #         else:
    #             # Additional check for advertisement dishes (multi-colored or specific patterns)
    #             if self._is_advertisement_dish(dish_roi):
    #                 return 'advertisement_dish'
    #             self.logger.debug(f"⚪ NORMAL DISH: red={red_percentage:.3f}, yellow={yellow_percentage:.3f}")
    #             return 'normal_dish'
                
    #     except Exception as e:
    #         self.logger.error(f"Error in color classification: {e}")
    #         return 'normal_dish'
    
    def _is_advertisement_dish(self, dish_roi: np.ndarray) -> bool:
        """
        Detect if dish is an advertisement dish based on color complexity
        
        Args:
            dish_roi: Region of interest containing the dish
            
        Returns:
            True if advertisement dish, False otherwise
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(dish_roi, cv2.COLOR_BGR2HSV)
            
            # Calculate color histogram
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            
            # Count number of significant color peaks
            peaks = 0
            for i, val in enumerate(hist):
                if val > 100:  # Threshold for significant color presence
                    peaks += 1
            
            # Advertisement dishes typically have more diverse colors
            return peaks > 5
            
        except Exception as e:
            self.logger.error(f"Error in advertisement detection: {e}")
            return False
    
    def draw_detections(self, frame: np.ndarray, detections: List[DishDetection]) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        # Color mapping for different dish types
        colors = {
            'normal_dish': (0, 255, 0),      # Green
            'red_dish': (0, 0, 255),         # Red
            'yellow_dish': (0, 255, 255),    # Yellow
            'advertisement_dish': (128, 128, 128)  # Gray
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(detection.dish_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.dish_type}: 1.00"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(annotated_frame, detection.center_point, 3, color, -1)
            
            # Draw counting point (1/3 vertical center) with different visual style
            if detection.counting_point:
                counting_color = (255, 255, 0)  # Cyan for counting point
                cv2.circle(annotated_frame, detection.counting_point, 4, counting_color, -1)
                cv2.circle(annotated_frame, detection.counting_point, 6, counting_color, 2)
                # Add small "C" text near counting point
                cv2.putText(annotated_frame, "C", 
                           (detection.counting_point[0] + 8, detection.counting_point[1] + 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, counting_color, 1)
        
        return annotated_frame
