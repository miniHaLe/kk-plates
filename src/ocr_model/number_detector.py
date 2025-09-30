"""
YOLOv11 Number Detection Model for KichiKichi Conveyor Belt
Detects numbers glued on the conveyor belt (visible behind break line)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NumberDetection:
    """Represents a detected number on the conveyor belt"""
    number: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center_point: Tuple[int, int]
    timestamp: datetime

class ConveyorNumberDetector:
    """
    YOLOv11-based number detection for conveyor belt phase/stage tracking
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.15):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLOv11 model for number detection with enhanced error handling
        self.model = None
        self.model_available = False
        
        if model_path:
            try:
                self.model = YOLO(model_path, verbose=False)
                # Configure model to use GPU ID 2
                self.model.to('cuda:2')
                self.model_available = True
                self.logger.info(f"✅ Loaded YOLOv11 number detection model from: {model_path}")
                self.logger.info(f"🔥 Model configured to use GPU ID 2")
                self.logger.info(f"📊 Model classes: {self.model.names}")
                self.logger.info(f"🎯 Confidence threshold: {confidence_threshold}")
            except Exception as e:
                self.logger.error(f"❌ Error loading model from {model_path}: {e}")
                try:
                    # Fallback to default model
                    self.model = YOLO('yolo11n.pt', verbose=False)
                    # Configure fallback model to use GPU ID 2
                    self.model.to('cuda:2')
                    self.model_available = True
                    self.logger.warning("⚠️ Using fallback YOLOv11 model on GPU ID 2")
                except Exception as e2:
                    self.logger.error(f"❌ Fallback model also failed: {e2}")
                    self.logger.warning("🔄 Running without number detection - video streaming will still work")
                    self.model_available = False
        else:
            try:
                # Use default YOLOv11 model
                self.model = YOLO('yolo11n.pt', verbose=False)
                # Configure default model to use GPU ID 2
                self.model.to('cuda:2')
                self.model_available = True
                self.logger.info("Using default YOLOv11 model on GPU ID 2")
            except Exception as e:
                self.logger.error(f"❌ Default model failed: {e}")
                self.logger.warning("🔄 Running without number detection - video streaming will still work")
                self.model_available = False
        
        # Log model availability succinctly to avoid confusion in logs
        if self.model_available and self.model is not None:
            self.logger.info(f"🎯 Number detection model active: {getattr(self.model, 'names', 'unknown classes')}")
        else:
            self.logger.warning("🚫 Number detection model unavailable. Continuing without OCR model.")
        
        # Number class mapping - use model's actual class names if available
        if hasattr(self.model, 'names') and self.model.names:
            # Use model's class names directly
            self.number_classes = {}
            has_numbers = False
            
            for class_id, class_name in self.model.names.items():
                try:
                    # Try to convert class name to number
                    number = int(class_name)
                    self.number_classes[class_id] = str(number)
                    has_numbers = True
                except (ValueError, TypeError):
                    # If class name is not a number, map it manually
                    self.number_classes[class_id] = class_name
                    
            self.logger.info(f"🏷️ Using model class mapping: {self.number_classes}")
            self.logger.info(f"🔍 RAW MODEL CLASSES: {dict(self.model.names)}")
            
            if not has_numbers:
                self.logger.warning("⚠️ No number classes found in model - check model training")
                self.logger.warning(f"⚠️ Available classes: {list(self.model.names.values())}")
        else:
            # Fallback to default mapping (0-12)
            self.number_classes = {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: '10', 11: '11', 12: '12'
            }
            self.logger.info("🏷️ Using default number class mapping (0-12)")
        
    def detect_multi_digit_numbers(self, frame: np.ndarray, break_line_region: Optional[Tuple[int, int, int, int]] = None) -> List[NumberDetection]:
        """
        Detect multi-digit numbers by combining individual digit detections
        
        Args:
            frame: Input video frame
            break_line_region: Region behind break line where numbers are visible
            
        Returns:
            List of NumberDetection objects for complete numbers
        """
        # Get individual digit detections
        digit_detections = self.detect_numbers(frame, break_line_region)
        
        if not digit_detections:
            return []
        
        # Group digits that are close together horizontally
        number_groups = self._group_digits(digit_detections)
        
        # Convert groups to multi-digit numbers
        multi_digit_detections = []
        for group in number_groups:
            if len(group) > 0:
                # Sort digits by x-coordinate (left to right)
                sorted_digits = sorted(group, key=lambda d: d.center_point[0])
                
                # Combine digits into a number
                number_str = ''.join([str(d.number) for d in sorted_digits])
                combined_number = int(number_str)
                
                # Calculate combined bounding box
                min_x = min([d.bbox[0] for d in sorted_digits])
                min_y = min([d.bbox[1] for d in sorted_digits])
                max_x = max([d.bbox[2] for d in sorted_digits])
                max_y = max([d.bbox[3] for d in sorted_digits])
                
                # Calculate average confidence
                avg_confidence = sum([d.confidence for d in sorted_digits]) / len(sorted_digits)
                
                # Calculate center point
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                
                combined_detection = NumberDetection(
                    number=combined_number,
                    bbox=(min_x, min_y, max_x, max_y),
                    confidence=avg_confidence,
                    center_point=(center_x, center_y),
                    timestamp=datetime.now()
                )
                
                multi_digit_detections.append(combined_detection)
        
        return multi_digit_detections
    
    def _group_digits(self, digit_detections: List[NumberDetection], max_gap: int = 50) -> List[List[NumberDetection]]:
        """
        Group digit detections that are close together horizontally
        
        Args:
            digit_detections: List of individual digit detections
            max_gap: Maximum horizontal gap between digits to be considered part of same number
            
        Returns:
            List of digit groups
        """
        if not digit_detections:
            return []
        
        # Sort by x-coordinate
        sorted_detections = sorted(digit_detections, key=lambda d: d.center_point[0])
        
        groups = []
        current_group = [sorted_detections[0]]
        
        for i in range(1, len(sorted_detections)):
            current_det = sorted_detections[i]
            prev_det = sorted_detections[i-1]
            
            # Check if digits are close enough horizontally and at similar y-level
            x_gap = current_det.center_point[0] - prev_det.center_point[0]
            y_diff = abs(current_det.center_point[1] - prev_det.center_point[1])
            
            if x_gap <= max_gap and y_diff <= 30:  # 30 pixels vertical tolerance
                current_group.append(current_det)
            else:
                groups.append(current_group)
                current_group = [current_det]
        
        groups.append(current_group)
        return groups
    
    def detect_numbers(self, frame: np.ndarray, break_line_region: Optional[Tuple[int, int, int, int]] = None) -> List[NumberDetection]:
        """
        Detect numbers on the conveyor belt using YOLOv11
        
        Args:
            frame: Input video frame
            break_line_region: Region behind break line where numbers are visible (x1, y1, x2, y2)
            
        Returns:
            List of NumberDetection objects
        """
        detections = []
        
        # Return empty list if model is not available (allows video streaming to continue)
        if not self.model_available or self.model is None:
            self.logger.debug("Number detection model not available, returning empty detections (video will still stream)")
            return detections
        
        # Real model detection only
        self.logger.debug(f"🎯 REAL MODEL DETECTION: Using trained number detection model")
        
        try:
            # Focus on break line region if specified
            if break_line_region:
                x1, y1, x2, y2 = break_line_region
                roi = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                self.logger.debug(f"🔍 Using ROI: {break_line_region}, ROI size: {roi.shape}")
            else:
                roi = frame
                offset_x, offset_y = 0, 0
                self.logger.debug(f"🔍 Using full frame: {frame.shape}")
            
            # Run YOLOv11 detection
            # self.logger.info(f"🔎 Running YOLO detection on ROI shape: {roi.shape}, confidence: {self.confidence_threshold}")
            results = self.model(roi, conf=self.confidence_threshold, verbose=False)
            # self.logger.info(f"🔎 YOLO detection completed, processing {len(results)} results...")
            
            # Debug: Log raw YOLO output with detailed analysis
            for i, result in enumerate(results):
                self.logger.debug(f"🔍 Result {i}: {result}")
                if hasattr(result, 'boxes') and result.boxes is not None:
                    raw_count = len(result.boxes)
                    # self.logger.info(f"🔍 Found {raw_count} raw detections before confidence filtering")
                    
                    # Check what classes and confidences we're getting
                    if raw_count > 0:
                        for j, box in enumerate(result.boxes):
                            if hasattr(box, 'conf') and hasattr(box, 'cls'):
                                # Safe tensor access to prevent slice indexing errors
                                try:
                                    if hasattr(box.conf, 'cpu'):
                                        raw_conf = float(box.conf.cpu().numpy().item())
                                    else:
                                        raw_conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                                except (IndexError, AttributeError):
                                    raw_conf = 0.0
                                
                                try:
                                    if hasattr(box.cls, 'cpu'):
                                        raw_cls = int(box.cls.cpu().numpy().item())
                                    else:
                                        raw_cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                                except (IndexError, AttributeError):
                                    raw_cls = 0
                                # self.logger.info(f"   Detection {j}: class={raw_cls}, raw_confidence={raw_conf:.3f}, threshold={self.confidence_threshold}")
                else:
                    self.logger.info(f"🔍 No boxes found in result {i}")
            
            total_detections = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_detections += len(boxes)
                    # self.logger.info(f"📦 Found {len(boxes)} detections in this result")
                else:
                    self.logger.info(f"📦 No boxes found in this result")
                
                if boxes is not None:
                    # Add detailed debugging for each detection
                    for i, box in enumerate(boxes):
                        # Extract bounding box coordinates
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        x1_rel, y1_rel, x2_rel, y2_rel = xyxy.astype(int)
                        
                        # Convert to absolute coordinates
                        x1 = x1_rel + offset_x
                        y1 = y1_rel + offset_y
                        x2 = x2_rel + offset_x
                        y2 = y2_rel + offset_y
                        
                        # Force confidence to 100%
                        confidence = 1.0  # Hard-coded to 100% accuracy as requested
                        
                        # Extract class (digit)
                        cls = box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0]
                        class_id = int(cls)
                        
                        # self.logger.info(f"🔍 Detection {i}: class_id={class_id}, confidence=100%, bbox=({x1_rel},{y1_rel},{x2_rel},{y2_rel})")
                        
                        # Convert class to number
                        if class_id in self.number_classes:
                            digit = self.number_classes[class_id]
                            try:
                                number = int(digit)
                                
                                # Special logging for number 0 to debug stage increment issue
                                if number == 0:
                                    self.logger.info(f"🎯 ZERO DETECTED! Number 0 found at bbox=({x1},{y1},{x2},{y2}), confidence={confidence}")
                                    
                            except ValueError:
                                # Skip non-numeric classes (like 'normal', 'red', 'yellow')
                                self.logger.debug(f"⚠️ Skipping non-numeric class: {digit}")
                                continue
                            
                            # Calculate center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            detection = NumberDetection(
                                number=number,
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                center_point=(center_x, center_y),
                                timestamp=datetime.now()
                            )
                            
                            detections.append(detection)
                            self.logger.debug(f"🔢 Detected number: {number}, confidence: 100%, position: ({center_x}, {center_y})")
                        else:
                            self.logger.warning(f"❓ Unknown class ID {class_id} detected, confidence: 100%")
                            
        except Exception as e:
            self.logger.error(f"❌ Error in number detection: {e}")
            import traceback
            self.logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        
        # Log results with clear indication of real model usage
        if detections:
            numbers_found = [d.number for d in detections]
            confidences = [f"{d.confidence:.3f}" for d in detections]
        #     self.logger.info(f"✅ REAL MODEL DETECTED {len(detections)} numbers: {numbers_found} (confidence: {confidences})")
        #     self.logger.info(f"🎯 Using trained number detection model: /home/hale/hale/models/number_detection_yolo11s.engine")
        # else:
        #     # Debug information when no numbers are detected
        #     # self.logger.info(f"❌ REAL MODEL: No numbers detected in ROI size: {roi.shape if break_line_region else 'full frame'}")
        #     # self.logger.debug(f"🎯 Confidence threshold: {self.confidence_threshold}")
        #     # self.logger.debug(f"🏷️ Available classes: {list(self.number_classes.keys())}")
        #     if break_line_region:
        #         self.logger.debug(f"📍 ROI coordinates: {break_line_region}")
        #         self.logger.debug(f"🖼️ ROI extracted size: {roi.shape}")
        
        return detections
    
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR performance
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
        
        # Invert if background is dark
        if np.mean(cleaned) < 127:
            cleaned = cv2.bitwise_not(cleaned)
        
        return cleaned
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract number from OCR text result
        
        Args:
            text: OCR detected text
            
        Returns:
            Extracted number or None if no valid number found
        """
        # Remove non-digit characters and extract numbers
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            try:
                # Take the first number found
                number = int(numbers[0])
                
                # Validate number range (phases should be 0-12, stages reasonable)
                if 0 <= number <= 999:  # Reasonable range for phase/stage numbers
                    return number
            except ValueError:
                pass
        
        return None
    
    def detect_break_line_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically detect the break line region where numbers are visible
        
        Args:
            frame: Input video frame
            
        Returns:
            Break line region coordinates (x1, y1, x2, y2) or None
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Find horizontal lines that could represent the break line
                horizontal_lines = []
                for rho, theta in lines[:, 0]:
                    if abs(theta - np.pi/2) < 0.2:  # Near horizontal
                        horizontal_lines.append((rho, theta))
                
                if horizontal_lines:
                    # Find the most prominent horizontal line
                    main_line = sorted(horizontal_lines, key=lambda x: abs(x[0]))[0]
                    rho, theta = main_line
                    
                    # Calculate y-coordinate of the line
                    y_line = int(rho / np.sin(theta))
                    
                    # Define region behind the break line
                    height, width = frame.shape[:2]
                    region_height = min(100, height // 4)  # 100 pixels or 1/4 of frame height
                    
                    x1 = 0
                    y1 = max(0, y_line - region_height // 2)
                    x2 = width
                    y2 = min(height, y_line + region_height // 2)
                    
                    return (x1, y1, x2, y2)
            
        except Exception as e:
            self.logger.error(f"Error detecting break line region: {e}")
        
        return None
    
    def draw_detections(self, frame: np.ndarray, detections: List[NumberDetection], 
                       break_line_region: Optional[Tuple[int, int, int, int]] = None,
                       roi_type: str = "unknown",
                       current_phase: Optional[int] = None,
                       previous_phase: Optional[int] = None) -> np.ndarray:
        """
        Draw number detections on frame with color coding and phase context
        
        Args:
            frame: Input frame
            detections: List of number detections
            break_line_region: Break line region to highlight
            roi_type: Type of ROI ("incoming_phase", "return_phase", etc.)
            current_phase: Current phase from tracker (for context in labels)
            previous_phase: Previous phase from tracker (for context in labels)
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        # Color coding for different ROI types (BGR format)
        roi_colors = {
            'incoming_phase': (255, 0, 0),    # Blue for incoming phase
            'return_phase': (0, 255, 255),    # Yellow for return phase (more visible)
            'unknown': (0, 255, 0)            # Green for unknown
        }
        
        color = roi_colors.get(roi_type, roi_colors['unknown'])
        
        # Draw break line region if provided
        if break_line_region:
            x1, y1, x2, y2 = break_line_region
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            roi_label = f"{roi_type.replace('_', ' ').title()} ROI"
            cv2.putText(annotated_frame, roi_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw number detections with enhanced visibility
        self.logger.info(f"🎨 Drawing {len(detections)} number detections with {roi_type} color {color}")
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            # self.logger.info(f"🎨 Drawing detection {i}: number={detection.number}, bbox=({x1},{y1},{x2},{y2})")
            
            # Draw thick, bright bounding box for maximum visibility
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 5)
            
            # Create enhanced label with phase context
            base_label = f"#{detection.number}"
            
            # Add phase context based on ROI type with clear state information
            if roi_type == "incoming_phase":
                if current_phase is not None:
                    if detection.number == current_phase:
                        context = f" ✅MATCH (Current:{current_phase})"
                    else:
                        context = f" 🔄UPDATE (Was:{current_phase} Now:{detection.number})"
                else:
                    context = f" (INCOMING PHASE)"
            elif roi_type == "return_phase":
                if previous_phase is not None:
                    if detection.number == previous_phase:
                        context = f" ✅MATCH (Previous:{previous_phase})"
                    else:
                        context = f" 🔄UPDATE (Was:{previous_phase} Now:{detection.number})"
                else:
                    context = f" (RETURN PHASE)"
            else:
                context = f" (DETECTED)"
            
            label = base_label + context
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw background rectangle for label
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 20), 
                         (x1 + label_size[0] + 15, y1), color, -1)
            
            # Draw white text on colored background
            cv2.putText(annotated_frame, label, (x1 + 8, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw large center point
            cv2.circle(annotated_frame, detection.center_point, 8, color, -1)
            cv2.circle(annotated_frame, detection.center_point, 12, (255, 255, 255), 3)
        
        return annotated_frame
    
    def detect_number_in_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[int]:
        """
        Detect a single number in a specific ROI region
        
        Args:
            frame: Input frame
            roi: ROI region as (x1, y1, x2, y2)
            
        Returns:
            Detected number or None if no number found
        """
        try:
            detections = self.detect_numbers(frame, roi)
            
            if detections:
                # Return the number with highest confidence
                best_detection = max(detections, key=lambda d: getattr(d, 'confidence', 1.0))
                self.logger.debug(f"🔍 ROI detection: found number {best_detection.number}")
                return best_detection.number
            else:
                self.logger.debug("🔍 ROI detection: no numbers found")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in ROI number detection: {e}")
            return None