"""Two-stage color classifier: HSV thresholds + CNN fallback."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import structlog

logger = structlog.get_logger()


class ColorClassifier:
    """Two-stage plate color classifier."""
    
    COLORS = ["red", "yellow", "normal"]
    
    def __init__(self, model_path: str, hsv_thresholds: Dict[str, Dict[str, List[int]]]):
        self.model_path = Path(model_path)
        self.hsv_thresholds = hsv_thresholds
        self.session: Optional[ort.InferenceSession] = None
        self.hsv_ambiguity_threshold = 0.15  # If HSV gives < 15% confidence diff, use CNN
        
    def load(self) -> None:
        """Load the ONNX model."""
        if self.model_path.exists():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            logger.info("Loaded color classifier", path=str(self.model_path))
        else:
            logger.warning("CNN model not found, using HSV only", path=str(self.model_path))
    
    def classify(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Classify plate color in given bounding box.
        
        Returns:
            (color_name, confidence)
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "normal", 0.5
        
        # Stage A: HSV classification
        hsv_color, hsv_conf = self._classify_hsv(roi)
        
        # Check if we need Stage B (CNN)
        if self.session is not None and hsv_conf < (1.0 - self.hsv_ambiguity_threshold):
            cnn_color, cnn_conf = self._classify_cnn(roi)
            # Use CNN if it's more confident
            if cnn_conf > hsv_conf:
                return cnn_color, cnn_conf
        
        return hsv_color, hsv_conf
    
    def _classify_hsv(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify using HSV thresholds."""
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate center region (more reliable than edges)
        h, w = roi.shape[:2]
        center_y1, center_y2 = h // 4, 3 * h // 4
        center_x1, center_x2 = w // 4, 3 * w // 4
        center_hsv = hsv[center_y1:center_y2, center_x1:center_x2]
        
        scores = {}
        for color, thresholds in self.hsv_thresholds.items():
            lower = np.array([thresholds["h"][0], thresholds["s"][0], thresholds["v"][0]])
            upper = np.array([thresholds["h"][1], thresholds["s"][1], thresholds["v"][1]])
            
            # Handle red hue wrap-around
            if color == "red" and thresholds["h"][0] > thresholds["h"][1]:
                mask1 = cv2.inRange(center_hsv, lower, np.array([179, upper[1], upper[2]]))
                mask2 = cv2.inRange(center_hsv, np.array([0, lower[1], lower[2]]), upper)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(center_hsv, lower, upper)
            
            # Calculate percentage of pixels matching
            scores[color] = np.sum(mask > 0) / mask.size
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        # Get best match
        best_color = max(scores, key=scores.get)
        confidence = scores[best_color]
        
        return best_color, confidence
    
    def _classify_cnn(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify using CNN model."""
        if self.session is None:
            return "normal", 0.5
        
        # Preprocess for CNN (resize to 64x64, normalize)
        input_size = (64, 64)
        roi_resized = cv2.resize(roi, input_size)
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(roi_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        outputs = self.session.run([output_name], {input_name: input_tensor})
        probs = outputs[0][0]
        
        # Apply softmax
        probs = np.exp(probs) / np.sum(np.exp(probs))
        
        # Get prediction
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        color = self.COLORS[class_idx]
        
        return color, confidence
    
    def create_color_mask(self, frame: np.ndarray, color: str) -> np.ndarray:
        """Create a binary mask for visualization."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color not in self.hsv_thresholds:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        thresholds = self.hsv_thresholds[color]
        lower = np.array([thresholds["h"][0], thresholds["s"][0], thresholds["v"][0]])
        upper = np.array([thresholds["h"][1], thresholds["s"][1], thresholds["v"][1]])
        
        if color == "red" and thresholds["h"][0] > thresholds["h"][1]:
            mask1 = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]]))
            mask2 = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]), upper)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, lower, upper)
        
        return mask