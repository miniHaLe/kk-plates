"""YOLOv8 detector wrapper for plate detection."""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
import structlog

logger = structlog.get_logger()


class PlateDetector:
    """YOLOv8-based plate detector."""
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.model_path = Path(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model: Optional[YOLO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load(self) -> None:
        """Load the model."""
        if not self.model_path.exists():
            # Try relative to data/models
            alt_path = Path("data/models") / self.model_path.name
            if alt_path.exists():
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        logger.info("Loaded detector model", path=str(self.model_path), device=self.device)
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Detect plates in frame.
        
        Returns:
            List of (x1, y1, x2, y2, confidence, class_id) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = self.model(
            frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=self.device
        )
        
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2, float(conf), int(cls)))
        
        return detections
    
    def warmup(self, size: Tuple[int, int] = (1920, 1080)) -> None:
        """Warmup the model with a dummy frame."""
        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("Model warmed up", size=size)