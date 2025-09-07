"""ROI (Region of Interest) drawing and editing tool."""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import cv2
import numpy as np
import yaml
import structlog

logger = structlog.get_logger()


class ROITool:
    """Interactive tool for drawing and editing ROI polygons."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.window_name = "ROI Editor - Click to draw, 'r' to reset, 's' to save, 'q' to quit"
        self.current_roi = "in_lane"
        self.rois: Dict[str, List[List[int]]] = {
            "in_lane": [],
            "out_lane": []
        }
        self.temp_points: List[List[int]] = []
        self.drawing = False
        self.frame: Optional[np.ndarray] = None
        
    def load_config(self) -> None:
        """Load existing ROI from config."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                if "roi" in config:
                    self.rois["in_lane"] = config["roi"].get("in_lane", [])
                    self.rois["out_lane"] = config["roi"].get("out_lane", [])
            logger.info("Loaded existing ROI config", path=str(self.config_path))
    
    def save_config(self) -> None:
        """Save ROI to config file."""
        # Load existing config
        config = {}
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
        
        # Update ROI section
        config["roi"] = {
            "in_lane": self.rois["in_lane"],
            "out_lane": self.rois["out_lane"]
        }
        
        # Save back
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Saved ROI config", path=str(self.config_path))
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append([x, y])
            self.drawing = True
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current polygon
            if len(self.temp_points) >= 3:
                self.rois[self.current_roi] = self.temp_points.copy()
                self.temp_points = []
                self.drawing = False
                # Switch to next ROI
                if self.current_roi == "in_lane":
                    self.current_roi = "out_lane"
                    logger.info("Finished in_lane, now draw out_lane")
                else:
                    logger.info("Finished out_lane")
    
    def draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI polygons on frame."""
        display = frame.copy()
        
        # Draw saved ROIs
        if len(self.rois["in_lane"]) >= 3:
            pts = np.array(self.rois["in_lane"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, "IN", tuple(self.rois["in_lane"][0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(self.rois["out_lane"]) >= 3:
            pts = np.array(self.rois["out_lane"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            cv2.putText(display, "OUT", tuple(self.rois["out_lane"][0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw temporary points
        for i, pt in enumerate(self.temp_points):
            cv2.circle(display, tuple(pt), 5, (255, 255, 0), -1)
            if i > 0:
                cv2.line(display, tuple(self.temp_points[i-1]), tuple(pt), (255, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            f"Current: {self.current_roi}",
            "Left click: Add point",
            "Right click: Finish polygon",
            "'r': Reset current",
            "'s': Save",
            "'q': Quit"
        ]
        
        y_offset = 30
        for inst in instructions:
            cv2.putText(display, inst, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return display
    
    def edit_on_frame(self, frame: np.ndarray) -> None:
        """Edit ROI on a given frame."""
        self.frame = frame
        self.load_config()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logger.info("ROI editor started. Draw in_lane first, then out_lane.")
        
        while True:
            display = self.draw_rois(frame)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset current ROI
                self.temp_points = []
                self.rois[self.current_roi] = []
                logger.info(f"Reset {self.current_roi}")
            elif key == ord('s'):
                # Save config
                if len(self.rois["in_lane"]) >= 3 and len(self.rois["out_lane"]) >= 3:
                    self.save_config()
                    logger.info("ROI configuration saved")
                else:
                    logger.warning("Both ROIs must have at least 3 points")
            elif key == ord('1'):
                self.current_roi = "in_lane"
                self.temp_points = []
                logger.info("Switched to in_lane")
            elif key == ord('2'):
                self.current_roi = "out_lane"
                self.temp_points = []
                logger.info("Switched to out_lane")
        
        cv2.destroyAllWindows()
    
    def edit_on_video(self, video_path: str) -> None:
        """Edit ROI using a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read frame from video")
        
        cap.release()
        self.edit_on_frame(frame)
    
    def edit_on_rtsp(self, rtsp_url: str) -> None:
        """Edit ROI using RTSP stream."""
        from ..capture.rtsp_reader import RTSPReader
        
        reader = RTSPReader(rtsp_url)
        reader.start()
        
        # Get first frame
        frame_data = reader.read_frame(timeout=5.0)
        if frame_data is None:
            reader.stop()
            raise ValueError("Cannot read frame from RTSP stream")
        
        _, frame = frame_data
        reader.stop()
        
        self.edit_on_frame(frame)