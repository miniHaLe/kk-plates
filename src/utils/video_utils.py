"""
Video utilities for KichiKichi Conveyor Belt System
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

class VideoProcessor:
    """Utility class for video processing operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_width: int = None, 
                    target_height: int = None, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame while optionally maintaining aspect ratio
        
        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if target_width is None and target_height is None:
            return frame
        
        if maintain_aspect:
            if target_width is not None:
                scale = target_width / w
                new_width = target_width
                new_height = int(h * scale)
            elif target_height is not None:
                scale = target_height / h
                new_height = target_height
                new_width = int(w * scale)
        else:
            new_width = target_width or w
            new_height = target_height or h
        
        return cv2.resize(frame, (new_width, new_height))
    
    @staticmethod
    def create_side_by_side(frame1: np.ndarray, frame2: np.ndarray, 
                           labels: Tuple[str, str] = None) -> np.ndarray:
        """
        Create side-by-side display of two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
            labels: Optional labels for frames
            
        Returns:
            Combined frame
        """
        if frame1 is None or frame2 is None:
            return frame1 if frame1 is not None else frame2
        
        # Resize frames to same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        target_height = min(h1, h2)
        frame1_resized = VideoProcessor.resize_frame(frame1, target_height=target_height)
        frame2_resized = VideoProcessor.resize_frame(frame2, target_height=target_height)
        
        # Combine horizontally
        combined = np.hstack([frame1_resized, frame2_resized])
        
        # Add labels if provided
        if labels:
            label1, label2 = labels
            # Add label to first frame
            cv2.putText(combined, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            # Add label to second frame
            w1_new = frame1_resized.shape[1]
            cv2.putText(combined, label2, (w1_new + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
        
        return combined
    
    @staticmethod
    def add_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Add FPS counter to frame
        
        Args:
            frame: Input frame
            fps: Current FPS value
            
        Returns:
            Frame with FPS counter
        """
        if frame is None:
            return frame
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Add background rectangle
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (frame.shape[1] - text_size[0] - 20, 10), 
                     (frame.shape[1] - 10, text_size[1] + 20), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, fps_text, (frame.shape[1] - text_size[0] - 15, text_size[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def create_roi_mask(frame_shape: Tuple[int, int], roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create a mask for region of interest
        
        Args:
            frame_shape: Shape of the frame (height, width)
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(frame_shape, dtype=np.uint8)
        x1, y1, x2, y2 = roi
        mask[y1:y2, x1:x2] = 255
        return mask
    
    @staticmethod
    def enhance_frame_for_detection(frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better detection performance
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        if frame is None:
            return frame
        
        # Convert to LAB color space for better lighting handling
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        lab = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

class FPSCounter:
    """Utility class for FPS calculation"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = cv2.getTickCount()
    
    def update(self) -> float:
        """
        Update FPS calculation and return current FPS
        
        Returns:
            Current FPS
        """
        current_time = cv2.getTickCount()
        
        if len(self.frame_times) > 0:
            frame_time = (current_time - self.last_time) / cv2.getTickFrequency()
            self.frame_times.append(frame_time)
            
            # Keep only recent frame times
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            return 0.0

class VideoWriter:
    """Utility class for writing processed video"""
    
    def __init__(self, output_path: str, fps: float = 30.0, 
                 frame_size: Tuple[int, int] = None):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, frame: np.ndarray) -> bool:
        """
        Initialize video writer with first frame
        
        Args:
            frame: First frame to determine properties
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.frame_size is None:
                h, w = frame.shape[:2]
                self.frame_size = (w, h)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            
            if not self.writer.isOpened():
                self.logger.error(f"Failed to open video writer for {self.output_path}")
                return False
            
            self.logger.info(f"Video writer initialized: {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing video writer: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to video
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful, False otherwise
        """
        if self.writer is None:
            if not self.initialize(frame):
                return False
        
        try:
            # Resize frame if necessary
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            self.writer.write(frame)
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing frame: {e}")
            return False
    
    def release(self):
        """Release video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.logger.info(f"Video writer released: {self.output_path}")
