"""
Configuration file for KichiKichi Synchronized Conveyor Belt Dish Counting System
Updated for advanced synchronized dual-camera system
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ConveyorConfig:
    """Configuration for conveyor belt parameters"""
    max_phases_per_stage: int = 12  # Phases 0-12 in each stage
    total_stages: int = 4  # Total number of stages (modifiable)
    break_line_threshold: float = 0.5  # Threshold for break line detection
    
    # New synchronized system parameters
    kitchen_delay_frames: int = 0  # Kitchen camera delay in frames
    dish_counting_tolerance: int = 10  # Tolerance for ROI crossing detection (pixels)
    phase_completion_timeout: int = 300  # Timeout for phase completion (frames)
    
@dataclass
class CameraConfig:
    """Configuration for camera settings"""
    # For POC, using video files instead of RTSP
    break_line_camera_source: str = "assets/videos/break_line_camera.mp4"
    kitchen_camera_source: str = "assets/videos/kitchen_camera.mp4"
    
    # RTSP URLs for production (commented out for POC)
    # break_line_rtsp: str = "rtsp://192.168.1.100:554/stream1"
    # kitchen_rtsp: str = "rtsp://192.168.1.101:554/stream1"
    
    fps: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Synchronized system settings
    enable_frame_display: bool = True  # Show OpenCV windows for debugging
    save_debug_frames: bool = False    # Save frames for debugging
    
@dataclass
class ModelConfig:
    """Configuration for AI models"""
    dish_detection_model_path: str = "models/dish_detection_yolo11s.engine"
    number_detection_model_path: str = "models/number_detection_yolo11s.engine"
    ocr_model_confidence: float = 0.2
    dish_detection_confidence: float = 0.3
    
    # Dish color classes
    dish_classes: Dict[str, int] = None
    
    def __post_init__(self):
        if self.dish_classes is None:
            self.dish_classes = {
                "normal_dish": 0,
                "red_dish": 1,
                "yellow_dish": 2,
                "advertisement_dish": 3  # Will be ignored in counting
            }

@dataclass
class ROIConfig:
    """Configuration for ROI settings"""
    roi_export_directory: str = "exports"
    
    # ROI configuration files
    roi_files: Dict[str, str] = None
    
    def __post_init__(self):
        if self.roi_files is None:
            self.roi_files = {
                'breakline_dish_count': 'break_line_ROI_dish_count.json',
                'breakline_current_phase': 'break_line_ROI_current_phase.json', 
                'breakline_return_phase': 'break_line_ROI_return_phase.json',
                'kitchen_dish_count': 'kitchen_dish_ROI_dish_count.json'
            }

@dataclass
class SynchronizationConfig:
    """Configuration for synchronized tracking system"""
    csv_timeline_path: str = "stage_phase.csv"
    enable_phase_signals: bool = True
    enable_return_dish_counting: bool = True
    enable_kitchen_trigger: bool = True
    
    # Calculation parameters
    enable_auto_calculation: bool = True  # Auto-calculate dishes taken out
    calculation_interval: int = 5  # Seconds between calculations
    
    # Stage table settings
    max_displayed_stages: int = 2  # Show latest 2 stages in dashboard
    max_phases_per_table: int = 13  # Show phases 0-12
    
@dataclass
class DashboardConfig:
    """Configuration for dashboard settings"""
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = True
    rate_calculation_window: int = 60  # seconds for rate calculation
    
    # New synchronized dashboard settings
    update_interval: int = 1000  # milliseconds between updates
    enable_charts: bool = True   # Enable real-time charts
    enable_tables: bool = True   # Enable stage tables
    enable_metrics: bool = True  # Enable metrics display
    
@dataclass
class AppConfig:
    """Main application configuration for synchronized system"""
    conveyor: ConveyorConfig = None
    camera: CameraConfig = None
    model: ModelConfig = None
    roi: ROIConfig = None
    sync: SynchronizationConfig = None
    dashboard: DashboardConfig = None
    
    def __post_init__(self):
        if self.conveyor is None:
            self.conveyor = ConveyorConfig()
        if self.camera is None:
            self.camera = CameraConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.roi is None:
            self.roi = ROIConfig()
        if self.sync is None:
            self.sync = SynchronizationConfig()
        if self.dashboard is None:
            self.dashboard = DashboardConfig()

# Global configuration instance
config = AppConfig()

# Environment variables override
if os.getenv("CONVEYOR_STAGES"):
    config.conveyor.total_stages = int(os.getenv("CONVEYOR_STAGES"))

if os.getenv("MAX_PHASES"):
    config.conveyor.max_phases_per_stage = int(os.getenv("MAX_PHASES"))

if os.getenv("DASHBOARD_PORT"):
    config.dashboard.port = int(os.getenv("DASHBOARD_PORT"))

if os.getenv("KITCHEN_DELAY_FRAMES"):
    config.conveyor.kitchen_delay_frames = int(os.getenv("KITCHEN_DELAY_FRAMES"))

if os.getenv("CSV_TIMELINE_PATH"):
    config.sync.csv_timeline_path = os.getenv("CSV_TIMELINE_PATH")
