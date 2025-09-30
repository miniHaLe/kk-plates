"""
Main Application for KichiKichi Conveyor Belt Dish Counting System
Coordinates all components: dish detection, OCR, tracking, and dashboard
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple
from datetime import datetime
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Disable YOLO verbose logging
import warnings
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_SILENT'] = 'True'

from dish_detection.dish_detector import DishDetector
from ocr_model.number_detector import ConveyorNumberDetector
from tracking.csv_conveyor_tracker import CSVConveyorTracker
from utils.roi_config_loader import get_tracker_roi_config
from dashboard.dashboard import KichiKichiDashboard
# from config.config import config  # Not needed for current implementation

class KichiKichiApp:
    """
    Main application class that orchestrates the entire system
    """
    
    def __init__(self, export_video=False, auto_restart=False):
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Video export settings
        self.export_video = export_video
        self.video_writers = {}
        
        # Auto-restart settings
        self.auto_restart = auto_restart
        self.user_connection_monitor = {
            'active_connections': set(),
            'queued_connections': set(),
            'last_connection_time': None,
            'restart_on_disconnect': True,
            'restart_on_connect': False,
            'connection_timeout': 5.0,  # Short timeout but less jittery
            'restart_cooldown': 5.0,     # 5 seconds between restarts
            'last_restart_time': 0,
            'startup_grace_period': 15.0  # Allow time for dashboard to connect after start
        }
        # Single-user gate
        self.allow_single_user = True
        
        # Initialize components with YOLOv11 models
        self.dish_detector = DishDetector(
            model_path="/home/hale/hale/models/dish_detection_yolo11s.engine",
            confidence_threshold=0.3
        )
        # Use specific trained number detection model for break line camera
        number_model_path = "/home/hale/hale/models/number_detection_yolo11s.engine"
        self.number_detector = ConveyorNumberDetector(
            model_path=number_model_path,
            confidence_threshold=0.1  # Extremely low to catch any detections
        )
        # Use CSV-driven tracker for synchronized phases/stages with kitchen delay
        csv_path = "/home/hale/hale/stage_phase.csv"
        kitchen_delay_frames = 60  # Kitchen camera delayed by 60 frames to match breakline timing (increased for better sync)
        phase_latency_compensation = 45  # Look ahead 45 frames (1.5 seconds at 30 FPS) for more responsive phase changes
        self.tracker = CSVConveyorTracker(csv_path, kitchen_delay_frames=kitchen_delay_frames, phase_latency_compensation=phase_latency_compensation)
        # Add reference to this app instance for demo restart functionality
        self.tracker.app_instance = self
        self.dashboard = KichiKichiDashboard(self.tracker)
        
        # Give dashboard reference to main app for system control
        self.dashboard.app_instance = self
        
        # Video sources
        self.break_line_cap = None
        self.kitchen_cap = None
        
        # Threading and control
        self.running = False
        self.paused = False
        self.break_line_thread = None
        self.kitchen_thread = None
        self.dashboard_thread = None
        
        # System status
        self.start_time = None
        self.pause_count = 0
        
        # Frame storage for processing
        self.current_break_line_frame = None
        self.current_kitchen_frame = None
        self.frame_lock = threading.Lock()
        
        # Video synchronization
        self.sync_lock = threading.Lock()
        self.master_frame_index = 0  # Breakline camera is master
        self.kitchen_frame_index = 0
        self.sync_offset = 0  # Kitchen camera offset in frames
        self.sync_tolerance = 3  # Allowed frame difference before sync correction
        self.last_sync_time = 0
        self.sync_status = {
            'is_synced': False,
            'frame_difference': 0,
            'last_sync_timestamp': None,
            'sync_corrections': 0
        }
        
        # Video loading synchronization
        self.video_load_sync = {
            'breakline_ready': threading.Event(),
            'kitchen_ready': threading.Event(),
            'both_ready': threading.Event(),
            'breakline_load_time': None,
            'kitchen_load_time': None,
            'load_timeout': 30.0,  # 30 seconds timeout for video loading
            'ready_frame_count': 5  # Number of frames to successfully read to consider "ready"
        }
        
        self.logger.info("KichiKichi application initialized successfully")
        
        # Start background connection watchdog to detect disconnects and restart cleanly
        try:
            self.connection_watchdog_thread = threading.Thread(
                target=self._connection_watchdog,
                name="ConnectionWatchdog",
                daemon=True
            )
            self.connection_watchdog_thread.start()
        except Exception as e:
            self.logger.error(f"Failed to start connection watchdog: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/kichikichi.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        # Quiet common noisy loggers
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('dash').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    def _validate_video_csv_alignment(self):
        """Validate that video properties align with CSV timeline expectations"""
        try:
            if not self.break_line_cap:
                return
            
            # Get video properties
            fps = self.break_line_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.break_line_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            # Get CSV timeline info
            csv_max_frame = self.tracker.timeline_parser.get_max_frame_index()
            csv_duration_seconds = csv_max_frame / 30.0 if csv_max_frame > 0 else 0  # Assume 30fps for CSV
            
            # self.logger.info(f"📊 Video validation:")
            # self.logger.info(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration_seconds:.1f}s")
            # self.logger.info(f"  CSV: max frame {csv_max_frame}, expected {csv_duration_seconds:.1f}s at 30fps")
            
            # Check alignment
            frame_diff = abs(total_frames - csv_max_frame)
            if frame_diff > 100:  # Allow 100 frame tolerance (~3.3 seconds at 30fps)
                self.logger.warning(f"⚠️ Video-CSV frame mismatch: {frame_diff} frames difference")
                self.logger.warning(f"⚠️ This may cause incorrect stage/phase detection")
            else:
                self.logger.info(f"✅ Video-CSV alignment looks good (diff: {frame_diff} frames)")
                
            # Check FPS
            if abs(fps - 30.0) > 1.0:
                self.logger.warning(f"⚠️ Video FPS ({fps:.1f}) differs from expected 30fps")
                self.logger.warning(f"⚠️ This may affect frame index calculations")
                
        except Exception as e:
            self.logger.error(f"❌ Error validating video-CSV alignment: {e}")
    
    def _sync_kitchen_camera(self, kitchen_frame_index: int) -> bool:
        """Synchronize kitchen camera with breakline camera (master)"""
        try:
            with self.sync_lock:
                # Calculate expected kitchen frame based on master (breakline) frame
                expected_kitchen_frame = self.master_frame_index + self.sync_offset
                frame_difference = kitchen_frame_index - expected_kitchen_frame
                
                # Update sync status
                self.sync_status['frame_difference'] = frame_difference
                self.sync_status['is_synced'] = abs(frame_difference) <= self.sync_tolerance
                
                # Log sync status periodically
                current_time = time.time()
                if current_time - self.last_sync_time > 5.0:  # Log every 5 seconds
                    # self.logger.info(f"🔄 SYNC STATUS: Master(breakline)={self.master_frame_index}, "
                    #                f"Kitchen={kitchen_frame_index}, Expected={expected_kitchen_frame}, "
                    #                f"Diff={frame_difference}, Synced={'✅' if self.sync_status['is_synced'] else '❌'}")
                    self.last_sync_time = current_time
                
                # Apply sync correction if needed
                if not self.sync_status['is_synced'] and self.kitchen_cap:
                    # Calculate correction
                    target_frame = max(0, expected_kitchen_frame)
                    current_frame = self.kitchen_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    
                    if abs(target_frame - current_frame) > self.sync_tolerance:
                        self.kitchen_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        self.sync_status['sync_corrections'] += 1
                        self.sync_status['last_sync_timestamp'] = datetime.now().isoformat()
                        
                        # self.logger.info(f"🔧 SYNC CORRECTION: Kitchen camera moved from frame {current_frame} to {target_frame}")
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in kitchen camera sync: {e}")
            return False
    
    def get_sync_status(self) -> dict:
        """Get current synchronization status for dashboard display"""
        with self.sync_lock:
            return {
                'master_frame': self.master_frame_index,
                'kitchen_frame': self.kitchen_frame_index,
                'sync_offset': self.sync_offset,
                'frame_difference': self.sync_status['frame_difference'],
                'is_synced': self.sync_status['is_synced'],
                'sync_tolerance': self.sync_tolerance,
                'sync_corrections': self.sync_status['sync_corrections'],
                'last_sync_timestamp': self.sync_status['last_sync_timestamp']
            }
    
    def initialize_cameras(self) -> bool:
        """
        Initialize video sources (files for POC, RTSP for production)
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize break line camera from video file
            break_line_video = "./assets/videos/break_line_camera.mp4"
            self.break_line_cap = cv2.VideoCapture(break_line_video)
            if not self.break_line_cap.isOpened():
                self.logger.warning(f"Cannot open break line video: {break_line_video}, running in mock mode")
                self.break_line_cap = None
            
            # Initialize kitchen camera from video file (corrected filename)
            kitchen_video = "./assets/videos/kitchen.mp4"
            self.kitchen_cap = cv2.VideoCapture(kitchen_video)
            if not self.kitchen_cap.isOpened():
                self.logger.warning(f"Cannot open kitchen video: {kitchen_video}, running in mock mode")
                self.kitchen_cap = None
            
            # Set camera properties for synchronized playback
            for cap in [self.break_line_cap, self.kitchen_cap]:
                if cap is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    # Ensure both videos start from the same frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Validate video properties against CSV timeline
            if self.break_line_cap is not None:
                self._validate_video_csv_alignment()
            
            # Test video loading synchronization
            # if not self._wait_for_videos_ready():
            #     self.logger.warning("⚠️ Video loading synchronization failed - continuing anyway")
            
            # if self.break_line_cap is None and self.kitchen_cap is None:
            #     self.logger.info("Running in full mock mode - no real cameras detected")
            # elif self.break_line_cap is None:
            #     self.logger.info("Running with kitchen camera only - break line in mock mode")
            # elif self.kitchen_cap is None:
            #     self.logger.info("Running with break line camera only - kitchen in mock mode")
            # else:
            #     self.logger.info("Cameras initialized successfully")

            # Initialize video writers if enabled
            self._initialize_video_writers()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing cameras: {e}")
            return False
    
    def _initialize_video_writers(self):
        """Initialize video writers if video export is enabled"""
        if not self.export_video:
            return
            
        # self.logger.info("🎥 Video export enabled. Initializing video writers...")
        
        # Create videos directory if it doesn't exist
        os.makedirs("./videos", exist_ok=True)
        
        # Get video properties from capture objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if self.break_line_cap:
            fps = self.break_line_cap.get(cv2.CAP_PROP_FPS)
            width = int(self.break_line_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.break_line_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width > 0 and height > 0:
                output_path = f"./videos/break_line_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self.video_writers['break_line'] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                # self.logger.info(f"Writing break line video to {output_path}")
        
        if self.kitchen_cap:
            fps = self.kitchen_cap.get(cv2.CAP_PROP_FPS)
            width = int(self.kitchen_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.kitchen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width > 0 and height > 0:
                output_path = f"./videos/kitchen_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self.video_writers['kitchen'] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                # self.logger.info(f"Writing kitchen video to {output_path}")

    def _test_video_readiness(self, cap, video_name: str) -> bool:
        """
        Test if a video source is ready by reading multiple frames
        
        Args:
            cap: OpenCV VideoCapture object
            video_name: Name of the video for logging
            
        Returns:
            True if video is ready, False otherwise
        """
        if cap is None or not cap.isOpened():
            return False
            
        try:
            # Save current position
            original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Test reading multiple frames
            successful_reads = 0
            for i in range(self.video_load_sync['ready_frame_count']):
                ret, frame = cap.read()
                if ret and frame is not None:
                    successful_reads += 1
                else:
                    break
            
            # Restore original position
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
            
            is_ready = successful_reads >= self.video_load_sync['ready_frame_count']
            if is_ready:
                self.logger.info(f"✅ {video_name} video is ready ({successful_reads}/{self.video_load_sync['ready_frame_count']} frames read successfully)")
            else:
                self.logger.warning(f"❌ {video_name} video not ready ({successful_reads}/{self.video_load_sync['ready_frame_count']} frames read)")
                
            return is_ready
            
        except Exception as e:
            self.logger.error(f"Error testing {video_name} video readiness: {e}")
            return False
    
    def _wait_for_videos_ready(self) -> bool:
        """
        Wait for both videos to be ready before proceeding
        
        Returns:
            True if both videos are ready within timeout, False otherwise
        """
        self.logger.info("🎬 Testing video loading synchronization...")
        
        # Reset events
        self.video_load_sync['breakline_ready'].clear()
        self.video_load_sync['kitchen_ready'].clear()
        self.video_load_sync['both_ready'].clear()
        
        start_time = time.time()
        
        # Test breakline video if available
        if self.break_line_cap is not None:
            self.logger.info("📹 Testing breakline camera readiness...")
            if self._test_video_readiness(self.break_line_cap, "Breakline"):
                self.video_load_sync['breakline_ready'].set()
                self.video_load_sync['breakline_load_time'] = time.time()
            else:
                self.logger.warning("⚠️ Breakline camera failed readiness test")
        else:
            # If no breakline camera, consider it "ready"
            self.video_load_sync['breakline_ready'].set()
            
        # Test kitchen video if available  
        if self.kitchen_cap is not None:
            self.logger.info("🍽️ Testing kitchen camera readiness...")
            if self._test_video_readiness(self.kitchen_cap, "Kitchen"):
                self.video_load_sync['kitchen_ready'].set()
                self.video_load_sync['kitchen_load_time'] = time.time()
            else:
                self.logger.warning("⚠️ Kitchen camera failed readiness test")
        else:
            # If no kitchen camera, consider it "ready"
            self.video_load_sync['kitchen_ready'].set()
        
        # Wait for both to be ready or timeout
        timeout_remaining = self.video_load_sync['load_timeout'] - (time.time() - start_time)
        
        if timeout_remaining > 0:
            self.logger.info(f"⏳ Waiting for both videos to be ready (timeout: {timeout_remaining:.1f}s)...")
            
            # Wait for both events to be set
            breakline_ready = self.video_load_sync['breakline_ready'].wait(timeout_remaining / 2)
            kitchen_ready = self.video_load_sync['kitchen_ready'].wait(timeout_remaining / 2)
            
            if breakline_ready and kitchen_ready:
                self.video_load_sync['both_ready'].set()
                load_time_diff = abs((self.video_load_sync['breakline_load_time'] or 0) - 
                                   (self.video_load_sync['kitchen_load_time'] or 0))
                
                self.logger.info(f"✅ Both videos are ready! Load time difference: {load_time_diff:.2f}s")
                
                # Small delay to ensure both videos are stable
                time.sleep(1.0)
                return True
            else:
                self.logger.warning(f"⏰ Video loading timeout - Breakline ready: {breakline_ready}, Kitchen ready: {kitchen_ready}")
                return False
        else:
            self.logger.error("⏰ Video loading timeout during initial testing")
            return False
    
    def get_video_sync_status(self) -> dict:
        """Get current video synchronization status for dashboard"""
        try:
            with self.sync_lock:
                return {
                    'videos_loaded': self.video_load_sync['both_ready'].is_set(),
                    'breakline_ready': self.video_load_sync['breakline_ready'].is_set(),
                    'kitchen_ready': self.video_load_sync['kitchen_ready'].is_set(),
                    'sync_validation_passed': self.video_load_sync['both_ready'].is_set(),
                    'master_frame': self.master_frame_index,
                    'kitchen_frame': self.kitchen_frame_index,
                    'frame_difference': abs(self.master_frame_index - self.kitchen_frame_index),
                    'load_time_breakline': self.video_load_sync['breakline_load_time'],
                    'load_time_kitchen': self.video_load_sync['kitchen_load_time'],
                    'sync_corrections': self.sync_status['sync_corrections']
                }
        except Exception as e:
            self.logger.error(f"Error getting video sync status: {e}")
            return {
                'videos_loaded': False,
                'breakline_ready': False,
                'kitchen_ready': False,
                'sync_validation_passed': False,
                'master_frame': 0,
                'kitchen_frame': 0,
                'frame_difference': 0,
                'load_time_breakline': None,
                'load_time_kitchen': None,
                'sync_corrections': 0
            }

    def register_user_connection(self, user_id: str = None) -> bool:
        """
        Register a new user connection and handle auto-restart logic
        
        Args:
            user_id: Unique identifier for the user (IP address, session ID, etc.)
            
        Returns:
            True if system should continue, False if restart is needed
        """
        if not self.auto_restart:
            return True
            
        current_time = time.time()
        
        # Generate user ID if not provided (using timestamp and random component)
        if user_id is None:
            user_id = f"user_{int(current_time)}_{hash(current_time) % 10000}"
        
        # Check restart cooldown
        time_since_last_restart = current_time - self.user_connection_monitor['last_restart_time']
        if time_since_last_restart < self.user_connection_monitor['restart_cooldown']:
            self.logger.debug(f"🕒 User connection registered but restart cooldown active ({time_since_last_restart:.1f}s)")
            return True
        
        # Enforce single-user access: if one is active, others are queued
        if self.allow_single_user and len(self.user_connection_monitor['active_connections']) >= 1:
            self.user_connection_monitor['queued_connections'].add(user_id)
            self.logger.info(f"🚦 User queued: {user_id} (active in use)")
            return True
        
        # First active user: do NOT auto-restart (we manage clean state elsewhere)
        
        # Add to active connections
        self.user_connection_monitor['active_connections'].add(user_id)
        self.user_connection_monitor['last_connection_time'] = current_time
        
        self.logger.info(f"👤 User registered: {user_id} (total: {len(self.user_connection_monitor['active_connections'])})")
        return True
    
    def unregister_user_connection(self, user_id: str) -> bool:
        """
        Unregister a user connection and handle auto-restart logic
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if system should continue, False if restart is needed
        """
        # Always handle disconnects (even without auto_restart)
            
        # Remove from active and queued connections
        self.user_connection_monitor['active_connections'].discard(user_id)
        self.user_connection_monitor['queued_connections'].discard(user_id)
        
        current_time = time.time()
        time_since_last_restart = current_time - self.user_connection_monitor['last_restart_time']
        
        # Check restart cooldown
        if time_since_last_restart < self.user_connection_monitor['restart_cooldown']:
            self.logger.debug(f"🕒 User disconnected but restart cooldown active ({time_since_last_restart:.1f}s)")
            return True
        
        # If no active user remains and auto_restart enabled, restart to free system for next in queue
        if len(self.user_connection_monitor['active_connections']) == 0 and self.auto_restart:
            self.logger.info(f"👤 Last user disconnected: {user_id} - triggering HARD restart for next session")
            # Clear queued; fresh process will handle new connection cleanly
            self.user_connection_monitor['queued_connections'].clear()
            self.hard_restart()
            return False
        
        self.logger.info(f"👤 User disconnected: {user_id} (remaining: {len(self.user_connection_monitor['active_connections'])})")
        return True
    
    def _trigger_auto_restart(self, reason: str) -> bool:
        """
        Trigger an automatic system restart
        
        Args:
            reason: Reason for the restart
            
        Returns:
            False to indicate restart is needed
        """
        current_time = time.time()
        self.user_connection_monitor['last_restart_time'] = current_time
        
        self.logger.info(f"🔄 AUTO-RESTART triggered: {reason}")
        self.logger.info("🛑 Stopping all system components for clean restart...")
        
        # Stop the system gracefully
        self.stop()
        
        # Clear all connections
        self.user_connection_monitor['active_connections'].clear()
        
        # The actual restart will be handled by the process manager or parent script
        return False
    
    def check_connection_timeout(self) -> bool:
        """
        Check for connection timeouts and handle auto-restart if needed
        
        Returns:
            True if system should continue, False if restart is needed
        """
        # Under supervisor, avoid in-process restarts on timeout; supervisor will handle full restarts
        if os.environ.get('KICHI_SUPERVISED') == '1':
            return True
        if not self.user_connection_monitor['last_connection_time']:
            return True
            
        current_time = time.time()
        time_since_last_connection = current_time - self.user_connection_monitor['last_connection_time']
        
        # Respect startup grace period to avoid immediate restarts after pressing Start
        if self.start_time and (time.time() - self.start_time) < self.user_connection_monitor.get('startup_grace_period', 15.0):
            return True
        
        # Check if connection timeout exceeded (only when auto_restart enabled)
        if self.auto_restart and time_since_last_connection > self.user_connection_monitor['connection_timeout']:
            self.logger.warning(f"⏰ Connection timeout detected ({time_since_last_connection:.1f}s) - HARD restart")
            self.hard_restart()
            return False
        
        return True

    def _connection_watchdog(self):
        """Background loop to monitor user connections and trigger restart on disconnect."""
        while True:
            try:
                time.sleep(1.0)
                # Under supervisor, let supervisor handle restarts; do not self-restart on heartbeat issues
                if os.environ.get('KICHI_SUPERVISED') == '1':
                    continue
                # Only enforce when streaming is active
                if not getattr(self, 'running', False):
                    continue
                # Single-user enforcement
                active = len(self.user_connection_monitor['active_connections'])
                last_seen = self.user_connection_monitor.get('last_connection_time')
                timeout = self.user_connection_monitor.get('connection_timeout', 2.0)
                now = time.time()
                # Respect startup grace to avoid immediate restarts right after Start
                if self.start_time and (now - self.start_time) < self.user_connection_monitor.get('startup_grace_period', 15.0):
                    continue
                # If auto_restart enabled and no active users OR heartbeat stale -> hard restart
                if self.auto_restart and (active == 0 or (last_seen is not None and (now - last_seen) > timeout)):
                    self.logger.info("🔌 No active user or heartbeat stale - hard restarting")
                    self.hard_restart()
                    return
            except Exception as e:
                try:
                    self.logger.error(f"Connection watchdog error: {e}")
                except Exception:
                    pass
    
    def get_connection_status(self) -> dict:
        """Get current connection monitoring status"""
        return {
            'auto_restart_enabled': self.auto_restart,
            'active_connections': len(self.user_connection_monitor['active_connections']),
            'connection_list': list(self.user_connection_monitor['active_connections']),
            'queued_connections': len(self.user_connection_monitor.get('queued_connections', set())),
            'last_connection_time': self.user_connection_monitor['last_connection_time'],
            'last_restart_time': self.user_connection_monitor['last_restart_time'],
            'restart_on_connect': self.user_connection_monitor['restart_on_connect'],
            'restart_on_disconnect': self.user_connection_monitor['restart_on_disconnect'],
            'allow_single_user': self.allow_single_user
        }

    def process_break_line_camera(self):
        """Process break line camera feed (main detection and OCR)"""
        self.logger.info("Starting break line camera processing")
        
        if not self.break_line_cap:
            self.logger.warning("❌ Break line camera is None - generating mock camera feed")
            self._process_mock_break_line_camera()
            return
            
        if not self.break_line_cap.isOpened():
            self.logger.warning("❌ Break line camera failed to open - generating mock camera feed")
            self._process_mock_break_line_camera()
            return
            
        self.logger.info(f"✅ Break line camera opened successfully")
        
        while self.running and self.break_line_cap and self.break_line_cap.isOpened():
            # Check for pause state
            while self.paused and self.running:
                time.sleep(0.1)  # Wait while paused
                
            if not self.running:
                break
                
            ret, frame = self.break_line_cap.read()
            if not ret:
                # End of video: do not loop; pause processing and await user action
                self.logger.info("⏹️ Break line video reached end - pausing processing (no loop)")
                # Mark demo as completed so UI can show completion modal even if CSV end not reached
                try:
                    current_state = self.tracker.get_current_state()
                    if hasattr(current_state, 'demo_completed') and not current_state.demo_completed:
                        current_state.demo_completed = True
                        current_state.demo_completion_time = datetime.now()
                        self.logger.info("🎉 Marked demo completed due to video EOF")
                except Exception as _e:
                    self.logger.debug(f"EOF completion mark error: {_e}")
                self.paused = True
                continue
            
            try:
                # Store frame for dashboard
                with self.frame_lock:
                    self.current_break_line_frame = frame.copy()
                    
                self.logger.debug(f"📹 Break line frame: {frame.shape} -> dashboard update")
                
                # Update CSV timeline position based on break-line frame index
                try:
                    frame_index = int(self.break_line_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # Update master frame index for synchronization
                    with self.sync_lock:
                        self.master_frame_index = frame_index
                    
                    # Debug: Log frame index every 100 frames to verify alignment with CSV
                    if frame_index % 100 == 0:
                        self.logger.info(f"🎬 Breakline camera at frame {frame_index}")
                except Exception as e:
                    frame_index = self.tracker.get_current_state().current_frame + 1
                    self.logger.warning(f"⚠️ Could not get frame index from video: {e}, using {frame_index}")
                
                self.tracker.update_breakline_frame_index(frame_index)
                timeline_updated = self.tracker.update_frame_position(frame_index, is_breakline_camera=True)
                
                # Debug: Log when timeline is updated
                if timeline_updated:
                    state = self.tracker.get_current_state()
                    # self.logger.info(f"🎯 Timeline updated at frame {frame_index}: Stage {state.current_stage}, Phase {state.current_phase}")

                # If demo completed, pause processing to stop counting and video progression
                current_state_for_stop = self.tracker.get_current_state()
                if hasattr(current_state_for_stop, 'demo_completed') and current_state_for_stop.demo_completed:
                    self.logger.info("🎬 Demo marked completed - pausing break line processing")
                    self.paused = True
                    continue

                # ROI-based detection using user-defined coordinates
                # BREAK LINE CAMERA: detects return dishes for backward line counts
                
                # ROI 1: Return dish detection (dishes coming back from customers)
                dish_detections = self.dish_detector.detect_dishes(frame)
                # Process break-line dish detections; tracker applies ROI and top-edge crossing internally
                self.tracker.process_dish_detections(dish_detections, roi_name="dish_detection")
                
                # Debug logging for return dish detection
                if dish_detections:
                    non_ad_dishes = [d for d in dish_detections if d.dish_type != 'advertisement_dish']
                    if non_ad_dishes:
                        self.logger.debug(f"🔄 BREAK LINE: Detected {len(non_ad_dishes)} return dishes: {[d.dish_type for d in non_ad_dishes]}")
                        self.logger.debug(f"🎯 Return dishes for ROI processing: {len(non_ad_dishes)} dishes")
                
                # ROI 2: Previous/Return phase number detection (completed phases from previous stages)
                self.logger.debug(f"🔍 Return phase ROI: {self.tracker.roi_return_phase}")
                return_phase_detections = self.number_detector.detect_numbers(
                    frame, self.tracker.roi_return_phase
                )
                if return_phase_detections:
                    phase_numbers = [d.number for d in return_phase_detections]
                    self.logger.info(f"🎯 BREAK LINE ROI: {phase_numbers} (updating previous phase)")
                else:
                    self.logger.debug(f"❌ No numbers detected in return ROI: {self.tracker.roi_return_phase}")
                
                # No secondary phase detection - only return phases detected by break line camera
                secondary_phase_detections = []
                
                # CRITICAL: DO NOT use full frame fallback - it contaminates ROI-based detection
                # Only use numbers that are actually detected within the specific return_phase ROI
                if not return_phase_detections:
                    self.logger.debug("❌ No numbers detected in return_phase ROI - this is normal")
                else:
                    # Verify all detections are truly within the return_phase ROI using bbox coordinates
                    verified_detections = []
                    for det in return_phase_detections:
                        if self._verify_detection_in_roi(det, self.tracker.roi_return_phase):
                            verified_detections.append(det)
                        else:
                            self.logger.warning(f"⚠️ Detection {det.number} at {det.center_point} not actually in return_phase ROI")
                    return_phase_detections = verified_detections
                        
                # REMOVED: Full frame detection fallback that was stealing incoming ROI numbers
                # This was causing break line camera to incorrectly update previous phase
                # when numbers were meant for kitchen camera's current phase update
                
                # Enhanced debugging for number detection on break line camera  
                self.logger.debug(f"Break line camera: secondary={len(secondary_phase_detections)}, return={len(return_phase_detections)}")
                
                if secondary_phase_detections:
                    numbers = [d.number for d in secondary_phase_detections]
                    confidences = [d.confidence for d in secondary_phase_detections]
                    self.logger.info(f"🔢 SECONDARY PHASE NUMBERS DETECTED: {numbers} (confidences: {confidences})")
                else:
                    self.logger.debug(f"🔍 No secondary phase numbers detected in ROI {self.tracker.roi_incoming_phase}")
                    
                if return_phase_detections:
                    numbers = [d.number for d in return_phase_detections]
                    confidences = [d.confidence for d in return_phase_detections]
                    self.logger.info(f"🔄 RETURN PHASE NUMBERS DETECTED: {numbers} (confidences: {confidences})")
                else:
                    self.logger.debug(f"🔍 No return phase numbers detected in ROI {self.tracker.roi_return_phase}")
                
                # CSV mode: phase/state come from CSV; use detections only for visualization
                
                # Create annotated frame for dashboard with ROI visualizations
                annotated_frame = self.dish_detector.draw_detections(frame, dish_detections)
                
                # Draw number detections with visible bounding boxes and phase context
                if return_phase_detections:
                    self.logger.info(f"🎨 DRAWING {len(return_phase_detections)} number detections on break line camera")
                    # Get current phase information from tracker
                    tracker_state = self.tracker.get_current_state()
                    annotated_frame = self.number_detector.draw_detections(
                        annotated_frame, return_phase_detections, self.tracker.roi_return_phase, "return_phase",
                        current_phase=tracker_state.current_phase,
                        previous_phase=tracker_state.last_return_phase
                    )
                else:
                    self.logger.debug("🔍 No numbers detected in return ROI")
                
                # Draw ROI rectangles on frame for visualization
                annotated_frame = self._draw_roi_rectangles(annotated_frame)
                
                # Add system info overlay
                annotated_frame = self._add_system_overlay(annotated_frame, "Break Line Camera")
                
                # Update dashboard
                self.dashboard.update_camera_frame('break_line', annotated_frame)
                
                # Write frame to video file if export is enabled
                if 'break_line' in self.video_writers:
                    self.video_writers['break_line'].write(annotated_frame)
                
                # Small delay to control processing rate (cap to 30 FPS to ensure UI updates)
                time.sleep(1.0 / 30)
                
            except Exception as e:
                self.logger.error(f"Error processing break line camera: {e}")
                time.sleep(0.1)
        
        self.logger.info("Break line camera processing stopped")
    
    def process_kitchen_camera(self):
        """Process kitchen camera feed (monitor dishes being served to customers)"""
        self.logger.info("Starting kitchen camera processing with ROI")
        self.logger.info(f"Kitchen counter ROI: {self.tracker.roi_kitchen_counter}")
        
        # Check if kitchen camera is available
        if not self.kitchen_cap or not self.kitchen_cap.isOpened():
            self.logger.warning("❌ Kitchen camera not available - generating mock camera feed")
            self._process_mock_kitchen_camera()
            return
        
        while self.running and self.kitchen_cap and self.kitchen_cap.isOpened():
            # Check for pause state
            while self.paused and self.running:
                time.sleep(0.1)  # Wait while paused
                
            if not self.running:
                break
                
            ret, frame = self.kitchen_cap.read()
            if not ret:
                # End of video: do not loop; pause processing and await user action
                self.logger.info("⏹️ Kitchen video reached end - pausing processing (no loop)")
                self.paused = True
                continue
            
            try:
                # Store frame for dashboard
                with self.frame_lock:
                    self.current_kitchen_frame = frame.copy()
                
                # KITCHEN CAMERA: Detects CURRENT stage dishes AND current phase numbers with delay compensation
                
                # Update kitchen frame index with delay compensation and synchronization
                kitchen_frame_index = int(self.kitchen_cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.kitchen_cap else 0
                
                # Store kitchen frame index for sync status
                with self.sync_lock:
                    self.kitchen_frame_index = kitchen_frame_index
                
                # Perform synchronization check and correction
                sync_corrected = self._sync_kitchen_camera(kitchen_frame_index)
                
                # Update kitchen frame index after potential sync correction
                if sync_corrected and self.kitchen_cap:
                    kitchen_frame_index = int(self.kitchen_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    with self.sync_lock:
                        self.kitchen_frame_index = kitchen_frame_index
                
                delayed_frame_index = self.tracker.update_kitchen_frame_index(kitchen_frame_index)
                
                # Check for phase change signal from breakline camera
                current_state = self.tracker.get_current_state()
                phase_changed = self.tracker.check_phase_change_signal(current_state.current_phase)
                if phase_changed:
                    self.logger.info(f"🚨 Kitchen Camera: Received phase change signal for Phase {current_state.current_phase}")
                
                # If demo completed, pause processing to stop counting and video progression
                current_state_for_stop = self.tracker.get_current_state()
                if hasattr(current_state_for_stop, 'demo_completed') and current_state_for_stop.demo_completed:
                    self.logger.info("🎬 Demo marked completed - pausing kitchen processing")
                    self.paused = True
                    continue

                # Incoming phase ROI detection removed per user request
                
                # 1. Detect dishes in full frame
                dish_detections = self.dish_detector.detect_dishes(frame)
                
                # Filter detections to kitchen counter ROI only
                self.logger.debug(f"Kitchen: {len(dish_detections)} total dishes")
                
                # Process kitchen detections through CSV tracker (applies ROI and top-edge crossing)
                self.tracker.process_dish_detections(dish_detections, roi_name="kitchen_counter")
                
                # Incoming phase detection processing removed
                
                # Debug: Log zero detection status periodically
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                
                if self._debug_counter % 100 == 0:  # Every 100 frames
                    zero_status = { 'csv_mode': True }
                    self.logger.info(f"🔍 ZERO DETECTION STATUS: {zero_status}")
                    state = self.tracker.get_current_state()
                    self.logger.info(f"📊 CURRENT STATE: Stage {state.current_stage}, Phase {state.current_phase}, Initialized: {state.is_phase_initialized}")
                
                # Manual test simulation removed - using real number detection only
                
                # Create annotated frame with ROI visualization
                annotated_frame = self.dish_detector.draw_detections(frame, dish_detections)
                annotated_frame = self._draw_kitchen_roi_rectangles(annotated_frame)
                
                # Incoming phase number detection drawing removed
                
                # Add system info overlay
                annotated_frame = self._add_system_overlay(annotated_frame, "Kitchen Camera")
                
                # Update dashboard
                self.dashboard.update_camera_frame('kitchen', annotated_frame)
                
                # Write frame to video file if export is enabled
                if 'kitchen' in self.video_writers:
                    self.video_writers['kitchen'].write(annotated_frame)
                
                # Log serving activity and rate calculation dishes
                if dish_detections: # Use dish_detections from the full frame
                    served_count = len([d for d in dish_detections 
                                      if d.dish_type != 'advertisement_dish'])
                    
                    # Count red and yellow dishes for rate calculation logging
                    red_count = len([d for d in dish_detections if d.dish_type == 'red_dish'])
                    yellow_count = len([d for d in dish_detections if d.dish_type == 'yellow_dish'])
                    
                    if served_count > 0:
                        self.logger.debug(f"Kitchen: {served_count} served")
                        
                    if red_count > 0 or yellow_count > 0:
                        self.logger.debug(f"Kitchen: +{red_count}R, +{yellow_count}Y")
                
                # Small delay to control processing rate (cap to 30 FPS to ensure UI updates)
                time.sleep(1.0 / 30)
                
            except Exception as e:
                self.logger.error(f"Error processing kitchen camera: {e}")
                time.sleep(0.1)
        
        self.logger.info("Kitchen camera processing stopped")
    
    def _process_mock_break_line_camera(self):
        """Generate mock break line camera feed when video files are missing"""
        self.logger.info("🎬 Starting mock break line camera feed")
        frame_counter = 0
        
        while self.running:
            # Check for pause state
            while self.paused and self.running:
                time.sleep(0.1)  # Wait while paused
                
            if not self.running:
                break
                
            try:
                # Create mock frame with annotations
                frame = self._create_mock_camera_frame("Break Line Camera", frame_counter)
                
                # Update CSV timeline position based on frame counter
                self.tracker.update_breakline_frame_index(frame_counter)
                self.tracker.update_frame_position(frame_counter, is_breakline_camera=True)
                
                # Generate mock dish detections occasionally
                dish_detections = []
                if frame_counter % 180 == 0:  # Every 3 seconds at 60fps
                    # Generate mock returned dish
                    x, y = 550, 350  # Center of mock ROI
                    dish_detections = [type('MockDish', (), {
                        'bbox': (x-30, y-20, x+30, y+20),
                        'confidence': 0.95,
                        'dish_type': 'normal_dish' if frame_counter % 360 == 0 else 'red_dish',
                        'center_point': (x, y),
                        'timestamp': datetime.now(),
                        'counting_point': (x, y+10)
                    })()]
                
                # Process mock detections
                if dish_detections:
                    self.tracker.process_dish_detections(dish_detections, roi_name="dish_detection")
                
                # Add system overlay and ROI visualization
                annotated_frame = self._draw_roi_rectangles(frame)
                annotated_frame = self._add_system_overlay(annotated_frame, "Break Line Camera (Mock)")
                
                # Update dashboard
                self.dashboard.update_camera_frame('break_line', annotated_frame)
                
                frame_counter += 1
                time.sleep(1.0 / 60)  # 60 FPS
                
            except Exception as e:
                self.logger.error(f"Error in mock break line camera: {e}")
                time.sleep(0.1)
    
    def _process_mock_kitchen_camera(self):
        """Generate mock kitchen camera feed when video files are missing"""
        self.logger.info("🎬 Starting mock kitchen camera feed")
        frame_counter = 0
        
        while self.running:
            # Check for pause state
            while self.paused and self.running:
                time.sleep(0.1)  # Wait while paused
                
            if not self.running:
                break
                
            try:
                # Create mock frame
                frame = self._create_mock_camera_frame("Kitchen Camera", frame_counter)
                
                # Update kitchen frame index with delay compensation
                delayed_frame_index = self.tracker.update_kitchen_frame_index(frame_counter)
                
                # Check for phase change signal
                current_state = self.tracker.get_current_state()
                phase_changed = self.tracker.check_phase_change_signal(current_state.current_phase)
                
                # Generate mock dish detections occasionally
                dish_detections = []
                if frame_counter % 240 == 0:  # Every 4 seconds at 60fps
                    # Generate mock kitchen dish
                    x, y = 950, 200  # Center of kitchen ROI
                    dish_detections = [type('MockDish', (), {
                        'bbox': (x-25, y-15, x+25, y+15),
                        'confidence': 0.90,
                        'dish_type': 'yellow_dish' if frame_counter % 480 == 0 else 'normal_dish',
                        'center_point': (x, y),
                        'timestamp': datetime.now(),
                        'counting_point': (x, y-8)
                    })()]
                
                # Process mock detections
                if dish_detections:
                    self.tracker.process_dish_detections(dish_detections, roi_name="kitchen_counter")
                
                # Add system overlay and ROI visualization
                annotated_frame = self._draw_kitchen_roi_rectangles(frame)
                annotated_frame = self._add_system_overlay(annotated_frame, "Kitchen Camera (Mock)")
                
                # Update dashboard
                self.dashboard.update_camera_frame('kitchen', annotated_frame)
                
                frame_counter += 1
                time.sleep(1.0 / 60)  # 60 FPS
                
            except Exception as e:
                self.logger.error(f"Error in mock kitchen camera: {e}")
                time.sleep(0.1)
    
    def _create_mock_camera_frame(self, camera_name: str, frame_counter: int) -> np.ndarray:
        """Create a mock camera frame with annotations"""
        # Create base frame (blue-gray background)
        frame = np.full((720, 1280, 3), (100, 120, 140), dtype=np.uint8)
        
        # Add title
        cv2.putText(frame, f"{camera_name} - Mock Feed", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_counter}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Add current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Add conveyor belt simulation (moving rectangle)
        belt_y = 350
        belt_width = 1000
        belt_height = 100
        belt_x = (frame_counter * 2) % (1280 + belt_width) - belt_width
        
        cv2.rectangle(frame, (belt_x, belt_y), (belt_x + belt_width, belt_y + belt_height), 
                     (80, 100, 120), -1)
        cv2.rectangle(frame, (belt_x, belt_y), (belt_x + belt_width, belt_y + belt_height), 
                     (60, 80, 100), 3)
        
        # Add mock dishes occasionally
        if frame_counter % 120 == 0:  # Every 2 seconds
            dish_x = belt_x + belt_width // 2
            dish_y = belt_y + belt_height // 2
            if 0 <= dish_x <= 1280:
                cv2.circle(frame, (dish_x, dish_y), 20, (50, 150, 50), -1)
                cv2.circle(frame, (dish_x, dish_y), 20, (30, 120, 30), 3)
        
        # Add status information
        try:
            state = self.tracker.get_current_state()
            status_text = f"Stage: {state.current_stage} | Phase: {state.current_phase}"
            cv2.putText(frame, status_text, (50, 680), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except:
            cv2.putText(frame, "Status: Initializing...", (50, 680), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        return frame
    
    def _scale_roi_coordinates(self, roi_coords: tuple, scale_factor: float) -> tuple:
        """Scale ROI coordinates based on frame scaling factor"""
        if roi_coords and scale_factor != 1.0:
            x1, y1, x2, y2 = roi_coords
            return (int(x1 * scale_factor), int(y1 * scale_factor), 
                    int(x2 * scale_factor), int(y2 * scale_factor))
        return roi_coords

    def _draw_roi_rectangles(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI rectangles on frame for visualization - using original scale coordinates"""
        try:
            # ROI coordinates are designed for dashboard display - use them as-is
            # No scaling applied since ROIs were drawn for the display size, not original video size
            
            # ROI colors (BGR format)
            roi_colors = {
                'dish_detection': (0, 255, 0),      # Green for dish detection
                'incoming_phase': (255, 0, 0),     # Blue for incoming phase  
                'return_phase': (0, 0, 255)        # Red for return phase
            }
            
            # ROI labels
            roi_labels = {
                'dish_detection': 'Dish Count ROI',
                'incoming_phase': 'Incoming Phase ROI', 
                'return_phase': 'Return Phase ROI'
            }
            
            # Get ROI coordinates from tracker (use original coordinates)
            roi_configs = {
                'dish_detection': self.tracker.roi_dish_detection,
                'incoming_phase': self.tracker.roi_incoming_phase,
                'return_phase': self.tracker.roi_return_phase
            }
            
            # Draw each ROI rectangle (exclude kitchen_counter for break line camera)
            for roi_name, roi_coords in roi_configs.items():
                if roi_coords:
                    # Use original coordinates - no scaling applied
                    x1, y1, x2, y2 = roi_coords
                    color = roi_colors.get(roi_name, (255, 255, 255))
                    label = roi_labels.get(roi_name, roi_name)
                    
                    # Draw rectangle with original thickness
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background with original font sizes
                    font_scale = 0.6
                    thickness = 2
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0] + 10, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing ROI rectangles: {e}")
            return frame
    
    def _draw_kitchen_roi_rectangles(self, frame: np.ndarray) -> np.ndarray:
        """Draw kitchen counter ROI rectangle on frame - using original scale"""
        try:
            # ROI coordinates are designed for dashboard display - use them as-is
            # No scaling applied since ROIs were drawn for the display size
            
            # Draw kitchen counter ROI
            x1, y1, x2, y2 = self.tracker.roi_kitchen_counter
            kitchen_color = (0, 255, 255)  # Yellow for kitchen counter
            kitchen_label = "Kitchen Counter ROI"
            
            font_scale = 0.6
            thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), kitchen_color, thickness)
            label_size = cv2.getTextSize(kitchen_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0] + 10, y1), kitchen_color, -1)
            cv2.putText(frame, kitchen_label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Incoming phase ROI drawing removed per user request
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing kitchen ROI rectangles: {e}")
            return frame
    
    def _draw_kitchen_roi_rectangle(self, frame: np.ndarray) -> np.ndarray:
        """Draw kitchen counter ROI rectangle on frame (legacy method)"""
        return self._draw_kitchen_roi_rectangles(frame)
    
    def _add_system_overlay(self, frame: np.ndarray, camera_name: str) -> np.ndarray:
        """Add system information overlay to frame"""
        overlay = frame.copy()
        
        # Get current state using new ROI-based methods
        try:
            phase_summary = self.tracker.get_phase_summary()
            total_dishes = self.tracker.get_total_dishes_on_belt()
            dish_serving_summary = self.tracker.get_dish_serving_summary()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Ensure all required keys exist
            phase_summary = {
                'current_stage': phase_summary.get('current_stage', 0),
                'current_phase': phase_summary.get('current_phase', 0),
                'last_return_phase': phase_summary.get('last_return_phase', 0),
                'is_initialized': phase_summary.get('is_initialized', False),
                'dishes_sent_current_phase': phase_summary.get('dishes_sent_current_phase', 0),
                'shifts_completed': phase_summary.get('shifts_completed', 0),
                'cycle_complete': phase_summary.get('cycle_complete', False)
            }
            
            dish_serving_summary = {
                'current_stage': dish_serving_summary.get('current_stage', 0),
                'current_phase': dish_serving_summary.get('current_phase', 0),
                'total_kitchen_dishes': dish_serving_summary.get('total_kitchen_dishes', 0),
                'total_returned_dishes': dish_serving_summary.get('total_returned_dishes', 0),
                'new_dishes_served': dish_serving_summary.get('new_dishes_served', 0),
                'equation': dish_serving_summary.get('equation', 'N/A')
            }
            
            # Different overlay content for different cameras
            if "Kitchen" in camera_name:
                # Kitchen camera - focus on serving statistics
                cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
                cv2.rectangle(overlay, (10, 10), (500, 180), (255, 255, 255), 2)
                
                texts = [
                    f"{camera_name} - {current_time}",
                    f"Stage: {dish_serving_summary['current_stage']} | Phase: {dish_serving_summary['current_phase']}",
                    f"Total Kitchen Dishes: {dish_serving_summary['total_kitchen_dishes']}",
                    f"Total Returned Dishes: {dish_serving_summary['total_returned_dishes']}",
                    f"New Dishes Served: {dish_serving_summary['new_dishes_served']}",
                    f"Equation: {dish_serving_summary['equation']}",
                    f"ROI: Kitchen Counter ({self.tracker.roi_kitchen_counter[0]}, {self.tracker.roi_kitchen_counter[1]}, {self.tracker.roi_kitchen_counter[2]}, {self.tracker.roi_kitchen_counter[3]})"
                ]
            else:
                # Break line camera - focus on conveyor belt tracking
                cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
                cv2.rectangle(overlay, (10, 10), (450, 200), (255, 255, 255), 2)
                
                texts = [
                    f"{camera_name} - {current_time}",
                    f"Stage: {phase_summary['current_stage']} | Phase: {phase_summary['current_phase']}",
                    f"Last Return: Phase {phase_summary['last_return_phase']}",
                    f"Phase Initialized: {'✓' if phase_summary['is_initialized'] else '✗'}",
                    f"Dishes to Customer - Normal: {total_dishes.get('normal_dish', 0)}",
                    f"Red: {total_dishes.get('red_dish', 0)} | Yellow: {total_dishes.get('yellow_dish', 0)}",
                    f"Current Phase Dishes: {phase_summary['dishes_sent_current_phase']}",
                    f"Shifts Completed: {phase_summary['shifts_completed']}",
                    f"Cycle Complete: {'✓' if phase_summary['cycle_complete'] else '✗'}"
                ]
            
            for i, text in enumerate(texts):
                y_pos = 25 + (i * 18)
                cv2.putText(overlay, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
            
        except Exception as e:
            self.logger.error(f"Error getting ROI tracking data: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to simple overlay with system state
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (400, 100), (255, 255, 255), 2)
            
            try:
                state = self.tracker.get_current_state()
                fallback_texts = [
                    f"{camera_name} - {current_time}",
                    f"Stage: {state.current_stage} | Phase: {state.current_phase}",
                    f"System Status: Running (Fallback Mode)"
                ]
            except:
                fallback_texts = [
                    f"{camera_name} - {current_time}",
                    "System Status: Initializing...",
                    "Data Loading..."
                ]
            
            for i, text in enumerate(fallback_texts):
                y_pos = 25 + (i * 18)
                cv2.putText(overlay, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.8
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def start_dashboard_server(self):
        """Start the dashboard server in a separate thread"""
        self.logger.info("Starting dashboard server")
        try:
            # Run dashboard with thread-safe settings
            self.dashboard.run(debug=False, threaded=True)
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
    
    def start(self, dashboard_in_main_thread: bool = False):
        """Start the entire system"""
        self.logger.info("Starting KichiKichi Conveyor Belt System")
        
        # Check if already running
        if self.running:
            self.logger.warning("System is already running")
            return True
        
        # Stop any existing threads first
        if self.break_line_thread and self.break_line_thread.is_alive():
            self.logger.info("Stopping existing break line thread")
            self.running = False
            self.break_line_thread.join(timeout=5)
        
        if self.kitchen_thread and self.kitchen_thread.is_alive():
            self.logger.info("Stopping existing kitchen thread")
            self.running = False
            self.kitchen_thread.join(timeout=5)
        
        # Initialize cameras
        if not self.initialize_cameras():
            self.logger.error("Failed to initialize cameras")
            return False
        
        # Clear any previous pause flag explicitly
        self.paused = False
        
        # Set running flag and start time
        self.running = True
        self.paused = False
        self.start_time = time.time()
        
        # Start processing threads
        self.break_line_thread = threading.Thread(
            target=self.process_break_line_camera,
            name="BreakLineProcessor"
        )
        self.kitchen_thread = threading.Thread(
            target=self.process_kitchen_camera,
            name="KitchenProcessor"
        )
        
        # Wait for both videos to be fully loaded before starting processing
        if self.video_load_sync['both_ready'].is_set():
            self.logger.info("✅ Both videos are already verified as ready - starting processing threads")
        else:
            self.logger.info("⏳ Performing final video synchronization check...")
            self._wait_for_videos_ready()
        
        # Start camera processing threads with synchronized loading
        self.logger.info("🎬 Starting synchronized video processing...")
        
        # Start both threads simultaneously since videos are pre-verified
        self.logger.info("📹 Starting breakline camera processing...")
        self.break_line_thread.start()
        
        self.logger.info("🍽️ Starting kitchen camera processing...")  
        self.kitchen_thread.start()
        
        # Update last connection time to avoid immediate watchdog restart
        try:
            self.user_connection_monitor['last_connection_time'] = time.time()
        except Exception:
            pass
        
        # Brief delay to let threads initialize
        time.sleep(0.5)
        
        self.logger.info("✅ Synchronized camera processing started successfully")
        self.logger.info(f"🌐 Dashboard available at: http://0.0.0.0:8050")
        
        # Log synchronization status
        sync_status = self.get_video_sync_status()
        if sync_status['videos_loaded']:
            load_diff = abs((sync_status['load_time_breakline'] or 0) - (sync_status['load_time_kitchen'] or 0))
            self.logger.info(f"📊 Video load synchronization: ✅ Success (difference: {load_diff:.2f}s)")
        else:
            self.logger.warning("📊 Video load synchronization: ⚠️ Warning - may experience bandwidth issues")
        
        if dashboard_in_main_thread:
            # Run dashboard in main thread (better for standalone dashboard)
            self.logger.info("Running dashboard in main thread")
            try:
                self.dashboard.run(debug=True)
            except KeyboardInterrupt:
                self.logger.info("Dashboard stopped by user")
                self.stop()
        else:
            # Run dashboard in background thread
            self.dashboard_thread = threading.Thread(
                target=self.start_dashboard_server,
                name="DashboardServer",
                daemon=True  # Allow main process to exit even if dashboard is running
            )
            self.dashboard_thread.start()
            self.logger.info("All systems started successfully")
        
        return True
    
    def _verify_detection_in_roi(self, detection, roi_coords) -> bool:
        """
        Verify that a number detection is actually within the specified ROI using bbox coordinates
        
        Args:
            detection: NumberDetection object with bbox coordinates
            roi_coords: Tuple (x1, y1, x2, y2) defining the ROI
            
        Returns:
            True if detection is within ROI, False otherwise
        """
        if not roi_coords or not detection.bbox:
            return False
            
        # Extract detection bbox
        det_x1, det_y1, det_x2, det_y2 = detection.bbox
        
        # Extract ROI coordinates
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
        
        # Check if detection center point is within ROI
        center_x = (det_x1 + det_x2) // 2
        center_y = (det_y1 + det_y2) // 2
        
        within_roi = (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2)
        
        if not within_roi:
            self.logger.debug(f"🚫 Detection {detection.number} center ({center_x},{center_y}) outside ROI ({roi_x1},{roi_y1},{roi_x2},{roi_y2})")
        else:
            self.logger.debug(f"✅ Detection {detection.number} center ({center_x},{center_y}) within ROI ({roi_x1},{roi_y1},{roi_x2},{roi_y2})")
            
        return within_roi
    
    def stop(self):
        """Stop the entire system"""
        self.logger.info("Stopping KichiKichi system")
        
        # Set running flag to False
        self.running = False
        
        # Wait for threads to finish
        if self.break_line_thread and self.break_line_thread.is_alive():
            self.break_line_thread.join(timeout=5)
        
        if self.kitchen_thread and self.kitchen_thread.is_alive():
            self.kitchen_thread.join(timeout=5)
        
        if hasattr(self, 'dashboard_thread') and self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=2)  # Shorter timeout for dashboard
        
        # Release camera resources
        if self.break_line_cap:
            self.break_line_cap.release()
        if self.kitchen_cap:
            self.kitchen_cap.release()
            
        # Release video writers
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        
        cv2.destroyAllWindows()
        
        self.logger.info("System stopped successfully")
    
    def pause(self):
        """Pause the system processing"""
        self.logger.info("Pausing KichiKichi system")
        self.paused = True
        self.pause_count += 1
        return True
    
    def resume(self):
        """Resume the system processing"""
        self.logger.info("Resuming KichiKichi system")
        self.paused = False
        return True
    
    def restart(self):
        """Restart the entire system"""
        self.logger.info("Restarting KichiKichi system")
        # Stop current operation
        was_running = self.running
        self.stop()
        
        # Reset tracker state
        self.tracker.reset_system()
        
        # Restart if it was running
        if was_running:
            return self.start()
        return True
    
    def reset_counts(self):
        """Reset all dish counts and tracking data"""
        self.logger.info("Resetting dish counts")
        self.tracker.reset_system()
        return True

    def hard_restart(self) -> None:
        """Hard restart the entire process by exec'ing run.py in dashboard mode"""
        try:
            # If under supervisor, prefer graceful soft restart to avoid port race
            if os.environ.get('KICHI_SUPERVISED') == '1':
                self.logger.warning("⚠️ Supervisor detected - performing soft restart instead of exec")
                try:
                    self.stop()
                    self.tracker.reset_system()
                except Exception:
                    pass
                # Let supervisor restart by exiting process with non-zero code
                sys.exit(1)
            self.logger.warning("⚠️ Hard restart requested - replacing current process")
            # Stop threads/resources gracefully
            try:
                self.stop()
            except Exception:
                pass
            # Exec replace current process
            python_exe = sys.executable
            run_path = "/home/hale/hale/run.py"
            os.execv(python_exe, [python_exe, run_path, "--mode", "dashboard"])
        except Exception as e:
            self.logger.error(f"Hard restart failed: {e}")
            return
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'running': self.running,
            'paused': self.paused,
            'uptime': uptime,
            'pause_count': self.pause_count,
            'cameras_active': {
                'breakline': self.break_line_cap is not None and (self.break_line_cap.isOpened() if self.break_line_cap else False),
                'kitchen': self.kitchen_cap is not None and (self.kitchen_cap.isOpened() if self.kitchen_cap else False)
            },
            'threads_active': {
                'breakline': self.break_line_thread.is_alive() if self.break_line_thread else False,
                'kitchen': self.kitchen_thread.is_alive() if self.kitchen_thread else False,
                'dashboard': self.dashboard_thread.is_alive() if self.dashboard_thread else False
            }
        }
    
    def restart_demo(self) -> bool:
        """Restart the POC demo from the beginning"""
        try:
            self.logger.info("🔄 Restarting POC demo...")
            
            # Stop current processing
            if hasattr(self, 'processing_active'):
                self.processing_active = False
            
            # Restart tracker
            if hasattr(self.tracker, 'restart_demo'):
                self.tracker.restart_demo()
            
            # Reset video positions
            if self.break_line_cap:
                self.break_line_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.logger.info("🎬 Break line camera reset to frame 0")
            
            if self.kitchen_cap:
                self.kitchen_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.logger.info("🎬 Kitchen camera reset to frame 0")
            
            # Reset sync counters
            with self.sync_lock:
                self.master_frame_index = 0
                self.kitchen_frame_index = 0
                self.sync_status = {
                    'is_synced': True,
                    'frame_difference': 0,
                    'sync_corrections': 0,
                    'last_sync_timestamp': datetime.now().isoformat()
                }
            
            # Restart processing
            if hasattr(self, 'processing_active'):
                self.processing_active = True
            
            self.logger.info("✅ POC demo restarted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error restarting demo: {e}")
            return False
    
    def run_console_mode(self):
        """Run in console mode with simple display and video sync status"""
        self.logger.info("Running in console mode")
        
        if not self.initialize_cameras():
            return
        
        # Show video synchronization status in console
        sync_status = self.get_video_sync_status()
        if sync_status['videos_loaded']:
            print("✅ Videos loaded and synchronized successfully")
            if sync_status['load_time_breakline'] and sync_status['load_time_kitchen']:
                load_diff = abs(sync_status['load_time_breakline'] - sync_status['load_time_kitchen'])
                print(f"📊 Load time difference: {load_diff:.2f}s")
        else:
            print("⚠️  Video synchronization warning - may experience bandwidth issues")
        
        self.running = True
        
        try:
            while self.running:
                # Process one frame from each camera
                ret1, frame1 = self.break_line_cap.read()
                ret2, frame2 = self.kitchen_cap.read()
                
                if ret1 and ret2:
                    # Process break line camera
                    dish_detections = self.dish_detector.detect_dishes(frame1)
                    break_line_region = self.number_detector.detect_break_line_region(frame1)
                    number_detections = self.number_detector.detect_numbers(frame1, break_line_region)
                    
                    # Update tracker
                    self.tracker.update_from_detections(dish_detections, number_detections)
                    
                    # Display info
                    state = self.tracker.get_current_state()
                    totals = self.tracker.get_total_dishes_on_belt()
                    
                    print(f"\rStage: {state.current_stage} | Phase: {state.current_phase} | "
                          f"Dishes: N:{totals['normal_dish']} R:{totals['red_dish']} Y:{totals['yellow_dish']} | "
                          f"Rates: R:{state.dishes_per_minute['red_dish']}/min Y:{state.dishes_per_minute['yellow_dish']}/min",
                          end='', flush=True)
                
                # Small delay
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping system...")
        finally:
            self.stop()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="KichiKichi Conveyor Belt Dish Counting System")
    parser.add_argument('--mode', choices=['dashboard', 'console'], default='dashboard',
                       help='Run mode: dashboard (web interface) or console (terminal output)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--export-video', action='store_true', help='Export annotated video streams to files')
    parser.add_argument('--auto-restart', action='store_true', 
                       help='Enable auto-restart on user connect/disconnect for clean sessions')
    
    args = parser.parse_args()
    
    # Create application with auto-restart capability
    app = KichiKichiApp(export_video=args.export_video, auto_restart=args.auto_restart)
    
    if args.auto_restart:
        print("🔄 Auto-restart enabled - system will restart on user connect/disconnect")
        print("👤 This ensures clean state for each user session")
    
    try:
        if args.mode == 'dashboard':
            # Start only the dashboard server, let user control system start via UI
            app.start_dashboard_server()
        else:
            # Run in console mode
            app.run_console_mode()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        app.stop()

if __name__ == "__main__":
    main()
