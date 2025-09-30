"""
CSV-Based Conveyor Tracker for KichiKichi
Replaces ROI-based phase detection with CSV timeline data
Maintains dish counting per phase functionality with ROI-based dish tracking
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from utils.csv_timeline_parser import CSVTimelineParser, TimelineEntry
from utils.roi_config_loader import get_tracker_roi_config
from dish_detection.dish_detector import DishDetection
from ocr_model.number_detector import NumberDetection
from tracking.backend_cache import BackendCache


# ROI configurations are loaded from /home/hale/hale/exports JSON files
ROI_CONFIG = get_tracker_roi_config()


@dataclass
class TrackedDish:
    """Track a dish's movement for ROI crossing detection (same as conveyor_tracker.py)"""
    id: str
    dish_type: str
    center_point: Tuple[int, int]
    # previous_center_point: Optional[Tuple[int, int]] = None
    last_seen: datetime
    roi_status: Dict[str, bool] = field(default_factory=dict)  # Which ROIs the dish is currently in
    crossed_rois: Set[str] = field(default_factory=set)       # Which ROIs the dish has crossed through
    count_point: Optional[Tuple[int, int]] = None             # Top-third counting point


@dataclass
class PhaseTrackingData:
    """Enhanced phase data with dish tracking"""
    phase_number: int
    frame_ranges: List[Tuple[int, int]]  # Frame ranges where this phase is active
    dish_counts: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0, 
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    total_dishes: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    is_active: bool = False
    dishes_processed: List[DishDetection] = field(default_factory=list)


@dataclass
class CSVConveyorState:
    """Conveyor state using CSV timeline data"""
    current_stage: int = 0
    current_phase: int = 0
    previous_stage: int = 0
    previous_phase: int = 0
    current_frame: int = 0
    
    # Demo completion state
    demo_completed: bool = False
    demo_completion_time: datetime = field(default_factory=datetime.now)
    
    # Dashboard compatibility fields
    is_phase_initialized: bool = False
    last_return_stage: int = 0
    last_return_phase: int = 0
    # Calibration flag: have we seen the first previous-phase (return window) update yet?
    first_prev_phase_seen: bool = False
    
    # Phase tracking data
    phase_data: Dict[int, PhaseTrackingData] = field(default_factory=dict)
    
    # Overall dish counts
    total_dishes_processed: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Kitchen and return dishes tracking (compatibility with existing system)
    dishes_to_customer: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    dishes_returning: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Current stage dishes (reset when stage changes)
    current_stage_dishes: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Kitchen dishes by type (compatibility)
    kitchen_dishes_served: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Persistent belt counters (not reset by phase changes)
    belt_counts: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0
    })
    belt_added_cumulative: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0
    })
    belt_returned_cumulative: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0
    })
    belt_calibration_phase10_entry_frame: int = -1
    belt_calibration_active: bool = False
    
    # Phase-specific dish tracking (stores dish counts for each phase)
    phase_dish_tracking: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # Rate calculations
    dishes_per_minute: Dict[str, float] = field(default_factory=lambda: {
        'red_dish': 0.0,
        'yellow_dish': 0.0
    })
    
    # Legacy compatibility fields
    total_kitchen_dishes_served: int = 0  # CUMULATIVE - never reset
    total_returned_dishes: int = 0
    new_dishes_served: int = 0
    last_calculation_time: datetime = field(default_factory=datetime.now)
    last_dish_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # New tracking fields for requirements
    total_dishes_counted_ever: int = 0  # Total dishes from kitchen camera (never reset)
    breakline_frame_index: int = 0  # Frame counter for breakline camera
    
    # Stage transition tracking
    previous_stage_dish_count: int = 0  # Dish count when previous stage finished
    current_stage_dish_count: int = 0   # Current stage dish count
    dishes_taken_out: int = 0           # Dishes removed from conveyor
    dishes_added_in: int = 0            # Dishes added to conveyor
    
    # Latest phases tracking for table display
    latest_phases: List[Any] = field(default_factory=list)  # Track last 2 phases (can be int or "waiting")
    latest_phase_dishes: Dict[Any, Dict[str, int]] = field(default_factory=dict)  # Dish counts for latest phases

    # Two-latest-stage tables (stage -> phase -> counts)
    stage_phase_tables: Dict[int, Dict[int, Dict[str, int]]] = field(default_factory=dict)
    latest_stages: List[int] = field(default_factory=list)
    # Per-stage running totals
    stage_totals: Dict[int, Dict[str, int]] = field(default_factory=dict)  # stage -> {kitchen_total, returned_total}
    stage_metrics: Dict[int, Dict[str, int]] = field(default_factory=dict)  # stage -> {taken_out, added_in}


class CSVConveyorTracker:
    """
    CSV-based conveyor tracker that uses pre-recorded timeline data
    instead of real-time ROI detection for phase and stage information.
    Uses ROI-based dish counting like conveyor_tracker.py for accurate dish tracking.
    """
    
    def __init__(self, csv_timeline_path: str, dishes_per_phase_estimate: float = 2.0, kitchen_delay_frames: int = 0, phase_latency_compensation: int = 15):
        """
        Initialize CSV-based conveyor tracker
        
        Args:
            csv_timeline_path: Path to the timeline CSV file
            dishes_per_phase_estimate: Average dishes processed per phase
            kitchen_delay_frames: Number of frames to delay kitchen camera relative to breakline
            phase_latency_compensation: Frames to anticipate phase changes for latency compensation
        """
        self.logger = logging.getLogger(__name__)
        self.timeline_parser = CSVTimelineParser(csv_timeline_path)
        self.dishes_per_phase_estimate = dishes_per_phase_estimate
        self.kitchen_delay_frames = kitchen_delay_frames
        self.phase_latency_compensation = phase_latency_compensation
        
        # Initialize backend cache for accuracy-first synchronization
        self.backend_cache = BackendCache()
        self.logger.info("🏪 CSV Tracker initialized with backend caching for accuracy")
        
        # ROI configurations (same as conveyor_tracker.py)
        self.roi_dish_detection = ROI_CONFIG['dish_detection']
        self.roi_incoming_phase = ROI_CONFIG['incoming_phase'] 
        self.roi_return_phase = ROI_CONFIG['return_phase']
        self.roi_kitchen_counter = ROI_CONFIG['kitchen_counter']
        
        # State management
        self.state = CSVConveyorState()
        self.lock = threading.Lock()
        
        # Camera delay management
        self.kitchen_frame_index = 0  # Kitchen camera frame counter
        
        # Dish tracking for ROI crossing detection (same as conveyor_tracker.py)
        self.tracked_dishes: Dict[str, TrackedDish] = {}
        self.dish_tracking_threshold = 100  # Distance threshold for tracking same dish
        self.dish_timeout = 3.0  # Seconds before forgetting a dish
        
        # Rate calculation data
        self.rate_calculation_data: Dict[str, deque] = {
            'red_dish': deque(maxlen=60),
            'yellow_dish': deque(maxlen=60)
        }
        
        # Tracking data
        self.dish_history: deque = deque(maxlen=1000)
        
        # Cumulative dish counter protection (never decreases)
        self.max_total_dishes_ever = 0
        
        # Breakline frame tracking 
        self.last_breakline_frame = 0
        
        # Rate calculation tracking
        self.dish_rate_history = {
            'red_dish': deque(maxlen=60),  # Last 60 data points for rate calculation
            'yellow_dish': deque(maxlen=60),
            'timestamps': deque(maxlen=60)
        }
        
        # Initialize phase tracking data
        self._initialize_phase_tracking()
        
        # Get maximum frame for end detection
        self.max_frame_index = self.timeline_parser.get_max_frame_index()
        self.logger.info(f"📊 CSV timeline covers frames up to {self.max_frame_index}")
        
        self.logger.info("✅ CSV-based conveyor tracker initialized")

    def _update_dish_rates(self):
        """Update dishes per minute calculation based on recent dish detections"""
        current_time = datetime.now()
        
        # Add current counts with timestamp
        red_count = self.state.current_stage_dishes.get('red_dish', 0)
        yellow_count = self.state.current_stage_dishes.get('yellow_dish', 0)
        
        self.dish_rate_history['red_dish'].append(red_count)
        self.dish_rate_history['yellow_dish'].append(yellow_count) 
        self.dish_rate_history['timestamps'].append(current_time)
        
        # Calculate rates if we have enough data points (at least 10 points)
        if len(self.dish_rate_history['timestamps']) >= 10:
            time_span = (self.dish_rate_history['timestamps'][-1] - self.dish_rate_history['timestamps'][0]).total_seconds()
            
            if time_span > 0:
                # Calculate dishes per minute based on recent activity
                red_recent = sum(list(self.dish_rate_history['red_dish'])[-10:])
                yellow_recent = sum(list(self.dish_rate_history['yellow_dish'])[-10:])
                
                minutes = time_span / 60.0
                self.state.dishes_per_minute['red_dish'] = red_recent / minutes if minutes > 0 else 0.0
                self.state.dishes_per_minute['yellow_dish'] = yellow_recent / minutes if minutes > 0 else 0.0
            else:
                self.state.dishes_per_minute['red_dish'] = 0.0
                self.state.dishes_per_minute['yellow_dish'] = 0.0
        else:
            # Not enough data points
            self.state.dishes_per_minute['red_dish'] = 0.0
            self.state.dishes_per_minute['yellow_dish'] = 0.0

    def _ensure_stage_table(self, stage: int) -> None:
        if stage not in self.state.stage_phase_tables:
            self.state.stage_phase_tables[stage] = {}
        if stage not in self.state.stage_totals:
            self.state.stage_totals[stage] = {"kitchen_total": 0, "returned_total": 0}

    def _ensure_stage_phase_entry(self, stage: int, phase: int) -> None:
        self._ensure_stage_table(stage)
        if phase not in self.state.stage_phase_tables[stage]:
            self.state.stage_phase_tables[stage][phase] = {
                'normal_dish': 0,
                'red_dish': 0,
                'yellow_dish': 0,
                'advertisement_dish': 0,
                'total': 0
            }
    
    def _generate_dish_id(self, center_point: Tuple[int, int], dish_type: str) -> str:
        """Generate unique ID for dish tracking (same as conveyor_tracker.py)"""
        return f"{dish_type}_{center_point[0]}_{center_point[1]}_{int(datetime.now().timestamp() * 1000) % 10000}"
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between two points (same as conveyor_tracker.py)"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def _is_point_in_roi(self, point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside ROI (same as conveyor_tracker.py)"""
        x, y = point
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _initialize_phase_tracking(self) -> None:
        """Initialize phase tracking data from CSV timeline"""
        all_phases = self.timeline_parser.get_all_phases()
        
        for phase in all_phases:
            # Get frame ranges for this phase
            frame_ranges = self.timeline_parser.get_phase_frame_ranges(phase)
            
            # Create phase tracking data
            self.state.phase_data[phase] = PhaseTrackingData(
                phase_number=phase,
                frame_ranges=frame_ranges
            )
            
            # Initialize phase dish tracking for compatibility
            self.state.phase_dish_tracking[phase] = {
                'normal_dish': 0,
                'red_dish': 0,
                'yellow_dish': 0,
                'advertisement_dish': 0
            }
        
        # Estimate initial dish counts per phase
        self._estimate_dishes_per_phase()
        
        self.logger.info(f"📊 Initialized tracking for {len(all_phases)} phases")
    
    def _estimate_dishes_per_phase(self) -> None:
        """Estimate dish counts per phase based on timeline statistics"""
        phase_stats = self.timeline_parser.phase_statistics
        
        for phase, tracking_data in self.state.phase_data.items():
            stats = phase_stats.get(phase)
            if not stats:
                continue
            
            # Estimate dishes based on frame count and stages
            base_estimate = max(1, int(stats.total_frames / 500))  # ~1 dish per 500 frames
            stage_multiplier = len(stats.stage_occurrences)
            estimated_total = base_estimate * max(1, stage_multiplier) * self.dishes_per_phase_estimate
            
            # Distribute among dish types (proportional estimation)
            tracking_data.dish_counts['normal_dish'] = int(estimated_total * 0.5)
            tracking_data.dish_counts['red_dish'] = int(estimated_total * 0.2)
            tracking_data.dish_counts['yellow_dish'] = int(estimated_total * 0.2)
            tracking_data.dish_counts['advertisement_dish'] = int(estimated_total * 0.1)
            tracking_data.total_dishes = int(estimated_total)
            
            # NOTE: phase_dish_tracking is reserved for direct kitchen ROI counts only
            # Do NOT populate with estimates - it must start at 0 and only increment on actual crossings
            
            self.logger.info(f"📈 Phase {phase}: Estimated {int(estimated_total)} total dishes")
    
    def update_frame_position(self, frame_index: int, is_breakline_camera: bool = True) -> bool:
        """
        Update current frame position and get stage/phase information
        Only updates state when breakline camera reaches the frame (for synchronization)
        
        Args:
            frame_index: Current frame number in the video
            is_breakline_camera: True if this update is from breakline camera
            
        Returns:
            True if stage/phase information was updated, False otherwise
        """
        with self.lock:
            # Only update timeline from breakline camera for proper synchronization
            if not is_breakline_camera:
                return False
                
            # Track breakline frame progression
            if frame_index <= self.last_breakline_frame:
                # Don't go backwards or stay on same frame
                return False
            self.last_breakline_frame = frame_index
            
            old_stage = self.state.current_stage
            old_phase = self.state.current_phase
            old_previous_phase = self.state.previous_phase
            
            # Get timeline data for current frame (no latency compensation for CSV mode)
            # CSV data is pre-recorded and accurate, latency compensation can cause wrong phase selection
            timeline_data = self.timeline_parser.get_stage_phase_for_frame(frame_index)
            
            if timeline_data:
                self.logger.debug(f"🎯 CSV SYNC: Frame {frame_index} -> Stage {timeline_data[0]}, Phase {timeline_data[1]}")
            else:
                # Try with small compensation only if no exact match (max 5 frames = 0.17 seconds)
                compensated_frame = frame_index + min(5, self.phase_latency_compensation)
                timeline_data = self.timeline_parser.get_stage_phase_for_frame(compensated_frame)
                if timeline_data:
                    self.logger.debug(f"🎯 CSV SYNC (compensated +{compensated_frame - frame_index}): Frame {frame_index} -> Stage {timeline_data[0]}, Phase {timeline_data[1]}")
            
            if timeline_data:
                current_stage, current_phase, previous_stage, previous_phase = timeline_data
                
                # Update state
                self.state.current_frame = frame_index
                self.state.current_stage = current_stage
                self.state.current_phase = current_phase
                self.state.previous_stage = previous_stage
                self.state.previous_phase = previous_phase
                self.state.last_return_stage = previous_stage
                self.state.last_return_phase = previous_phase
                
                # Mark phase as initialized
                if not self.state.is_phase_initialized:
                    self.state.is_phase_initialized = True
                    self.logger.info("✅ Phase tracking initialized from CSV timeline")
                
                # Check if stage changed
                stage_changed = (current_stage != old_stage)
                if stage_changed:
                    # Preserve last phase counts for the completed stage using the previous phase value
                    self._handle_stage_change(old_stage, current_stage, last_phase=old_phase)
                    # Avoid double-storing the completed phase in the subsequent phase change handler
                    self._skip_phase_storage_once = True
                
                # Check if phase changed
                if current_phase != old_phase:
                    self._handle_phase_change(old_phase, current_phase)

                # Previous phase change indicates return-phase window switch
                if previous_phase != old_previous_phase:
                    self._handle_previous_phase_change(old_previous_phase, previous_phase)
                
                # Update phase active status
                self._update_phase_active_status(current_phase)

                # Belt calibration: Stage 0 Phase 10, wait +100 frames before activating
                try:
                    if self.state.current_stage == 0 and self.state.current_phase == 10:
                        if self.state.belt_calibration_phase10_entry_frame < 0:
                            self.state.belt_calibration_phase10_entry_frame = frame_index
                        if (not self.state.belt_calibration_active and
                            self.state.belt_calibration_phase10_entry_frame >= 0 and
                            frame_index >= self.state.belt_calibration_phase10_entry_frame + 100):
                            self.state.belt_calibration_active = True
                            self.logger.info("📏 Belt calculation activated at Stage 0 Phase 10 (+100 frames)")
                except Exception:
                    pass

                # Demo completion condition: stop at Stage 4 Phase 0
                # This ensures we end the demo exactly when stage 4 begins (phase 0)
                if current_stage == 4 and current_phase == 0 and not self.state.demo_completed:
                    self.logger.info("🎉 POC DEMO COMPLETED - Stage 4 Phase 0 reached!")
                    self.state.demo_completed = True
                    self.state.demo_completion_time = datetime.now()
                    # Generate end-of-shift/business report
                    self.generate_end_shift_report()
                    self.logger.info("🛑 Demo stopped - waiting for user to restart")
                
                return True
            else:
                # No timeline data found - check if we've truly reached the end
                if frame_index >= self.max_frame_index:
                    # We've reached the end of CSV timeline
                    if self.state.is_phase_initialized and not self.state.demo_completed:
                        self.logger.info("🎉 POC DEMO COMPLETED - End of CSV timeline reached!")
                        self.state.demo_completed = True
                        self.state.demo_completion_time = datetime.now()
                        self.generate_end_shift_report()
                        self.logger.info("🛑 Demo stopped - waiting for user to restart")
                    return False
                else:
                    # Frame data not found but we haven't reached the end yet - continue with "waiting" state
                    self.logger.debug(f"⏳ No timeline data for frame {frame_index}, continuing with waiting state")
                    
                    # Set system to un-initialized when no phase data is available
                    if self.state.is_phase_initialized:
                        self.state.is_phase_initialized = False
                        self.logger.debug("⚠️ System set to un-initialized state (no phase data)")
                    
                    return True  # Continue processing, don't stop yet
    
    def _handle_phase_change(self, old_phase: int, new_phase: int) -> None:
        """Handle phase transition with camera counter resets"""
        self.logger.info(f"🔄 PHASE TRANSITION: {old_phase} → {new_phase}")
        
        # Reset camera counters on phase change
        self.logger.info("🔄 Resetting camera counters for new phase")
        
        # Store old counts for logging
        old_counts = self.state.current_stage_dishes.copy()
        old_return_counts = self.state.dishes_returning.copy()
        
        # Reset current stage dishes (Forward Line camera counters)
        for dish_type in self.state.current_stage_dishes.keys():
            self.state.current_stage_dishes[dish_type] = 0
        
        # Reset return dishes counter (Backward Line camera counters)  
        for dish_type in self.state.dishes_returning.keys():
            self.state.dishes_returning[dish_type] = 0
        
        # Log the reset
        total_reset_forward = sum(old_counts.values())
        total_reset_backward = sum(old_return_counts.values())
        self.logger.info(f"📊 Phase {old_phase}→{new_phase}: Reset Forward Line ({total_reset_forward}) and Backward Line ({total_reset_backward}) counters")
        
        # Ensure new phase is initialized in phase tracking
        if new_phase not in self.state.phase_dish_tracking:
            self.state.phase_dish_tracking[new_phase] = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }
    
    def _handle_stage_change(self, old_stage: int, new_stage: int, last_phase: Optional[int] = None) -> None:
        """Handle stage transition with dish tracking, counter resets, and phase data preservation"""
        self.logger.info(f"📊 STAGE TRANSITION: {old_stage} → {new_stage}")
        
        # Store current phase data before stage transition
        current_phase = self.state.current_phase if last_phase is None else last_phase
        if current_phase >= 0:
            # Ensure old stage exists in stage_phase_tables
            if old_stage not in self.state.stage_phase_tables:
                self.state.stage_phase_tables[old_stage] = {}
            
            # Store final phase data for completed stage
            current_phase_counts = self.state.phase_dish_tracking.get(current_phase, {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }).copy()
            
            self.state.stage_phase_tables[old_stage][current_phase] = current_phase_counts
            
            phase_total = sum(count for key, count in current_phase_counts.items() 
                             if key in ['normal_dish', 'red_dish', 'yellow_dish'] and isinstance(count, int))
            
            self.logger.info(f"📊 Preserved final Phase {current_phase} data for completed Stage {old_stage}: {phase_total} dishes")
        
        # Calculate stage transition dish tracking based on current stage total
        old_total = sum(self.state.current_stage_dishes.values())
        
        # Store previous stage dish count and current stage count
        self.state.previous_stage_dish_count = old_total
        self.state.current_stage_dish_count = old_total  # Update current stage total before reset
        
        # Calculate dishes taken out or added based on actual dish movement
        if old_total > 0:  # Had dishes in previous stage
            # All dishes from previous stage are considered taken out
            self.state.dishes_taken_out = old_total  
            self.state.dishes_added_in = 0  # New stage starts with 0 dishes
            self.logger.info(f"📤 Stage {old_stage} → {new_stage}: {old_total} dishes taken out of conveyor")
        else:
            self.state.dishes_taken_out = 0
            self.state.dishes_added_in = 0
        
        # Enhanced transition calculation logic
        try:
            self._ensure_stage_table(old_stage)
            totals = self.state.stage_totals.get(old_stage, {"kitchen_total": 0, "returned_total": 0})
            
            # Calculate more accurate transition metrics
            kitchen_total = totals.get("kitchen_total", 0)
            returned_total = totals.get("returned_total", 0)
            
            # Dishes taken out = dishes that were on conveyor but not returned
            # Dishes added in = new dishes from kitchen minus what was returned
            dishes_taken_by_customers = max(0, kitchen_total - returned_total)
            dishes_still_on_belt = returned_total  # These came back
            
            self.state.stage_metrics[old_stage] = {
                'taken_out': dishes_taken_by_customers,
                'added_in': kitchen_total,
                'returned': returned_total,
                'net_served': dishes_taken_by_customers
            }
            
            self.logger.info(f"📊 Stage {old_stage} final metrics: Added={kitchen_total}, Returned={returned_total}, Net served={dishes_taken_by_customers}")
        except Exception as e:
            self.logger.debug(f"Stage metrics finalize error: {e}")

        # Maintain latest stages list (keep last 2)
        if old_stage not in self.state.latest_stages and old_stage >= 0:
            self.state.latest_stages.append(old_stage)
            if len(self.state.latest_stages) > 2:
                self.state.latest_stages.pop(0)

        # Reset counters for new stage (Kitchen Counter and Break Line Camera)
        for dish_type in self.state.current_stage_dishes:
            self.state.current_stage_dishes[dish_type] = 0
        
        for dish_type in self.state.dishes_returning:
            self.state.dishes_returning[dish_type] = 0
        
        # Reset total returned dishes for new stage
        self.state.total_returned_dishes = 0
        
        # Reset current stage dish count to 0 for new stage
        self.state.current_stage_dish_count = 0
        
        # Reset phase dish tracking for new stage (all phases start fresh)
        self.state.phase_dish_tracking.clear()
        
        # Initialize new stage's first phase (usually 0)
        first_phase = self.state.current_phase
        self.state.phase_dish_tracking[first_phase] = {
            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
        }
        
        # Prepare tables/totals for new stage
        self._ensure_stage_table(new_stage)

        self.logger.info(f"🔄 New stage {new_stage}: Reset Kitchen Counter, Break Line Camera, and Phase counters")
        self.logger.info(f"✨ Phase {first_phase} reset to 0 for new stage incremental tracking")
        self.logger.info(f"📊 Stage tracking: Previous={self.state.previous_stage_dish_count}, Current=0, Taken out={self.state.dishes_taken_out}, Added={self.state.dishes_added_in}")
    
    def _handle_phase_change(self, old_phase: int, new_phase: int) -> None:
        """Handle phase transition with incremental phase counting and stage-phase table storage"""
        self.logger.info(f"🔄 PHASE TRANSITION: {old_phase} → {new_phase} (Stage {self.state.current_stage})")
        
        # Store completed phase data in stage_phase_tables for historical tracking
        if old_phase >= 0:  # Valid old phase
            # Skip once right after stage change because stage handler already stored last phase
            if hasattr(self, '_skip_phase_storage_once') and self._skip_phase_storage_once:
                delattr(self, '_skip_phase_storage_once')
            else:
                current_stage = self.state.current_stage
                
                # Ensure stage exists in stage_phase_tables
                if current_stage not in self.state.stage_phase_tables:
                    self.state.stage_phase_tables[current_stage] = {}
                
                # Get current dish counts for the completed phase
                old_phase_counts = self.state.phase_dish_tracking.get(old_phase, {
                    'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
                }).copy()
                
                # Store completed phase data
                self.state.stage_phase_tables[current_stage][old_phase] = old_phase_counts
                
                phase_total = sum(count for key, count in old_phase_counts.items() 
                                 if key in ['normal_dish', 'red_dish', 'yellow_dish'] and isinstance(count, int))
                
                self.logger.info(f"📊 Stored Phase {old_phase} data for Stage {current_stage}: {phase_total} dishes " +
                               f"(N:{old_phase_counts.get('normal_dish', 0)}, R:{old_phase_counts.get('red_dish', 0)}, Y:{old_phase_counts.get('yellow_dish', 0)})")
        
        # Reset phase dish tracking for new phase (incremental counting)
        self.state.phase_dish_tracking[new_phase] = {
            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
        }
        
        self.logger.info(f"✨ NEW PHASE {new_phase}: Reset counters to 0 for incremental tracking")
        
        # Deactivate old phase
        if old_phase in self.state.phase_data:
            self.state.phase_data[old_phase].is_active = False
            self.state.phase_data[old_phase].end_time = datetime.now()
        
        # Activate new phase
        if new_phase in self.state.phase_data:
            self.state.phase_data[new_phase].is_active = True
            if self.state.phase_data[new_phase].start_time is None:
                self.state.phase_data[new_phase].start_time = datetime.now()
        
        # Update latest phases tracking (keep last 2 phases)
        if new_phase not in self.state.latest_phases:
            self.state.latest_phases.append(new_phase)
            # Keep only last 2 phases
            if len(self.state.latest_phases) > 2:
                old_phase_to_remove = self.state.latest_phases.pop(0)
                # Remove from latest_phase_dishes if exists
                if old_phase_to_remove in self.state.latest_phase_dishes:
                    del self.state.latest_phase_dishes[old_phase_to_remove]
        
        # Initialize latest phase dishes tracking (starts at 0 for incremental counting)
        if new_phase not in self.state.latest_phase_dishes:
            self.state.latest_phase_dishes[new_phase] = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }
            if self.state.phase_data[new_phase].start_time is None:
                self.state.phase_data[new_phase].start_time = datetime.now()

        # Ensure stage-phase table exists for counting  
        self._ensure_stage_phase_entry(self.state.current_stage, new_phase)
    
    def _ensure_stage_phase_entry(self, stage: int, phase: int) -> None:
        """Ensure stage-phase entry exists in stage_phase_tables"""
        if stage not in self.state.stage_phase_tables:
            self.state.stage_phase_tables[stage] = {}
        
        if phase not in self.state.stage_phase_tables[stage]:
            self.state.stage_phase_tables[stage][phase] = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }

    def _handle_previous_phase_change(self, old_prev_phase: int, new_prev_phase: int) -> None:
        """Mark completion of return from previous phase window."""
        if old_prev_phase != new_prev_phase and self.state.is_phase_initialized:
            self.logger.info(f"✅ Return window completed for phase {old_prev_phase}; new previous phase: {new_prev_phase}")
            # Mark calibration point for belt count once we observe the first non-zero previous phase
            if not self.state.first_prev_phase_seen and new_prev_phase > 0:
                self.state.first_prev_phase_seen = True
                self.logger.info("📏 Belt count calibration active (first previous-phase update seen)")
    
    def _update_phase_active_status(self, current_phase: int) -> None:
        """Update which phase is currently active"""
        for phase, tracking_data in self.state.phase_data.items():
            tracking_data.is_active = (phase == current_phase)
    
    def update_breakline_frame_index(self, frame_index: int) -> None:
        """
        Update the breakline camera frame index for breakline frame counter
        
        Args:
            frame_index: Current breakline camera frame number
        """
        with self.lock:
            self.state.breakline_frame_index = frame_index
            self.logger.debug(f"📹 Breakline frame index updated: {frame_index}")
    
    def update_kitchen_frame_index(self, frame_index: int) -> None:
        """
        Update kitchen camera frame index with delay compensation
        
        Args:
            frame_index: Current kitchen camera frame number
        """
        with self.lock:
            self.kitchen_frame_index = frame_index
            # Apply delay to match with breakline timing
            delayed_frame_index = max(0, frame_index - self.kitchen_delay_frames)
            self.logger.debug(f"🎯 Kitchen frame index: {frame_index}, delayed: {delayed_frame_index}")
            return delayed_frame_index
    
    def sync_video_reset(self) -> None:
        """
        Synchronize video reset between cameras to prevent timing drift
        """
        with self.lock:
            self.kitchen_frame_index = 0
            self.state.breakline_frame_index = 0
            # Reset any frame-based tracking
            self.last_breakline_frame = -1
            self.logger.info("📹 Video sync reset: Both cameras synchronized to frame 0")
    
    def get_kitchen_frame_for_breakline_sync(self, breakline_frame: int) -> int:
        """
        Calculate which kitchen frame corresponds to a given breakline frame for synchronization
        
        Args:
            breakline_frame: Breakline camera frame number
            
        Returns:
            Corresponding kitchen camera frame number
        """
        # Kitchen camera should be ahead by delay_frames to match breakline timing
        return breakline_frame + self.kitchen_delay_frames
    
    def check_phase_change_signal(self, current_phase: int) -> bool:
        """
        Check if a phase change occurred and send signal to kitchen camera
        
        Args:
            current_phase: Current phase number
            
        Returns:
            True if phase changed, False otherwise
        """
        with self.lock:
            old_phase = getattr(self, '_last_signaled_phase', None)
            if old_phase is None or old_phase != current_phase:
                self._last_signaled_phase = current_phase
                self.logger.info(f"🚨 PHASE CHANGE SIGNAL: Kitchen camera should start counting for Phase {current_phase}")
                return True
            return False
    
    def _reset_system_to_initial_state(self) -> None:
        """Reset the entire system to initial state as if just started"""
        with self.lock:
            # Reset all counters to initial values
            self.state.current_stage_dishes = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }
            self.state.dishes_returning = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }
            self.state.kitchen_dishes_served = {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
            }
            
            # Reset cumulative counters
            self.state.total_kitchen_dishes_served = 0
            self.state.total_returned_dishes = 0
            self.state.new_dishes_served = 0
            # Also reset ever counters/history to meet "no number and no history" requirement
            self.state.total_dishes_counted_ever = 0
            
            # Reset new tracking fields (keep total_dishes_counted_ever as cumulative)
            self.state.breakline_frame_index = 0
            self.state.previous_stage_dish_count = 0
            self.state.current_stage_dish_count = 0
            self.state.dishes_taken_out = 0
            self.state.dishes_added_in = 0
            
            # Clear latest phases tracking
            self.state.latest_phases = []
            self.state.latest_phase_dishes = {}
            
            # Clear phase tracking
            self.state.phase_dish_tracking = {}
            
            # Reset phase data
            self.state.phase_data = {}

            # Clear stage/phase summary tables and metrics (no history)
            self.state.stage_phase_tables = {}
            self.state.latest_stages = []
            self.state.stage_totals = {}
            self.state.stage_metrics = {}

            # Reset persistent belt counters and calibration state
            self.state.belt_counts = {'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0}
            self.state.belt_added_cumulative = {'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0}
            self.state.belt_returned_cumulative = {'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0}
            self.state.belt_calibration_phase10_entry_frame = -1
            self.state.belt_calibration_active = False
            
            # Reset tracking dictionaries
            self.tracked_dishes = {}
            self.rate_calculation_data['red_dish'].clear()
            self.rate_calculation_data['yellow_dish'].clear()
            
            # Reset state values but keep current frame
            self.state.current_stage = 0
            self.state.current_phase = 0
            self.state.previous_stage = 0
            self.state.previous_phase = 0
            self.state.last_return_stage = 0
            self.state.last_return_phase = 0
            
            # Mark as uninitialized
            self.state.is_phase_initialized = False
            
            # Reset demo completion state
            self.state.demo_completed = False
            self.state.demo_completion_time = datetime.now()
            
            self.logger.info("🔄 System completely reset to initial state")
    
    def _update_dish_tracking(self, dish_detections: List[DishDetection]) -> List[str]:
        """
        Update dish tracking and detect ROI crossings for break line camera
        Same logic as conveyor_tracker.py
        
        Args:
            dish_detections: List of dish detections from break line camera
            
        Returns:
            List of dish IDs that crossed ROI boundaries
        """
        current_time = datetime.now()
        crossed_dishes = []
        
        # Clean up old tracked dishes
        self._cleanup_old_tracked_dishes(current_time)
        
        for detection in dish_detections:
            # Counting point: center at 1/3 from bottom of bbox
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) // 2
            center_y = y2 - (y2 - y1) // 3
            center_point = (center_x, center_y)
            dish_type = detection.dish_type
            
            # Find existing tracked dish or create new one
            existing_dish_id = None
            min_distance = float('inf')
            
            for dish_id, tracked_dish in self.tracked_dishes.items():
                if tracked_dish.dish_type == dish_type:
                    distance = self._distance(center_point, tracked_dish.center_point)
                    if distance < self.dish_tracking_threshold and distance < min_distance:
                        min_distance = distance
                        existing_dish_id = dish_id
            
            if existing_dish_id:
                # Update existing dish
                tracked_dish = self.tracked_dishes[existing_dish_id]
                tracked_dish.center_point = center_point
                tracked_dish.count_point = center_point
                tracked_dish.last_seen = current_time
            else:
                # Create new tracked dish
                dish_id = self._generate_dish_id(center_point, dish_type)
                tracked_dish = TrackedDish(
                    id=dish_id,
                    dish_type=dish_type,
                    center_point=center_point,
                    last_seen=current_time,
                    count_point=center_point
                )
                self.tracked_dishes[dish_id] = tracked_dish
                existing_dish_id = dish_id
            
            # Check ROI status for break line
            in_roi = self._is_point_in_roi(center_point, self.roi_dish_detection)
            
            # Update ROI status and detect crossings
            if 'dish_detection' not in tracked_dish.roi_status:
                tracked_dish.roi_status['dish_detection'] = False
            
            if in_roi and not tracked_dish.roi_status['dish_detection']:
                # Dish entered ROI
                tracked_dish.roi_status['dish_detection'] = True
                tracked_dish.crossed_rois.add('dish_detection')
                crossed_dishes.append(existing_dish_id)
                self.logger.debug(f"🔄 Dish {existing_dish_id} crossed ROI (return) at top-third point {center_point}")
            elif not in_roi and tracked_dish.roi_status['dish_detection']:
                # Dish left ROI
                tracked_dish.roi_status['dish_detection'] = False
        
        return crossed_dishes
    
    def _update_kitchen_dish_tracking(self, dish_detections: List[DishDetection]) -> List[str]:
        """
        Update dish tracking for kitchen camera and detect ROI crossings
        Same logic as conveyor_tracker.py but for kitchen ROI
        
        Args:
            dish_detections: List of dish detections from kitchen camera
            
        Returns:
            List of dish IDs that crossed kitchen ROI boundaries
        """
        current_time = datetime.now()
        crossed_dishes = []
        
        # Clean up old tracked dishes
        self._cleanup_old_tracked_dishes(current_time)
        
        for detection in dish_detections:
            # Counting point: center at 1/3 from bottom of bbox
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) // 2
            center_y = y2 - (y2 - y1) // 3
            center_point = (center_x, center_y)
            dish_type = detection.dish_type
            
            # Find existing tracked dish or create new one
            existing_dish_id = None
            min_distance = float('inf')
            
            for dish_id, tracked_dish in self.tracked_dishes.items():
                if tracked_dish.dish_type == dish_type:
                    distance = self._distance(center_point, tracked_dish.center_point)
                    if distance < self.dish_tracking_threshold and distance < min_distance:
                        min_distance = distance
                        existing_dish_id = dish_id
            
            if existing_dish_id:
                # Update existing dish
                tracked_dish = self.tracked_dishes[existing_dish_id]
                tracked_dish.center_point = center_point
                tracked_dish.count_point = center_point
                tracked_dish.last_seen = current_time
            else:
                # Create new tracked dish
                dish_id = self._generate_dish_id(center_point, dish_type)
                tracked_dish = TrackedDish(
                    id=dish_id,
                    dish_type=dish_type,
                    center_point=center_point,
                    last_seen=current_time,
                    count_point=center_point
                )
                self.tracked_dishes[dish_id] = tracked_dish
                existing_dish_id = dish_id
            
            # Check ROI status for kitchen
            in_roi = self._is_point_in_roi(center_point, self.roi_kitchen_counter)
            
            # Update ROI status and detect crossings
            if 'kitchen_counter' not in tracked_dish.roi_status:
                tracked_dish.roi_status['kitchen_counter'] = False
            
            if in_roi and not tracked_dish.roi_status['kitchen_counter']:
                # Dish entered ROI
                tracked_dish.roi_status['kitchen_counter'] = True
                tracked_dish.crossed_rois.add('kitchen_counter')
                crossed_dishes.append(existing_dish_id)
                self.logger.debug(f"🎯 Dish {existing_dish_id} crossed kitchen ROI at top-third point {center_point}")
            elif not in_roi and tracked_dish.roi_status['kitchen_counter']:
                # Dish left ROI
                tracked_dish.roi_status['kitchen_counter'] = False
        
        return crossed_dishes
    
    def _cleanup_old_tracked_dishes(self, current_time: datetime) -> None:
        """Remove dishes that haven't been seen for too long"""
        timeout_threshold = current_time - timedelta(seconds=self.dish_timeout)
        
        dishes_to_remove = []
        for dish_id, tracked_dish in self.tracked_dishes.items():
            if tracked_dish.last_seen < timeout_threshold:
                dishes_to_remove.append(dish_id)
        
        for dish_id in dishes_to_remove:
            del self.tracked_dishes[dish_id]
        
        if dishes_to_remove:
            self.logger.debug(f"🧹 Cleaned up {len(dishes_to_remove)} old tracked dishes")
    
    def _calculate_new_dishes_served(self):
        """Calculate new dishes served using: Kitchen total - Returned total"""
        new_dishes = max(0, self.state.total_kitchen_dishes_served - self.state.total_returned_dishes)
        
        if new_dishes != self.state.new_dishes_served:
            old_value = self.state.new_dishes_served
            self.state.new_dishes_served = new_dishes
            self.logger.info(f"📊 New dishes served updated: {old_value} → {new_dishes}")
            self.logger.info(f"📊 Calculation: {self.state.total_kitchen_dishes_served} (kitchen) - {self.state.total_returned_dishes} (returned) = {new_dishes}")
        
        return new_dishes
    
    def _update_rates(self):
        """Update dish serving rates based on recent activity"""
        current_time = datetime.now()
        
        # Calculate rates for red and yellow dishes
        for dish_type in ['red_dish', 'yellow_dish']:
            if dish_type in self.rate_calculation_data:
                # Filter recent data (last 60 seconds)
                recent_data = [
                    timestamp for timestamp in self.rate_calculation_data[dish_type]
                    if (current_time - timestamp).total_seconds() <= 60
                ]
                
                # Calculate rate per minute
                rate = len(recent_data)  # Already filtered to last 60 seconds
                self.state.dishes_per_minute[dish_type] = float(rate)
    
    def process_dish_detections(self, dish_detections: List[DishDetection], 
                              roi_name: str = "dish_detection") -> None:
        """
        Process dish detections using ROI crossing detection for accurate counting
        Only counts dishes that actually cross ROI boundaries (same logic as conveyor_tracker.py)
        Only process dishes if the system is properly initialized with phase data.
        
        Args:
            dish_detections: List of detected dishes
            roi_name: ROI name for tracking (break line vs kitchen)
        """
        if not dish_detections:
            return
        
        # Process dishes even if not phase initialized (for "waiting" phase tracking)  
        # Only skip if there are no dish detections
        if not self.state.is_phase_initialized:
            self.logger.debug(f"🔄 Processing dishes in 'waiting' phase - system not yet initialized")
        
        current_time = datetime.now()
        
        with self.lock:
            if roi_name == "kitchen_counter":
                # Process kitchen camera dishes (current stage dishes)
                crossed_dishes = self._update_kitchen_dish_tracking(dish_detections)
                self._process_kitchen_roi_crossings(crossed_dishes, current_time)
            else:
                # Process break line camera dishes (returning dishes)
                crossed_dishes = self._update_dish_tracking(dish_detections)
                self._process_break_line_roi_crossings(crossed_dishes, current_time)
    
    def _process_kitchen_roi_crossings(self, crossed_dishes: List[str], current_time: datetime) -> None:
        """Process dishes that crossed kitchen ROI - simple +1 count for each crossing"""
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if tracked_dish and 'kitchen_counter' in tracked_dish.crossed_rois:
                dish_type = tracked_dish.dish_type
                
                # Only count non-advertisement dishes
                if dish_type != 'advertisement_dish':
                    current_stage = self.state.current_stage
                    current_phase = self.state.current_phase
                    
                    # Simple +1 increment for current stage dishes
                    if dish_type in self.state.current_stage_dishes:
                        self.state.current_stage_dishes[dish_type] += 1
                    
                    # Simple +1 increment for current phase tracking (MAIN COUNTER FOR DASHBOARD)
                    if current_phase not in self.state.phase_dish_tracking:
                        self.state.phase_dish_tracking[current_phase] = {
                            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
                        }
                    
                    if dish_type in self.state.phase_dish_tracking[current_phase]:
                        self.state.phase_dish_tracking[current_phase][dish_type] += 1
                    
                    # Update stage totals for summary
                    self._ensure_stage_table(current_stage)
                    self.state.stage_totals[current_stage]['kitchen_total'] += 1
                    
                    # Update current stage dish count
                    self.state.current_stage_dish_count = sum(self.state.current_stage_dishes.values())
                    
                    # Simple +1 for overall kitchen serving counter
                    self.state.total_kitchen_dishes_served += 1
                    
                    # Persistent belt counters (added)
                    if dish_type in self.state.belt_added_cumulative:
                        self.state.belt_added_cumulative[dish_type] += 1
                    if dish_type in self.state.belt_counts:
                        self.state.belt_counts[dish_type] += 1
                    
                    # Simple +1 for dish type counters
                    if dish_type in self.state.kitchen_dishes_served:
                        self.state.kitchen_dishes_served[dish_type] += 1
                    
                    # Simple +1 for cumulative total (never resets)
                    self.state.total_dishes_counted_ever += 1
                    
                    # Protect cumulative counter from ever decreasing
                    if self.state.total_dishes_counted_ever > self.max_total_dishes_ever:
                        self.max_total_dishes_ever = self.state.total_dishes_counted_ever
                    else:
                        # Ensure total never decreases
                        self.state.total_dishes_counted_ever = self.max_total_dishes_ever
                        
                    # Track latest phases for UI display
                    display_phase = current_phase if self.state.is_phase_initialized else "waiting"
                    
                    # Update latest phases list
                    if display_phase not in self.state.latest_phases:
                        # Keep only the 2 most recent phases
                        if len(self.state.latest_phases) >= 2:
                            self.state.latest_phases.pop(0)  # Remove oldest
                        self.state.latest_phases.append(display_phase)
                    
                    # Update latest phase dishes tracking
                    if display_phase not in self.state.latest_phase_dishes:
                        self.state.latest_phase_dishes[display_phase] = {
                            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
                        }
                    if dish_type in self.state.latest_phase_dishes[display_phase]:
                        self.state.latest_phase_dishes[display_phase][dish_type] += 1
                    
                    # Log the simple kitchen ROI crossing
                    self.logger.info(f"🍽️ Kitchen ROI: +1 {dish_type} (Stage {current_stage}, Phase {current_phase}) - Total phase: {self.state.phase_dish_tracking[current_phase][dish_type]}")
                    
                    # Update rate calculation data for red and yellow dishes
                    if dish_type in ['red_dish', 'yellow_dish'] and dish_type in self.rate_calculation_data:
                        self.rate_calculation_data[dish_type].append(current_time)
                    
                    # Remove from crossed_rois to avoid double counting
                    tracked_dish.crossed_rois.discard('kitchen_counter')
        
        # Update calculations
        if crossed_dishes:
            self._calculate_new_dishes_served()
            self._update_dish_rates()
    
    def _process_break_line_roi_crossings(self, crossed_dishes: List[str], current_time: datetime) -> None:
        """
        Process dishes that crossed break line ROI - count as returning dishes and add to current phase
        
        According to requirements:
        - Breakline camera counts dishes coming back to the kitchen (previously served)
        - These returned dishes are added to the current phase for tracking
        """
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if tracked_dish and 'dish_detection' in tracked_dish.crossed_rois:
                dish_type = tracked_dish.dish_type
                
                # Only count non-advertisement dishes
                if dish_type != 'advertisement_dish':
                    # Count return dishes (backward line)
                    if dish_type in self.state.dishes_returning:
                        self.state.dishes_returning[dish_type] += 1
                        self.state.total_returned_dishes += 1
                        
                        # Persistent belt counters (returned)
                        if dish_type in self.state.belt_returned_cumulative:
                            self.state.belt_returned_cumulative[dish_type] += 1
                        # Decrement current belt counts only after calibration activation
                        if self.state.belt_calibration_active and dish_type in self.state.belt_counts:
                            self.state.belt_counts[dish_type] = max(0, self.state.belt_counts[dish_type] - 1)
                        
                        # Update per-stage totals
                        stage = self.state.current_stage
                        self._ensure_stage_table(stage)
                        self.state.stage_totals[stage]['returned_total'] += 1
                        
                        # ADD TO CURRENT PHASE: Key requirement implementation
                        current_phase = self.state.current_phase
                        # phase_dish_tracking is reserved for kitchen ROI only
                        
                        # Note: Break line dishes are returns - do NOT add to phase_dish_tracking
                        # phase_dish_tracking is used for "Direct Kitchen ROI Counts" display
                        # and should only contain kitchen ROI crossings as per user requirements
                        
                        # Update latest phase dishes tracking for UI
                        display_phase = current_phase if self.state.is_phase_initialized else "waiting"
                        if display_phase not in self.state.latest_phase_dishes:
                            self.state.latest_phase_dishes[display_phase] = {
                                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0, 'advertisement_dish': 0
                            }
                        if dish_type in self.state.latest_phase_dishes[display_phase]:
                            self.state.latest_phase_dishes[display_phase][dish_type] += 1
                        
                    # Note: phase_dish_tracking is for kitchen ROI only, not break line dishes
                    
                    # Note: Break line dishes are returns, but we track them in the phase they return to
                    # This gives us visibility into what dishes are coming back per phase
                        
                        phase_label = f"Phase {current_phase}" if self.state.is_phase_initialized else "waiting"
                        self.logger.info(f"🔄 Break line ROI: {dish_type} returned and added to {phase_label} (total returned: {self.state.total_returned_dishes})")
                
                # Remove from crossed_rois to avoid double counting
                tracked_dish.crossed_rois.discard('dish_detection')
        
        # Update calculations
        if crossed_dishes:
            self._calculate_new_dishes_served()
    
    def get_phase_dish_count(self, phase: int, dish_type: Optional[str] = None) -> int:
        """
        Get dish count for a specific phase - returns real-time camera tracking data
        
        Args:
            phase: Phase number
            dish_type: Specific dish type, or None for total count
            
        Returns:
            Dish count for the specified phase and type
        """
        # Use actual tracked dish counts from camera ROI crossings, not estimates
        if phase not in self.state.phase_dish_tracking:
            return 0
        
        phase_data = self.state.phase_dish_tracking[phase]
        
        if dish_type is None:
            # Return total for non-advertisement dishes only
            return sum([
                phase_data.get('normal_dish', 0),
                phase_data.get('red_dish', 0),
                phase_data.get('yellow_dish', 0)
            ])
        elif dish_type in phase_data:
            return phase_data.get(dish_type, 0)
        else:
            return 0
    
    def get_all_phase_dish_counts(self) -> Dict[int, Dict[str, int]]:
        """Get dish counts for all phases - returns cached accurate data prioritizing correctness"""
        # Try to get from cache first (accuracy priority)
        cached_phases = self.backend_cache.get_all_cached_phases()
        if cached_phases:
            result = {}
            for phase_id, phase_details in cached_phases.items():
                result[phase_id] = phase_details['dish_counts'].copy()
                # Calculate total accurately without double-counting
                result[phase_id]['total'] = sum([
                    phase_details['dish_counts'].get('normal_dish', 0),
                    phase_details['dish_counts'].get('red_dish', 0),
                    phase_details['dish_counts'].get('yellow_dish', 0)
                    # Don't count advertisement_dish in total
                ])
            
            self.logger.debug(f"🏪 Returning cached phase data for {len(result)} phases")
            return result
        
        # Fallback to real-time data if cache is empty, but update cache
        result = {}
        
        # Use actual tracked dish counts from camera ROI crossings, not estimates
        for phase, dish_counts in self.state.phase_dish_tracking.items():
            # Clean dish counts to prevent double-counting
            clean_counts = {
                'normal_dish': dish_counts.get('normal_dish', 0),
                'red_dish': dish_counts.get('red_dish', 0),
                'yellow_dish': dish_counts.get('yellow_dish', 0),
                'advertisement_dish': dish_counts.get('advertisement_dish', 0)
            }
            
            result[phase] = clean_counts.copy()
            result[phase]['total'] = sum([
                clean_counts.get('normal_dish', 0),
                clean_counts.get('red_dish', 0),
                clean_counts.get('yellow_dish', 0)
                # Don't count advertisement_dish in total
            ])
            
            # Cache this phase data for accuracy
            self.backend_cache.cache_phase_data(
                phase_id=phase,
                stage_id=self.state.current_stage,
                dish_counts=clean_counts,
                is_validated=True
            )
            
        # Also sync the phase_data with actual counts for consistency
        for phase in result:
            if phase in self.state.phase_data:
                clean_counts = {k: v for k, v in result[phase].items() if k != 'total'}
                self.state.phase_data[phase].dish_counts.update(clean_counts)
                self.state.phase_data[phase].total_dishes = result[phase]['total']
        
        # Update backend cache with current stage data for accuracy-first sync
        if hasattr(self, 'state') and self.state.current_stage is not None:
            stage_totals = {
                'kitchen_total': sum(self.state.current_stage_dishes.values()),
                'returned_total': sum(self.state.dishes_returning.values())
            }
            stage_metrics = {
                'taken_out': self.state.dishes_taken_out,
                'added_in': self.state.dishes_added_in
            }
            
            # Get current phase data for caching
            current_phase_data = {}
            if hasattr(self.state, 'stage_phase_tables') and self.state.current_stage in self.state.stage_phase_tables:
                current_phase_data = self.state.stage_phase_tables[self.state.current_stage].copy()
            
            # Cache current stage data for accuracy
            self.backend_cache.cache_stage_data(
                stage_id=self.state.current_stage,
                phase_data=current_phase_data,
                stage_totals=stage_totals,
                stage_metrics=stage_metrics,
                is_complete=False  # Current stage is not complete yet
            )
                
        self.logger.debug(f"🏪 Fallback: Generated and cached phase data for {len(result)} phases")
        return result
    
    def update_kitchen_data(self, frame_index: int, detections: List[DishDetection]) -> None:
        """
        Update kitchen camera data with dish detections and frame synchronization
        
        Args:
            frame_index: Current kitchen camera frame number
            detections: List of dish detections from kitchen camera
        """
        try:
            # Update kitchen frame index with delay compensation
            delayed_frame_index = self.update_kitchen_frame_index(frame_index)
            
            # Process dish detections through kitchen ROI
            if detections:
                self.process_dish_detections(detections, roi_name="kitchen_counter")
                self.logger.debug(f"🍽️ Kitchen: Processed {len(detections)} detections at frame {frame_index}")
            
            # Update last activity time
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating kitchen data: {e}")
    
    def update_break_line_data(self, frame_index: int, detections: List[DishDetection], 
                              current_phase: Optional[int] = None, return_phase: Optional[int] = None) -> None:
        """
        Update break line camera data with dish detections and phase information
        
        Args:
            frame_index: Current break line camera frame number
            detections: List of dish detections from break line camera
            current_phase: Detected current phase number (optional, CSV timeline takes precedence)
            return_phase: Detected return phase number (optional, CSV timeline takes precedence)
        """
        try:
            # Update frame position from CSV timeline (this is the primary source)
            timeline_updated = self.update_frame_position(frame_index, is_breakline_camera=True)
            
            # Process dish detections through break line ROI
            if detections:
                self.process_dish_detections(detections, roi_name="dish_detection")
                self.logger.debug(f"📹 Break line: Processed {len(detections)} detections at frame {frame_index}")
            
            # Log phase information for debugging (CSV timeline takes precedence over OCR)
            if current_phase is not None or return_phase is not None:
                csv_current = self.state.current_phase
                csv_previous = self.state.previous_phase
                self.logger.debug(f"📊 Phase comparison - OCR: current={current_phase}, return={return_phase} | CSV: current={csv_current}, previous={csv_previous}")
            
            # Update breakline frame index
            self.update_breakline_frame_index(frame_index)
            
            # Update last activity time
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating break line data: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current tracker status for dashboard/logging"""
        with self.lock:
            active_phases = [p for p, d in self.state.phase_data.items() if d.is_active]
            
            return {
                "current_frame": self.state.current_frame,
                "current_stage": self.state.current_stage,
                "current_phase": self.state.current_phase,
                "previous_stage": self.state.previous_stage, 
                "previous_phase": self.state.previous_phase,
                "active_phases": active_phases,
                "total_phases": len(self.state.phase_data),
                "total_dishes_processed": dict(self.state.total_dishes_processed),
                "current_stage_dishes": dict(self.state.current_stage_dishes),
                "phase_dish_counts": {
                    phase: {
                        "total": data.total_dishes,
                        "by_type": dict(data.dish_counts)
                    }
                    for phase, data in self.state.phase_data.items()
                    if data.total_dishes > 0  # Only show phases with dishes
                }
            }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        timeline_summary = self.timeline_parser.get_timeline_summary()
        current_status = self.get_current_status()
        
        # Calculate additional statistics
        total_estimated_dishes = sum(data.total_dishes for data in self.state.phase_data.values())
        phases_with_dishes = [p for p, d in self.state.phase_data.items() if d.total_dishes > 0]
        
        return {
            "timeline_info": timeline_summary,
            "tracking_status": current_status,
            "statistics": {
                "total_phases": len(self.state.phase_data),
                "phases_with_dishes": len(phases_with_dishes),
                "total_estimated_dishes": total_estimated_dishes,
                "avg_dishes_per_phase": total_estimated_dishes / max(1, len(phases_with_dishes)),
                "dish_distribution": {
                    dish_type: sum(data.dish_counts.get(dish_type, 0) 
                                 for data in self.state.phase_data.values())
                    for dish_type in ['normal_dish', 'red_dish', 'yellow_dish', 'advertisement_dish']
                }
            }
        }
    
    # Legacy compatibility methods
    def update_phase_numbers(self, incoming_detections: List[NumberDetection], 
                           return_detections: List[NumberDetection]) -> None:
        """Legacy compatibility method - now uses CSV data instead"""
        # This method is kept for compatibility but doesn't use the detections
        # Phase information comes from CSV timeline based on current frame
        pass
    
    def process_kitchen_dishes(self, dish_detections: List[DishDetection]) -> None:
        """Legacy compatibility for kitchen dish processing"""
        self.process_dish_detections(dish_detections, "kitchen_counter")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        return self.get_current_status()
    
    # Dashboard compatibility methods
    def get_current_state(self) -> 'CSVConveyorState':
        """Get current conveyor state - dashboard compatibility"""
        return self.state
    
    def get_total_dishes_on_belt(self) -> Dict[str, int]:
        """Get total dishes on belt - dashboard compatibility"""
        return self.state.dishes_to_customer.copy()
    
    def get_dishes_by_roi(self) -> Dict[str, Dict[str, int]]:
        """Get dishes by ROI - dashboard compatibility (simulated for CSV mode)"""
        return {
            'kitchen_counter': self.state.current_stage_dishes.copy(),
            'break_line': self.state.dishes_returning.copy()
        }

    def get_current_belt_counts(self) -> Dict[str, int]:
        """
        Phase-aware belt count with calibration:
        - Before calibration activation (Stage 0 Phase 10 +100 frames): accumulate belt_counts on each
          kitchen crossing; do not decrement on returns yet (stabilize baseline).
        - After activation: decrement belt_counts on each break-line return; belt_counts persist across
          phase updates and do not reset to 0 with history updates.
        Returns dict with per-type and 'total'.
        """
        with self.lock:
            counts = {
                'normal_dish': int(self.state.belt_counts.get('normal_dish', 0)),
                'red_dish': int(self.state.belt_counts.get('red_dish', 0)),
                'yellow_dish': int(self.state.belt_counts.get('yellow_dish', 0))
            }
            # Ensure non-negative
            for k in list(counts.keys()):
                counts[k] = max(0, counts[k])
            counts['total'] = counts['normal_dish'] + counts['red_dish'] + counts['yellow_dish']
            return counts
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get phase summary - dashboard compatibility"""
        with self.lock:
            # Calculate dishes sent in current phase
            current_phase_dishes = 0
            if self.state.current_phase in self.state.phase_dish_tracking:
                current_phase_dishes = sum(self.state.phase_dish_tracking[self.state.current_phase].values())
            
            # Calculate shifts completed (simplified as number of phase changes)
            shifts_completed = len([p for p in self.state.phase_dish_tracking.keys() if p != self.state.current_phase])
            
            return {
                'current_phase': self.state.current_phase,
                'current_stage': self.state.current_stage,
                'last_return_phase': self.state.last_return_phase,
                'last_return_stage': self.state.last_return_stage,
                'is_initialized': self.state.is_phase_initialized,
                'dishes_sent_current_phase': current_phase_dishes,
                'shifts_completed': shifts_completed,
                'cycle_complete': False,
                'total_phases': len(self.state.phase_data),
                'active_phases': [p for p, data in self.state.phase_data.items() if data.is_active],
                'phase_dish_counts': self.state.phase_dish_tracking.copy(),
                'last_updated': self.state.last_updated.isoformat() if hasattr(self.state, 'last_updated') else datetime.now().isoformat()
            }
    
    def get_dish_serving_summary(self) -> Dict[str, Any]:
        """Get dish serving summary - dashboard compatibility"""
        # Calculate new dishes served (kitchen total - returned total)
        kitchen_total = sum(self.state.current_stage_dishes.values())
        returned_total = sum(self.state.dishes_returning.values())
        net_served = max(0, kitchen_total - returned_total)
        
        return {
            'current_stage': self.state.current_stage,
            'current_phase': self.state.current_phase,
            'total_kitchen_dishes': kitchen_total,
            'total_returned_dishes': returned_total,
            'new_dishes_served': net_served,
            'kitchen_dishes_by_type': self.state.current_stage_dishes.copy(),
            'equation': f"{kitchen_total} (Forward Line - Current Phase/Stage) - {returned_total} (Backward Camera) = {net_served} (Added to Belt)"
        }
    
    def get_active_stage_phases(self) -> List[Tuple[int, int, int]]:
        """Get active stage phases with real dish counts for Stage Distribution chart"""
        active_phases = []
        
        # Use actual stage dish counts for a more accurate Stage Distribution
        stage_dish_data = {}
        
        # Collect dish counts by stage from stage_phase_tables
        for stage in self.state.stage_phase_tables:
            stage_total = 0
            for phase in self.state.stage_phase_tables[stage]:
                phase_dishes = self.state.stage_phase_tables[stage][phase]
                stage_total += sum(phase_dishes.values())
            if stage_total > 0:
                stage_dish_data[stage] = stage_total
        
        # Add current stage with current dishes
        current_stage_total = sum(self.state.current_stage_dishes.values())
        if current_stage_total > 0 or not stage_dish_data:
            stage_dish_data[self.state.current_stage] = current_stage_total
        
        # Convert to the expected format (stage, phase, count)
        for stage, count in stage_dish_data.items():
            # For stage distribution, we use the stage's primary phase or current phase
            primary_phase = self.state.current_phase if stage == self.state.current_stage else 0
            active_phases.append((stage, primary_phase, count))
        
        # Sort by stage for consistent display
        active_phases.sort(key=lambda x: x[0])
        
        self.logger.debug(f"🔍 Stage Distribution Data: {active_phases}")
        return active_phases
    
    def is_break_line_active(self) -> bool:
        """Check if break line is active - dashboard compatibility"""
        # In CSV mode, break line is active when there are returning dishes
        return sum(self.state.dishes_returning.values()) > 0
    
    def reset_session(self) -> None:
        """Reset current session data - dashboard compatibility"""
        with self.lock:
            # Reset current stage data but keep historical data
            self.state.current_stage_dishes = {
                'normal_dish': 0,
                'red_dish': 0,
                'yellow_dish': 0,
                'advertisement_dish': 0
            }
            self.state.dishes_returning = {
                'normal_dish': 0,
                'red_dish': 0,
                'yellow_dish': 0,
                'advertisement_dish': 0
            }
            self.logger.info("🔄 Session reset - current stage and return dishes cleared")
    
    def reset_system(self) -> None:
        """Reset entire system - dashboard compatibility"""
        with self.lock:
            self._reset_system_to_initial_state()
            self.logger.info("🔄 Full system reset completed")
    
    def restart_demo(self) -> None:
        """Restart the demo from the beginning"""
        with self.lock:
            self.logger.info("🔄 Restarting POC demo from beginning...")
            
            # Reset demo completion state
            self.state.demo_completed = False
            self.state.demo_completion_time = datetime.now()
            
            # Reset to initial system state
            self._reset_system_to_initial_state()
            
            self.logger.info("✅ POC demo restarted successfully")
    
    def generate_end_shift_report(self) -> Dict[str, Any]:
        """Generate comprehensive business report when shift ends"""
        from datetime import datetime
        import json
        
        self.logger.info("📊 Generating End-of-Shift Business Report")
        
        # Calculate total frames processed
        total_frames = self.state.current_frame
        
        # Calculate summary statistics
        total_dishes_served = sum(self.state.kitchen_dishes_served.values())
        total_dishes_returned = sum(self.state.dishes_returning.values())
        net_dishes_served = max(0, total_dishes_served - total_dishes_returned)
        
        # Stage analysis
        stage_analysis = {}
        for stage in self.state.stage_totals:
            stage_data = self.state.stage_totals[stage]
            stage_metrics = self.state.stage_metrics.get(stage, {})
            
            stage_analysis[f"Stage {stage}"] = {
                "forward_camera_total": stage_data.get("kitchen_total", 0),
                "backward_camera_total": stage_data.get("returned_total", 0),
                "dishes_added_to_belt": stage_metrics.get("net_served", stage_data.get("kitchen_total", 0) - stage_data.get("returned_total", 0)),
                "dishes_taken_by_customers": stage_metrics.get("taken_out", 0),
                "return_rate": round((stage_data.get("returned_total", 0) / max(stage_data.get("kitchen_total", 1), 1)) * 100, 1)
            }
        
        # Phase breakdown
        phase_breakdown = {}
        for phase in self.state.phase_dish_tracking:
            phase_data = self.state.phase_dish_tracking[phase]
            phase_breakdown[f"Phase {phase}"] = phase_data.copy()
        
        # Dish type analysis
        dish_type_analysis = {
            "normal_dishes": self.state.kitchen_dishes_served.get('normal_dish', 0),
            "red_dishes": self.state.kitchen_dishes_served.get('red_dish', 0),
            "yellow_dishes": self.state.kitchen_dishes_served.get('yellow_dish', 0),
            "advertisement_dishes": self.state.kitchen_dishes_served.get('advertisement_dish', 0)
        }
        
        # Calculate efficiency metrics
        served_rate = round((net_dishes_served / max(total_dishes_served, 1)) * 100, 1)
        return_rate = round((total_dishes_returned / max(total_dishes_served, 1)) * 100, 1)
        
        # Business report structure
        business_report = {
            "shift_summary": {
                "report_generated": datetime.now().isoformat(),
                "total_frames_processed": total_frames,
                "total_stages": len(self.state.stage_totals),
                "total_phases": len(self.state.phase_dish_tracking),
                "shift_duration_frames": total_frames
            },
            "operational_metrics": {
                "total_dishes_served": total_dishes_served,
                "total_dishes_returned": total_dishes_returned,
                "net_dishes_served": net_dishes_served,
                "service_efficiency": f"{served_rate}%",
                "return_rate": f"{return_rate}%"
            },
            "dish_type_breakdown": dish_type_analysis,
            "stage_performance": stage_analysis,
            "phase_breakdown": phase_breakdown,
            "key_insights": self._generate_business_insights(stage_analysis, dish_type_analysis, served_rate, return_rate)
        }
        
        # Save report to file
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kichikichi_shift_report_{timestamp}.json"
            filepath = f"reports/{filename}"
            
            # Create reports directory if it doesn't exist
            import os
            os.makedirs("reports", exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(business_report, f, indent=2)
            
            self.logger.info(f"📊 Business report saved to: {filepath}")
            
            # Also log summary to console
            self.logger.info("🎯 SHIFT SUMMARY:")
            self.logger.info(f"   Total Dishes Served: {total_dishes_served}")
            self.logger.info(f"   Net Dishes to Customers: {net_dishes_served}")
            self.logger.info(f"   Service Efficiency: {served_rate}%")
            self.logger.info(f"   Return Rate: {return_rate}%")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save business report: {e}")
        
        return business_report
    
    def _generate_business_insights(self, stage_analysis: Dict, dish_analysis: Dict, served_rate: float, return_rate: float) -> List[str]:
        """Generate business insights based on shift data"""
        insights = []
        
        # Efficiency insights
        if served_rate >= 80:
            insights.append("✅ Excellent service efficiency - minimal food waste")
        elif served_rate >= 60:
            insights.append("⚠️ Moderate service efficiency - consider optimizing dish timing")
        else:
            insights.append("🚨 Low service efficiency - review conveyor belt operations")
        
        # Return rate insights  
        if return_rate <= 20:
            insights.append("✅ Low return rate indicates good customer satisfaction")
        elif return_rate <= 40:
            insights.append("⚠️ Moderate return rate - monitor dish quality and customer preferences")
        else:
            insights.append("🚨 High return rate - investigate dish quality or customer satisfaction issues")
        
        # Stage performance insights
        best_stage = max(stage_analysis.keys(), key=lambda s: stage_analysis[s]["dishes_added_to_belt"]) if stage_analysis else None
        if best_stage:
            insights.append(f"🌟 Best performing stage: {best_stage}")
        
        # Dish type insights
        most_popular = max(dish_analysis.keys(), key=lambda d: dish_analysis[d]) if any(dish_analysis.values()) else None
        if most_popular:
            dish_name = most_popular.replace('_', ' ').title()
            insights.append(f"🍽️ Most popular dish type: {dish_name}")
        
        return insights
    
    def reset_counts(self) -> None:
        """Reset all counts - dashboard compatibility"""
        self.logger.info("🔄 Resetting CSV tracker counts")
        
        # Reset all dish counts
        for dish_type in self.state.dishes_to_customer:
            self.state.dishes_to_customer[dish_type] = 0
            self.state.dishes_returning[dish_type] = 0
            self.state.current_stage_dishes[dish_type] = 0
            self.state.kitchen_dishes_served[dish_type] = 0
            self.state.total_dishes_processed[dish_type] = 0
        
        # Reset phase data
        for phase_data in self.state.phase_data.values():
            for dish_type in phase_data.dish_counts:
                phase_data.dish_counts[dish_type] = 0
            phase_data.total_dishes = 0
            phase_data.dishes_processed = []
        
        # Reset other counters
        self.state.total_kitchen_dishes_served = 0
        self.state.total_returned_dishes = 0
        self.state.new_dishes_served = 0
        
        # Reset rates
        for dish_type in self.state.dishes_per_minute:
            self.state.dishes_per_minute[dish_type] = 0.0