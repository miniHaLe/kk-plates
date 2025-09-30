"""
Conveyor Belt Tracking System for KichiKichi
Handles ROI-based dish counting and phase tracking with automatic logic
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import threading
import time
import math

from dish_detection.dish_detector import DishDetection
from ocr_model.number_detector import NumberDetection
# from config.config import config  # Not needed for current implementation

# NUMBER DETECTION CLASSES: 0,1,2,3,4,5,6,7,8,9,10,11,12 with indices 0,1,2,3,4,5,6,7,8,9,10,11,12
# No position swapping - classes map directly to their numeric values
# PHASE ASSIGNMENT: Pure ROI-based detection, no calculations
# - incoming_phase ROI detection → current_phase
# - return_phase ROI detection → previous_phase (triggers dish movement)
# 
# STAGE RULE: previous_stage < current_stage
# - Special case: when current_stage = 0, previous_stage = 0
# - Normal case: when current_stage >= 1, previous_stage = current_stage - 1
#
# PHASE RULE: phases cannot decrease except for 12→0 cycle completion
# - Applies to both current_phase and previous_phase (last_return_phase)
# - Only allowed decrease: 12→0 (represents cycle completion)

# ROI Configurations from user's interactive tool
ROI_CONFIG = {
    'dish_detection': (415, 193, 854, 466),      # Main conveyor belt dish counting area    
    'incoming_phase': (1026, 174, 1252, 439),   # Phase number detection (incoming)
    'return_phase': (1068, 687, 1274, 890),     # Return phase number detection
    'kitchen_counter': (678, 139, 1334, 411)    # Kitchen counter area for served dishes
}

@dataclass
class TrackedDish:
    """Track a dish's movement for ROI crossing detection"""
    id: str
    dish_type: str
    center_point: Tuple[int, int]
    last_seen: datetime
    roi_status: Dict[str, bool] = field(default_factory=dict)  # Which ROIs the dish is currently in
    crossed_rois: Set[str] = field(default_factory=set)       # Which ROIs the dish has crossed through

@dataclass
class ConveyorState:
    """Current state of the conveyor belt with ROI-based tracking"""
    current_stage: int = 0
    current_phase: int = 0
    is_phase_initialized: bool = False
    
    # Separate tracking for return phase (previous stages)
    last_return_phase: int = 0
    last_return_stage: int = 0
    
    # Stage increment control - require 2 detections of number 0 before stage increment
    zero_detection_count: int = 0
    
    # Stage protection - stages only increase, never reduce except for no-dish cycles
    last_dish_detection_time: Optional[datetime] = None
    cycle_without_dishes: bool = False
    completed_cycles_without_dishes: int = 0
    
    # Dish counting for different areas
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
    
    # Current stage dishes only (reset when stage changes)
    current_stage_dishes: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Kitchen and return dishes tracking
    total_kitchen_dishes_served: int = 0  # Total dishes from kitchen camera
    total_returned_dishes: int = 0        # Total dishes returning from break line camera
    new_dishes_served: int = 0            # Kitchen total - Returned total
    last_calculation_time: datetime = field(default_factory=datetime.now)
    
    # Kitchen dishes by type
    kitchen_dishes_served: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0,
        'advertisement_dish': 0
    })
    
    # Phase-specific dish tracking (stores dish counts for each phase)
    phase_dish_tracking: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # Last detected phases for cycle tracking
    last_incoming_phase: int = 0
    last_return_phase_detection_time: Optional[datetime] = None
    
    # Rate calculations
    dishes_per_minute: Dict[str, float] = field(default_factory=lambda: {
        'red_dish': 0.0,
        'yellow_dish': 0.0
    })
    
    # Cycle tracking
    cycle_complete: bool = False
    shifts_completed: int = 0
    last_dish_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass 
class PhaseData:
    """Data for a specific phase"""
    phase_number: int
    dishes_sent: List[DishDetection] = field(default_factory=list)
    dishes_returned: List[DishDetection] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    expected_return_count: int = 0
    
    # Phase-specific dish counts (red, yellow, normal)
    phase_dish_counts: Dict[str, int] = field(default_factory=lambda: {
        'normal_dish': 0,
        'red_dish': 0,
        'yellow_dish': 0
    })

class ConveyorTracker:
    """
    ROI-based conveyor belt tracking system for KichiKichi
    Implements simplified ROI-based phase tracking without calculations
    
    Phase Assignment:
    - incoming_phase ROI detection → current_phase
    - return_phase ROI detection → previous_phase (triggers dish movement)
    - No mathematical relationships between phases
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ROI configurations
        self.roi_dish_detection = ROI_CONFIG['dish_detection']
        self.roi_incoming_phase = ROI_CONFIG['incoming_phase'] 
        self.roi_return_phase = ROI_CONFIG['return_phase']
        self.roi_kitchen_counter = ROI_CONFIG['kitchen_counter']
        
        # State management
        self.state = ConveyorState()
        self.phase_data: Dict[int, PhaseData] = {}  # Track data for each phase
        
        # Dish tracking for ROI crossing detection
        self.tracked_dishes: Dict[str, TrackedDish] = {}
        self.dish_tracking_threshold = 100  # Distance threshold for tracking same dish (increased for better tracking)
        self.dish_timeout = 3.0  # Seconds before forgetting a dish (increased for better tracking)
        
        # Tracking data
        self.dish_history: deque = deque(maxlen=1000)
        self.rate_calculation_data: Dict[str, deque] = {
            'red_dish': deque(maxlen=60),  # Reduced to avoid over-accumulation
            'yellow_dish': deque(maxlen=60)  # Reduced to avoid over-accumulation
        }
        
        # Phase detection tracking
        self.phase_initialization_attempts = 0
        self.max_initialization_attempts = 50  # Try for ~50 frames to find initial phase
        self.last_phase_detection_time = datetime.now()
        
        # Cycle detection
        self.no_dish_timeout = timedelta(minutes=5)  # Reset after 5 min of no dishes
        
        # Rate calculation settings
        self.rate_window = 60  # Time window for rate calculations in seconds
        self.last_rate_dish_time = {}  # Track when we last counted each dish type to avoid over-counting
        
        # Legacy attributes for backward compatibility
        self.total_stages = 999  # No limit on stages in simplified system
        self.stage_phase_data: Dict[Tuple[int, int], any] = {}  # Legacy stage-phase data
        self.break_line_detected = False
        self.last_break_line_time = datetime.now()
        self.break_line_cooldown = timedelta(seconds=5)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize phase 0 data (default phase when no numbers detected)
        self.phase_data[0] = PhaseData(phase_number=0)
        
        # self.logger.info("ConveyorTracker initialized - Stage 0: counting dishes until Phase 0 detected")
        # self.logger.info("Stage rule: previous_stage < current_stage (special case: stage 0 → previous_stage = 0)")
        # self.logger.info("Phase rule: phases cannot decrease except for 12→0 cycle completion")
    
    def _initialize_phase(self, phase: int):
        """Initialize a new phase"""
        if phase not in self.phase_data:
            self.phase_data[phase] = PhaseData(phase_number=phase)
            self.logger.debug(f"Initialized phase {phase}")
        
        # Initialize phase dish tracking in state
        if phase not in self.state.phase_dish_tracking:
            self.state.phase_dish_tracking[phase] = {
                'normal_dish': 0,
                'red_dish': 0,
                'yellow_dish': 0
            }
    
    def _initialize_stage_phase(self, stage: int, phase: int):
        """Initialize a new stage-phase combination (legacy compatibility)"""
        # This is for backward compatibility with old methods
        # In the simplified system, we don't use stage-phase tracking
        pass
    
    def _calculate_previous_stage(self, current_stage: int) -> int:
        """
        Calculate previous stage based on the rule: previous_stage < current_stage
        
        Special case: When current_stage = 0, previous_stage = 0
        Normal case: previous_stage = current_stage - 1
        
        Args:
            current_stage: The current stage number
            
        Returns:
            The appropriate previous stage number
        """
        if current_stage == 0:
            return 0  # Special case: stay at stage 0
        else:
            return max(0, current_stage - 1)  # Normal case: previous stage is current - 1
    
    def _is_phase_decrease_allowed(self, old_phase: int, new_phase: int) -> bool:
        """
        Check if phase decrease is allowed
        
        Rule: Phases cannot decrease except for the 12→0 cycle completion
        
        Args:
            old_phase: Current phase number
            new_phase: New phase number to validate
            
        Returns:
            True if the phase change is allowed, False otherwise
        """
        if new_phase >= old_phase:
            return True  # Increases or same phase are always allowed
        
        # Special case: 12→0 cycle completion is allowed
        if old_phase == 12 and new_phase == 0:
            return True
        
        # All other decreases are blocked
        return False
    
    def _set_current_phase_protected(self, new_phase: int, context: str = "UPDATE") -> bool:
        """
        Safely set current phase with protection logic
        
        Args:
            new_phase: The new phase number to set
            context: Context for logging (e.g., "INIT", "UPDATE", "RESET")
            
        Returns:
            True if phase was updated, False if blocked by protection
        """
        old_phase = self.state.current_phase
        
        # Apply protection logic
        if not self._is_phase_decrease_allowed(old_phase, new_phase):
            self.logger.info(f"🛡️ CURRENT PHASE PROTECTION ({context}): {old_phase} → {new_phase} BLOCKED")
            return False
        
        # Update phase
        self.state.current_phase = new_phase
        if old_phase != new_phase:
            self.logger.info(f"✅ CURRENT PHASE {context}: {old_phase} → {new_phase}")
        return True
    
    def _set_previous_phase_protected(self, new_phase: int, context: str = "UPDATE") -> bool:
        """
        Safely set previous phase with protection logic
        
        Args:
            new_phase: The new phase number to set
            context: Context for logging (e.g., "INIT", "UPDATE", "RESET")
            
        Returns:
            True if phase was updated, False if blocked by protection
        """
        old_phase = self.state.last_return_phase
        
        # Apply protection logic
        if not self._is_phase_decrease_allowed(old_phase, new_phase):
            self.logger.info(f"🛡️ PREVIOUS PHASE PROTECTION ({context}): {old_phase} → {new_phase} BLOCKED")
            return False
        
        # Update phase
        self.state.last_return_phase = new_phase
        if old_phase != new_phase:
            self.logger.info(f"✅ PREVIOUS PHASE {context}: {old_phase} → {new_phase}")
        return True
    
    def verify_phase_protection(self) -> Dict[str, any]:
        """
        Verify that phase protection is working correctly
        
        Returns:
            Dictionary with protection test results
        """
        results = {
            'protection_enabled': True,
            'current_phase_protection': True,
            'previous_phase_protection': True,
            'test_results': []
        }
        
        # Test current phase protection
        old_current = self.state.current_phase
        test_current = max(0, old_current - 1) if old_current > 0 else 11  # Try to decrease
        
        if test_current != old_current:  # Only test if it would be a change
            allowed = self._is_phase_decrease_allowed(old_current, test_current)
            results['test_results'].append({
                'test': f'Current phase {old_current} → {test_current}',
                'should_block': test_current < old_current and not (old_current == 12 and test_current == 0),
                'was_blocked': not allowed,
                'result': 'PASS' if (not allowed) == (test_current < old_current and not (old_current == 12 and test_current == 0)) else 'FAIL'
            })
        
        # Test previous phase protection  
        old_previous = self.state.last_return_phase
        test_previous = max(0, old_previous - 1) if old_previous > 0 else 11  # Try to decrease
        
        if test_previous != old_previous:  # Only test if it would be a change
            allowed = self._is_phase_decrease_allowed(old_previous, test_previous)
            results['test_results'].append({
                'test': f'Previous phase {old_previous} → {test_previous}',
                'should_block': test_previous < old_previous and not (old_previous == 12 and test_previous == 0),
                'was_blocked': not allowed,
                'result': 'PASS' if (not allowed) == (test_previous < old_previous and not (old_previous == 12 and test_previous == 0)) else 'FAIL'
            })
        
        # Test 12→0 cycle allowance
        if self.state.current_phase != 12:
            allowed_cycle = self._is_phase_decrease_allowed(12, 0)
            results['test_results'].append({
                'test': '12 → 0 cycle completion',
                'should_block': False,
                'was_blocked': not allowed_cycle,
                'result': 'PASS' if allowed_cycle else 'FAIL'
            })
        
        # Overall results
        all_tests_passed = all(test['result'] == 'PASS' for test in results['test_results'])
        results['overall_result'] = 'PASS' if all_tests_passed else 'FAIL'
        
        return results
    
    def _handle_phase_cycle_completion(self, return_phase: int):
        """
        Handle dish movement when a phase completes its cycle (return_phase ROI detection)
        
        When a number is detected crossing return_phase ROI, it means dishes from that phase
        have completed a cycle and should pass dishes to the current phase.
        
        Simplified Logic:
        - Use detected return_phase directly as the completed phase
        - Move dishes from completed phase to current phase
        - No calculations or validations
        """
        current_time = datetime.now()
        
        completed_phase = return_phase
        current_phase = self.state.current_phase
        
        self.logger.info(f"🔄 PHASE CYCLE COMPLETION: Return phase {completed_phase} → Current phase {current_phase}")
        
        # Initialize phases if needed
        self._initialize_phase(completed_phase)
        self._initialize_phase(current_phase)
        
        # Simple dish movement - no validation needed
        self._move_dishes_between_phases(completed_phase, current_phase)
        
        # Update last return phase detection time
        self.state.last_return_phase_detection_time = current_time
    
    def _move_dishes_between_phases(self, from_phase: int, to_phase: int):
        """
        Move dishes from one phase to another - simplified approach
        
        Args:
            from_phase: Phase that completed its cycle (detected in return_phase ROI)
            to_phase: Current phase receiving dishes (detected in incoming_phase ROI)
        """
        # Get current dish counts
        from_counts = self.state.phase_dish_tracking.get(from_phase, {
            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0
        })
        
        to_counts = self.state.phase_dish_tracking.get(to_phase, {
            'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0
        })
        
        # Add dishes from completed phase to current phase
        for dish_type in ['normal_dish', 'red_dish', 'yellow_dish']:
            if from_counts[dish_type] > 0:
                to_counts[dish_type] += from_counts[dish_type]
        
        # Update the tracking
        self.state.phase_dish_tracking[to_phase] = to_counts
        
        # Update phase data objects as well
        if to_phase in self.phase_data:
            self.phase_data[to_phase].phase_dish_counts = to_counts.copy()
        
        # Log result
        total_added = sum(from_counts.values())
        total_after = sum(to_counts.values())
        
        if total_added > 0:
            self.logger.info(f"📊 Phase {from_phase} → {to_phase}: +{total_added} dishes (total: {total_after})")
        
        # Clear the completed phase dishes
        for dish_type in ['normal_dish', 'red_dish', 'yellow_dish']:
            from_counts[dish_type] = 0
        
        self.state.phase_dish_tracking[from_phase] = from_counts
    
    def update_from_return_detections(self, return_dish_detections: List[DishDetection], 
                                    return_phase_detections: List[NumberDetection],
                                    secondary_phase_detections: List[NumberDetection]):
        """
        Update tracker state from BREAK LINE camera return detections
        
        Simplified approach: Use detected phases directly without calculations
        
        Args:
            return_dish_detections: Return dishes detected in break line ROI (from customers)
            return_phase_detections: Return phase numbers detected in return_phase ROI (triggers dish movement)
            secondary_phase_detections: Secondary phase number detections (fallback)
        """
        with self.lock:
            try:
                # Track dish activity for cycle monitoring
                self._track_dish_activity(has_dishes=len(return_dish_detections) > 0)
                
                # 1. Initialize phase if not done yet using return phase detections
                if not self.state.is_phase_initialized:
                    self._try_initialize_phase_from_return(return_phase_detections, secondary_phase_detections)
                
                # 2. Process return phase numbers (previous stage-phase numbers)
                self._process_return_phase_numbers(return_phase_detections)
                
                # 3. Process return dish detections (dishes coming back from customers)
                self._process_return_dish_detections(return_dish_detections)
                
                # 4. Update automatic phase progression
                self._update_automatic_phase_progression()
                
                # 5. Check for cycle completion
                self._check_cycle_completion()
                
                # 6. Update rates and cleanup
                self._update_rates()
                self._cleanup_old_data()
                
                self.state.last_updated = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error updating from return detections: {e}")
    
    def _try_initialize_phase(self, incoming_phase_detections: List[NumberDetection], return_phase_detections: List[NumberDetection]):
        """Initialize phase only when incoming_phase ROI detects a number"""
        if self.state.is_phase_initialized:
            # After initialization, incoming ROI detects current sequential phase numbers
            if incoming_phase_detections:
                best_detection = max(incoming_phase_detections, key=lambda x: x.confidence)
                detected_phase = best_detection.number
                
                self.logger.info(f"🎯 INCOMING ROI: Detected phase {detected_phase} (confidence: {best_detection.confidence:.3f})")
                
                # Update current phase when new number passes through
                if detected_phase != self.state.current_phase:
                    old_phase = self.state.current_phase
                    
                    # Check for stage progression BEFORE updating phase
                    stage_changed = False
                    if detected_phase < old_phase and old_phase >= 12:
                        # Stage transition detected (phase cycle completed)
                        self.state.current_stage += 1
                        stage_changed = True
                        self.logger.info(f"📊 STAGE TRANSITION: {self.state.current_stage-1} → {self.state.current_stage} (phase {old_phase} → {detected_phase})")
                        
                        # Reset current stage dishes for new stage
                        for dish_type in self.state.current_stage_dishes:
                            self.state.current_stage_dishes[dish_type] = 0
                        self.logger.info(f"🔄 Current stage dishes reset for new stage {self.state.current_stage}")
                    
                    # Update current phase (INITIALIZATION - bypassing protection for system startup)
                    self.state.current_phase = detected_phase
                    
                    if stage_changed:
                        self.logger.info(f"📍 STAGE {self.state.current_stage}: Phase UPDATED from {old_phase} to {detected_phase} (NEW STAGE)")
                    else:
                        self.logger.info(f"📍 STAGE {self.state.current_stage}: Phase UPDATED from {old_phase} to {detected_phase}")
                    
                    # Initialize phase data if needed
                    if detected_phase not in self.phase_data:
                        self.phase_data[detected_phase] = PhaseData(phase_number=detected_phase)
                        self.logger.info(f"🆕 Created new phase data for phase {detected_phase}")
                else:
                    self.logger.debug(f"🔄 STAGE {self.state.current_stage}: Phase {detected_phase} confirmed (no change)")
            else:
                self.logger.debug(f"🔍 INCOMING ROI: No numbers detected")
            return
        
        # CORRECTED INITIALIZATION LOGIC:
        # Return ROI shows higher numbers (completed phases like 11)
        # Incoming ROI should show current sequential phases (like 8, then 9, 10...)
        
        # Initialize based on incoming ROI detections
        if incoming_phase_detections:
            # Direct initialization from incoming ROI
            best_detection = max(incoming_phase_detections, key=lambda x: x.confidence)
            detected_phase = best_detection.number
            
            # Set current phase directly (INITIALIZATION - bypassing protection for system startup)
            self.state.current_phase = detected_phase
            
            # For initialization, start with stage based on detected phase context
            # But follow the stage rule: previous_stage < current_stage (special case for stage 0)
            estimated_stage = detected_phase // 12 if detected_phase > 0 else 0
            self.state.current_stage = estimated_stage
            self.state.is_phase_initialized = True
            
            if detected_phase not in self.phase_data:
                self.phase_data[detected_phase] = PhaseData(phase_number=detected_phase)
                
            self.logger.info(f"🚀 SYSTEM INITIALIZED: Stage {estimated_stage}, Phase {detected_phase} from incoming ROI")
                
            self.logger.info(f"✅ INCOMING ROI: Phase initialized at {detected_phase}")
        else:
            # No numbers detected - try to stay in phase 0 (with protection)
            self._set_current_phase_protected(0, "NO_DETECTION")
            # STAGE PROTECTION: Don't reset stage to 0 unless no-dish cycle allows it
            if self._should_allow_stage_change(0):
                self.state.current_stage = 0
                self.logger.info("🛡️ STAGE PROTECTION: Allowing stage reset to 0 (no numbers detected)")
            else:
                self.logger.info(f"🛡️ STAGE PROTECTION: Keeping stage at {self.state.current_stage} (no numbers, but no no-dish cycle)")
            
            if 0 not in self.phase_data:
                self.phase_data[0] = PhaseData(phase_number=0)
                
            self.phase_initialization_attempts += 1
            if self.phase_initialization_attempts % 100 == 0:
                self.logger.info(f"🔄 Waiting for phase detection - staying in phase 0 (attempts: {self.phase_initialization_attempts})")
    
    def _process_incoming_phase_numbers(self, incoming_phase_detections: List[NumberDetection]):
        """Process incoming phase numbers during initialization only"""
        # This method is only called during initialization phase
        # After initialization, we use automatic phase progression
        pass
    
    def _process_return_phase_numbers(self, return_phase_detections: List[NumberDetection]):
        """
        Process return phase numbers and handle phase cycle completion
        When a number crosses return_phase ROI, it means that phase completed a cycle
        
        CRITICAL: Only processes detections from 'return_phase' ROI (ROI restriction enforced)
        Args:
            return_phase_detections: Phase number detections from return_phase ROI ONLY
        """
        if not return_phase_detections:
            return
        
        # Get most confident return phase detection
        best_detection = max(return_phase_detections, key=lambda x: x.confidence)
        detected_return_phase = best_detection.number
        
        # Prevent phase numbers from decreasing (except for allowed cycle: 12→0)
        if not self._is_phase_decrease_allowed(self.state.last_return_phase, detected_return_phase):
            self.log_detection_vs_state("return_phase", detected_return_phase, "BLOCKED")
            self.logger.info(f"🛡️ RETURN PHASE PROTECTION: {self.state.last_return_phase} → {detected_return_phase} BLOCKED - ignoring detection")
            self.logger.debug(f"Protection rule: phases cannot decrease except 12→0 (detected: {detected_return_phase}, current: {self.state.last_return_phase})")
            return  # Completely ignore this detection
        
        self.logger.debug(f"Return ROI: Phase {detected_return_phase} detected")
        
        # Update last return phase and trigger cycle completion logic
        if detected_return_phase != self.state.last_return_phase:
            old_return_phase = self.state.last_return_phase
            old_return_stage = self.state.last_return_stage
            
            # UPDATE RETURN PHASE - Single clear place (with protection already applied above)
            self.state.last_return_phase = detected_return_phase
            
            # Apply stage rule using helper function
            self.state.last_return_stage = self._calculate_previous_stage(self.state.current_stage)
            
            # Log detection vs state for debugging
            self.log_detection_vs_state("return_phase", detected_return_phase, "UPDATED")
            
            # Log the phase update clearly
            if old_return_phase == 12 and detected_return_phase == 0:
                self.logger.info(f"✅ RETURN ROI: Previous phase cycle completion {old_return_phase} → {detected_return_phase}")
            else:
                self.logger.info(f"✅ RETURN ROI: Previous phase updated {old_return_phase} → {detected_return_phase}")
            
            self.logger.debug(f"Stage relationship: Current={self.state.current_stage}, Previous={self.state.last_return_stage}")
            
            # Handle the cycle completion and dish movement
            if self.state.is_phase_initialized:
                # Only apply complex phase cycle logic for stage 1+
                # For stage 0, just track the return phase without dish movement
                if self.state.current_stage >= 1:
                    self._handle_phase_cycle_completion(detected_return_phase)
                else:
                    self.logger.debug(f"Stage 0: Return phase {detected_return_phase} noted, no dish movement")
            else:
                self.logger.debug(f"System not initialized - storing return phase {detected_return_phase}")
            
            # No phase relationship validation - use detected phases directly
        else:
            # No phase change - detected phase matches current return phase
            self.log_detection_vs_state("return_phase", detected_return_phase, "NO_CHANGE")
            self.logger.debug(f"🔄 RETURN ROI: Phase {detected_return_phase} matches previous phase - no update needed")
    
    def _process_dish_detections_roi(self, dish_detections: List[DishDetection]):
        """Process dish detections using ROI crossing detection AND count all detected dishes"""
        if not dish_detections:
            return
        
        current_time = datetime.now()
        
        # Update dish tracking and get dishes that crossed ROI boundaries
        crossed_dishes = self._update_dish_tracking(dish_detections)
        
        # ONLY count dishes that have CROSSED the ROI boundary (not just detected in ROI)
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if tracked_dish and 'dish_detection' in tracked_dish.crossed_rois:
                dish_type = tracked_dish.dish_type
                # Count only dishes that crossed ROI boundary for current stage
                self.state.current_stage_dishes[dish_type] += 1
                self.logger.info(f"📊 CURRENT STAGE COUNT (ROI CROSSED): {dish_type} +1 → {self.state.current_stage_dishes[dish_type]} total")
                # Note: Don't remove from crossed_rois here as it's used later for other tracking
        
        # Process dishes that crossed ROI boundaries for advanced tracking
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if not tracked_dish:
                continue
                
            dish_type = tracked_dish.dish_type
            
            # Check which ROI was crossed
            if 'dish_detection' in tracked_dish.crossed_rois:
                # Dish crossed main detection ROI - add to advanced tracking
                self.state.dishes_to_customer[dish_type] += 1
                
                # Add to current phase data
                if self.state.current_phase in self.phase_data:
                    # Create a detection object for the tracked dish
                    detection = DishDetection(
                        bbox=(0, 0, 0, 0),  # We don't need bbox for this
                        confidence=1.0,
                        dish_type=dish_type,
                        center_point=tracked_dish.center_point,
                        timestamp=current_time
                    )
                    self.phase_data[self.state.current_phase].dishes_sent.append(detection)
                    self.phase_data[self.state.current_phase].expected_return_count += 1
                
                # Update last dish time
                self.state.last_dish_time = current_time
                
                self.logger.info(f"🎯 Dish crossed ROI boundary: {dish_type} in stage {self.state.current_stage} phase {self.state.current_phase}")
                self.logger.info(f"📊 Total current stage dishes: {sum(self.state.current_stage_dishes.values())}")
                
                # Remove from crossed_rois to avoid double counting
                tracked_dish.crossed_rois.discard('dish_detection')
        
        # Calculate new dishes served: Kitchen total - Returned total
        self._calculate_new_dishes_served()
        
        # Add to history (only dishes that actually crossed)
        crossing_detections = []
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if tracked_dish:
                crossing_detections.append(DishDetection(
                    bbox=(0, 0, 0, 0),
                    confidence=1.0,
                    dish_type=tracked_dish.dish_type,
                    center_point=tracked_dish.center_point,
                    timestamp=current_time
                ))
        self.dish_history.extend(crossing_detections)
    
    def _update_automatic_phase_progression(self):
        """Update phase progression - DISABLED for POC (stay at detected phase)"""
        if not self.state.is_phase_initialized:
            # Phase not initialized - system remains in phase 0 until incoming_phase detects a number
            return
        
        # For POC: Phase stays at detected number, no automatic progression
        # This ensures that when phase "9" is detected, it stays at "9"
        # Automatic progression is disabled
        return
    
    def _advance_phase_automatically(self):
        """Advance to the next phase automatically WITH PROTECTION"""
        old_phase = self.state.current_phase
        new_phase = (self.state.current_phase + 1) % 13  # Calculate next phase
        
        # Apply phase protection before updating
        if not self._set_current_phase_protected(new_phase, "AUTO_ADVANCE"):
            return  # Block automatic advancement if it would decrease phase illegally
        
        # Check for stage transition (phase coming back to 0)
        if self.state.current_phase == 0 and old_phase > 0:
            self.state.current_stage += 1
            self.logger.info(f"📊 Stage transition: {self.state.current_stage-1} → {self.state.current_stage}")
        
        # Update last return phase tracking (no longer auto-calculating expected)
        # Return phase is now tracked separately from incoming ROI detections
        
        # Initialize new phase data if needed
        if self.state.current_phase not in self.phase_data:
            self.phase_data[self.state.current_phase] = PhaseData(phase_number=self.state.current_phase)
        
        self.logger.info(f"🔄 Auto-advanced: Phase {old_phase} → {self.state.current_phase}")
    
    def _calculate_new_dishes_served(self):
        """Calculate new dishes served using simple equation: Kitchen total - Returned total"""
        # Simple calculation as requested by user
        new_dishes = max(0, self.state.total_kitchen_dishes_served - self.state.total_returned_dishes)
        
        if new_dishes != self.state.new_dishes_served:
            old_value = self.state.new_dishes_served
            self.state.new_dishes_served = new_dishes
            self.state.last_calculation_time = datetime.now()
            
            self.logger.info(f"🍽️ New Dishes Calculation:")
            self.logger.info(f"   Kitchen Total: {self.state.total_kitchen_dishes_served}")
            self.logger.info(f"   Returned Total: {self.state.total_returned_dishes}")
            self.logger.info(f"   New Dishes Served: {new_dishes} (was {old_value})")
        
        return new_dishes
    
    def update_kitchen_dish_count(self, dish_detections: List[DishDetection]):
        """Update kitchen served dish count using ROI crossing detection (kitchen camera)"""
        if not dish_detections:
            self.logger.debug("📥 No kitchen dish detections to process")
            return
        
        current_time = datetime.now()
        
        self.logger.info(f"📥 Kitchen processing {len(dish_detections)} dish detections")
        self.logger.debug(f"Kitchen tracker: {len(dish_detections)} dishes")
        
        # Use separate tracking for kitchen camera (different from break line camera)
        kitchen_crossed_dishes = self._update_kitchen_dish_tracking(dish_detections)
        
        self.logger.info(f"🎯 Kitchen ROI crossings detected: {len(kitchen_crossed_dishes)} dishes")
        
        # Only count dishes that actually crossed the kitchen ROI
        for dish_id in kitchen_crossed_dishes:
            # The dish_id contains the dish type information (format: "dish_type_x_y")
            # Need to extract "red_dish" or "yellow_dish", not just "red" or "yellow"
            dish_type_parts = dish_id.split('_')
            if len(dish_type_parts) >= 2:
                dish_type = '_'.join(dish_type_parts[:2])  # Extract "red_dish" or "yellow_dish"
            else:
                dish_type = dish_type_parts[0]  # Fallback for unexpected format
            
            self.logger.debug(f"Processing: {dish_id} → {dish_type}")
            
            if dish_type != 'advertisement_dish':
                # Count total dishes served (only when crossing ROI)
                old_total = self.state.total_kitchen_dishes_served
                self.state.total_kitchen_dishes_served += 1
                
                # Count by dish type
                if dish_type in self.state.kitchen_dishes_served:
                    self.state.kitchen_dishes_served[dish_type] += 1
                    self.logger.debug(f"Kitchen: {dish_type} → {self.state.kitchen_dishes_served[dish_type]}")
                else:
                    self.logger.warning(f"⚠️ Unknown dish type for kitchen tracking: {dish_type}")
                
                # Update rate calculation data for red and yellow dishes only
                if dish_type in ['red_dish', 'yellow_dish'] and dish_type in self.rate_calculation_data:
                    self.rate_calculation_data[dish_type].append(current_time)
                    self.logger.debug(f"Rate: +1 {dish_type[:3]}")
                
                self.logger.debug(f"Kitchen: {dish_type} served (total: {self.state.total_kitchen_dishes_served})")
        
        if kitchen_crossed_dishes:
            self.logger.info(f"🔄 Triggering rate update after {len(kitchen_crossed_dishes)} kitchen crossings")
            # Recalculate rates after kitchen update
            self._update_rates()
            
            # Recalculate new dishes served after kitchen update
            self._calculate_new_dishes_served()
        else:
            self.logger.debug("⚠️ No kitchen ROI crossings detected - no rate update triggered")
    
    def get_dish_serving_summary(self) -> Dict[str, any]:
        """Get summary of dish serving using simple equation approach"""
        return {
            'current_stage': self.state.current_stage,
            'current_phase': self.state.current_phase,
            'total_kitchen_dishes': self.state.total_kitchen_dishes_served,
            'total_returned_dishes': self.state.total_returned_dishes,
            'new_dishes_served': self.state.new_dishes_served,
            'last_calculation': self.state.last_calculation_time.isoformat(),
            'equation': f"{self.state.total_kitchen_dishes_served} - {self.state.total_returned_dishes} = {self.state.new_dishes_served}"
        }
    
    def get_kitchen_dish_totals(self) -> Dict[str, int]:
        """Get kitchen dish counts by type"""
        return {
            'normal_dish': self.state.kitchen_dishes_served.get('normal_dish', 0),
            'red_dish': self.state.kitchen_dishes_served.get('red_dish', 0),
            'yellow_dish': self.state.kitchen_dishes_served.get('yellow_dish', 0),
            'advertisement_dish': self.state.kitchen_dishes_served.get('advertisement_dish', 0)
        }
    
    def _handle_phase_return(self, return_phase: int):
        """Handle expected phase return"""
        if return_phase in self.phase_data:
            phase_data = self.phase_data[return_phase]
            self.logger.info(f"📤 Phase {return_phase} completed return - sent: {len(phase_data.dishes_sent)} dishes")
    
    def _handle_early_return(self, early_return_phase: int):
        """Handle early return detection"""
        if early_return_phase in self.phase_data:
            phase_data = self.phase_data[early_return_phase]
            remaining_dishes = len(phase_data.dishes_sent) - len(phase_data.dishes_returned)
            self.logger.info(f"⚡ Early return Phase {early_return_phase}: {remaining_dishes} dishes remaining")
    
    def _check_cycle_completion(self):
        """Check if a complete cycle is done and reset if needed"""
        current_time = datetime.now()
        time_since_last_dish = current_time - self.state.last_dish_time
        
        # If no dishes for the timeout period, consider cycle complete
        if time_since_last_dish > self.no_dish_timeout:
            if not self.state.cycle_complete:
                self.state.cycle_complete = True
                self.state.shifts_completed += 1
                self.logger.info(f"✅ Cycle completed! Shift #{self.state.shifts_completed} done")
                
                # Reset to stage 0 for next cycle
                self._reset_for_next_cycle()
    
    def _reset_for_next_cycle(self):
        """Reset system for next cycle with stage protection"""
        # STAGE PROTECTION: Only reset stage if no-dish cycle allows it
        if self._should_allow_stage_change(0):
            self.state.current_stage = 0
            self.logger.info("🛡️ STAGE PROTECTION: Allowing cycle reset to stage 0")
        else:
            self.logger.info(f"🛡️ STAGE PROTECTION: Keeping stage at {self.state.current_stage} during cycle reset")
            
        # PHASE PROTECTION: Only reset phase if protection allows it
        if self._set_current_phase_protected(0, "CYCLE_RESET"):
            self.logger.info("🛡️ PHASE PROTECTION: Allowing cycle reset to phase 0")
        else:
            self.logger.info(f"🛡️ PHASE PROTECTION: Keeping phase at {self.state.current_phase} during cycle reset")
            
        self.state.is_phase_initialized = False
        self.state.cycle_complete = False
        
        # Clear old phase data but keep recent history
        old_phases = [p for p, data in self.phase_data.items() 
                     if datetime.now() - data.start_time > timedelta(hours=1)]
        for phase in old_phases:
            del self.phase_data[phase]
        
        self.phase_initialization_attempts = 0
        self.logger.info("🔄 System reset for next cycle - back to phase 0, waiting for incoming_phase detection")
    
    def _reset_current_stage_dishes(self):
        """Reset the current stage dish count (called when stage advances)"""
        old_counts = self.state.current_stage_dishes.copy()
        total_old = sum(old_counts.values())
        
        # Reset to zero
        self.state.current_stage_dishes = {
            'normal_dish': 0,
            'red_dish': 0,
            'yellow_dish': 0,
            'advertisement_dish': 0
        }
        
        if total_old > 0:
            self.logger.info(f"🔄 Reset stage dishes: {old_counts} → {self.state.current_stage_dishes}")
    
    def _advance_stage(self):
        """Advance to the next stage and reset current stage dish count"""
        old_stage = self.state.current_stage
        self.state.current_stage += 1
        
        # Handle stage wraparound if needed
        if self.state.current_stage > self.total_stages:
            self.state.current_stage = 1
        
        # Reset current stage dish count when advancing to new stage
        self._reset_current_stage_dishes()
        
        # Trigger break line logic
        self._handle_break_line_transition(old_stage, self.state.current_stage)
        
        self.logger.info(f"Advanced from stage {old_stage} to stage {self.state.current_stage} (reset stage dish count)")
    
    def _handle_break_line_transition(self, old_stage: int, new_stage: int):
        """
        Handle break line transition logic
        Push dishes from previous stage-phase to new stage-phase
        """
        current_time = datetime.now()
        
        # Prevent rapid break line triggers
        if current_time - self.last_break_line_time < self.break_line_cooldown:
            return
        
        self.last_break_line_time = current_time
        self.break_line_detected = True
        
        # Find active stage-phases to transition
        source_phases = []
        for key, data in self.stage_phase_data.items():
            stage, phase = key
            if stage == old_stage and data.is_active and data.dishes:
                source_phases.append((stage, phase))
        
        # Move dishes to new stage
        if source_phases:
            # For simplicity, move dishes to new stage phase 0
            # In reality, this would depend on conveyor belt mechanics
            target_key = (new_stage, 0)
            self._initialize_stage_phase(new_stage, 0)
            
            for source_stage, source_phase in source_phases:
                source_key = (source_stage, source_phase)
                source_data = self.stage_phase_data[source_key]
                
                # Move dishes
                self.stage_phase_data[target_key].dishes.extend(source_data.dishes)
                
                # Mark source as inactive
                source_data.is_active = False
                
                self.logger.info(f"Moved {len(source_data.dishes)} dishes from "
                               f"stage {source_stage} phase {source_phase} to "
                               f"stage {new_stage} phase 0")
    
    def _process_dish_detections(self, dish_detections: List[DishDetection]):
        """Process current frame dish detections"""
        current_key = (self.state.current_stage, self.state.current_phase)
        self._initialize_stage_phase(self.state.current_stage, self.state.current_phase)
        
        # Add new dishes to current stage-phase
        new_dishes = []
        for detection in dish_detections:
            # Only count non-advertisement dishes in totals
            if detection.dish_type != 'advertisement_dish':
                self.state.dishes_to_customer[detection.dish_type] += 1
                new_dishes.append(detection)
                
                # Add to rate calculation data
                if detection.dish_type in self.rate_calculation_data:
                    self.rate_calculation_data[detection.dish_type].append(
                        detection.timestamp
                    )
            
            # Add to history for tracking
            self.dish_history.append(detection)
        
        # Add to current stage-phase
        if new_dishes:
            self.stage_phase_data[current_key].dishes.extend(new_dishes)
            self.logger.debug(f"Added {len(new_dishes)} dishes to stage "
                            f"{self.state.current_stage} phase {self.state.current_phase}")
    
    def _update_rates(self):
        """Update dishes per minute rates for red and yellow dishes (kitchen camera only)"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.rate_window)
        
        for dish_type in ['red_dish', 'yellow_dish']:
            # Filter recent entries (only from kitchen camera)
            recent_entries = [
                timestamp for timestamp in self.rate_calculation_data[dish_type]
                if timestamp > cutoff_time
            ]
            
            # Calculate rate (dishes per minute)
            if recent_entries:
                rate = len(recent_entries) * (60.0 / self.rate_window)
                old_rate = self.state.dishes_per_minute[dish_type]
                self.state.dishes_per_minute[dish_type] = round(rate, 2)
                
                # if rate != old_rate:
                #     # self.logger.debug(f"📊 Kitchen rate update: {dish_type} = {rate:.2f}/min (from {len(recent_entries)} detections)")
            else:
                self.state.dishes_per_minute[dish_type] = 0.0
    
    def _cleanup_old_data(self):
        """Clean up old inactive stage-phase data"""
        current_time = datetime.now()
        cleanup_threshold = timedelta(minutes=10)  # Keep data for 10 minutes
        
        keys_to_remove = []
        for key, data in self.stage_phase_data.items():
            if (not data.is_active and 
                current_time - data.entry_time > cleanup_threshold):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.stage_phase_data[key]
            self.logger.debug(f"Cleaned up old data for stage {key[0]} phase {key[1]}")
    
    def _generate_dish_id(self, center_point: Tuple[int, int], dish_type: str) -> str:
        """Generate a unique ID for a dish based on position and type"""
        return f"{dish_type}_{center_point[0]}_{center_point[1]}"
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _is_point_in_roi(self, point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        """Check if a point is inside an ROI"""
        x, y = point
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _update_dish_tracking(self, dish_detections: List[DishDetection]) -> List[str]:
        """
        Update dish tracking and return list of dishes that crossed ROIs
        
        Returns:
            List of dish IDs that crossed through ROI boundaries
        """
        current_time = datetime.now()
        crossed_dishes = []
        
        # Clean up old tracked dishes
        expired_ids = [
            dish_id for dish_id, tracked_dish in self.tracked_dishes.items()
            if (current_time - tracked_dish.last_seen).total_seconds() > self.dish_timeout
        ]
        for dish_id in expired_ids:
            del self.tracked_dishes[dish_id]
        
        # Process current detections
        total_detections = len([d for d in dish_detections if d.dish_type != 'advertisement_dish'])
        if total_detections > 0:
            self.logger.debug(f"🔍 Processing {total_detections} dish detections for ROI crossing")
        
        for detection in dish_detections:
            if detection.dish_type == 'advertisement_dish':
                continue
                
            center = detection.center_point
            self.logger.debug(f"🍽️ Processing dish: {detection.dish_type} at {center}")
            
            # Find matching tracked dish or create new one
            matched_dish_id = None
            min_distance = float('inf')
            
            for dish_id, tracked_dish in self.tracked_dishes.items():
                if tracked_dish.dish_type == detection.dish_type:
                    distance = self._distance(center, tracked_dish.center_point)
                    if distance < self.dish_tracking_threshold and distance < min_distance:
                        min_distance = distance
                        matched_dish_id = dish_id
            
            if matched_dish_id:
                # Update existing tracked dish
                tracked_dish = self.tracked_dishes[matched_dish_id]
                old_center = tracked_dish.center_point
                tracked_dish.center_point = center
                tracked_dish.last_seen = current_time
                
                # Check ROI status changes (ROI crossing detection)
                old_roi_status = tracked_dish.roi_status.copy()
                new_roi_status = {}
                
                # Check each ROI
                roi_names = ['dish_detection', 'incoming_phase', 'return_phase']
                for roi_name in roi_names:
                    roi_coords = ROI_CONFIG.get(roi_name)
                    if roi_coords:
                        new_roi_status[roi_name] = self._is_point_in_roi(center, roi_coords)
                
                tracked_dish.roi_status = new_roi_status
                
                # Detect ROI crossings (entering ROI from outside)
                for roi_name in roi_names:
                    old_in_roi = old_roi_status.get(roi_name, False)
                    new_in_roi = new_roi_status.get(roi_name, False)
                    
                    if not old_in_roi and new_in_roi:
                        # Dish entered ROI
                        tracked_dish.crossed_rois.add(roi_name)
                        crossed_dishes.append(matched_dish_id)
                        self.logger.info(f"🎯 Dish {matched_dish_id} crossed into {roi_name} ROI")
                        self.logger.debug(f"   Position: {old_center} → {center}")
                        self.logger.debug(f"   ROI coords: {ROI_CONFIG.get(roi_name)}")
                    elif old_in_roi and not new_in_roi:
                        # Dish exited ROI (for debugging)
                        self.logger.debug(f"🚪 Dish {matched_dish_id} exited {roi_name} ROI")
            else:
                # Create new tracked dish
                dish_id = self._generate_dish_id(center, detection.dish_type)
                
                # Check initial ROI status
                roi_status = {}
                roi_names = ['dish_detection', 'incoming_phase', 'return_phase']
                for roi_name in roi_names:
                    roi_coords = ROI_CONFIG.get(roi_name)
                    if roi_coords:
                        roi_status[roi_name] = self._is_point_in_roi(center, roi_coords)
                
                self.tracked_dishes[dish_id] = TrackedDish(
                    id=dish_id,
                    dish_type=detection.dish_type,
                    center_point=center,
                    last_seen=current_time,
                    roi_status=roi_status,
                    crossed_rois=set()
                )
                
                # If dish is created inside dish_detection ROI, count it as crossing
                # (this handles cases where dishes are first detected mid-crossing)
                if roi_status.get('dish_detection', False):
                    self.tracked_dishes[dish_id].crossed_rois.add('dish_detection')
                    crossed_dishes.append(dish_id)
                    self.logger.info(f"🎯 New dish detected inside dish_detection ROI, counting as crossing: {dish_id}")
                
                self.logger.debug(f"🆕 New dish tracked: {dish_id} at {center}")
        
        if total_detections > 0 and not crossed_dishes:
            self.logger.debug(f"⚠️ {total_detections} dishes detected but no ROI crossings detected")
            self.logger.debug(f"   Currently tracking {len(self.tracked_dishes)} dishes")
            self.logger.debug(f"   ROI coords: dish_detection={ROI_CONFIG.get('dish_detection')}")
        
        return crossed_dishes
    
    def _update_kitchen_dish_tracking(self, dish_detections: List[DishDetection]) -> List[str]:
        """
        Update kitchen dish tracking and return list of dishes that crossed kitchen ROI
        
        Returns:
            List of dish IDs that crossed through kitchen ROI boundary
        """
        current_time = datetime.now()
        crossed_dishes = []
        
        # Use a separate tracking dictionary for kitchen camera
        if not hasattr(self, 'kitchen_tracked_dishes'):
            self.kitchen_tracked_dishes = {}
        
        # Clean up old tracked dishes from kitchen
        expired_ids = [
            dish_id for dish_id, tracked_dish in self.kitchen_tracked_dishes.items()
            if (current_time - tracked_dish.last_seen).total_seconds() > self.dish_timeout
        ]
        for dish_id in expired_ids:
            del self.kitchen_tracked_dishes[dish_id]
        
        # Process current detections for kitchen ROI
        for detection in dish_detections:
            if detection.dish_type == 'advertisement_dish':
                continue
                
            center = detection.center_point
            
            # Find matching tracked dish or create new one
            matched_dish_id = None
            min_distance = float('inf')
            
            for dish_id, tracked_dish in self.kitchen_tracked_dishes.items():
                if tracked_dish.dish_type == detection.dish_type:
                    distance = self._distance(center, tracked_dish.center_point)
                    # Log dish matching for debugging
                    self.logger.debug(f"🔍 Checking match: {dish_id} ({tracked_dish.dish_type}) at {tracked_dish.center_point}, distance={distance:.1f}, threshold={self.dish_tracking_threshold}")
                    if distance < self.dish_tracking_threshold and distance < min_distance:
                        min_distance = distance
                        matched_dish_id = dish_id
                        # self.logger.debug(f"🔍 Found potential match: {dish_id}, distance={distance:.1f}")
            
            if matched_dish_id:
                # Update existing tracked dish
                tracked_dish = self.kitchen_tracked_dishes[matched_dish_id]
                old_center = tracked_dish.center_point
                tracked_dish.center_point = center
                tracked_dish.last_seen = current_time
                
                # self.logger.info(f"🔄 MATCHED EXISTING DISH: {matched_dish_id} ({tracked_dish.dish_type}) - updating position from {old_center} to {center}")
                
                # Check kitchen ROI status changes (proper movement tracking)
                old_in_kitchen = tracked_dish.roi_status.get('kitchen_counter', False)
                new_in_kitchen = self._is_point_in_roi(center, self.roi_kitchen_counter)
                tracked_dish.roi_status['kitchen_counter'] = new_in_kitchen
                
                # Detect kitchen ROI crossing (entering ROI from outside)
                if not old_in_kitchen and new_in_kitchen:
                    # Dish entered kitchen ROI - this is a genuine crossing
                    tracked_dish.crossed_rois.add('kitchen_counter')
                    crossed_dishes.append(matched_dish_id)
                    # self.logger.info(f"🍽️ GENUINE ROI CROSSING: Dish {matched_dish_id} ({tracked_dish.dish_type}) crossed INTO kitchen ROI")
                    # self.logger.info(f"🍽️ Position change: {old_center} → {center}")
                elif old_in_kitchen and new_in_kitchen:
                    # Dish moved within ROI - don't count again
                    self.logger.debug(f"🔄 WITHIN ROI: {matched_dish_id} moved within kitchen ROI (no new crossing)")
                elif old_in_kitchen and not new_in_kitchen:
                    # Dish left ROI
                    self.logger.debug(f"🔄 LEFT ROI: {matched_dish_id} moved out of kitchen ROI")
            else:
                # Create new tracked dish for kitchen
                dish_id = self._generate_dish_id(center, detection.dish_type)
                
                # Verify the dish is actually in the kitchen ROI (double-check filtering)
                in_kitchen = self._is_point_in_roi(center, self.roi_kitchen_counter)
                
                # # Detailed debugging for ROI detection
                # self.logger.info(f"🔍 NEW DISH: {detection.dish_type} at {center}")
                # self.logger.info(f"🔍 Kitchen ROI = {self.roi_kitchen_counter}")
                # self.logger.info(f"🔍 ROI verification: in_kitchen = {in_kitchen}")
                
                if in_kitchen:
                    self.kitchen_tracked_dishes[dish_id] = TrackedDish(
                        id=dish_id,
                        dish_type=detection.dish_type,
                        center_point=center,
                        last_seen=current_time,
                        roi_status={'kitchen_counter': in_kitchen},
                        crossed_rois=set()
                    )
                    
                    # For new dishes appearing directly in ROI, we consider this a crossing
                    # But only if they're actually in the ROI boundary
                    self.kitchen_tracked_dishes[dish_id].crossed_rois.add('kitchen_counter')
                    crossed_dishes.append(dish_id)
                    self.logger.info(f"🔥 NEW DISH IN KITCHEN ROI: {dish_id} ({detection.dish_type}) - counted as genuine crossing")
                else:
                    # This shouldn't happen if apply_roi_to_detections works correctly
                    self.logger.warning(f"⚠️ FILTERED DISH OUTSIDE ROI: {detection.dish_type} at {center} not in kitchen ROI - skipping")
                    continue
                
                self.logger.debug(f"🆕 New kitchen dish tracked: {dish_id} at {center}")
        
        return crossed_dishes
    
    def _update_rates(self):
        """Update dishes per minute rates for red and yellow dishes (kitchen camera only)"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.rate_window)
        
        self.logger.debug(f"📊 Updating rates - window: {self.rate_window}s, cutoff: {cutoff_time}")
        
        for dish_type in ['red_dish', 'yellow_dish']:
            # Filter recent entries (only from kitchen camera)
            all_entries = list(self.rate_calculation_data[dish_type])
            recent_entries = [
                timestamp for timestamp in all_entries
                if timestamp > cutoff_time
            ]
            
            self.logger.debug(f"📊 {dish_type}: {len(all_entries)} total entries, {len(recent_entries)} recent entries")
            
            # Calculate rate (dishes per minute)
            if recent_entries:
                rate = len(recent_entries) * (60.0 / self.rate_window)
                old_rate = self.state.dishes_per_minute[dish_type]
                self.state.dishes_per_minute[dish_type] = round(rate, 2)
                
                # self.logger.info(f"📊 Kitchen rate update: {dish_type} = {rate:.2f}/min (from {len(recent_entries)} detections, was {old_rate:.2f})")
            else:
                old_rate = self.state.dishes_per_minute[dish_type]
                self.state.dishes_per_minute[dish_type] = 0.0
                # if old_rate > 0:
                #     self.logger.info(f"📊 Kitchen rate reset: {dish_type} = 0.0/min (no recent detections)")
        
        self.logger.debug(f"Rate calc: red={self.state.dishes_per_minute['red_dish']:.2f}/min, yellow={self.state.dishes_per_minute['yellow_dish']:.2f}/min")
    
    def get_active_stage_phases(self) -> List[Tuple[int, int, int]]:
        """
        Get list of active stage-phases with dish counts for Stage Distribution chart
        
        Returns:
            List of tuples (stage, phase, dish_count)
        """
        active_phases = []
        
        # If phase is initialized, show current stage-phase with current dish count
        if self.state.is_phase_initialized:
            total_current_dishes = sum(self.state.current_stage_dishes.values())
            active_phases.append((
                self.state.current_stage, 
                self.state.current_phase, 
                total_current_dishes
            ))
            
            self.logger.debug(f"🔍 Active stage-phase: Stage {self.state.current_stage}, Phase {self.state.current_phase}, Dishes: {total_current_dishes}")
        else:
            # If not initialized, show phase 0 with current dishes
            total_current_dishes = sum(self.state.current_stage_dishes.values()) 
            active_phases.append((0, 0, total_current_dishes))
            self.logger.debug(f"🔍 Pre-initialization: Stage 0, Phase 0, Dishes: {total_current_dishes}")
        
        return active_phases
    
    def get_current_state(self) -> ConveyorState:
        """Get current conveyor state"""
        return self.state
    
    
    def get_total_dishes_on_belt(self) -> Dict[str, int]:
        """
        Get total count of dishes currently on the conveyor belt for CURRENT STAGE ONLY
        Returns current stage dishes (not cumulative)
        
        Returns:
            Dictionary with dish type counts for current stage only (excluding advertisements)
        """
        totals = {
            'normal_dish': self.state.current_stage_dishes['normal_dish'],
            'red_dish': self.state.current_stage_dishes['red_dish'], 
            'yellow_dish': self.state.current_stage_dishes['yellow_dish']
        }
        return totals
    
    def get_dishes_by_roi(self) -> Dict[str, Dict[str, int]]:
        """
        Get dish counts separated by ROI area
        
        Returns:
            Dictionary with 'to_customer' and 'returning' counts
        """
        return {
            'to_customer': {
                'normal_dish': self.state.dishes_to_customer['normal_dish'],
                'red_dish': self.state.dishes_to_customer['red_dish'],
                'yellow_dish': self.state.dishes_to_customer['yellow_dish']
            },
            'returning': {
                'normal_dish': self.state.dishes_returning['normal_dish'],
                'red_dish': self.state.dishes_returning['red_dish'],
                'yellow_dish': self.state.dishes_returning['yellow_dish']
            }
        }
    
    def get_phase_summary(self) -> Dict[str, any]:
        """
        Get summary of current phase status
        
        Returns:
            Dictionary with phase information
        """
        current_phase_data = self.phase_data.get(self.state.current_phase)
        
        return {
            'current_phase': self.state.current_phase,
            'current_stage': self.state.current_stage,
            'last_return_phase': self.state.last_return_phase,
            'last_return_stage': self.state.last_return_stage,
            'is_initialized': self.state.is_phase_initialized,
            'dishes_sent_current_phase': len(current_phase_data.dishes_sent) if current_phase_data else 0,
            'expected_returns': current_phase_data.expected_return_count if current_phase_data else 0,
            'cycle_complete': self.state.cycle_complete,
            'shifts_completed': self.state.shifts_completed
        }
    
    def get_phase_dish_tracking(self) -> Dict[int, Dict[str, int]]:
        """
        Get phase-specific dish tracking data
        
        Returns:
            Dictionary mapping phase numbers to dish counts by type
            Format: {phase: {'normal_dish': count, 'red_dish': count, 'yellow_dish': count}}
        """
        return self.state.phase_dish_tracking.copy()
    
    def get_kitchen_dish_summary(self) -> Dict[str, any]:
        """
        Get summary of kitchen dish tracking with dual variables
        
        Returns:
            Dictionary with current stage dishes and total dishes passed through
        """
        return {
            # Current stage dishes (resets when stage changes)
            'current_stage_dishes': self.state.current_stage_dishes.copy(),
            'current_stage_total': sum(self.state.current_stage_dishes.values()),
            
            # Total dishes that have passed through (cumulative)
            'total_dishes_served': self.state.total_kitchen_dishes_served,
            'total_returned_dishes': self.state.total_returned_dishes,
            'new_dishes_served': self.state.new_dishes_served,
            
            # Kitchen dishes by type (cumulative)
            'kitchen_dishes_by_type': self.state.kitchen_dishes_served.copy(),
            
            # Current phase tracking
            'current_phase': self.state.current_phase,
            'current_phase_dishes': self.state.phase_dish_tracking.get(self.state.current_phase, {
                'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0
            }),
            
            # Phase tracking overview
            'active_phases': list(self.state.phase_dish_tracking.keys()),
            'total_phases_tracked': len(self.state.phase_dish_tracking)
        }
    
    def get_stage_behavior_info(self) -> Dict[str, any]:
        """
        Get information about current stage behavior
        
        Returns:
            Dictionary explaining current stage logic
        """
        current_stage = self.state.current_stage
        previous_stage = self.state.last_return_stage
        
        base_info = {
            'current_stage': current_stage,
            'previous_stage': previous_stage,
            'stage_rule': f'previous_stage ({previous_stage}) < current_stage ({current_stage})',
            'stage_rule_status': 'Valid' if previous_stage <= current_stage else 'Invalid',
            'current_phase': self.state.current_phase,
            'previous_phase': self.state.last_return_phase,
            'phase_protection': 'Enabled - phases cannot decrease except 12→0'
        }
        
        if current_stage == 0:
            stage_specific = {
                'behavior': 'Simple dish counting until Phase 0 detected',
                'next_trigger': 'Phase 0 detection in incoming_phase ROI will advance to Stage 1',
                'phase_movement_logic': 'Disabled - no complex phase-to-phase dish movement',
                'cycle_completion_logic': 'Disabled - simple return phase tracking only',
                'stage_rule_note': 'Special case: previous_stage = 0 when current_stage = 0'
            }
        else:
            stage_specific = {
                'behavior': 'Advanced tracking with phase-to-phase dish movement',
                'next_trigger': f'2 detections of Phase 0 will advance to Stage {current_stage + 1}',
                'phase_movement_logic': 'Enabled - dishes move from detected return_phase to detected current_phase',
                'cycle_completion_logic': 'Enabled - return_phase ROI triggers dish movement',
                'stage_rule_note': f'Normal case: previous_stage = current_stage - 1 = {current_stage - 1}'
            }
        
        return {**base_info, **stage_specific}
    
    @staticmethod
    def get_number_detection_classes() -> Dict[str, any]:
        """
        Get information about number detection classes
        
        Returns:
            Dictionary explaining number detection class mapping
        """
        return {
            'classes': list(range(13)),  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            'indices': list(range(13)),  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            'mapping': 'Direct mapping - class value equals index (no position swapping)',
            'description': 'Number detection classes 0-12 map directly to their numeric values',
            'phase_assignment': 'incoming_phase ROI detection → current_phase, return_phase ROI detection → previous_phase',
            'no_calculations': 'Pure ROI-based detection, no mathematical relationships between phases',
            'phase_protection': 'Phases cannot decrease except for 12→0 cycle completion'
        }
    
    def get_stage_relationship_info(self) -> Dict[str, any]:
        """
        Get information about stage relationships and rules
        
        Returns:
            Dictionary explaining stage logic and current state
        """
        current_stage = self.state.current_stage
        previous_stage = self.state.last_return_stage
        calculated_previous = self._calculate_previous_stage(current_stage)
        
        return {
            'stage_rule': 'previous_stage < current_stage (except when current_stage = 0)',
            'special_case': 'When current_stage = 0, previous_stage = 0',
            'normal_case': 'When current_stage >= 1, previous_stage = current_stage - 1',
            'current_state': {
                'current_stage': current_stage,
                'previous_stage': previous_stage,
                'calculated_previous': calculated_previous,
                'rule_compliant': previous_stage == calculated_previous
            },
            'examples': {
                'stage_0': 'Current=0, Previous=0 (special case)',
                'stage_1': 'Current=1, Previous=0',
                'stage_2': 'Current=2, Previous=1',
                'stage_3': 'Current=3, Previous=2'
            }
        }
    
    def get_phase_protection_info(self) -> Dict[str, any]:
        """
        Get information about phase protection rules and current status
        
        Returns:
            Dictionary explaining phase protection logic and current state
        """
        current_phase = self.state.current_phase
        previous_phase = self.state.last_return_phase
        
        return {
            'phase_rule': 'Phases cannot decrease except for 12→0 cycle completion',
            'allowed_transitions': {
                'forward': 'Any phase increase (e.g., 5→6, 11→12)',
                'same': 'Same phase detection (e.g., 8→8)',
                'cycle_completion': '12→0 transition only'
            },
            'blocked_transitions': 'All other decreases (e.g., 8→7, 5→3, 10→2)',
            'current_state': {
                'current_phase': current_phase,
                'previous_phase': previous_phase,
                'current_can_advance_to': list(range(current_phase, 13)) + ([0] if current_phase == 12 else []),
                'previous_can_advance_to': list(range(previous_phase, 13)) + ([0] if previous_phase == 12 else [])
            },
            'protection_examples': {
                'valid': ['0→1', '5→8', '11→12', '12→0'],
                'blocked': ['8→7', '12→11', '5→3', '10→2']
            },
            'helper_function': '_is_phase_decrease_allowed(old_phase, new_phase)'
        }
    
    def get_phase_info_for_display(self) -> Dict[str, int]:
        """
        Get current phase information for display purposes (e.g., bbox labels)
        
        Returns:
            Dictionary with current and previous phase information
        """
        return {
            'current_phase': self.state.current_phase,
            'previous_phase': self.state.last_return_phase,
            'current_stage': self.state.current_stage,
            'previous_stage': self.state.last_return_stage
        }
    
    def debug_phase_state(self, context: str = "DEBUG") -> None:
        """
        Debug method to log current phase state for troubleshooting UI mismatches
        
        Args:
            context: Context string to identify where this debug was called from
        """
        # Quick snapshot without lock to avoid UI freezing
        current_phase = self.state.current_phase
        previous_phase = self.state.last_return_phase
        current_stage = self.state.current_stage
        previous_stage = self.state.last_return_stage
        last_incoming = self.state.last_incoming_phase
        is_initialized = self.state.is_phase_initialized
        
        self.logger.info(f"🔍 {context} - PHASE STATE:")
        self.logger.info(f"  📍 Current Phase: {current_phase} (from incoming ROI)")
        self.logger.info(f"  📍 Previous Phase: {previous_phase} (from return ROI)")
        self.logger.info(f"  📍 Current Stage: {current_stage}")
        self.logger.info(f"  📍 Previous Stage: {previous_stage}")
        self.logger.info(f"  📍 Last Incoming: {last_incoming}")
        self.logger.info(f"  📍 Is Initialized: {is_initialized}")
    
    def log_detection_vs_state(self, roi_type: str, detected_number: int, action_taken: str) -> None:
        """
        Log detection vs current state for debugging UI mismatches
        
        Args:
            roi_type: "incoming_phase" or "return_phase"
            detected_number: The number that was detected
            action_taken: What action was taken (e.g., "UPDATED", "BLOCKED", "IGNORED")
        """
        # Quick snapshot without lock to avoid UI freezing
        if roi_type == "incoming_phase":
            current_value = self.state.current_phase
            self.logger.info(f"🎯 INCOMING ROI: Detected #{detected_number}, Current Phase: {current_value}, Action: {action_taken}")
        elif roi_type == "return_phase":
            current_value = self.state.last_return_phase
            self.logger.info(f"🎯 RETURN ROI: Detected #{detected_number}, Previous Phase: {current_value}, Action: {action_taken}")
        else:
            self.logger.info(f"🎯 {roi_type.upper()}: Detected #{detected_number}, Action: {action_taken}")
    
    def apply_roi_to_detections(self, detections: List, roi_coordinates: Tuple[int, int, int, int]) -> List:
        """
        Filter detections to only include those within ROI
        
        Args:
            detections: List of detections (DishDetection or NumberDetection)
            roi_coordinates: ROI as (x1, y1, x2, y2)
            
        Returns:
            Filtered list of detections within ROI
        """
        if not detections or not roi_coordinates:
            return detections
        
        x1, y1, x2, y2 = roi_coordinates
        filtered_detections = []
        
        self.logger.debug(f"🔍 ROI FILTER: Processing {len(detections)} detections with ROI {roi_coordinates}")
        
        for detection in detections:
            # Get center point of detection
            if hasattr(detection, 'center_point'):
                center_x, center_y = detection.center_point
                # Check if center point is within ROI
                is_within_roi = x1 <= center_x <= x2 and y1 <= center_y <= y2
                
                if is_within_roi:
                    filtered_detections.append(detection)
                    self.logger.debug(f"✅ ROI FILTER: {detection.dish_type if hasattr(detection, 'dish_type') else 'detection'} at {detection.center_point} is WITHIN ROI")
                else:
                    self.logger.debug(f"❌ ROI FILTER: {detection.dish_type if hasattr(detection, 'dish_type') else 'detection'} at {detection.center_point} is OUTSIDE ROI")
        
        self.logger.info(f"🔍 ROI FILTER RESULT: {len(filtered_detections)}/{len(detections)} detections within ROI {roi_coordinates}")
        return filtered_detections
    
    def get_current_state(self) -> ConveyorState:
        """
        Get the current conveyor state including all tracking data and rates
        
        Returns:
            ConveyorState: Current state with all tracking information
        """
        # Return the current state without lock to avoid UI freezing
        return self.state
    
    def get_zero_detection_status(self) -> dict:
        """Get current zero detection count for debugging stage increment logic"""
        return {
            'zero_detection_count': self.state.zero_detection_count,
            'current_stage': self.state.current_stage,
            'current_phase': self.state.current_phase,
            'needs_detections': max(0, 2 - self.state.zero_detection_count),
            'ready_for_stage_increment': self.state.zero_detection_count >= 2,
            'completed_cycles_without_dishes': self.state.completed_cycles_without_dishes,
            'cycle_without_dishes': self.state.cycle_without_dishes
        }
    
    def _should_allow_stage_change(self, new_stage: int) -> bool:
        """
        Determine if stage change should be allowed based on protection rules:
        - Stages only increase, never decrease
        - Exception: Can reset if 1 complete cycle passed without dishes
        """
        current_stage = self.state.current_stage
        
        # Always allow increases
        if new_stage > current_stage:
            self.logger.info(f"🛡️ STAGE PROTECTION: Allowing increase {current_stage} → {new_stage}")
            return True
        
        # Check if this is a reset after no-dish cycle
        if new_stage < current_stage:
            if self.state.completed_cycles_without_dishes >= 1:
                self.logger.info(f"🛡️ STAGE PROTECTION: Allowing reset {current_stage} → {new_stage} (no dishes for {self.state.completed_cycles_without_dishes} cycle(s))")
                return True
            else:
                self.logger.warning(f"🛡️ STAGE PROTECTION: Blocking decrease {current_stage} → {new_stage} (no no-dish cycle)")
                return False
        
        # Same stage - allow (no change)
        return True
    
    def _track_dish_activity(self, has_dishes: bool):
        """Track dish activity to detect cycles without dishes"""
        current_time = datetime.now()
        
        if has_dishes:
            # Update last dish detection time
            self.state.last_dish_detection_time = current_time
            self.state.cycle_without_dishes = False
            self.logger.debug(f"🍽️ DISH ACTIVITY: Dishes detected, cycle reset")
        else:
            # Check if we've gone a full cycle without dishes
            if self.state.last_dish_detection_time:
                time_since_last_dish = current_time - self.state.last_dish_detection_time
                # Consider 1 cycle = 60 seconds (adjust as needed)
                cycle_duration = timedelta(seconds=60)
                
                if time_since_last_dish >= cycle_duration:
                    if not self.state.cycle_without_dishes:
                        self.state.cycle_without_dishes = True
                        self.state.completed_cycles_without_dishes += 1
                        self.logger.info(f"🔄 NO-DISH CYCLE COMPLETED: {self.state.completed_cycles_without_dishes} cycle(s) without dishes")
            else:
                # First time tracking, set initial time
                self.state.last_dish_detection_time = current_time
    
    def reset_counts(self):
        """Reset all dish counts and tracking data"""
        with self.lock:
            self.state = ConveyorState()
            self.stage_phase_data.clear()
            self.dish_history.clear()
            for deque_data in self.rate_calculation_data.values():
                deque_data.clear()
            
            # Re-initialize first stage-phase
            self._initialize_stage_phase(1, 0)
            
            self.logger.info("Conveyor tracker reset successfully")
    
    def is_break_line_active(self) -> bool:
        """Check if break line was recently detected"""
        return (self.break_line_detected and 
                datetime.now() - self.last_break_line_time < timedelta(seconds=2))
    
    def clear_break_line_flag(self):
        """Clear the break line detection flag"""
        self.break_line_detected = False
    
    def update_current_phase(self, incoming_phase_detections: List[NumberDetection]):
        """
        Update CURRENT phase from incoming_phase ROI detections - simplified approach
        
        Uses detected phase numbers directly without calculations
        
        CRITICAL: This method ONLY updates self.state.current_phase
        CRITICAL: This method NEVER updates self.state.last_return_phase (previous phase)
        CRITICAL: Only processes detections from 'incoming_phase' ROI
        
        Args:
            incoming_phase_detections: Phase number detections from incoming_phase ROI ONLY
        """
        if not incoming_phase_detections:
            return
        
        # Use the highest confidence detection
        best_detection = max(incoming_phase_detections, key=lambda x: x.confidence)
        detected_phase = best_detection.number
        
        self.logger.debug(f"Incoming ROI: Phase {detected_phase} detected")
        
        # Initialize system if not yet initialized
        if not self.state.is_phase_initialized:
            # INITIALIZATION - bypassing protection for system startup
            self.state.current_phase = detected_phase
            
            # For stage 0: Stay in stage 0 until phase 0 is detected
            # Only advance to stage 1+ when phase 0 passes through incoming_phase ROI
            if detected_phase == 0:
                # Phase 0 detected - this starts the new stage logic
                self.state.current_stage = 1  # Move to stage 1 when phase 0 detected
                self.logger.info(f"Phase 0 detected - advancing to Stage 1")
            else:
                # Any other phase detected during initialization - stay in stage 0
                self.state.current_stage = 0
                self.logger.info(f"System initialized in Stage 0 with Phase {detected_phase}")
            
            self.state.is_phase_initialized = True
            self.logger.info(f"System initialized: Phase {detected_phase}, Stage {self.state.current_stage}")
            
            # Initialize phase data
            if detected_phase not in self.phase_data:
                self.phase_data[detected_phase] = PhaseData(phase_number=detected_phase)
            return
        
        # Update current phase when new number passes through
        if detected_phase != self.state.current_phase:
            old_phase = self.state.current_phase
            old_stage = self.state.current_stage
            
            # Prevent current phase from decreasing (except for allowed cycle: 12→0)
            if not self._is_phase_decrease_allowed(old_phase, detected_phase):
                self.log_detection_vs_state("incoming_phase", detected_phase, "BLOCKED")
                self.logger.info(f"🛡️ INCOMING PHASE PROTECTION: {old_phase} → {detected_phase} BLOCKED - ignoring detection")
                self.logger.debug(f"Protection rule: phases cannot decrease except 12→0 (detected: {detected_phase}, current: {old_phase})")
                return  # Completely ignore this detection
            
            # Update last incoming phase tracking
            self.state.last_incoming_phase = detected_phase
            
            # SINGLE PLACE FOR CURRENT PHASE UPDATE - No duplication (with protection already applied above)
            self.state.current_phase = detected_phase
            
            # Log detection vs state for debugging
            self.log_detection_vs_state("incoming_phase", detected_phase, "UPDATED")
            
            # Log the phase update clearly
            if old_phase == 12 and detected_phase == 0:
                self.logger.info(f"✅ INCOMING ROI: Current phase cycle completion {old_phase} → {detected_phase}")
            else:
                self.logger.info(f"✅ INCOMING ROI: Current phase updated {old_phase} → {detected_phase}")
            
            # Handle stage logic based on detected phase
            if detected_phase == 0:
                # Handle phase 0 detection - stage increment logic
                if self.state.current_stage == 0:
                    # Stage 0 → Stage 1 when phase 0 is detected
                    self.state.current_stage = 1
                    self.logger.info(f"🎯 STAGE ADVANCE: Phase 0 detected in Stage 0 - advancing to Stage 1")
                    
                    # Reset current stage dishes for new stage
                    self.state.current_stage_dishes = {
                        'normal_dish': 0,
                        'red_dish': 0,
                        'yellow_dish': 0,
                        'advertisement_dish': 0
                    }
                else:
                    # For stage 1+, use the 2-detection logic for stage increment
                    self.state.zero_detection_count += 1
                    self.logger.debug(f"Zero count: {self.state.zero_detection_count}/2 from phase {old_phase}")
                    
                    # Increment stage after 2 detections of number 0
                    if self.state.zero_detection_count >= 2:
                        # Check if we should increment stage (protected - only increase)
                        new_stage = self.state.current_stage + 1
                        
                        # Apply stage protection logic
                        if self._should_allow_stage_change(new_stage):
                            self.state.current_stage = new_stage
                            self.state.zero_detection_count = 0  # Reset counter
                            
                            self.logger.info(f"🎯 STAGE INCREMENT: {old_stage} → {self.state.current_stage}")
                            
                            # Reset current stage dishes for new stage
                            self.state.current_stage_dishes = {
                                'normal_dish': 0,
                                'red_dish': 0,
                                'yellow_dish': 0,
                                'advertisement_dish': 0
                            }
                            
                            # Reset cycle tracking since we had dishes and progressed
                            self.state.completed_cycles_without_dishes = 0
                            self.state.cycle_without_dishes = False
                        else:
                            self.logger.debug(f"Stage increment blocked by protection")
                            self.state.zero_detection_count = 0  # Reset counter anyway
            else:
                # Reset zero detection counter for non-zero numbers
                if self.state.zero_detection_count > 0:
                    self.logger.debug(f"Reset zero counter (was {self.state.zero_detection_count}) - detected number {detected_phase}")
                    self.state.zero_detection_count = 0
            
            # Initialize new phase data if needed
            if self.state.current_phase not in self.phase_data:
                self.phase_data[self.state.current_phase] = PhaseData(phase_number=self.state.current_phase)
            
            # Check for cycle completion
            self._check_cycle_completion()
        else:
            # No phase change - detected phase matches current phase
            self.log_detection_vs_state("incoming_phase", detected_phase, "NO_CHANGE")
            self.logger.debug(f"🔄 INCOMING ROI: Phase {detected_phase} matches current phase - no update needed")

    def update_current_stage_dishes(self, dish_detections: List[DishDetection]):
        """Update current stage dish count from KITCHEN camera (only dishes crossing ROI)"""
        if not dish_detections:
            # Track no dish activity
            self._track_dish_activity(has_dishes=False)
            return
        
        # Track dish activity
        self._track_dish_activity(has_dishes=True)
        
        current_time = datetime.now()
        
        self.logger.debug(f"Kitchen processing {len(dish_detections)} dish detections")
        
        # Use separate tracking for kitchen camera current stage dishes
        kitchen_crossed_dishes = self._update_kitchen_dish_tracking(dish_detections)
        
        if kitchen_crossed_dishes:
            self.logger.debug(f"Kitchen ROI crossings: {len(kitchen_crossed_dishes)} dishes")
        
        # Count dishes that actually crossed the kitchen ROI for current stage
        for dish_id in kitchen_crossed_dishes:
            dish_type_parts = dish_id.split('_')
            if len(dish_type_parts) >= 2:
                dish_type = '_'.join(dish_type_parts[:2])
            else:
                dish_type = dish_type_parts[0]
            
            self.logger.debug(f"Processing dish: {dish_id} → {dish_type}")
            
            if dish_type != 'advertisement_dish':
                # Count for current stage dishes
                if dish_type in self.state.current_stage_dishes:
                    self.state.current_stage_dishes[dish_type] += 1
                
                # Also add to current phase tracking (only for stage 1+)
                if self.state.current_stage >= 1:
                    current_phase = self.state.current_phase
                    self._initialize_phase(current_phase)  # Ensure phase is initialized
                    
                    if current_phase in self.state.phase_dish_tracking:
                        if dish_type in self.state.phase_dish_tracking[current_phase]:
                            self.state.phase_dish_tracking[current_phase][dish_type] += 1
                        
                        # Update phase data object as well
                        if current_phase in self.phase_data:
                            self.phase_data[current_phase].phase_dish_counts = self.state.phase_dish_tracking[current_phase].copy()
                else:
                    # Stage 0: Simple dish counting, no phase tracking
                    self.logger.debug(f"Stage 0: Simple dish counting for {dish_type}")
                
                # Count for overall kitchen serving
                old_total = self.state.total_kitchen_dishes_served
                self.state.total_kitchen_dishes_served += 1
                
                # Count by dish type
                if dish_type in self.state.kitchen_dishes_served:
                    self.state.kitchen_dishes_served[dish_type] += 1
                    self.logger.debug(f"Kitchen: {dish_type} → {self.state.kitchen_dishes_served[dish_type]}")
                
                # Update rate calculation data for red and yellow dishes (for rate metrics)
                if dish_type in ['red_dish', 'yellow_dish']:
                    if dish_type in self.rate_calculation_data:
                        self.rate_calculation_data[dish_type].append(current_time)
                        self.logger.info(f"🔥 RATE UPDATE (Current Stage): +1 {dish_type} at {current_time}, total entries: {len(self.rate_calculation_data[dish_type])}")
                    else:
                        self.logger.warning(f"⚠️ {dish_type} not found in rate_calculation_data: {list(self.rate_calculation_data.keys())}")
                
                self.logger.debug(f"Kitchen: {dish_type} served (total: {self.state.total_kitchen_dishes_served})")
        
        if kitchen_crossed_dishes:
            # Recalculate new dishes served after kitchen update
            self._calculate_new_dishes_served()
            # Update rates based on new crossings
            self._update_rates()
        else:
            self.logger.debug("⚠️ No kitchen ROI crossings detected - no update triggered")
    
    def _try_initialize_phase_from_return(self, return_phase_detections: List[NumberDetection], 
                                         secondary_phase_detections: List[NumberDetection]):
        """Initialize phase from return/previous phase detections - simplified approach"""
        # Use return phase detections first, then secondary as fallback
        phase_detections = return_phase_detections or secondary_phase_detections
        
        if phase_detections:
            best_detection = max(phase_detections, key=lambda x: x.confidence)
            detected_return_phase = best_detection.number
            
            # Simple approach: Just store the return phase, don't calculate current phase
            # Current phase will be set when incoming_phase ROI detects a number
            
            # Set system state - keep current phase at 0 until incoming ROI detection
            # INITIALIZATION - bypassing protection for system startup 
            self.state.last_return_phase = detected_return_phase
            self.state.current_stage = 0  # Stay in stage 0 until proper initialization
            self.state.current_phase = 0  # Stay in phase 0 until incoming ROI detection
            
            # Apply stage rule using helper function
            self.state.last_return_stage = self._calculate_previous_stage(self.state.current_stage)
            # Don't set is_phase_initialized - wait for incoming ROI
            
            self.logger.info(f"Return phase {detected_return_phase} detected - waiting for incoming phase detection")
        else:
            self.logger.debug("No return phase detections for initialization")
    
    def _process_return_dish_detections(self, return_dish_detections: List[DishDetection]):
        """Process dishes returning from customers (break line camera)"""
        if not return_dish_detections:
            return
        
        current_time = datetime.now()
        
        # Update dish tracking and get dishes that crossed ROI boundaries
        crossed_dishes = self._update_dish_tracking(return_dish_detections)
        
        # Process dishes that crossed ROI boundaries for return tracking
        for dish_id in crossed_dishes:
            tracked_dish = self.tracked_dishes.get(dish_id)
            if tracked_dish and 'dish_detection' in tracked_dish.crossed_rois:
                dish_type = tracked_dish.dish_type
                
                # Count return dishes
                if dish_type in self.state.dishes_returning:
                    self.state.dishes_returning[dish_type] += 1
                    self.state.total_returned_dishes += 1
                    
                    self.logger.info(f"🔄 Return dish crossed ROI: {dish_type}, total returned: {self.state.total_returned_dishes}")
                
                # Remove from crossed_rois to avoid double counting
                tracked_dish.crossed_rois.discard('dish_detection')
        
        # Calculate new dishes served: Kitchen total - Returned total
        self._calculate_new_dishes_served()
