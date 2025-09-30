"""
Tracker Adapter for KichiKichi Conveyor Belt System
Provides a unified interface that can switch between ROI-based and CSV-based tracking
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os

from tracking.conveyor_tracker import ConveyorTracker
from tracking.csv_conveyor_tracker import CSVConveyorTracker
from dish_detection.dish_detector import DishDetection
from ocr_model.number_detector import NumberDetection


class ConveyorTrackerAdapter:
    """
    Unified adapter that can use either ROI-based or CSV-based tracking
    Provides backward compatibility with existing code
    """
    
    def __init__(self, csv_timeline_path: Optional[str] = None, 
                 use_csv_tracking: bool = False,
                 dishes_per_phase_estimate: float = 2.0):
        """
        Initialize the tracker adapter
        
        Args:
            csv_timeline_path: Path to CSV timeline file (required if use_csv_tracking=True)
            use_csv_tracking: Whether to use CSV-based tracking instead of ROI-based
            dishes_per_phase_estimate: Average dishes per phase for CSV tracking
        """
        self.logger = logging.getLogger(__name__)
        self.use_csv_tracking = use_csv_tracking
        self.csv_timeline_path = csv_timeline_path
        self.current_frame_index = 0
        
        # Initialize the appropriate tracker
        if use_csv_tracking:
            if not csv_timeline_path or not os.path.exists(csv_timeline_path):
                raise ValueError(f"CSV timeline file required for CSV tracking: {csv_timeline_path}")
            
            self.csv_tracker = CSVConveyorTracker(csv_timeline_path, dishes_per_phase_estimate)
            self.roi_tracker = None
            self.tracking_mode = "CSV"
            self.logger.info(f"✅ Initialized CSV-based tracking with {csv_timeline_path}")
            
        else:
            self.roi_tracker = ConveyorTracker()
            self.csv_tracker = None
            self.tracking_mode = "ROI"
            self.logger.info("✅ Initialized ROI-based tracking")
        
        # Expose common attributes for backward compatibility
        self._expose_common_attributes()
    
    def _expose_common_attributes(self) -> None:
        """Expose common attributes for backward compatibility"""
        if self.use_csv_tracking and self.csv_tracker:
            # ROI configurations from CSV tracker
            self.roi_dish_detection = getattr(self.csv_tracker, 'roi_dish_detection', (415, 193, 854, 466))
            self.roi_incoming_phase = getattr(self.csv_tracker, 'roi_incoming_phase', (1026, 174, 1252, 439))
            self.roi_return_phase = getattr(self.csv_tracker, 'roi_return_phase', (1068, 687, 1274, 890))
            self.roi_kitchen_counter = getattr(self.csv_tracker, 'roi_kitchen_counter', (678, 139, 1334, 411))
            self.state = self.csv_tracker.state
        elif self.roi_tracker:
            # ROI configurations from ROI tracker
            self.roi_dish_detection = getattr(self.roi_tracker, 'roi_dish_detection', (415, 193, 854, 466))
            self.roi_incoming_phase = getattr(self.roi_tracker, 'roi_incoming_phase', (1026, 174, 1252, 439))
            self.roi_return_phase = getattr(self.roi_tracker, 'roi_return_phase', (1068, 687, 1274, 890))
            self.roi_kitchen_counter = getattr(self.roi_tracker, 'roi_kitchen_counter', (678, 139, 1334, 411))
            self.state = self.roi_tracker.state
    
    def update_frame_index(self, frame_index: int) -> bool:
        """
        Update current frame index for CSV tracking
        
        Args:
            frame_index: Current video frame number
            
        Returns:
            True if tracking was updated, False otherwise
        """
        self.current_frame_index = frame_index
        
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.update_frame_position(frame_index)
        else:
            # For ROI tracking, frame index is not directly used
            return True
    
    def update_frame_position(self, frame_index: int) -> bool:
        """
        Update current frame position (same as update_frame_index for compatibility)
        
        Args:
            frame_index: Current video frame number
            
        Returns:
            True if tracking was updated, False otherwise
        """
        return self.update_frame_index(frame_index)
    
    def update_breakline_frame_index(self, frame_index: int) -> None:
        """
        Update the breakline camera frame index
        
        Args:
            frame_index: Current breakline camera frame number
        """
        if self.use_csv_tracking and self.csv_tracker:
            self.csv_tracker.update_breakline_frame_index(frame_index)
        # ROI tracker doesn't need breakline frame index tracking
    
    def update_phase_numbers(self, incoming_detections: List[NumberDetection], 
                           return_detections: List[NumberDetection]) -> None:
        """
        Update phase numbers - behavior depends on tracking mode
        
        Args:
            incoming_detections: Phase numbers detected in incoming ROI
            return_detections: Phase numbers detected in return ROI
        """
        if self.use_csv_tracking and self.csv_tracker:
            # CSV tracking doesn't use ROI detections, but we can log them
            if incoming_detections or return_detections:
                incoming_nums = [d.number for d in incoming_detections]
                return_nums = [d.number for d in return_detections]
                self.logger.debug(f"🔍 CSV Mode: Ignoring ROI detections - "
                                f"Incoming: {incoming_nums}, Return: {return_nums}")
        elif self.roi_tracker:
            # ROI tracking uses the detections
            self.roi_tracker.update_current_phase(incoming_detections)
            if return_detections:
                self.roi_tracker.update_from_return_detections([], return_detections, [])
    
    def process_kitchen_dishes(self, dish_detections: List[DishDetection]) -> None:
        """Process dish detections from kitchen camera"""
        if self.use_csv_tracking and self.csv_tracker:
            self.csv_tracker.process_dish_detections(dish_detections, "kitchen_counter")
        elif self.roi_tracker:
            # Use the actual method available in ConveyorTracker
            self.roi_tracker.update_kitchen_dish_count(dish_detections)
    
    def process_dish_detections(self, dish_detections: List[DishDetection], 
                              roi_name: str = "dish_detection") -> None:
        """Process general dish detections"""
        if self.use_csv_tracking and self.csv_tracker:
            self.csv_tracker.process_dish_detections(dish_detections, roi_name)
        elif self.roi_tracker:
            # For ROI tracker, route based on roi_name
            if roi_name == "kitchen_counter":
                self.roi_tracker.update_kitchen_dish_count(dish_detections)
            else:
                # Process as current stage dishes
                self.roi_tracker.update_current_stage_dishes(dish_detections)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current tracking status"""
        if self.use_csv_tracking and self.csv_tracker:
            status = self.csv_tracker.get_current_status()
            status['tracking_mode'] = 'CSV'
            status['csv_timeline_path'] = self.csv_timeline_path
        elif self.roi_tracker:
            # For ROI tracker, build compatible status using available methods
            current_state = self.roi_tracker.get_current_state()
            status = {
                'tracking_mode': 'ROI',
                'current_stage': current_state.current_stage,
                'current_phase': current_state.current_phase,
                'current_frame': self.current_frame_index,
                'total_dishes_processed': dict(current_state.dishes_to_customer),
                'current_stage_dishes': dict(current_state.current_stage_dishes),
                'kitchen_dishes_served': dict(current_state.kitchen_dishes_served),
                'phase_dish_tracking': dict(current_state.phase_dish_tracking)
            }
        else:
            status = {'error': 'No tracker initialized'}
        
        return status
    
    def get_phase_dish_count(self, phase: int, dish_type: Optional[str] = None) -> int:
        """Get dish count for a specific phase"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_phase_dish_count(phase, dish_type)
        elif self.roi_tracker:
            # For ROI tracker, use phase_dish_tracking
            current_state = self.roi_tracker.get_current_state()
            if phase not in current_state.phase_dish_tracking:
                return 0
            
            phase_data = current_state.phase_dish_tracking[phase]
            if dish_type is None:
                return sum(phase_data.values())
            else:
                return phase_data.get(dish_type, 0)
        return 0
    
    def get_all_phase_dish_counts(self) -> Dict[int, Dict[str, int]]:
        """Get dish counts for all phases"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_all_phase_dish_counts()
        elif self.roi_tracker:
            # For ROI tracker, use phase_dish_tracking
            current_state = self.roi_tracker.get_current_state()
            result = {}
            for phase, counts in current_state.phase_dish_tracking.items():
                result[phase] = counts.copy()
                result[phase]['total'] = sum(counts.values())
            return result
        return {}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_dashboard_data()
        elif self.roi_tracker:
            # For ROI tracker, build dashboard-compatible data
            current_state = self.roi_tracker.get_current_state()
            return {
                'current_stage': current_state.current_stage,
                'current_phase': current_state.current_phase,
                'dishes_to_customer': dict(current_state.dishes_to_customer),
                'dishes_returning': dict(current_state.dishes_returning),
                'kitchen_dishes_served': dict(current_state.kitchen_dishes_served),
                'current_stage_dishes': dict(current_state.current_stage_dishes),
                'phase_dish_tracking': dict(current_state.phase_dish_tracking),
                'tracking_mode': 'ROI'
            }
        return {}
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_summary_report()
        elif self.roi_tracker:
            # For ROI tracker, build basic report
            current_state = self.roi_tracker.get_current_state()
            total_dishes = sum(current_state.dishes_to_customer.values())
            phases_with_dishes = len([p for p, counts in current_state.phase_dish_tracking.items() 
                                    if sum(counts.values()) > 0])
            
            return {
                'tracking_mode': 'ROI',
                'current_status': self.get_current_status(),
                'statistics': {
                    'total_dishes_processed': total_dishes,
                    'phases_with_dishes': phases_with_dishes,
                    'current_stage': current_state.current_stage,
                    'current_phase': current_state.current_phase
                }
            }
        return {'error': 'No tracker initialized'}
    
    def apply_roi_to_detections(self, detections, roi_coordinates) -> List:
        """
        Filter detections to only those within ROI bounds
        
        Args:
            detections: List of detections to filter
            roi_coordinates: ROI coordinates (x1, y1, x2, y2)
            
        Returns:
            Filtered list of detections within ROI
        """
        if self.roi_tracker:
            # Use the original tracker's method
            return self.roi_tracker.apply_roi_to_detections(detections, roi_coordinates)
        else:
            # Simple ROI filtering for CSV tracker
            x1, y1, x2, y2 = roi_coordinates
            filtered_detections = []
            
            for detection in detections:
                # Check if detection bbox overlaps with ROI
                det_x1, det_y1, det_x2, det_y2 = detection.bbox
                det_center_x = (det_x1 + det_x2) / 2
                det_center_y = (det_y1 + det_y2) / 2
                
                if x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2:
                    filtered_detections.append(detection)
            
            return filtered_detections
    
    def get_current_state(self):
        """Get current tracker state - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_current_state()
        elif self.roi_tracker:
            return self.roi_tracker.get_current_state()
        else:
            raise RuntimeError("No tracker initialized")
    
    def get_dish_serving_summary(self) -> Dict[str, Any]:
        """Get dish serving summary - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            return self.csv_tracker.get_dish_serving_summary()
        elif self.roi_tracker:
            return self.roi_tracker.get_dish_serving_summary()
        else:
            return {}
    
    def get_total_dishes_on_belt(self) -> Dict[str, int]:
        """Get total dishes on belt by type - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            # For CSV tracker, get current stage dishes
            state = self.csv_tracker.get_current_state()
            return {
                'normal_dish': state.current_stage_dishes.get('normal_dish', 0),
                'red_dish': state.current_stage_dishes.get('red_dish', 0),
                'yellow_dish': state.current_stage_dishes.get('yellow_dish', 0)
            }
        elif self.roi_tracker:
            state = self.roi_tracker.get_current_state()
            return {
                'normal_dish': state.current_stage_dishes.get('normal_dish', 0),
                'red_dish': state.current_stage_dishes.get('red_dish', 0),
                'yellow_dish': state.current_stage_dishes.get('yellow_dish', 0)
            }
        else:
            return {'normal_dish': 0, 'red_dish': 0, 'yellow_dish': 0}
    
    def get_dishes_by_roi(self) -> Dict[str, Dict[str, int]]:
        """Get dishes by ROI region - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            # CSV mode doesn't have traditional ROI data, return empty dict
            return {}
        elif self.roi_tracker:
            # ROI tracker would have this method
            if hasattr(self.roi_tracker, 'get_dishes_by_roi'):
                return self.roi_tracker.get_dishes_by_roi()
            else:
                return {}
        else:
            return {}
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get phase summary - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            # For CSV tracker, build phase summary from available data
            state = self.csv_tracker.get_current_state()
            return {
                'current_phase': state.current_phase,
                'current_stage': state.current_stage,
                'total_phases': len(state.phase_dish_tracking) if hasattr(state, 'phase_dish_tracking') else 0,
                'is_initialized': state.is_phase_initialized
            }
        elif self.roi_tracker:
            if hasattr(self.roi_tracker, 'get_phase_summary'):
                return self.roi_tracker.get_phase_summary()
            else:
                state = self.roi_tracker.get_current_state()
                return {
                    'current_phase': state.current_phase,
                    'current_stage': state.current_stage,
                    'total_phases': len(state.phase_dish_tracking),
                    'is_initialized': True
                }
        else:
            return {'current_phase': 0, 'current_stage': 0, 'total_phases': 0, 'is_initialized': False}
    
    def is_break_line_active(self) -> bool:
        """Check if break line is active - delegates to the appropriate tracker"""
        if self.use_csv_tracking and self.csv_tracker:
            # For CSV mode, break line is "active" if we have return detections recently
            if hasattr(self.csv_tracker, 'is_break_line_active'):
                return self.csv_tracker.is_break_line_active()
            else:
                # Default implementation - could be enhanced
                return False
        elif self.roi_tracker:
            if hasattr(self.roi_tracker, 'is_break_line_active'):
                return self.roi_tracker.is_break_line_active()
            else:
                return False
        else:
            return False

    def log_current_state(self) -> None:
        """Log current tracking state for debugging"""
        status = self.get_current_status()
        self.logger.info(f"🔍 {self.tracking_mode} Tracking Status:")
        self.logger.info(f"  Frame: {status.get('current_frame', 'N/A')}")
        self.logger.info(f"  Stage: {status.get('current_stage', 'N/A')}")
        self.logger.info(f"  Phase: {status.get('current_phase', 'N/A')}")
        
        if 'phase_dish_counts' in status:
            active_phases = [p for p, data in status['phase_dish_counts'].items() 
                           if data.get('total', 0) > 0]
            self.logger.info(f"  Active Phases with Dishes: {active_phases}")


def create_tracker_adapter(csv_timeline_path: Optional[str] = None,
                         use_csv_tracking: bool = False,
                         dishes_per_phase_estimate: float = 2.0) -> ConveyorTrackerAdapter:
    """
    Factory function to create a tracker adapter
    
    Args:
        csv_timeline_path: Path to CSV timeline file
        use_csv_tracking: Whether to use CSV-based tracking
        dishes_per_phase_estimate: Average dishes per phase estimate
        
    Returns:
        Configured ConveyorTrackerAdapter instance
    """
    return ConveyorTrackerAdapter(csv_timeline_path, use_csv_tracking, dishes_per_phase_estimate)