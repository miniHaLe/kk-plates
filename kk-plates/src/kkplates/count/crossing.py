"""ROI crossing detection logic with debouncing."""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import cv2
import structlog

logger = structlog.get_logger()


@dataclass
class CrossingEvent:
    """Represents a crossing event."""
    track_id: int
    direction: str  # "in" or "out"
    color: str
    timestamp: float
    position: Tuple[float, float]


class CrossingDetector:
    """Detects when tracked objects cross ROI polygons."""
    
    def __init__(self, 
                 in_roi: List[List[int]], 
                 out_roi: List[List[int]],
                 min_travel_distance: float = 20.0,
                 debounce_frames: int = 10):
        """
        Initialize crossing detector.
        
        Args:
            in_roi: Polygon points for incoming lane
            out_roi: Polygon points for outgoing lane
            min_travel_distance: Minimum distance to travel before considering crossing
            debounce_frames: Number of frames to wait before allowing re-crossing
        """
        self.in_roi = np.array(in_roi, dtype=np.int32)
        self.out_roi = np.array(out_roi, dtype=np.int32)
        self.min_travel_distance = min_travel_distance
        self.debounce_frames = debounce_frames
        
        # Track states
        self.track_states: Dict[int, str] = {}  # "outside", "in_zone", "out_zone"
        self.track_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.track_debounce: Dict[int, int] = defaultdict(int)
        self.crossed_tracks: Set[int] = set()
        
        # Counters
        self.total_in = 0
        self.total_out = 0
        self.current_on_belt = 0
        self.color_counts = {"red": 0, "yellow": 0, "normal": 0}
        
    def update(self, tracks: List, colors: Dict[int, str], timestamp: float) -> List[CrossingEvent]:
        """
        Update with current tracks and detect crossings.
        
        Args:
            tracks: List of Track objects with track_id and center properties
            colors: Dict mapping track_id to color
            timestamp: Current timestamp
            
        Returns:
            List of crossing events
        """
        events = []
        current_track_ids = set()
        
        for track in tracks:
            track_id = track.track_id
            center = track.center
            color = colors.get(track_id, "normal")
            current_track_ids.add(track_id)
            
            # Update position history
            self.track_positions[track_id].append(center)
            if len(self.track_positions[track_id]) > 30:
                self.track_positions[track_id].pop(0)
            
            # Update debounce counter
            if self.track_debounce[track_id] > 0:
                self.track_debounce[track_id] -= 1
            
            # Check if we have enough history
            if len(self.track_positions[track_id]) < 3:
                continue
            
            # Get previous state
            prev_state = self.track_states.get(track_id, "outside")
            
            # Check current position
            in_in_zone = self._point_in_polygon(center, self.in_roi)
            in_out_zone = self._point_in_polygon(center, self.out_roi)
            
            # Determine current state
            if in_in_zone:
                current_state = "in_zone"
            elif in_out_zone:
                current_state = "out_zone"
            else:
                current_state = "outside"
            
            # Check for state transitions (crossings)
            if prev_state != current_state and self.track_debounce[track_id] == 0:
                # Check if traveled minimum distance
                if self._has_traveled_min_distance(track_id):
                    # Detect crossing direction
                    if prev_state == "outside" and current_state == "in_zone":
                        # Entering from kitchen to belt
                        event = self._handle_in_crossing(track_id, color, center, timestamp)
                        if event:
                            events.append(event)
                    
                    elif prev_state == "in_zone" and current_state == "outside":
                        # Completed IN crossing
                        pass
                    
                    elif prev_state == "outside" and current_state == "out_zone":
                        # Entering OUT zone from belt
                        event = self._handle_out_crossing(track_id, color, center, timestamp)
                        if event:
                            events.append(event)
                    
                    elif prev_state == "out_zone" and current_state == "outside":
                        # Completed OUT crossing
                        pass
            
            # Update state
            self.track_states[track_id] = current_state
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_track_ids)
        
        return events
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """Check if point is inside polygon."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def _has_traveled_min_distance(self, track_id: int) -> bool:
        """Check if track has traveled minimum distance."""
        positions = self.track_positions[track_id]
        if len(positions) < 2:
            return False
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return total_distance >= self.min_travel_distance
    
    def _get_crossing_direction(self, track_id: int, roi: np.ndarray) -> Optional[str]:
        """Determine crossing direction based on trajectory."""
        positions = self.track_positions[track_id]
        if len(positions) < 5:
            return None
        
        # Get positions before and after crossing
        before_positions = []
        after_positions = []
        
        for i, pos in enumerate(positions):
            if self._point_in_polygon(pos, roi):
                before_positions = positions[:i]
                after_positions = positions[i+1:]
                break
        
        if len(before_positions) < 2 or len(after_positions) < 2:
            return None
        
        # Calculate average positions
        before_avg = np.mean(before_positions[-3:], axis=0)
        after_avg = np.mean(after_positions[:3], axis=0)
        
        # Determine direction based on movement
        # Assuming camera is top-down and belt moves horizontally
        if roi is self.in_roi:
            # For IN zone: moving from left (kitchen) to right (belt) is IN
            return "in" if after_avg[0] > before_avg[0] else None
        else:
            # For OUT zone: moving from right (belt) to left (kitchen) is OUT
            return "out" if after_avg[0] < before_avg[0] else None
    
    def _handle_in_crossing(self, track_id: int, color: str, position: Tuple[float, float], 
                          timestamp: float) -> Optional[CrossingEvent]:
        """Handle IN crossing event."""
        # Check if already crossed
        if track_id in self.crossed_tracks:
            return None
        
        # Verify crossing direction
        direction = self._get_crossing_direction(track_id, self.in_roi)
        if direction != "in":
            return None
        
        # Update counters
        self.total_in += 1
        self.current_on_belt += 1
        self.color_counts[color] = self.color_counts.get(color, 0) + 1
        
        # Mark as crossed and set debounce
        self.crossed_tracks.add(track_id)
        self.track_debounce[track_id] = self.debounce_frames
        
        event = CrossingEvent(
            track_id=track_id,
            direction="in",
            color=color,
            timestamp=timestamp,
            position=position
        )
        
        logger.info("IN crossing detected", 
                   track_id=track_id, 
                   color=color,
                   total_in=self.total_in,
                   on_belt=self.current_on_belt)
        
        return event
    
    def _handle_out_crossing(self, track_id: int, color: str, position: Tuple[float, float],
                           timestamp: float) -> Optional[CrossingEvent]:
        """Handle OUT crossing event."""
        # Verify crossing direction
        direction = self._get_crossing_direction(track_id, self.out_roi)
        if direction != "out":
            return None
        
        # Update counters
        self.total_out += 1
        self.current_on_belt = max(0, self.current_on_belt - 1)  # Never negative
        
        # Set debounce
        self.track_debounce[track_id] = self.debounce_frames
        
        event = CrossingEvent(
            track_id=track_id,
            direction="out",
            color=color,
            timestamp=timestamp,
            position=position
        )
        
        logger.info("OUT crossing detected",
                   track_id=track_id,
                   color=color,
                   total_out=self.total_out,
                   on_belt=self.current_on_belt)
        
        return event
    
    def _cleanup_old_tracks(self, current_track_ids: Set[int]) -> None:
        """Remove data for tracks that no longer exist."""
        all_track_ids = set(self.track_states.keys())
        old_track_ids = all_track_ids - current_track_ids
        
        for track_id in old_track_ids:
            self.track_states.pop(track_id, None)
            self.track_positions.pop(track_id, None)
            self.track_debounce.pop(track_id, None)
            self.crossed_tracks.discard(track_id)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "current_on_belt": self.current_on_belt,
            "color_counts": self.color_counts.copy()
        }
    
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROIs and stats on frame for debugging."""
        display = frame.copy()
        
        # Draw ROIs
        cv2.polylines(display, [self.in_roi], True, (0, 255, 0), 2)
        cv2.polylines(display, [self.out_roi], True, (0, 0, 255), 2)
        
        # Draw labels
        in_center = np.mean(self.in_roi, axis=0).astype(int)
        out_center = np.mean(self.out_roi, axis=0).astype(int)
        
        cv2.putText(display, "IN", tuple(in_center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "OUT", tuple(out_center),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw stats
        stats_text = [
            f"In: {self.total_in}",
            f"Out: {self.total_out}",
            f"On Belt: {self.current_on_belt}",
            f"R:{self.color_counts['red']} Y:{self.color_counts['yellow']} N:{self.color_counts['normal']}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return display