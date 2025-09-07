"""Metrics aggregation with sliding window calculations."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import structlog

logger = structlog.get_logger()


@dataclass
class MetricSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: float
    total_in: int
    total_out: int
    current_on_belt: int
    color_counts: Dict[str, int]
    
    # Sliding window metrics
    window_seconds: int
    plates_per_minute: float
    color_frequencies: Dict[str, float]  # plates/min per color
    color_ratios: Dict[str, float]  # percentage per color
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "current_on_belt": self.current_on_belt,
            "color_counts": self.color_counts,
            "window_seconds": self.window_seconds,
            "plates_per_minute": round(self.plates_per_minute, 2),
            "color_frequencies": {k: round(v, 2) for k, v in self.color_frequencies.items()},
            "color_ratios": {k: round(v, 3) for k, v in self.color_ratios.items()}
        }


@dataclass
class TimestampedEvent:
    """Event with timestamp for sliding window."""
    timestamp: float
    event_type: str  # "in" or "out"
    color: str


class MetricsAggregator:
    """Aggregates plate counting metrics with sliding window calculations."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        
        # Event queues for sliding window
        self.in_events: deque = deque()
        self.out_events: deque = deque()
        
        # Current totals
        self.total_in = 0
        self.total_out = 0
        self.current_on_belt = 0
        self.color_counts = {"red": 0, "yellow": 0, "normal": 0}
        
        # Last snapshot
        self.last_snapshot: Optional[MetricSnapshot] = None
        self.last_snapshot_time = 0
        
    def add_event(self, event_type: str, color: str, timestamp: Optional[float] = None) -> None:
        """
        Add a crossing event.
        
        Args:
            event_type: "in" or "out"
            color: Plate color
            timestamp: Event timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        event = TimestampedEvent(timestamp, event_type, color)
        
        if event_type == "in":
            self.in_events.append(event)
            self.total_in += 1
            self.current_on_belt += 1
            self.color_counts[color] = self.color_counts.get(color, 0) + 1
            
        elif event_type == "out":
            self.out_events.append(event)
            self.total_out += 1
            self.current_on_belt = max(0, self.current_on_belt - 1)
        
        # Clean old events
        self._clean_old_events(timestamp)
    
    def update_from_crossing_stats(self, stats: Dict) -> None:
        """Update from crossing detector stats."""
        self.total_in = stats["total_in"]
        self.total_out = stats["total_out"]
        self.current_on_belt = stats["current_on_belt"]
        self.color_counts = stats["color_counts"]
    
    def get_snapshot(self, force: bool = False) -> Optional[MetricSnapshot]:
        """
        Get current metrics snapshot.
        
        Args:
            force: Force snapshot even if recently generated
            
        Returns:
            MetricSnapshot or None if too soon since last snapshot
        """
        current_time = time.time()
        
        # Rate limit snapshots (unless forced)
        if not force and current_time - self.last_snapshot_time < 1.0:
            return None
        
        # Clean old events
        self._clean_old_events(current_time)
        
        # Calculate window metrics
        window_start = current_time - self.window_seconds
        
        # Count events in window
        in_window_events = [e for e in self.in_events if e.timestamp >= window_start]
        
        # Calculate plates per minute
        window_duration_minutes = self.window_seconds / 60.0
        plates_per_minute = len(in_window_events) / window_duration_minutes if window_duration_minutes > 0 else 0
        
        # Calculate per-color frequencies and ratios
        color_counts_window = {"red": 0, "yellow": 0, "normal": 0}
        for event in in_window_events:
            color_counts_window[event.color] += 1
        
        total_window = sum(color_counts_window.values())
        
        color_frequencies = {}
        color_ratios = {}
        
        for color in ["red", "yellow", "normal"]:
            count = color_counts_window[color]
            color_frequencies[color] = count / window_duration_minutes if window_duration_minutes > 0 else 0
            color_ratios[color] = count / total_window if total_window > 0 else 0
        
        snapshot = MetricSnapshot(
            timestamp=current_time,
            total_in=self.total_in,
            total_out=self.total_out,
            current_on_belt=self.current_on_belt,
            color_counts=self.color_counts.copy(),
            window_seconds=self.window_seconds,
            plates_per_minute=plates_per_minute,
            color_frequencies=color_frequencies,
            color_ratios=color_ratios
        )
        
        self.last_snapshot = snapshot
        self.last_snapshot_time = current_time
        
        return snapshot
    
    def _clean_old_events(self, current_time: float) -> None:
        """Remove events older than window."""
        window_start = current_time - self.window_seconds - 10  # Keep 10s buffer
        
        # Clean in events
        while self.in_events and self.in_events[0].timestamp < window_start:
            self.in_events.popleft()
        
        # Clean out events  
        while self.out_events and self.out_events[0].timestamp < window_start:
            self.out_events.popleft()
    
    def get_health_status(self) -> Dict:
        """Get health check status."""
        current_time = time.time()
        
        # Check if we have recent events
        last_in_time = self.in_events[-1].timestamp if self.in_events else 0
        last_out_time = self.out_events[-1].timestamp if self.out_events else 0
        last_event_time = max(last_in_time, last_out_time)
        
        time_since_last_event = current_time - last_event_time if last_event_time > 0 else float('inf')
        
        return {
            "healthy": time_since_last_event < 300,  # No events for 5 min = unhealthy
            "last_event_seconds_ago": round(time_since_last_event, 1),
            "current_on_belt": self.current_on_belt,
            "total_processed": self.total_in
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.in_events.clear()
        self.out_events.clear()
        self.total_in = 0
        self.total_out = 0
        self.current_on_belt = 0
        self.color_counts = {"red": 0, "yellow": 0, "normal": 0}
        self.last_snapshot = None
        self.last_snapshot_time = 0
        logger.info("Metrics reset")