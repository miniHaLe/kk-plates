"""Tests for metrics aggregation."""

import pytest
import time
from unittest.mock import patch
from kkplates.metrics.aggregator import MetricsAggregator, MetricSnapshot


class TestMetricsAggregator:
    """Test suite for metrics aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        """Create metrics aggregator with 60s window."""
        return MetricsAggregator(window_seconds=60)
    
    def test_init(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.window_seconds == 60
        assert aggregator.total_in == 0
        assert aggregator.total_out == 0
        assert aggregator.current_on_belt == 0
        assert all(count == 0 for count in aggregator.color_counts.values())
    
    def test_add_in_event(self, aggregator):
        """Test adding IN events."""
        aggregator.add_event("in", "red", timestamp=1.0)
        aggregator.add_event("in", "yellow", timestamp=2.0)
        aggregator.add_event("in", "normal", timestamp=3.0)
        
        assert aggregator.total_in == 3
        assert aggregator.current_on_belt == 3
        assert aggregator.color_counts["red"] == 1
        assert aggregator.color_counts["yellow"] == 1
        assert aggregator.color_counts["normal"] == 1
    
    def test_add_out_event(self, aggregator):
        """Test adding OUT events."""
        # First add some IN events
        aggregator.add_event("in", "red", timestamp=1.0)
        aggregator.add_event("in", "yellow", timestamp=2.0)
        
        # Then OUT events
        aggregator.add_event("out", "red", timestamp=3.0)
        
        assert aggregator.total_in == 2
        assert aggregator.total_out == 1
        assert aggregator.current_on_belt == 1
    
    def test_current_on_belt_never_negative(self, aggregator):
        """Test that current_on_belt never goes negative."""
        # Add more OUT than IN
        aggregator.add_event("out", "red", timestamp=1.0)
        aggregator.add_event("out", "yellow", timestamp=2.0)
        
        assert aggregator.total_out == 2
        assert aggregator.current_on_belt == 0  # Not negative
    
    @patch('time.time')
    def test_get_snapshot_basic(self, mock_time, aggregator):
        """Test basic snapshot generation."""
        mock_time.return_value = 100.0
        
        # Add events
        aggregator.add_event("in", "red", timestamp=95.0)
        aggregator.add_event("in", "yellow", timestamp=96.0)
        aggregator.add_event("in", "red", timestamp=97.0)
        
        snapshot = aggregator.get_snapshot(force=True)
        
        assert snapshot is not None
        assert snapshot.timestamp == 100.0
        assert snapshot.total_in == 3
        assert snapshot.plates_per_minute == 3.0  # 3 plates in 60s = 3/min
    
    @patch('time.time')
    def test_sliding_window_calculation(self, mock_time, aggregator):
        """Test sliding window metrics calculation."""
        # Current time is 200
        mock_time.return_value = 200.0
        
        # Add events: some inside window (140-200), some outside
        aggregator.add_event("in", "red", timestamp=100.0)    # Outside window
        aggregator.add_event("in", "red", timestamp=150.0)    # Inside window
        aggregator.add_event("in", "yellow", timestamp=180.0) # Inside window
        aggregator.add_event("in", "normal", timestamp=190.0) # Inside window
        
        snapshot = aggregator.get_snapshot(force=True)
        
        # Only 3 events in window (150, 180, 190)
        assert snapshot.plates_per_minute == 3.0
        assert snapshot.color_ratios["red"] == pytest.approx(1/3)
        assert snapshot.color_ratios["yellow"] == pytest.approx(1/3)
        assert snapshot.color_ratios["normal"] == pytest.approx(1/3)
    
    @patch('time.time')
    def test_color_frequencies(self, mock_time, aggregator):
        """Test per-color frequency calculation."""
        mock_time.return_value = 120.0
        
        # Add events in last 60 seconds
        for i in range(6):
            aggregator.add_event("in", "red", timestamp=65.0 + i * 5)
        for i in range(3):
            aggregator.add_event("in", "yellow", timestamp=70.0 + i * 10)
        
        snapshot = aggregator.get_snapshot(force=True)
        
        assert snapshot.color_frequencies["red"] == 6.0  # 6 per minute
        assert snapshot.color_frequencies["yellow"] == 3.0  # 3 per minute
        assert snapshot.color_frequencies["normal"] == 0.0
    
    @patch('time.time')
    def test_snapshot_rate_limiting(self, mock_time, aggregator):
        """Test snapshot rate limiting."""
        mock_time.return_value = 100.0
        
        # First snapshot should work
        snapshot1 = aggregator.get_snapshot()
        assert snapshot1 is not None
        
        # Immediate second call should be rate limited
        snapshot2 = aggregator.get_snapshot()
        assert snapshot2 is None
        
        # But force should bypass rate limit
        snapshot3 = aggregator.get_snapshot(force=True)
        assert snapshot3 is not None
    
    def test_update_from_crossing_stats(self, aggregator):
        """Test updating from external crossing detector stats."""
        stats = {
            "total_in": 10,
            "total_out": 7,
            "current_on_belt": 3,
            "color_counts": {"red": 5, "yellow": 3, "normal": 2}
        }
        
        aggregator.update_from_crossing_stats(stats)
        
        assert aggregator.total_in == 10
        assert aggregator.total_out == 7
        assert aggregator.current_on_belt == 3
        assert aggregator.color_counts == stats["color_counts"]
    
    @patch('time.time')
    def test_old_events_cleanup(self, mock_time, aggregator):
        """Test that old events are cleaned up."""
        # Add old events
        for i in range(100):
            aggregator.add_event("in", "red", timestamp=i)
        
        # Move time forward
        mock_time.return_value = 200.0
        
        # Get snapshot (should trigger cleanup)
        snapshot = aggregator.get_snapshot(force=True)
        
        # Check that old events were cleaned
        assert len(aggregator.in_events) < 100
        # All remaining events should be within window + buffer
        for event in aggregator.in_events:
            assert event.timestamp >= 200.0 - 60 - 10  # window + 10s buffer
    
    def test_health_status(self, aggregator):
        """Test health status reporting."""
        # Add recent event
        aggregator.add_event("in", "red", timestamp=time.time() - 10)
        
        health = aggregator.get_health_status()
        
        assert health["healthy"] is True
        assert health["last_event_seconds_ago"] < 20
        assert health["current_on_belt"] == 1
        assert health["total_processed"] == 1
    
    def test_health_status_unhealthy(self, aggregator):
        """Test unhealthy status when no recent events."""
        # Add old event
        aggregator.add_event("in", "red", timestamp=time.time() - 400)
        
        health = aggregator.get_health_status()
        
        assert health["healthy"] is False
        assert health["last_event_seconds_ago"] > 300
    
    def test_reset(self, aggregator):
        """Test metrics reset."""
        # Add some data
        aggregator.add_event("in", "red", timestamp=1.0)
        aggregator.add_event("out", "red", timestamp=2.0)
        
        # Reset
        aggregator.reset()
        
        assert aggregator.total_in == 0
        assert aggregator.total_out == 0
        assert aggregator.current_on_belt == 0
        assert len(aggregator.in_events) == 0
        assert len(aggregator.out_events) == 0
        assert all(count == 0 for count in aggregator.color_counts.values())
    
    def test_snapshot_serialization(self, aggregator):
        """Test that snapshot can be serialized to dict."""
        aggregator.add_event("in", "red", timestamp=1.0)
        
        snapshot = aggregator.get_snapshot(force=True)
        snapshot_dict = snapshot.to_dict()
        
        # Check required fields
        assert "timestamp" in snapshot_dict
        assert "total_in" in snapshot_dict
        assert "total_out" in snapshot_dict
        assert "current_on_belt" in snapshot_dict
        assert "color_counts" in snapshot_dict
        assert "plates_per_minute" in snapshot_dict
        assert "color_frequencies" in snapshot_dict
        assert "color_ratios" in snapshot_dict
        
        # Check types
        assert isinstance(snapshot_dict["plates_per_minute"], float)
        assert isinstance(snapshot_dict["color_frequencies"], dict)
        assert isinstance(snapshot_dict["color_ratios"], dict)
    
    @patch('time.time')
    def test_zero_window_events(self, mock_time, aggregator):
        """Test snapshot when no events in window."""
        mock_time.return_value = 200.0
        
        # Add only old events
        aggregator.add_event("in", "red", timestamp=50.0)
        
        snapshot = aggregator.get_snapshot(force=True)
        
        assert snapshot.plates_per_minute == 0.0
        assert all(freq == 0.0 for freq in snapshot.color_frequencies.values())
        assert all(ratio == 0.0 for ratio in snapshot.color_ratios.values())