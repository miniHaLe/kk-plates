"""Tests for ROI crossing detection logic."""

import pytest
import numpy as np
from unittest.mock import Mock
from kkplates.count.crossing import CrossingDetector, CrossingEvent


class MockTrack:
    """Mock track object for testing."""
    def __init__(self, track_id: int, positions: list):
        self.track_id = track_id
        self.positions = positions
        self.position_idx = 0
        
    @property
    def center(self):
        if self.position_idx < len(self.positions):
            return self.positions[self.position_idx]
        return self.positions[-1]
    
    def advance(self):
        """Move to next position."""
        if self.position_idx < len(self.positions) - 1:
            self.position_idx += 1


class TestCrossingDetector:
    """Test suite for crossing detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a crossing detector with test ROIs."""
        in_roi = [[100, 100], [200, 100], [200, 150], [100, 150]]
        out_roi = [[100, 200], [200, 200], [200, 250], [100, 250]]
        return CrossingDetector(in_roi, out_roi, min_travel_distance=10.0, debounce_frames=5)
    
    def test_init(self, detector):
        """Test detector initialization."""
        assert detector.total_in == 0
        assert detector.total_out == 0
        assert detector.current_on_belt == 0
        assert len(detector.color_counts) == 3
    
    def test_single_in_crossing(self, detector):
        """Test a single plate crossing IN."""
        # Create track moving from left to right through IN zone
        positions = [
            (50, 125),   # Outside
            (80, 125),   # Outside
            (110, 125),  # Inside IN zone
            (150, 125),  # Inside IN zone
            (190, 125),  # Inside IN zone
            (220, 125),  # Outside (right)
        ]
        
        track = MockTrack(track_id=1, positions=positions)
        colors = {1: "red"}
        events_total = []
        
        # Simulate track movement
        for i in range(len(positions)):
            track.advance()
            events = detector.update([track], colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have exactly one IN event
        assert len(events_total) == 1
        assert events_total[0].direction == "in"
        assert events_total[0].color == "red"
        assert detector.total_in == 1
        assert detector.total_out == 0
        assert detector.current_on_belt == 1
        assert detector.color_counts["red"] == 1
    
    def test_single_out_crossing(self, detector):
        """Test a single plate crossing OUT."""
        # First add a plate to the belt
        detector.total_in = 1
        detector.current_on_belt = 1
        
        # Create track moving from right to left through OUT zone
        positions = [
            (250, 225),  # Outside (right)
            (220, 225),  # Outside
            (190, 225),  # Inside OUT zone
            (150, 225),  # Inside OUT zone
            (110, 225),  # Inside OUT zone
            (80, 225),   # Outside (left)
            (50, 225),   # Outside (left)
        ]
        
        track = MockTrack(track_id=2, positions=positions)
        colors = {2: "yellow"}
        events_total = []
        
        # Simulate track movement
        for i in range(len(positions)):
            track.advance()
            events = detector.update([track], colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have exactly one OUT event
        assert len(events_total) == 1
        assert events_total[0].direction == "out"
        assert events_total[0].color == "yellow"
        assert detector.total_out == 1
        assert detector.current_on_belt == 0  # Was 1, now 0
    
    def test_no_crossing_short_travel(self, detector):
        """Test that short movements don't trigger crossings."""
        # Create track with very small movement inside IN zone
        positions = [
            (140, 125),  # Inside IN zone
            (142, 125),  # Still inside, moved only 2 pixels
            (144, 125),  # Still inside, moved only 4 pixels total
        ]
        
        track = MockTrack(track_id=3, positions=positions)
        colors = {3: "normal"}
        events_total = []
        
        for i in range(len(positions)):
            track.advance()
            events = detector.update([track], colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have no events (didn't travel minimum distance)
        assert len(events_total) == 0
        assert detector.total_in == 0
    
    def test_debounce_prevents_double_counting(self, detector):
        """Test that debounce prevents double counting."""
        # Create track that enters and quickly re-enters
        positions = [
            (50, 125),   # Outside
            (150, 125),  # Inside IN zone (big jump to ensure min distance)
            (250, 125),  # Outside
            (150, 125),  # Back inside IN zone
            (250, 125),  # Outside again
        ]
        
        track = MockTrack(track_id=4, positions=positions)
        colors = {4: "red"}
        events_total = []
        
        for i in range(len(positions)):
            track.advance()
            events = detector.update([track], colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have only one event due to debounce
        assert len(events_total) == 1
        assert detector.total_in == 1
    
    def test_multiple_simultaneous_tracks(self, detector):
        """Test multiple tracks crossing simultaneously."""
        # Create three tracks
        track1 = MockTrack(track_id=10, positions=[
            (50, 125), (150, 125), (250, 125)  # IN crossing
        ])
        track2 = MockTrack(track_id=11, positions=[
            (250, 225), (150, 225), (50, 225)  # OUT crossing
        ])
        track3 = MockTrack(track_id=12, positions=[
            (300, 300), (310, 310), (320, 320)  # No crossing
        ])
        
        colors = {10: "red", 11: "yellow", 12: "normal"}
        
        # Set up initial state for OUT crossing
        detector.total_in = 1
        detector.current_on_belt = 1
        
        events_total = []
        tracks = [track1, track2, track3]
        
        # Simulate movement
        for i in range(3):
            for track in tracks:
                track.advance()
            events = detector.update(tracks, colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have one IN and one OUT event
        in_events = [e for e in events_total if e.direction == "in"]
        out_events = [e for e in events_total if e.direction == "out"]
        
        assert len(in_events) == 1
        assert len(out_events) == 1
        assert detector.total_in == 2  # Was 1, now 2
        assert detector.total_out == 1
        assert detector.current_on_belt == 1  # 2 - 1 = 1
    
    def test_current_on_belt_never_negative(self, detector):
        """Test that current_on_belt never goes negative."""
        # Try to trigger more OUT than IN events
        positions = [
            (250, 225), (150, 225), (50, 225)  # OUT crossing
        ]
        
        track = MockTrack(track_id=20, positions=positions)
        colors = {20: "red"}
        
        # Process OUT crossing when belt is already empty
        assert detector.current_on_belt == 0
        
        for i in range(len(positions)):
            track.advance()
            detector.update([track], colors, timestamp=i * 0.1)
        
        # Should still be 0, not negative
        assert detector.current_on_belt == 0
        assert detector.total_out == 1
    
    def test_color_counting(self, detector):
        """Test color counting accuracy."""
        # Create tracks for each color
        red_track = MockTrack(track_id=30, positions=[
            (50, 125), (150, 125), (250, 125)
        ])
        yellow_track = MockTrack(track_id=31, positions=[
            (50, 125), (150, 125), (250, 125)
        ])
        normal_track = MockTrack(track_id=32, positions=[
            (50, 125), (150, 125), (250, 125)
        ])
        
        # Process each color separately to avoid position conflicts
        for track, color in [(red_track, "red"), (yellow_track, "yellow"), (normal_track, "normal")]:
            colors = {track.track_id: color}
            for i in range(3):
                track.advance()
                detector.update([track], colors, timestamp=i * 0.1 + track.track_id)
        
        # Check color counts
        assert detector.color_counts["red"] == 1
        assert detector.color_counts["yellow"] == 1
        assert detector.color_counts["normal"] == 1
        assert detector.total_in == 3
    
    def test_get_stats(self, detector):
        """Test statistics retrieval."""
        # Add some crossings
        track = MockTrack(track_id=40, positions=[
            (50, 125), (150, 125), (250, 125)
        ])
        colors = {40: "red"}
        
        for i in range(3):
            track.advance()
            detector.update([track], colors, timestamp=i * 0.1)
        
        stats = detector.get_stats()
        
        assert stats["total_in"] == 1
        assert stats["total_out"] == 0
        assert stats["current_on_belt"] == 1
        assert stats["color_counts"]["red"] == 1
    
    def test_wrong_direction_ignored(self, detector):
        """Test that crossings in wrong direction are ignored."""
        # Track moving right to left through IN zone (wrong direction)
        positions = [
            (250, 125),  # Outside (right)
            (190, 125),  # Inside IN zone
            (150, 125),  # Inside IN zone
            (110, 125),  # Inside IN zone
            (50, 125),   # Outside (left)
        ]
        
        track = MockTrack(track_id=50, positions=positions)
        colors = {50: "red"}
        events_total = []
        
        for i in range(len(positions)):
            track.advance()
            events = detector.update([track], colors, timestamp=i * 0.1)
            events_total.extend(events)
        
        # Should have no events (wrong direction)
        assert len(events_total) == 0
        assert detector.total_in == 0