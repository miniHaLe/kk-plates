"""
CSV Timeline Parser for KichiKichi Conveyor Belt System
Replaces ROI-based phase and stage detection with pre-recorded timeline data
"""

import csv
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path


@dataclass
class TimelineEntry:
    """Single timeline entry from CSV"""
    frame_index: int
    current_stage: int
    current_phase: int
    previous_stage: int
    previous_phase: int


@dataclass
class PhaseStatistics:
    """Statistics for a specific phase"""
    phase_number: int
    total_frames: int
    dish_count: int = 0
    start_frame: int = 0
    end_frame: int = 0
    stage_occurrences: List[int] = field(default_factory=list)


class CSVTimelineParser:
    """
    Parses video timeline CSV data to provide stage/phase information
    Replaces ROI-based detection with frame-indexed lookup
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize CSV timeline parser
        
        Args:
            csv_file_path: Path to the video timeline CSV file
        """
        self.logger = logging.getLogger(__name__)
        self.csv_file_path = Path(csv_file_path)
        self.timeline_data: List[TimelineEntry] = []
        self.phase_statistics: Dict[int, PhaseStatistics] = {}
        self.current_frame_index = 0
        
        # Performance optimization: pre-computed lookup tables
        self.frame_lookup: Dict[int, TimelineEntry] = {}
        self.phase_frame_ranges: Dict[int, List[Tuple[int, int]]] = {}  # phase -> [(start, end), ...]
        
        # Load and process the CSV data
        self._load_timeline_data()
        self._compute_phase_statistics()
        self._build_lookup_tables()
    
    def _load_timeline_data(self) -> None:
        """Load timeline data from CSV file"""
        try:
            if not self.csv_file_path.exists():
                raise FileNotFoundError(f"Timeline CSV file not found: {self.csv_file_path}")
            
            # Read CSV with built-in csv module
            self.timeline_data = []
            with open(self.csv_file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Validate required columns
                required_columns = ['Frame Index', 'Current Stage', 'Current Phase', 'Previous Stage', 'Previous Phase']
                fieldnames = reader.fieldnames or []
                if not all(col in fieldnames for col in required_columns):
                    missing = [col for col in required_columns if col not in fieldnames]
                    raise ValueError(f"Missing required columns in CSV: {missing}")
                
                # Convert rows to timeline entries
                for row in reader:
                    entry = TimelineEntry(
                        frame_index=int(row['Frame Index']),
                        current_stage=int(row['Current Stage']),
                        current_phase=int(row['Current Phase']),
                        previous_stage=int(row['Previous Stage']),
                        previous_phase=int(row['Previous Phase'])
                    )
                    self.timeline_data.append(entry)
            
            self.logger.info(f"✅ Loaded {len(self.timeline_data)} timeline entries from {self.csv_file_path}")
            
            # Log data range information
            if self.timeline_data:
                first_frame = self.timeline_data[0].frame_index
                last_frame = self.timeline_data[-1].frame_index
                stages = set(entry.current_stage for entry in self.timeline_data)
                phases = set(entry.current_phase for entry in self.timeline_data)
                
                self.logger.info(f"📊 Timeline range: Frame {first_frame} to {last_frame}")
                self.logger.info(f"📈 Stages found: {sorted(stages)}")
                self.logger.info(f"🔄 Phases found: {sorted(phases)}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load timeline data: {e}")
            raise
    
    def _compute_phase_statistics(self) -> None:
        """Compute statistics for each phase"""
        phase_frames: Dict[int, List[int]] = {}
        phase_stages: Dict[int, List[int]] = {}
        
        # Collect frame data for each phase
        for entry in self.timeline_data:
            phase = entry.current_phase
            if phase not in phase_frames:
                phase_frames[phase] = []
                phase_stages[phase] = []
            
            phase_frames[phase].append(entry.frame_index)
            if entry.current_stage not in phase_stages[phase]:
                phase_stages[phase].append(entry.current_stage)
        
        # Create phase statistics
        for phase, frames in phase_frames.items():
            self.phase_statistics[phase] = PhaseStatistics(
                phase_number=phase,
                total_frames=len(frames),
                start_frame=min(frames),
                end_frame=max(frames),
                stage_occurrences=sorted(phase_stages[phase])
            )
        
        self.logger.info(f"📋 Computed statistics for {len(self.phase_statistics)} phases")
    
    def _build_lookup_tables(self) -> None:
        """Build performance optimization lookup tables"""
        # Frame-based lookup
        for entry in self.timeline_data:
            self.frame_lookup[entry.frame_index] = entry
        
        # Phase frame ranges (for efficient phase-based queries)
        for phase, stats in self.phase_statistics.items():
            # Find continuous ranges for this phase
            phase_entries = [e for e in self.timeline_data if e.current_phase == phase]
            if not phase_entries:
                continue
            
            ranges = []
            current_start = phase_entries[0].frame_index
            last_frame = current_start
            
            for entry in phase_entries[1:]:
                if entry.frame_index - last_frame > 1:  # Gap detected
                    ranges.append((current_start, last_frame))
                    current_start = entry.frame_index
                last_frame = entry.frame_index
            
            ranges.append((current_start, last_frame))
            self.phase_frame_ranges[phase] = ranges
        
        self.logger.info(f"🚀 Built lookup tables for {len(self.frame_lookup)} frames")
    
    def get_stage_phase_for_frame(self, frame_index: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get stage and phase information for a specific frame
        
        Args:
            frame_index: Frame number to lookup
            
        Returns:
            Tuple of (current_stage, current_phase, previous_stage, previous_phase) or None if not found
        """
        entry = self.frame_lookup.get(frame_index)
        if entry:
            return (entry.current_stage, entry.current_phase, entry.previous_stage, entry.previous_phase)
        
        # If exact frame not found, find nearest
        return self._find_nearest_frame_data(frame_index)
    
    def _find_nearest_frame_data(self, frame_index: int) -> Optional[Tuple[int, int, int, int]]:
        """Find the nearest frame data when exact frame is not available
        
        For sparse CSV data (entries every ~600-800 frames), use the most recent entry
        that is still before or at the current frame to maintain timeline accuracy
        """
        if not self.timeline_data:
            return None
        
        # Find all frames that are less than or equal to the current frame
        valid_entries = [entry for entry in self.timeline_data if entry.frame_index <= frame_index]
        
        if not valid_entries:
            # If frame_index is before the first CSV entry, don't return anything
            # This prevents using future timeline data
            self.logger.debug(f"🎯 Frame {frame_index} is before first CSV entry ({self.timeline_data[0].frame_index})")
            return None
        
        # Get the entry with the highest frame index that's still <= current frame
        nearest_lower_entry = max(valid_entries, key=lambda x: x.frame_index)
        
        # Check if the gap is reasonable (not more than 1000 frames = ~33 seconds at 30fps)
        frame_gap = frame_index - nearest_lower_entry.frame_index
        if frame_gap > 1000:
            self.logger.warning(f"🎯 Large frame gap: {frame_gap} frames between CSV entry {nearest_lower_entry.frame_index} and current {frame_index}")
        
        self.logger.debug(f"🎯 Frame {frame_index} -> Using CSV entry at frame {nearest_lower_entry.frame_index} (gap: {frame_gap})")
        
        return (nearest_lower_entry.current_stage, nearest_lower_entry.current_phase,
               nearest_lower_entry.previous_stage, nearest_lower_entry.previous_phase)
    
    def get_phase_statistics(self, phase: int) -> Optional[PhaseStatistics]:
        """Get statistics for a specific phase"""
        return self.phase_statistics.get(phase)
    
    def get_all_phases(self) -> List[int]:
        """Get list of all phases in the timeline"""
        return sorted(self.phase_statistics.keys())
    
    def get_phase_frame_ranges(self, phase: int) -> List[Tuple[int, int]]:
        """Get frame ranges where a specific phase is active"""
        return self.phase_frame_ranges.get(phase, [])
    
    def estimate_dishes_per_phase(self, dishes_per_frame: float = 0.1) -> Dict[int, int]:
        """
        Estimate dish count per phase based on frame duration
        
        Args:
            dishes_per_frame: Estimated dishes processed per frame (default 0.1)
            
        Returns:
            Dictionary mapping phase number to estimated dish count
        """
        dish_estimates = {}
        
        for phase, stats in self.phase_statistics.items():
            # Simple estimation based on frame count
            estimated_dishes = int(stats.total_frames * dishes_per_frame)
            
            # Minimum of 1 dish per phase if phase exists
            if estimated_dishes == 0 and stats.total_frames > 0:
                estimated_dishes = 1
                
            dish_estimates[phase] = estimated_dishes
            stats.dish_count = estimated_dishes
        
        return dish_estimates
    
    def get_timeline_summary(self) -> Dict[str, Any]:
        """Get summary information about the timeline"""
        if not self.timeline_data:
            return {"error": "No timeline data loaded"}
        
        stages = [entry.current_stage for entry in self.timeline_data]
        phases = [entry.current_phase for entry in self.timeline_data]
        
        return {
            "total_frames": len(self.timeline_data),
            "frame_range": (self.timeline_data[0].frame_index, self.timeline_data[-1].frame_index),
            "stages": {
                "unique": sorted(set(stages)),
                "range": (min(stages), max(stages)),
                "total": len(set(stages))
            },
            "phases": {
                "unique": sorted(set(phases)),
                "range": (min(phases), max(phases)),
                "total": len(set(phases))
            },
            "phase_statistics": {
                phase: {
                    "total_frames": stats.total_frames,
                    "estimated_dishes": stats.dish_count,
                    "frame_range": (stats.start_frame, stats.end_frame),
                    "stages": stats.stage_occurrences
                }
                for phase, stats in self.phase_statistics.items()
            }
        }
    
    def update_current_frame(self, frame_index: int) -> None:
        """Update the current frame index for tracking"""
        self.current_frame_index = frame_index
    
    def get_current_stage_phase(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current stage/phase based on current frame index"""
        return self.get_stage_phase_for_frame(self.current_frame_index)
    
    def get_max_frame_index(self) -> int:
        """Get the maximum frame index in the timeline data"""
        if not self.timeline_data:
            return 0
        return max(entry.frame_index for entry in self.timeline_data)


# Convenience functions for easy integration
def create_timeline_parser(csv_file_path: str) -> CSVTimelineParser:
    """Create and initialize a CSV timeline parser"""
    return CSVTimelineParser(csv_file_path)


def load_timeline_data(csv_file_path: str) -> Dict[str, Any]:
    """Load timeline data and return summary"""
    parser = CSVTimelineParser(csv_file_path)
    return parser.get_timeline_summary()