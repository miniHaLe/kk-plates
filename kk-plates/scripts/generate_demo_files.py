#!/usr/bin/env python3
"""Generate demo files for testing without OpenCV dependency."""

import json
from pathlib import Path

# Create sample ground truth JSON
def create_sample_json():
    sample_data = {
        "video_info": {
            "duration": 10,
            "fps": 25,
            "total_frames": 250,
            "total_plates": 15
        },
        "expected_metrics": {
            "min_in_events": 8,
            "min_total_plates": 10,
            "color_distribution": {
                "red": 0.33,
                "yellow": 0.33,
                "normal": 0.34
            }
        },
        "sample_events": [
            {"timestamp": 1.0, "event": "in", "color": "red", "track_id": 1},
            {"timestamp": 2.0, "event": "in", "color": "yellow", "track_id": 2},
            {"timestamp": 3.0, "event": "in", "color": "normal", "track_id": 3},
            {"timestamp": 4.0, "event": "out", "color": "red", "track_id": 1},
            {"timestamp": 5.0, "event": "in", "color": "red", "track_id": 4}
        ]
    }
    
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "sample_ground_truth.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample ground truth: {output_dir / 'sample_ground_truth.json'}")

# Create sample metrics export
def create_sample_metrics():
    metrics = {
        "capture_duration": 10,
        "metrics": [
            {
                "timestamp": 1234567890.0,
                "total_in": 45,
                "total_out": 42,
                "current_on_belt": 3,
                "color_counts": {"red": 15, "yellow": 15, "normal": 15},
                "window_seconds": 60,
                "plates_per_minute": 12.5,
                "color_frequencies": {"red": 4.2, "yellow": 4.1, "normal": 4.2},
                "color_ratios": {"red": 0.333, "yellow": 0.333, "normal": 0.334}
            }
        ],
        "stats": {
            "total_in": 45,
            "total_out": 42,
            "current_on_belt": 3,
            "color_counts": {"red": 15, "yellow": 15, "normal": 15}
        }
    }
    
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "sample_metrics_export.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Created sample metrics: {output_dir / 'sample_metrics_export.json'}")

# Create sample event log
def create_sample_logs():
    events = [
        {
            "timestamp": "2024-01-15T10:30:00.123",
            "event": "crossing_event",
            "track_id": 42,
            "direction": "in",
            "color": "red",
            "position": [250, 400]
        },
        {
            "timestamp": "2024-01-15T10:30:01.456",
            "event": "metrics_snapshot",
            "total_in": 100,
            "total_out": 97,
            "current_on_belt": 3,
            "plates_per_minute": 15.2,
            "color_ratios": {"red": 0.35, "yellow": 0.32, "normal": 0.33}
        },
        {
            "timestamp": "2024-01-15T10:30:05.789",
            "event": "alert_generated",
            "alert_type": "color_ratio_deviation",
            "severity": "warning",
            "message": "Red plate ratio exceeds target",
            "details": {"color": "red", "actual": 0.42, "target": 0.33}
        }
    ]
    
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "sample_events.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    
    print(f"Created sample logs: {output_dir / 'sample_events.jsonl'}")

if __name__ == "__main__":
    create_sample_json()
    create_sample_metrics()
    create_sample_logs()
    print("\nAll demo files created successfully!")