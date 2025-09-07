#!/usr/bin/env python3
"""Generate sample conveyor belt video with colored plates."""

import cv2
import numpy as np
import random
from pathlib import Path
import argparse


class Plate:
    """Represents a plate on the conveyor."""
    def __init__(self, x, y, color, plate_id):
        self.x = x
        self.y = y
        self.color = color
        self.plate_id = plate_id
        self.width = random.randint(70, 90)
        self.height = random.randint(40, 50)
        self.speed = random.uniform(1.5, 2.5)  # pixels per frame
        
    def move(self):
        """Move plate along conveyor."""
        self.x += self.speed
        
    def draw(self, frame):
        """Draw plate on frame."""
        color_map = {
            'red': (0, 0, 200),
            'yellow': (0, 200, 200),
            'normal': (220, 220, 220)
        }
        
        color = color_map[self.color]
        cv2.rectangle(frame, 
                     (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)),
                     color, -1)
        
        # Add border
        cv2.rectangle(frame, 
                     (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)),
                     (50, 50, 50), 2)
        
        # Add plate number
        cv2.putText(frame, str(self.plate_id), 
                   (int(self.x + 10), int(self.y + 25)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def create_conveyor_background(width, height):
    """Create conveyor belt background."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Draw conveyor belt
    belt_y1 = 350
    belt_height = 250
    cv2.rectangle(frame, (0, belt_y1), (width, belt_y1 + belt_height), (100, 100, 100), -1)
    
    # Draw lanes
    in_lane_y = 400
    out_lane_y = 540
    lane_height = 40
    
    # In lane
    cv2.rectangle(frame, (0, in_lane_y), (width, in_lane_y + lane_height), (120, 120, 120), -1)
    cv2.putText(frame, "IN LANE", (10, in_lane_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Out lane
    cv2.rectangle(frame, (0, out_lane_y), (width, out_lane_y + lane_height), (120, 120, 120), -1)
    cv2.putText(frame, "OUT LANE", (10, out_lane_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw ROI zones
    in_roi = np.array([[220, 380], [420, 380], [420, 430], [220, 430]], np.int32)
    out_roi = np.array([[220, 520], [420, 520], [420, 570], [220, 570]], np.int32)
    
    cv2.polylines(frame, [in_roi], True, (0, 255, 0), 2)
    cv2.polylines(frame, [out_roi], True, (0, 0, 255), 2)
    
    return frame


def generate_sample_video(output_path, duration_seconds=10, fps=25):
    """Generate sample video with plates crossing ROIs."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    plates_in = []
    plates_out = []
    plate_counter = 1
    
    # Generate plates at intervals
    colors = ['red', 'yellow', 'normal']
    color_weights = [0.33, 0.33, 0.34]
    
    for frame_num in range(total_frames):
        # Create background
        frame = create_conveyor_background(width, height)
        
        # Add timestamp
        cv2.putText(frame, f"Frame: {frame_num} / Time: {frame_num/fps:.1f}s",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Generate new plates (IN lane)
        if frame_num % 30 == 0 and frame_num > 0:  # Every ~1 second
            color = random.choices(colors, weights=color_weights)[0]
            plate = Plate(50, 395, color, plate_counter)
            plates_in.append(plate)
            plate_counter += 1
        
        # Generate plates (OUT lane) - some plates moving back
        if frame_num % 40 == 0 and frame_num > 60:
            color = random.choices(colors, weights=color_weights)[0]
            plate = Plate(width - 100, 535, color, plate_counter)
            plate.speed = -plate.speed  # Move left
            plates_out.append(plate)
            plate_counter += 1
        
        # Update and draw plates
        for plate in plates_in[:]:
            plate.move()
            if plate.x > width:
                plates_in.remove(plate)
            else:
                plate.draw(frame)
        
        for plate in plates_out[:]:
            plate.move()
            if plate.x < -100:
                plates_out.remove(plate)
            else:
                plate.draw(frame)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Generated {output_path} ({duration_seconds}s, {total_frames} frames)")
    
    # Generate ground truth
    return {
        'duration': duration_seconds,
        'fps': fps,
        'total_frames': total_frames,
        'total_plates': plate_counter - 1
    }


def generate_ground_truth(video_info, output_path):
    """Generate ground truth JSON for the video."""
    import json
    
    ground_truth = {
        "video_info": video_info,
        "expected_metrics": {
            "min_in_events": 8,  # At least 8 IN crossings in 10s
            "min_total_plates": 10,
            "color_distribution": {
                "red": 0.33,
                "yellow": 0.33,
                "normal": 0.34
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Generated ground truth: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample conveyor video")
    parser.add_argument("--output", type=str, default="data/samples/conveyor_sample.mp4")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate video
    video_info = generate_sample_video(output_path, args.duration, args.fps)
    
    # Generate ground truth
    gt_path = output_path.with_suffix('.json')
    generate_ground_truth(video_info, gt_path)


if __name__ == "__main__":
    main()