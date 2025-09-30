#!/usr/bin/env python3
"""
RTSP Stream Analyzer
===================

A comprehensive tool for analyzing RTSP video streams and generating detailed reports
on stream properties including resolution, bitrate, bandwidth, FPS, and more.

Author: AI Assistant
Date: 2025-09-12
"""

import cv2
import time
import json
import threading
import psutil
import subprocess
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class StreamMetrics:
    """Data class to store stream metrics."""
    url: str
    resolution: Tuple[int, int]
    fps: float
    actual_fps: float
    bitrate_kbps: float
    bandwidth_mbps: float
    codec: str
    duration_analyzed: float
    frames_analyzed: int
    frames_dropped: int
    avg_frame_size: float
    peak_bandwidth: float
    min_bandwidth: float
    connection_time: float
    error_count: int
    stability_score: float
    timestamp: str


class RTSPAnalyzer:
    """Comprehensive RTSP stream analyzer."""
    
    def __init__(self):
        self.is_analyzing = False
        self.bandwidth_history = []
        self.frame_times = []
        self.frame_sizes = []
        self.error_log = []
        
    def get_stream_info_ffprobe(self, url: str) -> Dict[str, Any]:
        """Get detailed stream information using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            self.error_log.append(f"ffprobe error: {str(e)}")
        
        return {}
    
    def calculate_bandwidth(self, frame_data: np.ndarray, time_diff: float) -> float:
        """Calculate bandwidth based on frame data size and time difference."""
        if time_diff <= 0:
            return 0.0
        
        # Estimate compressed frame size (rough approximation)
        frame_size_bytes = frame_data.nbytes * 0.1  # Assuming ~10:1 compression
        bandwidth_bps = (frame_size_bytes * 8) / time_diff
        return bandwidth_bps / (1024 * 1024)  # Convert to Mbps
    
    def analyze_stability(self, fps_history: List[float], bandwidth_history: List[float]) -> float:
        """Calculate stream stability score based on FPS and bandwidth consistency."""
        if not fps_history or not bandwidth_history:
            return 0.0
        
        fps_variance = np.var(fps_history) if len(fps_history) > 1 else 0
        bandwidth_variance = np.var(bandwidth_history) if len(bandwidth_history) > 1 else 0
        
        # Lower variance = higher stability (score out of 100)
        fps_stability = max(0, 100 - (fps_variance * 10))
        bandwidth_stability = max(0, 100 - (bandwidth_variance * 100))
        
        return (fps_stability + bandwidth_stability) / 2
    
    def test_connection_speed(self, url: str) -> float:
        """Test initial connection speed to the RTSP stream."""
        start_time = time.time()
        
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                connection_time = time.time() - start_time
                cap.release()
                return connection_time
        
        cap.release()
        return -1.0  # Failed to connect
    
    def analyze_stream(self, url: str, duration: int = 30) -> StreamMetrics:
        """
        Analyze RTSP stream for specified duration and return comprehensive metrics.
        
        Args:
            url: RTSP stream URL
            duration: Analysis duration in seconds
            
        Returns:
            StreamMetrics object containing all analysis results
        """
        print(f"🎥 Starting RTSP stream analysis...")
        print(f"📡 URL: {url}")
        print(f"⏱️  Duration: {duration} seconds")
        print("-" * 60)
        
        # Initialize metrics
        start_time = time.time()
        frames_analyzed = 0
        frames_dropped = 0
        fps_history = []
        bandwidth_history = []
        frame_size_history = []
        last_frame_time = None
        
        # Test connection speed
        connection_time = self.test_connection_speed(url)
        if connection_time < 0:
            raise ConnectionError(f"Failed to connect to RTSP stream: {url}")
        
        # Get stream info using ffprobe
        stream_info = self.get_stream_info_ffprobe(url)
        
        # Open video capture
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP stream: {url}")
        
        # Get basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📺 Resolution: {width}x{height}")
        print(f"🎯 Declared FPS: {declared_fps:.2f}")
        print(f"🔗 Connection time: {connection_time:.3f}s")
        print("\n📊 Real-time analysis:")
        
        self.is_analyzing = True
        
        try:
            while self.is_analyzing and (time.time() - start_time) < duration:
                ret, frame = cap.read()
                current_time = time.time()
                
                if ret:
                    frames_analyzed += 1
                    
                    # Calculate actual FPS
                    if last_frame_time is not None:
                        time_diff = current_time - last_frame_time
                        if time_diff > 0:
                            current_fps = 1.0 / time_diff
                            fps_history.append(current_fps)
                            
                            # Calculate bandwidth
                            bandwidth = self.calculate_bandwidth(frame, time_diff)
                            bandwidth_history.append(bandwidth)
                            
                            # Track frame size
                            frame_size = frame.nbytes
                            frame_size_history.append(frame_size)
                    
                    last_frame_time = current_time
                    
                    # Real-time reporting every 5 seconds
                    elapsed = current_time - start_time
                    if frames_analyzed % max(1, int(declared_fps * 5)) == 0:
                        avg_fps = np.mean(fps_history[-int(declared_fps * 5):]) if fps_history else 0
                        avg_bw = np.mean(bandwidth_history[-int(declared_fps * 5):]) if bandwidth_history else 0
                        print(f"  ⏱️  {elapsed:.1f}s | 🎞️  {avg_fps:.1f} FPS | 📡 {avg_bw:.2f} Mbps | 📦 {frames_analyzed} frames")
                
                else:
                    frames_dropped += 1
                    self.error_log.append(f"Frame dropped at {current_time - start_time:.2f}s")
                    
                    # If too many consecutive drops, break
                    if frames_dropped > frames_analyzed and frames_dropped > 100:
                        print("❌ Too many dropped frames, stopping analysis...")
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            self.error_log.append(f"Analysis error: {str(e)}")
        finally:
            cap.release()
            self.is_analyzing = False
        
        # Calculate final metrics
        total_time = time.time() - start_time
        actual_fps = frames_analyzed / total_time if total_time > 0 else 0
        avg_bandwidth = np.mean(bandwidth_history) if bandwidth_history else 0
        peak_bandwidth = np.max(bandwidth_history) if bandwidth_history else 0
        min_bandwidth = np.min(bandwidth_history) if bandwidth_history else 0
        avg_frame_size = np.mean(frame_size_history) if frame_size_history else 0
        
        # Estimate bitrate (rough calculation)
        estimated_bitrate = (avg_frame_size * actual_fps * 8) / 1024  # kbps
        
        # Calculate stability score
        stability_score = self.analyze_stability(fps_history, bandwidth_history)
        
        # Extract codec information
        codec = "Unknown"
        if stream_info and 'streams' in stream_info:
            for stream in stream_info['streams']:
                if stream.get('codec_type') == 'video':
                    codec = stream.get('codec_name', 'Unknown')
                    break
        
        return StreamMetrics(
            url=url,
            resolution=(width, height),
            fps=declared_fps,
            actual_fps=actual_fps,
            bitrate_kbps=estimated_bitrate,
            bandwidth_mbps=avg_bandwidth,
            codec=codec,
            duration_analyzed=total_time,
            frames_analyzed=frames_analyzed,
            frames_dropped=frames_dropped,
            avg_frame_size=avg_frame_size,
            peak_bandwidth=peak_bandwidth,
            min_bandwidth=min_bandwidth,
            connection_time=connection_time,
            error_count=len(self.error_log),
            stability_score=stability_score,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_report(self, metrics: StreamMetrics, output_file: Optional[str] = None) -> str:
        """Generate a detailed analysis report."""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           RTSP STREAM ANALYSIS REPORT                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 STREAM INFORMATION
├─ URL: {metrics.url}
├─ Timestamp: {metrics.timestamp}
├─ Analysis Duration: {metrics.duration_analyzed:.2f} seconds
└─ Connection Time: {metrics.connection_time:.3f} seconds

📺 VIDEO PROPERTIES
├─ Resolution: {metrics.resolution[0]}x{metrics.resolution[1]} ({metrics.resolution[0] * metrics.resolution[1] / 1000000:.1f}MP)
├─ Codec: {metrics.codec}
├─ Declared FPS: {metrics.fps:.2f}
├─ Actual FPS: {metrics.actual_fps:.2f}
└─ FPS Accuracy: {(metrics.actual_fps / metrics.fps * 100) if metrics.fps > 0 else 0:.1f}%

📡 BANDWIDTH & BITRATE
├─ Average Bandwidth: {metrics.bandwidth_mbps:.2f} Mbps
├─ Peak Bandwidth: {metrics.peak_bandwidth:.2f} Mbps
├─ Minimum Bandwidth: {metrics.min_bandwidth:.2f} Mbps
├─ Estimated Bitrate: {metrics.bitrate_kbps:.0f} kbps
└─ Bandwidth Efficiency: {(metrics.bitrate_kbps / 1024 / metrics.bandwidth_mbps * 100) if metrics.bandwidth_mbps > 0 else 0:.1f}%

📦 FRAME ANALYSIS
├─ Total Frames Analyzed: {metrics.frames_analyzed:,}
├─ Frames Dropped: {metrics.frames_dropped:,}
├─ Drop Rate: {(metrics.frames_dropped / (metrics.frames_analyzed + metrics.frames_dropped) * 100) if (metrics.frames_analyzed + metrics.frames_dropped) > 0 else 0:.2f}%
├─ Average Frame Size: {metrics.avg_frame_size / 1024:.1f} KB
└─ Data Rate: {metrics.avg_frame_size * metrics.actual_fps / 1024 / 1024:.2f} MB/s

🔍 QUALITY METRICS
├─ Stream Stability Score: {metrics.stability_score:.1f}/100
├─ Error Count: {metrics.error_count}
├─ Overall Rating: {self._get_overall_rating(metrics)}
└─ Recommendation: {self._get_recommendation(metrics)}

📈 PERFORMANCE SUMMARY
{self._get_performance_summary(metrics)}

{self._get_error_summary() if self.error_log else "✅ No errors detected during analysis"}

╔══════════════════════════════════════════════════════════════════════════════╗
║ Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        
        return report
    
    def _get_overall_rating(self, metrics: StreamMetrics) -> str:
        """Calculate overall stream rating."""
        score = 0
        
        # FPS consistency (30%)
        fps_accuracy = (metrics.actual_fps / metrics.fps) if metrics.fps > 0 else 0
        score += min(30, fps_accuracy * 30)
        
        # Drop rate (25%)
        total_frames = metrics.frames_analyzed + metrics.frames_dropped
        drop_rate = metrics.frames_dropped / total_frames if total_frames > 0 else 0
        score += max(0, 25 - (drop_rate * 250))
        
        # Stability (25%)
        score += metrics.stability_score * 0.25
        
        # Error rate (20%)
        error_rate = metrics.error_count / metrics.duration_analyzed if metrics.duration_analyzed > 0 else 0
        score += max(0, 20 - (error_rate * 20))
        
        if score >= 90:
            return "🟢 Excellent"
        elif score >= 75:
            return "🟡 Good"
        elif score >= 60:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    def _get_recommendation(self, metrics: StreamMetrics) -> str:
        """Get improvement recommendations."""
        issues = []
        
        if metrics.frames_dropped > metrics.frames_analyzed * 0.05:
            issues.append("High frame drop rate")
        
        if metrics.stability_score < 70:
            issues.append("Unstable stream quality")
        
        if metrics.actual_fps < metrics.fps * 0.9:
            issues.append("FPS below declared rate")
        
        if metrics.error_count > 0:
            issues.append("Stream errors detected")
        
        if not issues:
            return "✅ Stream quality is good"
        else:
            return f"⚠️  Issues: {', '.join(issues)}"
    
    def _get_performance_summary(self, metrics: StreamMetrics) -> str:
        """Generate performance summary."""
        resolution_class = self._classify_resolution(metrics.resolution)
        bitrate_efficiency = self._classify_bitrate_efficiency(metrics)
        
        return f"""├─ Resolution Class: {resolution_class}
├─ Bitrate Efficiency: {bitrate_efficiency}
├─ Stream Consistency: {'High' if metrics.stability_score > 80 else 'Medium' if metrics.stability_score > 60 else 'Low'}
└─ Network Requirements: {self._estimate_network_requirements(metrics)}"""
    
    def _classify_resolution(self, resolution: Tuple[int, int]) -> str:
        """Classify resolution."""
        width, height = resolution
        pixels = width * height
        
        if pixels >= 3840 * 2160:
            return "4K UHD"
        elif pixels >= 1920 * 1080:
            return "Full HD (1080p)"
        elif pixels >= 1280 * 720:
            return "HD (720p)"
        elif pixels >= 854 * 480:
            return "SD (480p)"
        else:
            return "Low Resolution"
    
    def _classify_bitrate_efficiency(self, metrics: StreamMetrics) -> str:
        """Classify bitrate efficiency."""
        width, height = metrics.resolution
        pixels = width * height
        bits_per_pixel = (metrics.bitrate_kbps * 1024) / (pixels * metrics.actual_fps) if pixels > 0 and metrics.actual_fps > 0 else 0
        
        if bits_per_pixel < 0.1:
            return "🟢 Highly Efficient"
        elif bits_per_pixel < 0.3:
            return "🟡 Efficient"
        elif bits_per_pixel < 0.5:
            return "🟠 Moderate"
        else:
            return "🔴 Inefficient"
    
    def _estimate_network_requirements(self, metrics: StreamMetrics) -> str:
        """Estimate network requirements."""
        bandwidth_mbps = metrics.bandwidth_mbps
        
        if bandwidth_mbps < 1:
            return "Low bandwidth (< 1 Mbps)"
        elif bandwidth_mbps < 5:
            return "Medium bandwidth (1-5 Mbps)"
        elif bandwidth_mbps < 15:
            return "High bandwidth (5-15 Mbps)"
        else:
            return "Very high bandwidth (> 15 Mbps)"
    
    def _get_error_summary(self) -> str:
        """Generate error summary."""
        if not self.error_log:
            return ""
        
        return f"""
❌ ERRORS DETECTED:
{chr(10).join([f"   • {error}" for error in self.error_log[:10]])}
{f"   ... and {len(self.error_log) - 10} more errors" if len(self.error_log) > 10 else ""}"""


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RTSP Stream Analyzer")
    parser.add_argument("url", help="RTSP stream URL")
    parser.add_argument("-d", "--duration", type=int, default=30, help="Analysis duration in seconds")
    parser.add_argument("-o", "--output", help="Output report file")
    
    args = parser.parse_args()
    
    analyzer = RTSPAnalyzer()
    
    try:
        metrics = analyzer.analyze_stream(args.url, args.duration)
        report = analyzer.generate_report(metrics, args.output)
        print(report)
        
        # Save metrics as JSON
        json_file = args.output.replace('.txt', '.json') if args.output else 'rtsp_analysis.json'
        with open(json_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"📊 Metrics saved to: {json_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
