"""Command-line interface for KK-Plates system."""

import sys
import time
import signal
from pathlib import Path
from typing import Optional
import cv2
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import structlog

from .config import Settings
from .capture.rtsp_reader import RTSPReader
from .detect.model import PlateDetector
from .classify.color_model import ColorClassifier
from .roi.roi_tool import ROITool
from .track.tracker import ByteTracker
from .count.crossing import CrossingDetector
from .metrics.aggregator import MetricsAggregator
from .alert.preset_check import PresetChecker
from .sinks.powerbi import PowerBISink
from .logging.jsonlogger import configure_logging, EventLogger, MetricsFileWriter

app = typer.Typer(help="KK-Plates: Real-time plate counting and classification system")
console = Console()
logger = structlog.get_logger()


class PlateCountingPipeline:
    """Main pipeline for plate counting."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.running = False
        
        # Configure logging
        configure_logging(
            level=config.logging.level,
            log_file=Path("logs/kkplates.jsonl"),
            json_output=config.logging.json
        )
        
        # Initialize components
        self.reader = RTSPReader(config.rtsp_url, config.rtsp_latency_ms)
        self.detector = PlateDetector(
            config.detector.model,
            config.detector.conf_thres,
            config.detector.iou_thres
        )
        self.classifier = ColorClassifier(
            config.classifier.model_path,
            config.classifier.hsv_thresholds
        )
        self.tracker = ByteTracker()
        self.crossing_detector = CrossingDetector(
            config.roi.in_lane,
            config.roi.out_lane
        )
        self.metrics = MetricsAggregator(config.metrics.window_seconds)
        self.preset_checker = PresetChecker(
            config.preset.target_ratio,
            config.preset.tolerance.relative
        )
        self.powerbi = PowerBISink(config.powerbi.endpoint, config.powerbi.api_key)
        
        # Logging
        self.event_logger = EventLogger("pipeline")
        self.metrics_writer = MetricsFileWriter(Path("logs"))
        
        # Stats
        self.frame_count = 0
        self.start_time = 0
        self.last_metrics_time = 0
        
    def start(self) -> None:
        """Start the pipeline."""
        logger.info("Starting plate counting pipeline")
        
        # Load models
        self.detector.load()
        self.detector.warmup()
        self.classifier.load()
        
        # Test Power BI connection
        if not self.powerbi.test_connection():
            logger.warning("Power BI connection test failed, continuing anyway")
        
        # Start capture
        self.reader.start()
        self.running = True
        self.start_time = time.time()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.stop()
    
    def stop(self) -> None:
        """Stop the pipeline."""
        self.running = False
        self.reader.stop()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped")
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Process a single frame."""
        # Detect plates
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Classify colors for each track
        colors = {}
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color, conf = self.classifier.classify(frame, (x1, y1, x2, y2))
            colors[track.track_id] = color
        
        # Check crossings
        events = self.crossing_detector.update(tracks, colors, timestamp)
        
        # Log events
        for event in events:
            self.metrics.add_event(event.direction, event.color, event.timestamp)
            event_dict = {
                "track_id": event.track_id,
                "direction": event.direction,
                "color": event.color,
                "timestamp": event.timestamp,
                "position": event.position
            }
            self.event_logger.log_crossing_event(event_dict)
            self.metrics_writer.write_event(event_dict)
            self.powerbi.send_event(event_dict)
        
        # Update metrics from crossing detector
        self.metrics.update_from_crossing_stats(self.crossing_detector.get_stats())
        
        # Draw visualization
        display = self.crossing_detector.draw_debug(frame)
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = colors.get(track.track_id, "normal")
            
            # Color mapping for visualization
            color_bgr = {
                "red": (0, 0, 255),
                "yellow": (0, 255, 255),
                "normal": (255, 255, 255)
            }.get(color, (255, 255, 255))
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(display, f"{track.track_id}:{color}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        
        # Add FPS
        fps = self.frame_count / (time.time() - self.start_time) if self.start_time > 0 else 0
        cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
    
    def run(self, show_video: bool = True) -> None:
        """Run the main processing loop."""
        self.start()
        
        try:
            frame_skip = 0
            for timestamp, frame in self.reader.frames():
                if not self.running:
                    break
                
                self.frame_count += 1
                
                # Skip frames based on stride
                frame_skip += 1
                if frame_skip < self.config.frame_stride:
                    continue
                frame_skip = 0
                
                # Process frame
                display = self.process_frame(frame, timestamp)
                
                # Show video if requested
                if show_video:
                    cv2.imshow("KK-Plates Monitor", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Periodic metrics update (every second)
                current_time = time.time()
                if current_time - self.last_metrics_time >= 1.0:
                    self._update_metrics()
                    self.last_metrics_time = current_time
                    
        except Exception as e:
            logger.error("Pipeline error", error=str(e), exc_info=True)
            self.event_logger.log_error("pipeline_error", str(e))
        finally:
            self.stop()
    
    def _update_metrics(self) -> None:
        """Update and send metrics."""
        # Get metrics snapshot
        snapshot = self.metrics.get_snapshot(force=True)
        if snapshot:
            snapshot_dict = snapshot.to_dict()
            
            # Log metrics
            self.event_logger.log_metrics_snapshot(snapshot_dict)
            self.metrics_writer.write_metric(snapshot_dict)
            
            # Send to Power BI
            self.powerbi.send_metrics(snapshot_dict)
            
            # Check presets
            alerts = self.preset_checker.check(snapshot_dict)
            for alert in alerts:
                alert_dict = alert.to_dict()
                self.event_logger.log_alert(alert_dict)
                self.metrics_writer.write_alert(alert_dict)
                self.powerbi.send_alert(alert_dict)


@app.command()
def roi(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    source: Optional[str] = typer.Option(None, "--source", "-s", 
                                         help="Video file or RTSP URL (uses config if not specified)")
):
    """Edit ROI (Region of Interest) polygons."""
    settings = Settings.from_yaml(config)
    roi_tool = ROITool(config)
    
    if source:
        if source.startswith("rtsp://"):
            roi_tool.edit_on_rtsp(source)
        else:
            roi_tool.edit_on_video(source)
    else:
        # Use RTSP from config
        roi_tool.edit_on_rtsp(settings.rtsp_url)


@app.command()
def run(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    no_video: bool = typer.Option(False, "--no-video", help="Run without video display"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging")
):
    """Run the real-time plate counting pipeline."""
    settings = Settings.from_yaml(config)
    
    if debug:
        settings.logging.level = "DEBUG"
    
    console.print(Panel.fit(
        f"[bold green]Starting KK-Plates Pipeline[/bold green]\n"
        f"RTSP: {settings.rtsp_url}\n"
        f"Press Ctrl+C to stop",
        title="KK-Plates"
    ))
    
    pipeline = PlateCountingPipeline(settings)
    pipeline.run(show_video=not no_video)


@app.command()
def export_metrics(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    seconds: int = typer.Option(10, "--seconds", "-s", help="Duration to capture metrics"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (stdout if not specified)")
):
    """Export metrics for a specified duration."""
    import json
    
    settings = Settings.from_yaml(config)
    pipeline = PlateCountingPipeline(settings)
    
    console.print(f"[bold]Capturing metrics for {seconds} seconds...[/bold]")
    
    pipeline.start()
    metrics_list = []
    
    try:
        start_time = time.time()
        for timestamp, frame in pipeline.reader.frames():
            if time.time() - start_time > seconds:
                break
            
            # Process frame
            pipeline.process_frame(frame, timestamp)
            
            # Collect metrics every second
            if int(time.time() - start_time) > len(metrics_list):
                snapshot = pipeline.metrics.get_snapshot(force=True)
                if snapshot:
                    metrics_list.append(snapshot.to_dict())
                    console.print(f"Collected {len(metrics_list)} snapshots")
    
    finally:
        pipeline.stop()
    
    # Output results
    result = {
        "capture_duration": seconds,
        "metrics": metrics_list,
        "stats": pipeline.crossing_detector.get_stats()
    }
    
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]Exported metrics to {output}[/green]")
    else:
        console.print_json(data=result)


@app.command()
def status(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c")
):
    """Check system status and configuration."""
    settings = Settings.from_yaml(config)
    
    # Create status table
    table = Table(title="KK-Plates System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    # Check RTSP
    try:
        reader = RTSPReader(settings.rtsp_url)
        reader.start()
        frame_data = reader.read_frame(timeout=5.0)
        reader.stop()
        
        if frame_data:
            _, frame = frame_data
            h, w = frame.shape[:2]
            table.add_row("RTSP Stream", "✓ Connected", f"{w}x{h} @ {settings.rtsp_url}")
        else:
            table.add_row("RTSP Stream", "✗ No frames", settings.rtsp_url)
    except Exception as e:
        table.add_row("RTSP Stream", "✗ Failed", str(e))
    
    # Check models
    detector_path = Path(settings.detector.model)
    if not detector_path.exists():
        detector_path = Path("data/models") / detector_path.name
    
    table.add_row(
        "Detector Model",
        "✓ Found" if detector_path.exists() else "✗ Missing",
        str(detector_path)
    )
    
    classifier_path = Path(settings.classifier.model_path)
    table.add_row(
        "Classifier Model", 
        "✓ Found" if classifier_path.exists() else "✗ Missing",
        str(classifier_path)
    )
    
    # Check Power BI
    powerbi = PowerBISink(settings.powerbi.endpoint, settings.powerbi.api_key)
    pb_status = "✓ Connected" if powerbi.test_connection() else "✗ Failed"
    table.add_row("Power BI", pb_status, settings.powerbi.endpoint)
    
    # Check ROI
    roi_configured = len(settings.roi.in_lane) >= 3 and len(settings.roi.out_lane) >= 3
    table.add_row(
        "ROI Configuration",
        "✓ Configured" if roi_configured else "✗ Not configured",
        f"In: {len(settings.roi.in_lane)} points, Out: {len(settings.roi.out_lane)} points"
    )
    
    console.print(table)
    
    # Show target ratios
    console.print("\n[bold]Target Color Ratios:[/bold]")
    for color, ratio in settings.preset.target_ratio.items():
        console.print(f"  {color}: {ratio:.1%} (±{settings.preset.tolerance.relative:.0%})")


@app.command()
def train_detector(
    data_dir: Path = typer.Argument(..., help="Directory with YOLO format data"),
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(16, "--batch", "-b"),
    output_dir: Path = typer.Option("data/models", "--output", "-o")
):
    """Train YOLOv8 detector on plate dataset."""
    from ultralytics import YOLO
    
    console.print(f"[bold]Training detector on {data_dir}[/bold]")
    
    # Load base model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=data_dir / "dataset.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name="kk_plates",
        project=str(output_dir),
        exist_ok=True
    )
    
    # Save best model
    best_model_path = output_dir / "kk_plates" / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, output_dir / "kk_plates_detector.pt")
        console.print(f"[green]✓ Saved best model to {output_dir}/kk_plates_detector.pt[/green]")


if __name__ == "__main__":
    app()