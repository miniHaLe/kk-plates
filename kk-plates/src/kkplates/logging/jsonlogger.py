"""Structured JSON logging configuration."""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.stdlib import LoggerFactory, add_logger_name


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_output: Whether to output JSON format
    """
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        stream=sys.stdout
    )
    
    # Processors for structlog
    processors = [
        TimeStamper(fmt="iso"),
        add_log_level,
        add_logger_name,
    ]
    
    if json_output:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Format for file output (always JSON)
        if json_output:
            file_formatter = logging.Formatter("%(message)s")
        else:
            # Even in console mode, use JSON for files
            file_formatter = logging.Formatter("%(message)s")
        
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


class EventLogger:
    """Logger for plate events with consistent structure."""
    
    def __init__(self, component: str):
        self.logger = structlog.get_logger(component)
    
    def log_crossing_event(self, event: Dict[str, Any]) -> None:
        """Log a crossing event."""
        self.logger.info(
            "crossing_event",
            event_type="crossing",
            track_id=event["track_id"],
            direction=event["direction"],
            color=event["color"],
            position=event["position"],
            timestamp=event["timestamp"]
        )
    
    def log_metrics_snapshot(self, metrics: Dict[str, Any]) -> None:
        """Log a metrics snapshot."""
        self.logger.info(
            "metrics_snapshot",
            event_type="metrics",
            total_in=metrics["total_in"],
            total_out=metrics["total_out"],
            current_on_belt=metrics["current_on_belt"],
            plates_per_minute=metrics["plates_per_minute"],
            color_ratios=metrics["color_ratios"],
            timestamp=metrics["timestamp"]
        )
    
    def log_alert(self, alert: Dict[str, Any]) -> None:
        """Log an alert."""
        self.logger.warning(
            "alert_generated",
            event_type="alert",
            alert_type=alert["alert_type"],
            severity=alert["severity"],
            message=alert["message"],
            details=alert["details"],
            timestamp=alert["timestamp"]
        )
    
    def log_error(self, error_type: str, message: str, **kwargs) -> None:
        """Log an error."""
        self.logger.error(
            "error_occurred",
            event_type="error",
            error_type=error_type,
            message=message,
            **kwargs
        )
    
    def log_performance(self, metric_name: str, value: float, unit: str = "ms") -> None:
        """Log a performance metric."""
        self.logger.debug(
            "performance_metric",
            event_type="performance",
            metric=metric_name,
            value=value,
            unit=unit
        )


class MetricsFileWriter:
    """Write metrics to JSONL file for analysis."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log files
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        self.metrics_file = self.output_dir / f"metrics_{date_str}.jsonl"
        self.events_file = self.output_dir / f"events_{date_str}.jsonl"
        self.alerts_file = self.output_dir / f"alerts_{date_str}.jsonl"
    
    def write_metric(self, metric: Dict[str, Any]) -> None:
        """Write metric to file."""
        import json
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metric) + "\n")
    
    def write_event(self, event: Dict[str, Any]) -> None:
        """Write event to file."""
        import json
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def write_alert(self, alert: Dict[str, Any]) -> None:
        """Write alert to file."""
        import json
        with open(self.alerts_file, "a") as f:
            f.write(json.dumps(alert) + "\n")