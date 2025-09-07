"""Configuration management using Pydantic."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ROIConfig(BaseModel):
    """Region of Interest configuration."""
    in_lane: List[List[int]] = Field(..., description="Polygon points for incoming lane")
    out_lane: List[List[int]] = Field(..., description="Polygon points for outgoing lane")
    
    @validator("in_lane", "out_lane")
    def validate_polygon(cls, v):
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for point in v:
            if len(point) != 2:
                raise ValueError("Each point must have exactly 2 coordinates")
        return v


class DetectorConfig(BaseModel):
    """Object detector configuration."""
    model: str = Field("yolov8n.pt", description="Path to YOLO model")
    conf_thres: float = Field(0.25, ge=0.0, le=1.0)
    iou_thres: float = Field(0.45, ge=0.0, le=1.0)


class HSVThreshold(BaseModel):
    """HSV color threshold ranges."""
    h: List[int] = Field(..., min_items=2, max_items=2)
    s: List[int] = Field(..., min_items=2, max_items=2)
    v: List[int] = Field(..., min_items=2, max_items=2)


class ClassifierConfig(BaseModel):
    """Color classifier configuration."""
    model_path: str = Field(..., description="Path to ONNX model")
    hsv_thresholds: Dict[str, HSVThreshold]


class MetricsConfig(BaseModel):
    """Metrics aggregation configuration."""
    window_seconds: int = Field(60, ge=1)


class ToleranceConfig(BaseModel):
    """Alert tolerance configuration."""
    relative: float = Field(0.20, ge=0.0, le=1.0)


class PresetConfig(BaseModel):
    """Preset target ratios and tolerances."""
    target_ratio: Dict[str, float]
    tolerance: ToleranceConfig
    
    @validator("target_ratio")
    def validate_ratios(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Target ratios must sum to 1.0, got {total}")
        return v


class PowerBIConfig(BaseModel):
    """Power BI sink configuration."""
    endpoint: str
    api_key: str


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json: bool = Field(True)


class Settings(BaseSettings):
    """Main application settings."""
    rtsp_url: str
    rtsp_latency_ms: int = Field(60, ge=0)
    frame_stride: int = Field(1, ge=1)
    roi: ROIConfig
    detector: DetectorConfig
    classifier: ClassifierConfig
    metrics: MetricsConfig
    preset: PresetConfig
    powerbi: PowerBIConfig
    logging: LoggingConfig
    
    class Config:
        env_prefix = "KKPLATES_"
        case_sensitive = False
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save settings to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)