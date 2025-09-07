# KK-Plates: Real-time Plate Counting & Classification System

A production-ready computer vision system for counting and classifying plates on conveyor belts at Kichi-Kichi restaurants. The system uses real-time video analysis to track plates, classify them by color (red/yellow/normal), and maintain accurate counts with configurable alerts.

## Features

- **Real-time Detection**: Process RTSP streams at 25 FPS with YOLOv8
- **Color Classification**: Two-stage classification (HSV + CNN) for red/yellow/normal plates
- **Accurate Counting**: Track plates crossing IN/OUT zones with debouncing
- **Metrics & Alerts**: Monitor color ratios and alert on deviations
- **Power BI Integration**: Stream metrics to dashboards
- **Production Ready**: Structured logging, health checks, and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional but recommended)
- IP Camera: Hikvision DS-2CD2146G2-I(SU) or compatible RTSP stream

### Installation

```bash
# Clone repository
git clone https://github.com/kichi-kichi/kk-plates.git
cd kk-plates

# Install dependencies
pip install -e ".[dev]"

# Or using make
make setup
```

### Camera Setup Guide

1. **Mount Position**:
   - Install camera **directly above** the kitchen doorway
   - Minimum height: **50cm** from conveyor belt
   - Ensure both IN and OUT lanes are centered in view
   - Camera should be perpendicular to belt (top-down view)

2. **Lighting**:
   - Use consistent white lighting
   - Avoid colored lights that may affect plate color detection
   - Minimize shadows and glare on plates

3. **Camera Configuration**:
   - Resolution: 1920x1080 (1080p)
   - Frame rate: 25 FPS
   - Codec: H.264
   - Bitrate: ~4 Mbps

### Configuration

1. **Define ROI (Regions of Interest)**:
```bash
# Interactive ROI editor
kkplates roi --config configs/default.yaml

# Or use RTSP stream directly
kkplates roi --source rtsp://192.168.10.45:554/...
```

2. **Edit Configuration**:
```yaml
# configs/default.yaml
rtsp_url: "rtsp://192.168.10.45:554/Streaming/Unicast/channels/101"
roi:
  in_lane: [[220, 380], [700, 380], [700, 430], [220, 430]]
  out_lane: [[220, 520], [700, 520], [700, 570], [220, 570]]
preset:
  target_ratio: {red: 0.33, yellow: 0.33, normal: 0.34}
  tolerance: {relative: 0.20}  # ±20% tolerance
```

### Running the System

```bash
# Run real-time pipeline
kkplates run --config configs/default.yaml

# Run without video display (production)
kkplates run --config configs/default.yaml --no-video

# Check system status
kkplates status --config configs/default.yaml

# Export metrics for testing
kkplates export-metrics --config configs/default.yaml --seconds 60
```

## Data Collection & Training

### Collecting Training Data

1. **Recording Requirements**:
   - Record during normal operation with variety of plates
   - Each color should appear **≥150 times** in one session
   - Include plates with/without lids
   - Capture different lighting conditions

2. **Belt Phase Recording**:
   - Record 1-2 minute video walking along the belt
   - Capture printed number sequence (1, 2, 3, ...) for cycle tracking

### Data Labeling

Use the labeling protocol in `notebooks/01_label_protocol.ipynb`:
- Label plates with bounding boxes
- Add `color` attribute: red/yellow/normal
- Use LabelMe or CVAT for annotation

### Training Models

1. **Prepare Dataset**:
```bash
python scripts/prepare_data.py data/raw data/yolo --format labelme
```

2. **Train Detector**:
```bash
python src/kkplates/detect/train.py data/yolo --epochs 100
```

3. **Train Color Classifier**:
```bash
python src/kkplates/classify/train_color_cnn.py data/colors --epochs 50
```

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌────────────┐
│ RTSP Stream │────▶│ Detector │────▶│  Tracker   │
└─────────────┘     └──────────┘     └────────────┘
                                            │
                    ┌──────────┐            ▼
                    │ Classifier│◀────┌────────────┐
                    └──────────┘      │  Crossing  │
                                      │  Detector  │
                                      └────────────┘
                                            │
                    ┌──────────┐            ▼
                    │ Power BI │◀────┌────────────┐
                    └──────────┘      │  Metrics   │
                                      │ Aggregator │
                    ┌──────────┐      └────────────┘
                    │   Logs   │◀───────────┘
                    └──────────┘
```

## Metrics & Monitoring

### Real-time Metrics
- **Current plates on belt**: IN count - OUT count
- **Plates per minute**: Sliding 60s window
- **Color distribution**: Real-time ratios vs preset targets

### Alerts
- Triggered when color ratios deviate beyond tolerance
- Configurable violation threshold (default: 2 consecutive windows)
- Sent to Power BI and logged

### JSON Log Format
```json
{
  "timestamp": 1234567890.123,
  "event_type": "crossing",
  "track_id": 42,
  "direction": "in",
  "color": "red",
  "position": [250, 400]
}
```

## Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_roi_crossing.py -v

# Run with coverage
pytest --cov=kkplates tests/
```

## Troubleshooting

### Common Issues

1. **Blurry Plates**:
   - Check camera focus
   - Ensure adequate lighting
   - Reduce conveyor speed if possible

2. **Missed Detections**:
   - Adjust `detector.conf_thres` (lower = more sensitive)
   - Ensure ROI covers entire crossing area
   - Check for occlusions

3. **Wrong Color Classification**:
   - Recalibrate HSV thresholds using ROI tool
   - Collect more training data for CNN
   - Check lighting consistency

4. **Double Counting**:
   - Increase `min_travel_distance` in crossing detector
   - Adjust `debounce_frames` parameter
   - Ensure ROI zones don't overlap

### Performance Tuning

- **GPU**: Ensure CUDA is available: `nvidia-smi`
- **Frame Stride**: Increase `frame_stride` to process fewer frames
- **Model Size**: Use smaller YOLO variant (yolov8n → yolov8s)

## API Reference

### Configuration Schema

See `src/kkplates/config.py` for full schema with validation.

### Power BI Integration

Metrics are sent as JSON to the configured endpoint:
```json
{
  "timestamp": 1234567890,
  "total_in": 150,
  "total_out": 147,
  "current_on_belt": 3,
  "plates_per_minute": 12.5,
  "red_ratio": 0.35,
  "yellow_ratio": 0.32,
  "normal_ratio": 0.33
}
```

## Development

### Project Structure
```
kk-plates/
├── src/kkplates/       # Main package
├── configs/            # Configuration files
├── data/              # Sample data and models
├── tests/             # Test suite
├── scripts/           # Utility scripts
└── notebooks/         # Jupyter notebooks
```

### Contributing

1. Install dev dependencies: `pip install -e ".[dev]"`
2. Run linters: `make lint`
3. Run tests: `make test`
4. Format code: `black src tests`

## License

Proprietary - Kichi-Kichi Restaurants

## Support

For issues or questions, contact: engineering@kichi-kichi.com