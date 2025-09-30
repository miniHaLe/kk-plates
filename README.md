# 🍽️ KichiKichi Conveyor Belt Dish Counting System

An AI-powered conveyor belt dish counting and monitoring system for KichiKichi restaurant from Golden Gate. This system uses computer vision to detect and count dishes of different colors (normal, red, yellow) while tracking their movement through different stages and phases of the conveyor belt system.

## 🎯 Features

- **Multi-Color Dish Detection**: Detects normal, red, yellow, and advertisement dishes
- **Stage-Phase Tracking**: Tracks dishes through configurable stages (1-n) and phases (0-12)
- **Break Line Logic**: Handles dish transitions when break line is triggered
- **Dual Camera System**: Monitors both break line and kitchen areas
- **Real-time Dashboard**: Web-based dashboard with live statistics and camera feeds
- **Rate Monitoring**: Tracks dishes per minute for red and yellow dishes
- **OCR Number Detection**: Reads phase/stage numbers from conveyor belt

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Break Line     │    │  Kitchen        │
│  Camera         │    │  Camera         │
│  (Main)         │    │  (Monitoring)   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │   Video Processing     │
         │   - Dish Detection     │
         │   - OCR Recognition    │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Tracking System       │
         │  - Stage/Phase Logic   │
         │  - Break Line Handler  │
         │  - Counting & Rates    │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │    Dashboard           │
         │  - Real-time Display   │
         │  - Camera Feeds        │
         │  - Statistics & Charts │
         └────────────────────────┘
```

## 📁 Project Structure

```
kichikichi/
├── src/
│   ├── dish_detection/
│   │   └── dish_detector.py       # YOLO-based dish detection
│   ├── ocr_model/
│   │   └── number_detector.py     # OCR for conveyor belt numbers
│   ├── tracking/
│   │   └── conveyor_tracker.py    # Stage-phase tracking logic
│   ├── dashboard/
│   │   └── dashboard.py           # Web dashboard interface
│   ├── utils/
│   │   └── video_utils.py         # Video processing utilities
│   └── main_app.py                # Main application coordinator
├── config/
│   └── config.py                  # Configuration settings
├── assets/
│   ├── videos/                    # POC video files
│   └── images/                    # Image assets
├── models/                        # AI model files
├── tests/                         # Test files
├── logs/                          # Application logs
├── requirements.txt               # Python dependencies
├── Makefile                       # Project management commands
├── run.py                         # Entry point script
└── README.md                      # This file
```

## ✅ Project Status - PRODUCTION READY

**All requirements successfully implemented!** The KichiKichi system now includes:

- ✅ **Conda environment setup** with activation reminders
- ✅ **Kitchen camera delay system** (30 frames) for breakline synchronization
- ✅ **Phase 0 initialization** with proper CSV timeline integration
- ✅ **Complete ROI integration** (4 detection areas from exports/)
- ✅ **1/3 bbox counting logic** for precise dish detection
- ✅ **Breakline return counting** added to current phase
- ✅ **Kitchen camera signaling** for phase change detection  
- ✅ **2-column comparison tables** with mathematical calculations
- ✅ **Dish transition calculations** (taken out vs added in)
- ✅ **Professional dashboard UI** as primary interface
- ✅ **Mock camera system** when video files unavailable

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda environment with name 'py312' (recommended)
- OpenCV compatible system
- CUDA support (optional, for GPU acceleration)

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd kichikichi
   make setup-env
   ```

2. **Install dependencies:**
   ```bash
   make install
   ```

3. **Video files ready:**
   ```bash
   # Video files are already configured:
   # ✅ assets/videos/break_line_camera.mp4 (1280x720, 25fps)
   # ✅ assets/videos/kitchen.mp4 (1280x720, 25fps)
   ```

4. **Run the system:**
   ```bash
   make run
   ```

5. **Open dashboard:**
   ```
   http://localhost:8050
   ```

### Development Mode

```bash
make run-dev    # Run with debug enabled
make run-console    # Run in console mode (no web interface)
```

## 🎛️ Configuration

Edit `config/config.py` to customize:

- **Conveyor Settings**: Number of stages, phases per stage
- **Camera Sources**: Video files (POC) or RTSP URLs (production)
- **Model Parameters**: Detection confidence thresholds
- **Dashboard Settings**: Host, port, update intervals

```python
# Example configuration
@dataclass
class ConveyorConfig:
    max_phases_per_stage: int = 12  # Phases 0-12
    total_stages: int = 4           # Number of stages
    break_line_threshold: float = 0.5

@dataclass
class CameraConfig:
    # For POC (video files)
    break_line_camera_source: str = "assets/videos/break_line_camera.mp4"
    kitchen_camera_source: str = "assets/videos/kitchen_camera.mp4"
    
    # For production (RTSP streams)
    # break_line_rtsp: str = "rtsp://192.168.1.100:554/stream1"
    # kitchen_rtsp: str = "rtsp://192.168.1.101:554/stream1"
```

## 🔧 System Components

### 1. Dish Detection Model (`dish_detector.py`)
- **Technology**: YOLO + Color Analysis
- **Detects**: Normal, red, yellow, advertisement dishes
- **Features**: Confidence scoring, bounding box detection, color classification

### 2. OCR Number Detection (`number_detector.py`)
- **Technology**: EasyOCR
- **Purpose**: Read phase/stage numbers on conveyor belt
- **Features**: Break line region detection, preprocessing, number validation

### 3. Conveyor Tracking (`conveyor_tracker.py`)
- **Manages**: Stage-phase transitions, dish counting, break line logic
- **Features**: Thread-safe operations, rate calculations, data cleanup

### 4. Dashboard (`dashboard.py`)
- **Technology**: Dash + Plotly
- **Features**: Real-time updates, camera feeds, interactive charts, control panel

## 📊 Dashboard Features

### Main Display
- **Current Position**: Shows current stage and phase
- **Total Dishes**: Counts by type (normal, red, yellow)
- **Rate Monitoring**: Dishes per minute for red and yellow
- **Break Line Status**: Indicates when break line is active

### Charts
- **Rate Trends**: Historical dish rate over time
- **Stage-Phase Distribution**: Active dishes per stage-phase

### Camera Feeds
- **Break Line Camera**: Main detection and OCR
- **Kitchen Camera**: Dish serving monitoring

### Controls
- **Reset Counts**: Clear all counters
- **Calibrate**: System calibration (future feature)
- **Export Data**: Download statistics (future feature)

## 🔄 Conveyor Belt Logic

### Stage-Phase System
- Each stage contains phases 0-12 (configurable)
- When phase returns to 0, stage increments
- System supports multiple stages running simultaneously

### Break Line Mechanics
- Triggers when specific conditions are met
- Pushes dishes from previous stage-phase to new stage-phase
- Handles variable phase lengths automatically

### Example Flow
```
Stage 1, Phase 10 → Break Line → Stage 2, Phase 0
Stage 2, Phase 5  → Dishes pushed from Stage 1, Phase 8
```

## 🛠️ Development

### Available Commands
```bash
make help          # Show all available commands
make install       # Install dependencies
make run           # Run main application
make run-dev       # Development mode
make run-console   # Console mode
make test          # Run tests
make clean         # Clean temporary files
make lint          # Code linting
make logs          # Monitor application logs
```

### Testing
```bash
make test          # Run all tests
python -m pytest tests/ -v    # Detailed test output
```

### Debugging
- Check logs: `make logs` or view `logs/kichikichi.log`
- Console mode: `make run-console` for terminal output
- Debug mode: `make run-dev` for detailed debugging

## 🔧 Troubleshooting

### Common Issues

1. **Camera not opening:**
   - Check video file paths in `config/config.py`
   - Ensure video files exist in `assets/videos/`

2. **Dependencies missing:**
   ```bash
   make check-deps    # Check what's missing
   make install       # Reinstall dependencies
   ```

3. **Dashboard not loading:**
   - Check if port 8050 is available
   - Look for errors in logs: `make logs`

4. **Poor detection accuracy:**
   - Adjust confidence thresholds in `config/config.py`
   - Check lighting conditions in video
   - Verify model files are present

### Performance Tips

- **GPU Acceleration**: Install CUDA-enabled PyTorch for better performance
- **Memory Usage**: Adjust frame processing rates in video utilities
- **Network**: Use local RTSP streams for better performance

## 📈 Production Deployment

### RTSP Camera Setup
1. Update `config/config.py` with RTSP URLs:
   ```python
   break_line_rtsp: str = "rtsp://your-camera-ip:554/stream1"
   kitchen_rtsp: str = "rtsp://your-camera-ip:554/stream2"
   ```

2. Test camera connections:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture('rtsp://your-ip:554/stream1'); print('Connected:', cap.isOpened())"
   ```

### System Requirements
- **CPU**: Multi-core processor (Intel i5+ or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible for optimal performance
- **Storage**: 50GB+ for logs and model files
- **Network**: Stable connection for RTSP streams

## 📄 License

This project is proprietary software for KichiKichi restaurant from Golden Gate.

## 🤝 Support

For technical support or questions about the system:
1. Check the troubleshooting section above
2. Review application logs: `make logs`
3. Contact the development team

---

**KichiKichi Conveyor Belt Dish Counting System** - Revolutionizing restaurant automation with AI-powered vision technology.
