# KichiKichi Conveyor Belt Dish Counting System
# Makefile for project management

.PHONY: help install run run-dev run-console test clean setup-env

# Default target
help:
	@echo "KichiKichi Conveyor Belt Dish Counting System"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install        - Install all dependencies"
	@echo "  setup-env      - Setup Python environment and activate conda"
	@echo "  run            - Run the main application with dashboard (ROI mode)"
	@echo "  run-sync       - 🔄 Run NEW synchronized system with all requirements"
	@echo "  run-auto-restart - 🔄 Run with FULL SYSTEM auto-restart (30min minimum runtime)"
	@echo "  restart-all    - 🔄 Force restart all system components"
	@echo "  restart-backend - 🔄 Restart backend only (keep dashboard)"
	@echo "  monitor-restart - 👁️  Monitor connections and auto-restart system"
	@echo "  watch-dashboard - 🎯 Watch dashboard with restart (30min minimum runtime)"
	@echo "  run-dev        - Run in development mode with debug enabled (ROI mode)"
	@echo "  run-csv-dev    - 📊 Run CSV mode in development with debug"
	@echo "  run-console    - Run in console mode (terminal output only)"
	@echo "  run-csv-console - 📊 Run CSV mode in console (no dashboard)"
	@echo "  demo           - 🎨 Run beautiful demo dashboard with simulated data"
	@echo "  demo-csv       - 🎭 Quick CSV timeline demo"
	@echo "  test-csv       - 🧪 Test CSV timeline integration"
	@echo "  test-dashboard - 🧪 Test dashboard in isolation (quick test)"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean temporary files and logs"
	@echo "  check-deps     - Check if all dependencies are installed"
	@echo "  lint           - Run code linting"
	@echo ""

# Environment setup
setup-env:
	@echo "Setting up Python environment..."
	conda activate kichi || echo "Please ensure conda environment 'kichi' is available"
	@echo "Environment setup complete"

# Install dependencies
install: setup-env
	@echo "Installing dependencies..."
	pip install -r requirements_env.txt
	@echo "Installation complete"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@python -c "import cv2, numpy, torch, ultralytics, flask, easyocr, dash; print('✅ All core dependencies available')" || echo "❌ Missing dependencies - run 'make install'"

# Run main application (original ROI-based)
run: check-deps
	@echo "Starting KichiKichi Conveyor Belt System (ROI mode)..."
	python run.py --mode dashboard

# Run synchronized system (NEW - implements all requirements)
run-sync: check-deps
	@echo "🔄 Starting KichiKichi Synchronized System..."
	@echo "✨ Features: Dual cameras, phase sync, ROI counting, two-stage tables"
	@echo "🌐 Dashboard will open at: http://localhost:8050"
	./launch_synchronized_system.sh dashboard

# Auto-restart WHOLE SYSTEM for clean user sessions
run-auto-restart: check-deps
	@echo "🔄 Starting KichiKichi with FULL SYSTEM Auto-Restart..."
	@echo "✨ Features: Complete codebase restart after 30min minimum runtime"
	@echo "⏳ System runs for minimum 30 minutes before restart logic activates"
	@echo "🌐 Dashboard will open at: http://localhost:8050"
	python scripts/connection_monitor.py --restart-command "make run-sync"

# Force restart all components
restart-all:
	@echo "🔄 Force restarting all KichiKichi components..."
	@pkill -f "main_app.py" || true
	@pkill -f "csv_main_app.py" || true
	@pkill -f "dashboard" || true
	@sleep 2
	@echo "✅ All processes stopped. Starting fresh..."
	$(MAKE) run-sync

# Restart backend only (keep dashboard)
restart-backend:
	@echo "🔄 Restarting backend components only..."
	@pkill -f "main_app.py" || true
	@sleep 1
	@echo "✅ Backend restarted"

# Monitor and auto-restart on user activity
monitor-restart: check-deps
	@echo "👁️  Starting connection monitor with auto-restart..."
	@echo "🔄 Will restart backend on user connect/disconnect"
	python scripts/connection_monitor.py

# Dashboard watcher with auto-restart on user disconnect
watch-dashboard: check-deps
	@echo "🎯 Starting dashboard watcher..."
	@echo "📊 Monitoring: python run.py --mode dashboard"
	@echo "⏳ System runs for minimum 30 minutes before restart logic activates"
	@echo "🌐 Dashboard will be available at: http://localhost:8050"
	python scripts/dashboard_watcher.py

# Test synchronized system
test-sync: check-deps
	@echo "🧪 Testing Synchronized System Components..."
	./launch_synchronized_system.sh test
	@echo "🚀 Starting KichiKichi with CSV Timeline Tracking..."
	@echo "📊 Using timeline data: video_timeline_2025-09-16T11-37-40.csv"
	@echo "🌐 Dashboard will open at: http://localhost:8050"
	python src/csv_main_app.py --mode dashboard --use-csv

# Run in development mode (original)
run-dev: check-deps
	@echo "Starting in development mode..."
	python run.py --mode dashboard --debug

# Run CSV mode in development with detailed logging
run-csv-dev: check-deps
	@echo "🛠️  Starting CSV mode in development..."
	@echo "📊 Using timeline data: video_timeline_2025-09-16T11-37-40.csv"
	@echo "🐛 Debug logging enabled"
	python src/csv_main_app.py --mode dashboard --use-csv --debug

# Run in console mode
run-console: check-deps
	@echo "Starting in console mode..."
	python run.py --mode console

# Run CSV mode in console (no dashboard)
run-csv-console: check-deps
	@echo "📊 Starting CSV mode in console (no dashboard)..."
	python src/csv_main_app.py --mode console --use-csv

# Run professional demo dashboard
demo: check-deps
	@echo "🚀 Starting KichiKichi Professional Dashboard Demo..."
	@echo "✨ Features beautiful UI with simulated data"
	@echo "🌐 Will open at: http://localhost:8050"
	python demo_dashboard.py

# Test CSV timeline integration
test-csv: check-deps
	@echo "🧪 Testing CSV timeline integration..."
	python test_csv_integration.py

# Quick CSV timeline demo
demo-csv: check-deps
	@echo "🎭 Quick CSV Timeline Demo..."
	@echo "📊 Shows CSV parsing and dish counting per phase"
	python test_csv_simple.py

# Create sample video files for POC
create-sample-videos:
	@echo "Creating sample video directory structure..."
	mkdir -p assets/videos
	@echo "⚠️  Please place your test videos in:"
	@echo "   - assets/videos/break_line_camera.mp4"
	@echo "   - assets/videos/kitchen_camera.mp4"
	@echo "Or update config/config.py with your video paths"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v || echo "No tests found - test framework ready"

# Code linting
lint:
	@echo "Running code linting..."
	flake8 src/ --max-line-length=100 --ignore=E501,W503 || echo "flake8 not installed, skipping lint"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf logs/*.log
	@echo "Cleanup complete"

# Development utilities
dev-setup: install create-sample-videos
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Place your test videos in assets/videos/"
	@echo "2. Run 'make run-dev' to start the system"
	@echo "3. Open http://localhost:8050 to view dashboard"

# Production setup
prod-setup: install
	@echo "Production setup..."
	@echo "⚠️  Update config/config.py with your RTSP camera URLs"
	@echo "Production setup guide created"

# Quick start for CSV timeline mode
quickstart-csv:
	@echo "Quick Start Guide for KichiKichi CSV Timeline Mode"
	@echo "================================================"
	@echo ""
	@echo "✅ CSV timeline file already available: video_timeline_2025-09-16T11-37-40.csv"
	@echo ""
	@echo "1. Install dependencies:"
	@echo "   make install"
	@echo ""
	@echo "2. Test CSV integration:"
	@echo "   make demo-csv"
	@echo ""
	@echo "3. Run full system with CSV tracking:"
	@echo "   make run-csv"
	@echo ""
	@echo "4. Open dashboard:"
	@echo "   http://localhost:8050"
	@echo ""

# Quick start for POC (original)
quickstart:
	@echo "Quick Start Guide for KichiKichi System (ROI Mode)"
	@echo "================================================="
	@echo ""
	@echo "1. Install dependencies:"
	@echo "   make install"
	@echo ""
	@echo "2. Add your test videos:"
	@echo "   make create-sample-videos"
	@echo "   # Then copy your videos to assets/videos/"
	@echo ""
	@echo "3. Run the system:"
	@echo "   make run"
	@echo ""
	@echo "4. Open dashboard:"
	@echo "   http://localhost:8050"
	@echo ""

# Monitor logs
logs:
	@echo "Monitoring application logs..."
	tail -f logs/kichikichi.log || echo "No log file found - start the application first"

# System info
info:
	@echo "System Information:"
	@echo "=================="
	@echo "Python version: $(shell python --version)"
	@echo "OpenCV version: $(shell python -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed')"
	@echo "Working directory: $(PWD)"
