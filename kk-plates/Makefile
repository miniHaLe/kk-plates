.PHONY: setup lint test run benchmark clean help

help:
	@echo "Available targets:"
	@echo "  setup      - Install dependencies and set up environment"
	@echo "  lint       - Run code linters (ruff, black, mypy)"
	@echo "  test       - Run test suite"
	@echo "  run        - Start the plate counting pipeline"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  clean      - Clean up generated files"

setup:
	pip install -e ".[dev]"
	# Download YOLOv8n base model if not present
	@if [ ! -f "data/models/yolov8n.pt" ]; then \
		echo "Downloading YOLOv8n base model..."; \
		mkdir -p data/models; \
		python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('data/models/yolov8n.pt')"; \
	fi

lint:
	ruff check src tests
	black --check src tests
	mypy src

test:
	pytest -v --maxfail=1 --cov=kkplates tests/

run:
	kkplates run --config configs/default.yaml

benchmark:
	@echo "Running performance benchmarks..."
	python -m pytest tests/test_benchmark.py -v

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete