"""Train YOLOv8 detector for plate detection."""

import sys
from pathlib import Path
from typing import Optional
import yaml
from ultralytics import YOLO
import structlog

logger = structlog.get_logger()


def create_dataset_yaml(data_dir: Path, output_path: Path) -> None:
    """Create dataset.yaml for YOLO training."""
    dataset_config = {
        "path": str(data_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "plate"
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(dataset_config, f)
    
    logger.info("Created dataset config", path=str(output_path))


def train_detector(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None
) -> Path:
    """
    Train YOLOv8 detector.
    
    Args:
        data_dir: Directory with train/val splits
        output_dir: Output directory for models
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Image size for training
        device: Device to use (None for auto)
        
    Returns:
        Path to best model
    """
    # Create dataset.yaml if not exists
    dataset_yaml = data_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        create_dataset_yaml(data_dir, dataset_yaml)
    
    # Load base model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        name="kk_plates_detector",
        project=str(output_dir),
        exist_ok=True,
        
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Other settings
        save=True,
        save_period=-1,
        cache=True,
        workers=8,
        amp=True,
        close_mosaic=10,
        resume=False,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    # Get best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    # Copy to output directory with clear name
    final_path = output_dir / "kk_plates_detector.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, final_path)
        logger.info("Saved best model", path=str(final_path))
    
    return final_path


def export_to_onnx(model_path: Path, output_path: Path, imgsz: int = 640) -> Path:
    """Export model to ONNX format."""
    model = YOLO(str(model_path))
    
    # Export to ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False
    )
    
    # Move to desired location
    if Path(onnx_path).exists():
        import shutil
        shutil.move(onnx_path, output_path)
        logger.info("Exported to ONNX", path=str(output_path))
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train plate detector")
    parser.add_argument("data_dir", type=Path, help="Dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--export-onnx", action="store_true")
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_detector(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_path = args.output_dir / "kk_plates_detector.onnx"
        export_to_onnx(model_path, onnx_path)