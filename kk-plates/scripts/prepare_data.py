#!/usr/bin/env python3
"""Prepare dataset for training from various annotation formats."""

import json
import yaml
from pathlib import Path
import shutil
import random
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import argparse


def cvat_to_yolo(cvat_json_path: Path, output_dir: Path, img_width: int, img_height: int):
    """Convert CVAT JSON annotations to YOLO format."""
    with open(cvat_json_path) as f:
        data = json.load(f)
    
    annotations = {}
    
    # Process each image
    for image in data.get('images', []):
        img_id = image['id']
        img_name = image['file_name']
        annotations[img_name] = []
        
        # Find annotations for this image
        for ann in data.get('annotations', []):
            if ann['image_id'] == img_id:
                # Get bbox in CVAT format (x, y, w, h)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                cx = (x + w/2) / img_width
                cy = (y + h/2) / img_height
                nw = w / img_width
                nh = h / img_height
                
                # Get class (assuming single class 'plate')
                class_id = 0
                
                annotations[img_name].append(f"{class_id} {cx} {cy} {nw} {nh}")
    
    return annotations


def labelme_to_yolo(labelme_json_path: Path) -> List[str]:
    """Convert LabelMe JSON annotation to YOLO format."""
    with open(labelme_json_path) as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    yolo_annotations = []
    
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Calculate center and dimensions
            cx = (x1 + x2) / 2 / img_width
            cy = (y1 + y2) / 2 / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            
            # Get color attribute
            color = shape.get('label', 'normal').lower()
            # For detection, we use single class
            class_id = 0
            
            yolo_annotations.append(f"{class_id} {cx} {cy} {w} {h}")
    
    return yolo_annotations


def create_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    annotation_format: str,
    train_ratio: float = 0.8
):
    """Create YOLO format dataset with train/val split."""
    
    # Create output directories
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val")
    
    # Process files
    for split, files in [('train', train_files), ('val', val_files)]:
        for img_path in files:
            # Copy image
            dst_img = output_dir / 'images' / split / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Convert annotation
            if annotation_format == 'labelme':
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    yolo_annotations = labelme_to_yolo(json_path)
                    
                    # Write YOLO label file
                    label_path = output_dir / 'labels' / split / img_path.stem + '.txt'
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
            
            elif annotation_format == 'cvat':
                # CVAT format would be handled differently
                # This is a placeholder for CVAT conversion
                pass
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'plate'
        }
    }
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"Dataset created at: {output_dir}")
    print(f"Config file: {output_dir / 'dataset.yaml'}")


def create_color_dataset(
    input_dir: Path,
    output_dir: Path,
    annotation_format: str
):
    """Create color classification dataset from annotations."""
    
    # Create color directories
    for color in ['red', 'yellow', 'normal']:
        (output_dir / color).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in input_dir.glob('*.jpg'):
        if annotation_format == 'labelme':
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                
                img = cv2.imread(str(img_path))
                
                for i, shape in enumerate(data.get('shapes', [])):
                    if shape['shape_type'] == 'rectangle':
                        # Get color from label or attributes
                        color = shape.get('label', 'normal').lower()
                        if color not in ['red', 'yellow', 'normal']:
                            color = 'normal'
                        
                        # Extract ROI
                        points = shape['points']
                        x1, y1 = map(int, points[0])
                        x2, y2 = map(int, points[1])
                        
                        roi = img[y1:y2, x1:x2]
                        if roi.size > 0:
                            # Save cropped plate
                            crop_path = output_dir / color / f"{img_path.stem}_crop{i}.jpg"
                            cv2.imwrite(str(crop_path), roi)
    
    # Print statistics
    for color in ['red', 'yellow', 'normal']:
        count = len(list((output_dir / color).glob('*.jpg')))
        print(f"{color}: {count} samples")


def analyze_dataset(dataset_dir: Path):
    """Analyze dataset and print statistics."""
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'images_per_split': {'train': 0, 'val': 0},
        'avg_objects_per_image': 0
    }
    
    for split in ['train', 'val']:
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split
        
        if img_dir.exists():
            images = list(img_dir.glob('*'))
            stats['images_per_split'][split] = len(images)
            stats['total_images'] += len(images)
            
            # Count annotations
            for img in images:
                label_file = label_dir / (img.stem + '.txt')
                if label_file.exists():
                    with open(label_file) as f:
                        annotations = f.readlines()
                        stats['total_annotations'] += len(annotations)
    
    if stats['total_images'] > 0:
        stats['avg_objects_per_image'] = stats['total_annotations'] / stats['total_images']
    
    print("\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Train images: {stats['images_per_split']['train']}")
    print(f"Val images: {stats['images_per_split']['val']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Avg objects per image: {stats['avg_objects_per_image']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("input_dir", type=Path, help="Input directory with images and annotations")
    parser.add_argument("output_dir", type=Path, help="Output directory for YOLO dataset")
    parser.add_argument("--format", choices=['labelme', 'cvat'], default='labelme',
                       help="Annotation format")
    parser.add_argument("--task", choices=['detection', 'classification'], default='detection',
                       help="Task type")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Train/val split ratio")
    parser.add_argument("--analyze", action='store_true',
                       help="Analyze dataset after creation")
    
    args = parser.parse_args()
    
    if args.task == 'detection':
        create_yolo_dataset(
            args.input_dir,
            args.output_dir,
            args.format,
            args.train_ratio
        )
    else:
        create_color_dataset(
            args.input_dir,
            args.output_dir,
            args.format
        )
    
    if args.analyze and args.task == 'detection':
        analyze_dataset(args.output_dir)


if __name__ == "__main__":
    import cv2  # Import here to avoid issues if not needed
    main()