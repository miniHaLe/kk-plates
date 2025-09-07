"""Train CNN color classifier for plates."""

import sys
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import structlog

logger = structlog.get_logger()


class PlateColorDataset(Dataset):
    """Dataset for plate color classification."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], 
                 transform_size: Tuple[int, int] = (64, 64)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_size = transform_size
        self.augment = True
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Resize
        img = cv2.resize(img, self.transform_size)
        
        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            # Random brightness/contrast
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
        
        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor, self.labels[idx]


class ColorCNN(nn.Module):
    """Lightweight CNN for color classification."""
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def prepare_dataset(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """Load dataset from directory structure."""
    color_map = {"red": 0, "yellow": 1, "normal": 2}
    
    image_paths = []
    labels = []
    
    for color_name, label in color_map.items():
        color_dir = data_dir / color_name
        if not color_dir.exists():
            logger.warning(f"Color directory not found: {color_dir}")
            continue
        
        for img_path in color_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(label)
        
        for img_path in color_dir.glob("*.png"):
            image_paths.append(img_path)
            labels.append(label)
    
    logger.info(f"Loaded dataset", 
                total=len(image_paths),
                red=labels.count(0),
                yellow=labels.count(1),
                normal=labels.count(2))
    
    return image_paths, labels


def train_color_classifier(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Optional[str] = None
) -> Path:
    """Train color classifier."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare dataset
    image_paths, labels = prepare_dataset(data_dir)
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = PlateColorDataset(train_paths, train_labels)
    val_dataset = PlateColorDataset(val_paths, val_labels)
    val_dataset.augment = False
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    
    # Initialize model
    model = ColorCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop
    best_val_acc = 0
    best_model_path = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs}",
                   train_loss=f"{avg_train_loss:.4f}",
                   train_acc=f"{train_acc:.2f}%",
                   val_loss=f"{avg_val_loss:.4f}",
                   val_acc=f"{val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / f"color_cnn_epoch{epoch+1}_acc{val_acc:.1f}.pth"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': ['red', 'yellow', 'normal']
            }, best_model_path)
            logger.info(f"Saved best model", path=str(best_model_path))
    
    # Save final model
    final_path = output_dir / "kk_color_cnn.pth"
    if best_model_path and best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, final_path)
    
    return final_path


def export_to_onnx(model_path: Path, output_path: Path, input_size: int = 64) -> Path:
    """Export PyTorch model to ONNX."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    model = ColorCNN(num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    logger.info("Exported to ONNX", path=str(output_path))
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train color classifier")
    parser.add_argument("data_dir", type=Path, help="Dataset directory with red/yellow/normal subdirs")
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-onnx", action="store_true")
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_color_classifier(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_path = args.output_dir / "kk_color_cnn.onnx"
        export_to_onnx(model_path, onnx_path)