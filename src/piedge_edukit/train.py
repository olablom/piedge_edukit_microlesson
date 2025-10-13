#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# train.py - Training script for MobileNetV2 classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import json
import csv
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from .model import create_model, export_to_onnx, verify_onnx_model
from .preprocess import PreprocessConfig, validate_preprocessing_consistency
from .labels import LabelManager, validate_labels_integrity


def ensure_dir(p: str) -> Path:
    """Ensure directory exists, create if needed."""
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_training_metrics(out_dir: Path, train_losses: list, val_losses: list, train_accs: list, val_accs: list):
    """Save training metrics as CSV, JSON and PNG plot."""
    ensure_dir(str(out_dir))

    # Prepare data
    epochs = list(range(1, len(train_losses) + 1))
    history = []
    for i, epoch in enumerate(epochs):
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses[i],
                "val_loss": val_losses[i],
                "train_acc": train_accs[i],
                "val_acc": val_accs[i],
            }
        )

    # Save CSV
    csv_path = out_dir / "training_log.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        w.writeheader()
        for row in history:
            w.writerow(row)

    # Save JSON
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))

    # Create PNG plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # ---- Loss ----
    axs[0].plot(epochs, train_losses, marker="o", label="train loss")
    axs[0].plot(epochs, val_losses, marker="o", label="val loss")

    yvals = list(train_losses) + list(val_losses)
    ymin, ymax = min(yvals), max(yvals)
    pad = max(1e-3, 0.02 * (ymax - ymin or 1.0))
    axs[0].set_ylim(ymin - pad, ymax + pad)

    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training & Validation Loss")
    axs[0].legend()

    # ---- Accuracy ----
    axs[1].plot(epochs, train_accs, marker="o", label="train acc")
    axs[1].plot(epochs, val_accs, marker="o", label="val acc")

    avals = list(train_accs) + list(val_accs)
    amin, amax = min(avals), max(avals)
    apad = max(0.2, 0.02 * (amax - amin or 1.0))
    axs[1].set_ylim(amin - apad, amax + apad)

    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("Training & Validation Accuracy")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Training metrics saved to {out_dir}")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {out_dir / 'training_log.json'}")
    print(f"  - Plot: {out_dir / 'training_curves.png'}")


class FakeImageDataset(Dataset):
    """Fake dataset using torchvision.datasets.FakeData for testing."""

    def __init__(self, size: int = 100, num_classes: int = 2, transform=None):
        self.fake_data = datasets.FakeData(
            size=size,
            image_size=(3, 64, 64),
            num_classes=num_classes,
            transform=transform,
        )
        self.classes = [f"class{i}" for i in range(num_classes)]

    def __len__(self):
        return len(self.fake_data)

    def __getitem__(self, idx):
        image, label = self.fake_data[idx]
        # Convert numeric label to string label
        string_label = f"class{label}"
        return image, string_label


class ImageDataset(Dataset):
    """Custom dataset for image classification."""

    def __init__(self, data_dir: str, transform=None, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training

        # Find all image files
        self.image_paths = []
        self.labels = []

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # Scan for images - handle both flat structure (data/class/) and train/val structure (data/train/class/)
        if (self.data_dir / "train").exists():
            # Train/val structure
            split_dir = self.data_dir / ("train" if is_training else "val")
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_path in class_dir.iterdir():
                            if img_path.suffix.lower() in extensions:
                                self.image_paths.append(str(img_path))
                                self.labels.append(class_name)
        else:
            # Flat structure
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in extensions:
                            self.image_paths.append(str(img_path))
                            self.labels.append(class_name)

        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")

        print(f"[OK] Found {len(self.image_paths)} images in {len(set(self.labels))} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        from PIL import Image

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class Trainer:
    """MobileNetV2 trainer with deterministic settings."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "models",
        seed: int = 42,
        use_fakedata: bool = False,
        use_pretrained: bool = True,
        ci_fast: bool = False,
        fake_train_size: int = 100,
        fake_val_size: int = 20,
        num_workers: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_fakedata = use_fakedata
        self.use_pretrained = use_pretrained
        self.ci_fast = ci_fast
        # shrink dataset and workers in CI fast mode (only relevant for FakeData)
        if self.ci_fast and self.use_fakedata:
            fake_train_size = 64
            fake_val_size = 16
            num_workers = 0
        self.fake_train_size = fake_train_size
        self.fake_val_size = fake_val_size
        self.num_workers = num_workers

        # Set seeds for reproducibility
        self._set_seeds(seed)

        # Initialize components
        self.preprocess_config = PreprocessConfig()
        self.label_manager = LabelManager()

        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.use_fakedata:
            print("[INFO] Using FakeData for training (no real images)")

        # Training hyperparameters
        self.batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.weight_decay = 1e-4

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, LabelManager]:
        """Prepare training and validation data."""

        # Create datasets
        train_transform = self.preprocess_config.get_transform(is_training=True)
        val_transform = self.preprocess_config.get_transform(is_training=False)

        if self.use_fakedata:
            # Use FakeData for testing
            train_dataset = FakeImageDataset(size=self.fake_train_size, num_classes=2, transform=train_transform)
            val_dataset = FakeImageDataset(size=self.fake_val_size, num_classes=2, transform=val_transform)
            unique_classes = train_dataset.classes
        else:
            # Use real image data
            train_dataset = ImageDataset(self.data_dir, train_transform, is_training=True)
            val_dataset = ImageDataset(self.data_dir, val_transform, is_training=False)
            unique_classes = sorted(list(set(train_dataset.labels)))

        # Set up label manager
        self.label_manager.set_classes(unique_classes)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, self.label_manager

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)

            # Convert string labels to indices
            label_indices = torch.tensor([self.label_manager.get_class_index(label) for label in labels]).to(
                self.device
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, label_indices)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_indices.size(0)
            correct += (predicted == label_indices).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        return avg_loss, accuracy

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)

                # Convert string labels to indices
                label_indices = torch.tensor([self.label_manager.get_class_index(label) for label in labels]).to(
                    self.device
                )

                outputs = model(images)
                loss = criterion(outputs, label_indices)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label_indices.size(0)
                correct += (predicted == label_indices).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy

    def train(self) -> nn.Module:
        """Train MobileNetV2 model."""

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        print("Preparing data...")
        train_loader, val_loader, label_manager = self.prepare_data()

        # Create model (avoid downloading pretrained weights for fakedata or when --no-pretrained is set)
        num_classes = label_manager.get_num_classes()
        model = create_model(num_classes=num_classes, pretrained=self.use_pretrained)
        model = model.to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        print(f"Training model with {num_classes} classes...")
        print(f"Classes: {label_manager.get_classes()}")

        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # Initialize CSV logging
        csv_path = self.output_dir / "train_metrics.csv"
        csv_header_written = False

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)

            # Update learning rate
            scheduler.step()

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log to CSV
            import csv

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not csv_header_written:
                    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                    csv_header_written = True
                writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.output_dir / "model_best.pth")

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Save training metrics and plots
        _save_training_metrics(Path("reports"), train_losses, val_losses, train_accs, val_accs)

        # Load best model
        model.load_state_dict(torch.load(self.output_dir / "model_best.pth"))

        return model

    def export_model(self, model: nn.Module) -> bool:
        """Export trained model to ONNX."""

        print("Exporting model to ONNX...")

        # Export to ONNX
        onnx_path = self.output_dir / "model.onnx"
        success = export_to_onnx(
            model,
            str(onnx_path),
            input_shape=(1, 3, 64, 64),
            opset_version=17,
            dynamic_batch=True,
        )

        if success:
            # Verify ONNX model
            verify_onnx_model(str(onnx_path))

            # Validate preprocessing and labels after export
            if not validate_preprocessing_consistency():
                print("[ERROR] Preprocessing validation failed")
                return False

            if not validate_labels_integrity():
                print("[ERROR] Labels validation failed")
                return False

            # Save training info
            training_info = {
                "num_classes": self.label_manager.get_num_classes(),
                "classes": self.label_manager.get_classes(),
                "input_shape": [1, 3, 64, 64],
                "opset_version": 17,
                "dynamic_batch": True,
            }

            with open(self.output_dir / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)

            print(f"[OK] Model exported successfully to {onnx_path}")
            return True

        return False


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 classifier")
    parser.add_argument("--data-path", help="Path to training data (not needed with --fakedata)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (1=Smoke Test, 5=Pretty Demo)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (256=Smoke Test, 16=Pretty Demo)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fakedata", action="store_true", help="Use FakeData instead of real images")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained weights (speeds up CI and offline runs)",
    )
    parser.add_argument(
        "--ci-fast",
        action="store_true",
        help="Tiny run for CI (skip pretrained, tiny fake dataset, 1 epoch)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()

    # Validate arguments
    if not args.fakedata and not args.data_path:
        parser.error("--data-path is required unless --fakedata is used")

    # Create trainer
    trainer = Trainer(
        data_dir=args.data_path or "dummy",
        output_dir=args.output_dir,
        seed=args.seed,
        use_fakedata=args.fakedata,
        use_pretrained=(not args.fakedata and not args.no_pretrained and not args.ci_fast),
        ci_fast=args.ci_fast,
        num_workers=args.num_workers,
    )

    # Update training parameters
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.lr

    # Clamp CI fast mode params
    if args.ci_fast:
        args.epochs = min(args.epochs, 1)
        args.batch_size = max(args.batch_size, 128)

    # Train model
    model = trainer.train()

    # Export to ONNX
    success = trainer.export_model(model)

    if success:
        print("[OK] Training and export completed successfully!")
    else:
        print("[ERROR] Export failed!")
        exit(1)


if __name__ == "__main__":
    main()
