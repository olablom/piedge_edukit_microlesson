#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# preprocess.py - Central preprocessing with hash validation for PiEdge EduKit

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class PreprocessConfig:
    """Central preprocessing configuration with hash validation."""

    def __init__(self, config_path: str = "models/preprocess_config.json"):
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default preprocessing parameters
        self.params = {
            "input_size": (64, 64),
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear",
            "color_space": "RGB",
        }

        self._load_or_create_config()

    def _load_or_create_config(self):
        """Load existing config or create new one with hash."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                data = json.load(f)
                self.params = data.get("params", self.params)
                self.config_hash = data.get("config_hash")
        else:
            self.config_hash = self._compute_hash()
            self._save_config()

    def _compute_hash(self) -> str:
        """Compute hash of preprocessing parameters."""
        # Create deterministic string representation
        params_str = json.dumps(self.params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    def _save_config(self):
        """Save configuration with hash."""
        config_data = {
            "params": self.params,
            "config_hash": self.config_hash,
            "description": "PiEdge EduKit preprocessing configuration",
            "opset": 17,  # Hard-locked ONNX opset version
        }
        with open(self.config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def validate_hash(self) -> bool:
        """Validate that current params match stored hash."""
        current_hash = self._compute_hash()
        return current_hash == self.config_hash

    def get_transform(self, is_training: bool = False) -> transforms.Compose:
        """Get preprocessing transform pipeline."""
        if not self.validate_hash():
            raise RuntimeError(
                f"Preprocessing hash mismatch! "
                f"Expected: {self.config_hash}, "
                f"Got: {self._compute_hash()}. "
                f"Delete {self.config_path} to reset."
            )

        transform_list = [
            transforms.Resize(self.params["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.params["mean"], std=self.params["std"]),
        ]

        if is_training:
            # Add data augmentation for training
            transform_list.insert(-2, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(-2, transforms.RandomRotation(degrees=10))

        return transforms.Compose(transform_list)

    def preprocess_image(
        self, image_path: str, is_training: bool = False
    ) -> torch.Tensor:
        """Preprocess single image with validation."""
        transform = self.get_transform(is_training)

        try:
            image = Image.open(image_path).convert(self.params["color_space"])
            tensor = transform(image)
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess {image_path}: {e}")

    def preprocess_batch(
        self, image_paths: list, is_training: bool = False
    ) -> torch.Tensor:
        """Preprocess batch of images."""
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path, is_training)
            tensors.append(tensor)
        return torch.stack(tensors)

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get expected input shape (C, H, W)."""
        return (3, *self.params["input_size"])


def validate_preprocessing_consistency(
    config_path: str = "models/preprocess_config.json",
) -> bool:
    """Validate that all preprocessing is consistent across the project."""
    config = PreprocessConfig(config_path)

    # Check if config exists and is valid
    if not config.config_path.exists():
        print(f"ERROR: No preprocessing config found at {config_path}")
        print("Run training first to create preprocessing configuration.")
        return False

    # Validate hash
    if not config.validate_hash():
        print(f"ERROR: Preprocessing hash mismatch!")
        print(f"Expected: {config.config_hash}")
        print(f"Current: {config._compute_hash()}")
        print("This indicates preprocessing parameters have changed.")
        print("Delete the config file to reset or ensure consistency.")
        return False

    print(f"[OK] Preprocessing configuration valid (hash: {config.config_hash})")
    return True


def reset_preprocessing_config(config_path: str = "models/preprocess_config.json"):
    """Reset preprocessing configuration (use with caution!)."""
    config_file = Path(config_path)
    if config_file.exists():
        config_file.unlink()
        print(f"Reset preprocessing config at {config_path}")
    else:
        print(f"No config file found at {config_path}")


if __name__ == "__main__":
    # Test preprocessing system
    config = PreprocessConfig()
    print(f"Preprocessing config hash: {config.config_hash}")
    print(f"Input shape: {config.get_input_shape()}")
    print(f"Validation: {config.validate_hash()}")

# --- Compatibility shim: export torchvision FakeData via our package ---
try:
    from torchvision.datasets import FakeData as _TVFakeData

    class FakeData(_TVFakeData):
        """Re-export torchvision.datasets.FakeData for notebooks that import
        it from piedge_edukit.preprocess."""

        pass
except Exception:  # torchvision might be unavailable in some envs
    FakeData = None

# --- Compatibility shim: expose torchvision FakeData and accept legacy args ---
try:
    from torchvision.datasets import FakeData as _TVFakeData

    class FakeData(_TVFakeData):
        """Compatibility wrapper around torchvision.datasets.FakeData.

        Accepts:
          - num_samples=... (alias for size=...)
          - image_size=64  (int) -> auto-expand to (3,64,64)
        """

        def __init__(
            self,
            size=None,
            num_samples=None,
            image_size=(3, 64, 64),
            num_classes=2,
            **kwargs,
        ):
            # Map legacy 'num_samples' to 'size'
            if num_samples is not None and size is None:
                size = num_samples
            # If user passed image_size as an int (e.g., 64), expand to (3,64,64)
            if isinstance(image_size, int):
                image_size = (3, image_size, image_size)
            # Provide a safe default transform to ensure tensors are returned
            if "transform" not in kwargs or kwargs["transform"] is None:
                kwargs["transform"] = transforms.ToTensor()
            # torchvision FakeData requires size and image_size
            super().__init__(
                size=size, image_size=image_size, num_classes=num_classes, **kwargs
            )
except Exception:
    FakeData = None

# --- Register alias into torchvision.datasets for notebooks expecting PEDFakeData ---
try:
    import torchvision.datasets as _tv_datasets  # type: ignore

    if FakeData is not None:
        setattr(_tv_datasets, "PEDFakeData", FakeData)
except Exception:
    # If torchvision is not available, silently skip aliasing
    pass
