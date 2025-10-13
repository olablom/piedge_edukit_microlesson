#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# labels.py - Label management and validation for PiEdge EduKit

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class LabelManager:
    """Manages class labels with index↔class validation."""

    def __init__(self, labels_path: str = "models/labels.json"):
        self.labels_path = Path(labels_path)
        self.labels_dir = self.labels_path.parent
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_or_create_labels()

    def _load_or_create_labels(self):
        """Load existing labels or create empty structure."""
        if self.labels_path.exists():
            with open(self.labels_path, "r") as f:
                data = json.load(f)
                self.classes = data.get("classes", [])
                self.class_to_idx = data.get("class_to_idx", {})
                # Convert string keys back to integers for idx_to_class
                idx_to_class_raw = data.get("idx_to_class", {})
                self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
        else:
            self._save_labels()

    def _save_labels(self):
        """Save labels to JSON file."""
        labels_data = {
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "num_classes": len(self.classes),
            "description": "PiEdge EduKit class labels",
        }
        with open(self.labels_path, "w") as f:
            json.dump(labels_data, f, indent=2)

    def set_classes(self, classes: List[str]):
        """Set class labels and create mappings."""
        if not classes:
            raise ValueError("Classes list cannot be empty")

        # Remove duplicates while preserving order
        self.classes = list(dict.fromkeys(classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        self._save_labels()
        print(f"Set {len(self.classes)} classes: {self.classes}")

    def add_class(self, class_name: str) -> int:
        """Add a new class and return its index."""
        if class_name in self.class_to_idx:
            return self.class_to_idx[class_name]

        idx = len(self.classes)
        self.classes.append(class_name)
        self.class_to_idx[class_name] = idx
        self.idx_to_class[idx] = class_name

        self._save_labels()
        return idx

    def get_class_index(self, class_name: str) -> int:
        """Get index for class name."""
        if class_name not in self.class_to_idx:
            raise KeyError(f"Class '{class_name}' not found. Available: {self.classes}")
        return self.class_to_idx[class_name]

    def get_class_name(self, index: int) -> str:
        """Get class name for index."""
        if index not in self.idx_to_class:
            raise KeyError(
                f"Index {index} not found. Available indices: {list(self.idx_to_class.keys())}"
            )
        return self.idx_to_class[index]

    def validate_labels(self) -> bool:
        """Validate label consistency."""
        if not self.classes:
            print("Error: No classes defined")
            return False

        # Check consistency
        if len(self.classes) != len(self.class_to_idx):
            print("Error: Classes and class_to_idx length mismatch")
            return False

        if len(self.classes) != len(self.idx_to_class):
            print("Error: Classes and idx_to_class length mismatch")
            return False

        # Check mappings
        for idx, cls in enumerate(self.classes):
            if self.class_to_idx.get(cls) != idx:
                print(f"Error: class_to_idx mapping incorrect for '{cls}'")
                return False

            if self.idx_to_class.get(idx) != cls:
                print(f"Error: idx_to_class mapping incorrect for index {idx}")
                return False

        print(f"[OK] Labels valid: {len(self.classes)} classes")
        return True

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)

    def get_classes(self) -> List[str]:
        """Get list of class names."""
        return self.classes.copy()

    def create_sample_labels(self, num_classes: int = 3) -> List[str]:
        """Create sample labels for testing."""
        sample_classes = [f"class_{i}" for i in range(num_classes)]
        self.set_classes(sample_classes)
        return sample_classes


def validate_labels_integrity(labels_path: str = "models/labels.json") -> bool:
    """Validate labels integrity before export, benchmark, and GPIO."""
    if not Path(labels_path).exists():
        print(f"Error: Labels file not found at {labels_path}")
        return False

    manager = LabelManager(labels_path)
    return manager.validate_labels()


def load_labels(labels_path: str = "models/labels.json") -> LabelManager:
    """Load labels manager from file."""
    if not Path(labels_path).exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    manager = LabelManager(labels_path)
    if not manager.validate_labels():
        raise ValueError("Labels validation failed")

    return manager


if __name__ == "__main__":
    # Test label management
    manager = LabelManager()

    # Create sample labels
    sample_classes = ["cat", "dog", "bird"]
    manager.set_classes(sample_classes)

    # Test validation
    print(f"Validation: {manager.validate_labels()}")
    print(f"Classes: {manager.get_classes()}")
    print(f"Num classes: {manager.get_num_classes()}")

    # Test mappings
    print(f"Index of 'dog': {manager.get_class_index('dog')}")
    print(f"Class at index 0: {manager.get_class_name(0)}")
