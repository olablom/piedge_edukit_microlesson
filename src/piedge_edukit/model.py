#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# model.py - MobileNetV2 model definition and ONNX export for PiEdge EduKit

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2-based image classifier for edge deployment."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()

        # Load MobileNetV2 with optional pretrained weights.
        # Newer torchvision prefers explicit weights over deprecated 'pretrained'.
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
        except Exception:
            # Fallback for older torchvision versions where 'weights' may not exist
            self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.backbone.last_channel, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def get_feature_extractor(self):
        """Get feature extractor without classifier."""
        return nn.Sequential(*list(self.backbone.features.children()))


def create_model(
    num_classes: int = 3, pretrained: bool = True
) -> MobileNetV2Classifier:
    """Create MobileNetV2 classifier model."""
    model = MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
    opset_version: int = 17,  # Hard-locked to opset 17
    dynamic_batch: bool = True,
) -> bool:
    """Export PyTorch model to ONNX format."""

    model.eval()

    # Move model to CPU for ONNX export
    model = model.cpu()

    # Create dummy input on CPU
    dummy_input = torch.randn(input_shape)

    # Dynamic batch size if requested
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        print(f"[OK] Model exported to {output_path} (opset {opset_version})")
        return True

    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False


def verify_onnx_model(
    onnx_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64)
) -> bool:
    """Verify ONNX model with ONNX Runtime."""

    if not Path(onnx_path).exists():
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return False

    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Test with ONNX Runtime
        session = ort.InferenceSession(onnx_path)

        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Run inference
        outputs = session.run(None, {"input": test_input})

        print(f"[OK] ONNX model verified successfully")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output dtype: {outputs[0].dtype}")

        return True

    except Exception as e:
        print(f"[ERROR] ONNX verification failed: {e}")
        return False


def get_model_info(onnx_path: str) -> dict:
    """Get information about ONNX model."""
    if not Path(onnx_path).exists():
        return {"error": "Model file not found"}

    try:
        # Load model
        onnx_model = onnx.load(onnx_path)

        # Get model info
        info = {
            "file_size_mb": Path(onnx_path).stat().st_size / (1024 * 1024),
            "opset_version": onnx_model.opset_import[0].version,
            "input_shape": None,
            "output_shape": None,
            "num_parameters": 0,
        }

        # Get input/output shapes
        for input_tensor in onnx_model.graph.input:
            if input_tensor.name == "input":
                shape = [
                    dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim
                ]
                info["input_shape"] = shape

        for output_tensor in onnx_model.graph.output:
            if output_tensor.name == "output":
                shape = [
                    dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim
                ]
                info["output_shape"] = shape

        # Count parameters (approximate)
        for node in onnx_model.graph.node:
            if node.op_type in ["Conv", "Gemm", "BatchNormalization"]:
                info["num_parameters"] += 1  # Simplified count

        return info

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test model creation and export
    print("Creating MobileNetV2 model...")
    model = create_model(num_classes=3, pretrained=True)

    print(f"Model created with {model.num_classes} classes")

    # Test export
    output_path = "models/test_model.onnx"
    Path("models").mkdir(exist_ok=True)

    success = export_to_onnx(model, output_path)
    if success:
        verify_onnx_model(output_path)
        info = get_model_info(output_path)
        print(f"Model info: {info}")
