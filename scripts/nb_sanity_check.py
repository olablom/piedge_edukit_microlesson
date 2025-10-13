#!/usr/bin/env python3
"""
PiEdge EduKit - ONNX Sanity Check
Verifies that ONNX models can be loaded and run correctly
"""

import os
import sys
from pathlib import Path

def check_onnx_model(model_path):
    """Check if ONNX model exists and can be loaded"""
    if not Path(model_path).exists():
        print(f"❌ ONNX model not found: {model_path}")
        return False
    
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        # Load and check model
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX model loaded and validated: {model_path}")
        
        # Test inference
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        output = outputs[0]
        
        print(f"✅ ONNX inference successful: input {dummy_input.shape} -> output {output.shape} {output.dtype}")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ ONNX model error: {e}")
        return False

def main():
    print("🔍 PiEdge EduKit - ONNX Sanity Check")
    print("=" * 50)
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found. Run training first.")
        sys.exit(1)
    
    # Check for ONNX model
    onnx_model = models_dir / "model.onnx"
    if not onnx_model.exists():
        print("❌ ONNX model not found. Run training first.")
        sys.exit(1)
    
    # Check ONNX model
    if check_onnx_model(onnx_model):
        print("\n🎉 ONNX sanity check passed!")
    else:
        print("\n⚠️  ONNX sanity check failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
