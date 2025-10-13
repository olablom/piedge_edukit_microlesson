#!/usr/bin/env python3
"""
PiEdge EduKit - Preflight Check
Validates environment and dependencies before running notebooks
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python version {version.major}.{version.minor} is too old. Need Python 3.8+")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_jupyter_kernel():
    """Check if Jupyter kernel is available"""
    try:
        result = subprocess.run([sys.executable, "-m", "jupyter", "kernelspec", "list"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Jupyter kernels available")
            return True
        else:
            print("❌ Jupyter kernels not available")
            return False
    except Exception as e:
        print(f"❌ Error checking Jupyter kernels: {e}")
        return False

def main():
    print("🔍 PiEdge EduKit - Preflight Check")
    print("=" * 50)
    
    all_good = True
    
    # Check Python version
    print("\n🐍 Python Environment:")
    all_good &= check_python_version()
    
    # Check essential packages
    print("\n📦 Essential Packages:")
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("jupyter", "jupyter"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    ]
    
    for package, import_name in packages:
        all_good &= check_package(package, import_name)
    
    # Check essential files
    print("\n📁 Essential Files:")
    files = [
        ("notebooks/00_run_everything.ipynb", "Main notebook"),
        ("notebooks/01_training_and_export.ipynb", "Training notebook"),
        ("notebooks/02_latency_benchmark.ipynb", "Benchmark notebook"),
        ("notebooks/03_quantization.ipynb", "Quantization notebook"),
        ("notebooks/04_evaluate_and_verify.ipynb", "Evaluation notebook"),
        ("src/piedge_edukit/__init__.py", "Package init"),
        ("requirements.txt", "Requirements file"),
        ("pyproject.toml", "Project config"),
    ]
    
    for filepath, description in files:
        all_good &= check_file_exists(filepath, description)
    
    # Check Jupyter
    print("\n🌐 Jupyter Environment:")
    all_good &= check_jupyter_kernel()
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All preflight checks passed! Ready to run PiEdge EduKit.")
        print("📖 Start with: python main.py")
    else:
        print("⚠️  Some preflight checks failed. Please fix the issues above.")
        print("💡 Try running: bash scripts/setup_venv.sh")
        sys.exit(1)

if __name__ == "__main__":
    main()
