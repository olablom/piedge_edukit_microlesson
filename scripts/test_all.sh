#!/usr/bin/env bash
set -euo pipefail

# PiEdge EduKit - Complete Test Suite
# Runs: Local dev → Pretty demo → Clean-room → Artifact check

echo "🧪 PiEdge EduKit - Complete Test Suite"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS: $1${NC}"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL: $1${NC}"
    ((TESTS_FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN: $1${NC}"
}

# Test A: Local dev smoke test
echo ""
echo "🔬 Test A: Local Dev Smoke Test"
echo "-------------------------------"

# Clean start
if [ -d ".venv" ]; then
    echo "Cleaning existing .venv..."
    rm -rf .venv
fi

# Create venv
echo "Creating fresh .venv..."
python -m venv .venv
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

# Install
echo "Installing requirements..."
pip install -r requirements.txt >/dev/null 2>&1
pip install -e . >/dev/null 2>&1

# Clean artifacts
rm -rf models reports progress
mkdir -p models reports progress

# Run smoke pipeline
echo "Running smoke pipeline..."
if python -m piedge_edukit.train --fakedata --no-pretrained --epochs 1 --batch-size 256 --output-dir ./models >/dev/null 2>&1; then
    pass "Training"
else
    fail "Training"
fi

if python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 1 --runs 3 --providers CPUExecutionProvider >/dev/null 2>&1; then
    pass "Benchmark"
else
    fail "Benchmark"
fi

if python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 >/dev/null 2>&1; then
    pass "Quantization (INT8)"
elif python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 --fallback >/dev/null 2>&1; then
    pass "Quantization (FP32 fallback)"
else
    warn "Quantization failed - creating synthetic data..."
    python - <<'PY'
from PIL import Image
import numpy as np, os
for cls in ['class0','class1']:
    d=os.path.join('data','train',cls); os.makedirs(d, exist_ok=True)
    for i in range(16):
        arr=(np.random.rand(64,64,3)*255).astype('uint8')
        Image.fromarray(arr).save(os.path.join(d,f'{i}.png'))
PY
    if python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 >/dev/null 2>&1; then
        pass "Quantization (with synthetic data)"
    else
        fail "Quantization (even with synthetic data)"
    fi
fi

if python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 32 >/dev/null 2>&1; then
    pass "Evaluation"
else
    fail "Evaluation"
fi

if python verify.py >/dev/null 2>&1; then
    pass "Verification"
else
    fail "Verification"
fi

# Test B: Pretty Demo
echo ""
echo "🎨 Test B: Pretty Demo"
echo "----------------------"

if bash scripts/demo_pretty.sh >/dev/null 2>&1; then
    pass "Pretty Demo"
else
    fail "Pretty Demo"
fi

if python verify.py >/dev/null 2>&1; then
    pass "Pretty Demo Verification"
else
    fail "Pretty Demo Verification"
fi

# Test C: Clean-room test
echo ""
echo "🏠 Test C: Clean-room Test"
echo "-------------------------"

WORK_DIR="/tmp/lesson_test_$$"
echo "Using work dir: $WORK_DIR"

# Clean up any existing test
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Copy and extract zip
if [ -f "$HOME/Documents/GitHub/piedge_edukit/VG_Ola_Blom_20251005.zip" ]; then
    ZIP_PATH="$HOME/Documents/GitHub/piedge_edukit/VG_Ola_Blom_20251005.zip"
elif [ -f "/c/Users/olabl/Documents/GitHub/piedge_edukit/VG_Ola_Blom_20251005.zip" ]; then
    ZIP_PATH="/c/Users/olabl/Documents/GitHub/piedge_edukit/VG_Ola_Blom_20251005.zip"
else
    fail "Could not find VG_Ola_Blom_20251005.zip"
    ZIP_PATH=""
fi

if [ -n "$ZIP_PATH" ] && [ -f "$ZIP_PATH" ]; then
    echo "Extracting $ZIP_PATH..."
    unzip -q "$ZIP_PATH"
    
    # Create fresh venv
    python -m venv .venv
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
    
    # Install
    pip install -r requirements.txt >/dev/null 2>&1
    
    # Run lesson
    if bash run_lesson.sh >/dev/null 2>&1; then
        pass "Clean-room lesson run"
    else
        fail "Clean-room lesson run"
    fi
    
    # Verify
    if python verify.py >/dev/null 2>&1; then
        pass "Clean-room verification"
    else
        fail "Clean-room verification"
    fi
else
    fail "Clean-room test (zip not found)"
fi

# Return to original directory
cd - >/dev/null

# Test D: Artifact check
echo ""
echo "📁 Test D: Artifact Check"
echo "------------------------"

# Check for required artifacts
ARTIFACTS_OK=true

if [ -f "models/model.onnx" ]; then
    pass "ONNX model exists"
else
    fail "ONNX model missing"
    ARTIFACTS_OK=false
fi

if [ -f "reports/training_curves.png" ]; then
    pass "Training curves plot"
else
    fail "Training curves plot missing"
    ARTIFACTS_OK=false
fi

if [ -f "reports/latency_plot.png" ]; then
    pass "Latency plot"
else
    fail "Latency plot missing"
    ARTIFACTS_OK=false
fi

if [ -f "reports/quantization_comparison.png" ]; then
    pass "Quantization comparison plot"
else
    fail "Quantization comparison plot missing"
    ARTIFACTS_OK=false
fi

if [ -f "reports/eval_summary.txt" ]; then
    pass "Evaluation summary"
else
    fail "Evaluation summary missing"
    ARTIFACTS_OK=false
fi

if [ -f "progress/receipt.json" ]; then
    pass "Receipt exists"
    # Check if receipt shows PASS
    if grep -q '"pass": true' progress/receipt.json; then
        pass "Receipt shows PASS"
    else
        fail "Receipt shows FAIL"
        ARTIFACTS_OK=false
    fi
else
    fail "Receipt missing"
    ARTIFACTS_OK=false
fi

# Final summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! Ready for submission.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check output above.${NC}"
    exit 1
fi
