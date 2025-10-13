@echo off
setlocal enabledelayedexpansion

REM PiEdge EduKit - Complete Test Suite (Windows)
REM Runs: Local dev → Pretty demo → Clean-room → Artifact check

echo 🧪 PiEdge EduKit - Complete Test Suite
echo ======================================

set TESTS_PASSED=0
set TESTS_FAILED=0

REM Test A: Local dev smoke test
echo.
echo 🔬 Test A: Local Dev Smoke Test
echo -------------------------------

REM Clean start
if exist .venv (
    echo Cleaning existing .venv...
    rmdir /s /q .venv
)

REM Create venv
echo Creating fresh .venv...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install
echo Installing requirements...
pip install -r requirements.txt >nul 2>&1
pip install -e . >nul 2>&1

REM Clean artifacts
if exist models rmdir /s /q models
if exist reports rmdir /s /q reports
if exist progress rmdir /s /q progress
mkdir models reports progress

REM Run smoke pipeline
echo Running smoke pipeline...
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 1 --batch-size 256 --output-dir ./models >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Training
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Training
    set /a TESTS_FAILED+=1
)

python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 1 --runs 3 --providers CPUExecutionProvider >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Benchmark
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Benchmark
    set /a TESTS_FAILED+=1
)

python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Quantization (INT8)
    set /a TESTS_PASSED+=1
) else (
    echo ⚠️  WARN: Quantization failed - creating synthetic data...
    python -c "from PIL import Image; import numpy as np, os; [os.makedirs(os.path.join('data','train',cls), exist_ok=True) for cls in ['class0','class1']]; [Image.fromarray((np.random.rand(64,64,3)*255).astype('uint8')).save(os.path.join('data','train',cls,f'{i}.png')) for cls in ['class0','class1'] for i in range(16)]"
    python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ PASS: Quantization (with synthetic data)
        set /a TESTS_PASSED+=1
    ) else (
        echo ❌ FAIL: Quantization (even with synthetic data)
        set /a TESTS_FAILED+=1
    )
)

python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 32 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Evaluation
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Evaluation
    set /a TESTS_FAILED+=1
)

python verify.py >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Verification
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Verification
    set /a TESTS_FAILED+=1
)

REM Test B: Pretty Demo
echo.
echo 🎨 Test B: Pretty Demo
echo ----------------------

scripts\demo_pretty.bat >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Pretty Demo
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Pretty Demo
    set /a TESTS_FAILED+=1
)

python verify.py >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PASS: Pretty Demo Verification
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Pretty Demo Verification
    set /a TESTS_FAILED+=1
)

REM Test C: Clean-room test
echo.
echo 🏠 Test C: Clean-room Test
echo -------------------------

set WORK_DIR=%TEMP%\lesson_test_%RANDOM%
echo Using work dir: %WORK_DIR%

REM Clean up any existing test
if exist "%WORK_DIR%" rmdir /s /q "%WORK_DIR%"
mkdir "%WORK_DIR%"
cd /d "%WORK_DIR%"

REM Copy and extract zip
set ZIP_PATH=%USERPROFILE%\Documents\GitHub\piedge_edukit\VG_Ola_Blom_20251005.zip
if exist "%ZIP_PATH%" (
    echo Extracting %ZIP_PATH%...
    powershell -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '.' -Force"
    
    REM Create fresh venv
    python -m venv .venv
    call .venv\Scripts\activate.bat
    
    REM Install
    pip install -r requirements.txt >nul 2>&1
    
    REM Run lesson
    bash run_lesson.sh >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ PASS: Clean-room lesson run
        set /a TESTS_PASSED+=1
    ) else (
        echo ❌ FAIL: Clean-room lesson run
        set /a TESTS_FAILED+=1
    )
    
    REM Verify
    python verify.py >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ PASS: Clean-room verification
        set /a TESTS_PASSED+=1
    ) else (
        echo ❌ FAIL: Clean-room verification
        set /a TESTS_FAILED+=1
    )
) else (
    echo ❌ FAIL: Clean-room test (zip not found)
    set /a TESTS_FAILED+=1
)

REM Return to original directory
cd /d "%~dp0"

REM Test D: Artifact check
echo.
echo 📁 Test D: Artifact Check
echo ------------------------

REM Check for required artifacts
if exist models\model.onnx (
    echo ✅ PASS: ONNX model exists
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: ONNX model missing
    set /a TESTS_FAILED+=1
)

if exist reports\training_curves.png (
    echo ✅ PASS: Training curves plot
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Training curves plot missing
    set /a TESTS_FAILED+=1
)

if exist reports\latency_plot.png (
    echo ✅ PASS: Latency plot
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Latency plot missing
    set /a TESTS_FAILED+=1
)

if exist reports\quantization_comparison.png (
    echo ✅ PASS: Quantization comparison plot
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Quantization comparison plot missing
    set /a TESTS_FAILED+=1
)

if exist reports\eval_summary.txt (
    echo ✅ PASS: Evaluation summary
    set /a TESTS_PASSED+=1
) else (
    echo ❌ FAIL: Evaluation summary missing
    set /a TESTS_FAILED+=1
)

if exist progress\receipt.json (
    echo ✅ PASS: Receipt exists
    set /a TESTS_PASSED+=1
    findstr /c:"\"pass\": true" progress\receipt.json >nul
    if %errorlevel% equ 0 (
        echo ✅ PASS: Receipt shows PASS
        set /a TESTS_PASSED+=1
    ) else (
        echo ❌ FAIL: Receipt shows FAIL
        set /a TESTS_FAILED+=1
    )
) else (
    echo ❌ FAIL: Receipt missing
    set /a TESTS_FAILED+=1
)

REM Final summary
echo.
echo 📊 Test Summary
echo ===============
echo Tests passed: %TESTS_PASSED%
echo Tests failed: %TESTS_FAILED%

if %TESTS_FAILED% equ 0 (
    echo 🎉 ALL TESTS PASSED! Ready for submission.
    exit /b 0
) else (
    echo ❌ Some tests failed. Check output above.
    exit /b 1
)
