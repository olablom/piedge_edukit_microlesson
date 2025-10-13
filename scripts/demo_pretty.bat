@echo off
REM PiEdge EduKit - Pretty Demo Script (Windows)
REM Runs a complete demo with visualizations

echo 🎨 PiEdge EduKit - Pretty Demo
echo ===============================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Run 'bash scripts/setup_venv.sh' first.
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo 🚀 Starting demo...

REM Run all notebooks
echo 📓 Running all notebooks...
python scripts/run_notebooks.py

REM Generate HTML reports
echo 📊 Generating HTML reports...
python -m jupyter nbconvert --to html --output-dir reports/nbhtml notebooks/*.ipynb

REM Run validation
echo ✅ Running validation...
python scripts/nb_validate.py

REM Run sanity check
echo 🔍 Running sanity check...
python scripts/nb_sanity_check.py

echo.
echo 🎉 Demo completed successfully!
echo ===============================
echo 📁 Check the following directories for results:
echo    - reports/nbexec/     (executed notebooks)
echo    - reports/nbhtml/     (HTML reports)
echo    - models/             (trained models)
echo    - reports/            (plots and metrics)
echo.
echo 🌐 Open reports/nbhtml/ in your browser to view the results!
