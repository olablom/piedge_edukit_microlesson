@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

REM Create .venv if missing (Python 3.12 required)
if not exist .venv (
  echo Creating .venv (Python 3.12 required)...
  py -3.12 -m venv .venv
)

call .venv\Scripts\activate

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
)

python main.py --open-report

