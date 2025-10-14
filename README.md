# PiEdge EduKit — Micro-Lesson (English)

Start here: **index.html**

This repository delivers a self-contained ~30-minute interactive lesson with automatic verification and a JSON receipt.

Requirements: Python 3.12, a local `.venv`, pinned `requirements.txt`. GPU optional.

# PiEdge EduKit

![CI](https://github.com/olablom/piedge_edukit_microlesson/actions/workflows/ci.yml/badge.svg)

**Start here → [`index.html`](index.html)** | Swedish: **[README.sv.md](README.sv.md)**

A **self-contained 30-minute micro-lesson** for edge ML: train a tiny image classifier → export to ONNX → benchmark latency → drive a GPIO LED with hysteresis.

## 🚀 One-click Start

**Windows:** Double-click `run_lesson.bat` _or_ run `python main.py`  
**macOS/Linux:** `bash run_lesson.sh` _or_ `python3 main.py`

### Quickstart (Windows)

```bash
git clone https://github.com/olablom/piedge_edukit_microlesson.git
cd piedge_edukit_microlesson
python -m venv .venv

# Activate venv (choose one):
# - Git Bash:      source .venv/Scripts/activate
# - PowerShell:    .\.venv\Scripts\Activate.ps1
# - CMD:           .\.venv\Scripts\activate.bat

pip install -r requirements.txt
pip install -e .
copy .env.example .env
python main.py --open-report
```

This will:

- Create `.venv` if it doesn't exist
- Install all requirements (including `ipywidgets==8.1.5` for notebook UI)
- Register Jupyter kernel "piedge"
- Attempts to open Jupyter Lab automatically (falls back to printing a URL if it can't)

> **Note:** If `data/train` is missing, a small synthetic dataset is created automatically during execution.

### Create submission ZIP (tracked files only)

```bash
# From repo root
git pull --ff-only
git archive --format=zip \
  --output ../VG_<Firstname>_<Lastname>_<YYYYMMDD>.zip \
  --prefix=piedge_edukit_microlesson/ \
  main
```

> **Note:** Use `git archive` – the ZIP should normally be ≪ 50 MB.

## Build lesson + auto-checksum

```bash
# Build lesson + auto-checksum
bash scripts/lesson_pack.sh && git commit -m "chore: refresh lesson zip & checksum" CHECKSUMS.txt && git push
```

## Manual Setup (if needed)

> **Prerequisites (hard requirement)**
>
> - **Python 3.12.x** (inte 3.11, inte 3.13)
> - Git Bash (Windows) eller bash (macOS/Linux)
> - 3–4 GB ledigt disk-utrymme
>
> **Snabbinstallation**  
> **Windows (Git Bash):**
>
> ```bash
> winget install --id Python.Python.3.12 -e
> # öppna NY Git Bash efter installation
> ```
>
> **macOS (Homebrew):**
>
> ```bash
> brew install python@3.12
> echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```
>
> **Ubuntu:**
>
> ```bash
> sudo add-apt-repository ppa:deadsnakes/ppa -y
> sudo apt update && sudo apt install -y python3.12 python3.12-venv
> ```

## Quick start (Python 3.12 only)

### Quick Start (Smoke Test)

```bash
# Create and activate venv
bash scripts/setup_venv.sh
source .venv/bin/activate      # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows

# Run Smoke Test (1 epoch, fast pipeline verification)
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 1 --batch-size 256 --output-dir ./models
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 1 --runs 3 --providers CPUExecutionProvider
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 32
python verify.py
```

### Optional: Pretty Demo (Nice Graphs)

For demonstration purposes with clear training curves and stable confusion matrix:

```bash
# Windows (Git Bash eller PowerShell)
scripts\demo_pretty.bat

# macOS / Linux
bash scripts/demo_pretty.sh
```

This runs:

1. Training (5 epochs, batch-size 16) → ONNX export
2. Benchmarking (50 warmup, 200 runs) → latency analysis
3. Evaluation (200 samples) → confusion matrix
4. Automatically opens the generated PNG images

**Note:** Smoke Test shows 1 point (with markers), Pretty Demo shows 5-point curves. Both give **PASS** in `verify.py`.

### Alternative: Jupyter Notebook

```bash
# Start Jupyter Notebook with the lesson
python main.py
```

This opens `notebooks/00_run_everything.ipynb` directly in Jupyter Notebook - the complete interactive lesson!

### Alternative: Direct CLI

**Note:** If `data/train` is missing, a small synthetic dataset is created automatically during execution.

```bash
# Create and activate venv
bash scripts/setup_venv.sh
source .venv/bin/activate      # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows

# Run the micro-lesson (self-contained, FakeData)
bash run_lesson.sh              # Linux/macOS
# or: run_lesson.bat              # Windows

# Auto-verify (JSON receipt)
python verify.py
# See progress/receipt.json
```

### Quantization (robust, with fallback)

```bash
# Quantize FP32 → INT8 (static). Uses data/calib if present, else synthetic samples.
python scripts/quantize_static.py \
  --fp32-model models/model.onnx \
  --calib-dir data/train \
  --num-calib 64 \
  --img-size 64 \
  --seed 42

# Outputs:
#  - models_quant/model_static.onnx (if static quant succeeds)
#  - models_quant/model_dynamic.onnx (fallback if static fails)
```

## What you'll learn

- Deterministic preprocessing + ONNX export (opset=17)
- Latency benchmarking (p50/p95/mean/std)
- (Optional) INT8 static quantization with comparison
- GPIO inference with hysteresis + debounce (simulate/real)

## Data options

The pipeline accepts both layouts:

- **Flat structure:** `data/<class>/*.{jpg,png}`
- **Train/val structure:** `data/{train,val}/<class>/*.{jpg,png}`

**No images needed:** Use `--fakedata` flag for quick testing.

## CLI commands

```bash
# Training
python -m piedge_edukit.train --fakedata --output-dir ./models

# Benchmarking
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200

# Quantization (optional)
python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 25

# Evaluation
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata
```

## Raspberry Pi (aarch64)

```bash
sudo bash pi_setup/install_pi_requirements.sh
python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200
python -m piedge_edukit.gpio_control --no-simulate --model-path ./models/model.onnx --data-path ./data --target class1 --duration 10
```

## Dashboard (optional)

```bash
streamlit run app.py
```

## Structure

```
index.html  run_lesson.sh  verify.py  scripts/  notebooks/  src/
requirements.txt  progress/  LICENSE  DATA_LICENSES.md  .env.example
piedge_edukit/  models/  reports/  pi_setup/  data/
```

## Troubleshooting

### Kernel Selection

**Important:** In Jupyter Notebook, select **"Python 3.12 (.venv piedge)"** as your kernel. This ensures you're using the correct virtual environment.

### Problems Panel

Notebooks are partially exempted from Ruff/Pyright linting to reduce noise. Focus on fixing warnings in `src/` and `scripts/` directories.

### Windows Tips

- Open generated images: `cmd.exe /c start "" "reports\\file.png"`
- Use Git Bash for all bash commands
- PowerShell works for Python commands

### Quantization Issues

On some Windows setups, static INT8 quantization currently fails. The lesson **accepts FP32 fallback** and verify still **PASS**. This is expected behavior.

### Common Issues

- **"Module not found"**: Ensure you're in the repo root and virtual environment is activated
- **"Permission denied"**: Run `bash scripts/setup_venv.sh` to recreate the virtual environment
- **Slow performance**: Use `--fakedata` flag for quick testing

## License

Apache-2.0. See `LICENSE`.

## Submission & Policy

- Solo work only for this assignment.
- Submit a single ZIP named `VG_<Firstname>_<Lastname>_<YYYYMMDD>.zip`.
- Keep within ZIP limits (≤0.5 GB with downloads or ≤1 GB self-contained).
- Everything must run inside `.venv` (no system Python installs).
- Python 3.12 is required across OS: Windows 11, macOS (latest), Ubuntu 22.04.
