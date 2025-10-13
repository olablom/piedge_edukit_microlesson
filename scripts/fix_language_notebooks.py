#!/usr/bin/env python3
"""
Fix remaining Swedish phrases in notebooks by applying targeted replacements
to common end-of-notebook sections. Usage:

  python scripts/fix_language_notebooks.py notebooks/01_training_and_export.ipynb notebooks/02_latency_benchmark.ipynb

This script is idempotent – running multiple times is safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPLACEMENTS = [
    # 01 — sections and headings
    (
        "# 🧠 Träning & ONNX Export - Förstå vad som händer",
        "# 🧠 Training & ONNX Export - Understand what's happening",
    ),
    (
        "**Mål**: Förstå hur träning fungerar och experimentera med olika inställningar.",
        "**Goal**: Understand how training works and experiment with different settings.",
    ),
    ("I detta notebook kommer vi att:", "In this notebook we will:"),
    (
        "- Förstå vad FakeData är och varför vi använder det",
        "- Understand what FakeData is and why we use it",
    ),
    (
        "- Se hur dataset-pipeline → modell → loss/accuracy fungerar",
        "- See how dataset-pipeline → model → loss/accuracy works",
    ),
    (
        "- Experimentera med olika hyperparametrar",
        "- Experiment with different hyperparameters",
    ),
    ("- Förstå varför vi exporterar till ONNX", "- Understand why we export to ONNX"),
    (
        "> **💡 Tips**: Kör cellerna i ordning och läs förklaringarna. Experimentera gärna med värdena!",
        "> **💡 Tip**: Run the cells in order and read the explanations. Feel free to experiment with the values!",
    ),
    (
        "## 🤔 Vad är FakeData och varför använder vi det?",
        "## 🤔 What is FakeData and why do we use it?",
    ),
    (
        "**FakeData** är syntetiska bilder som PyTorch genererar automatiskt. Det är perfekt för:",
        "**FakeData** are synthetic images that PyTorch generates automatically. It's perfect for:",
    ),
    (
        "- **Snabb prototyping** - ingen nedladdning av stora dataset",
        "- **Quick prototyping** - no downloading of large datasets",
    ),
    (
        "- **Reproducerbarhet** - samma data varje gång",
        "- **Reproducibility** - same data every time",
    ),
    (
        "- **Undervisning** - fokus på algoritmer, inte datahantering",
        "- **Teaching** - focus on algorithms, not data management",
    ),
    (
        "<summary>🔍 Klicka för att se vad FakeData innehåller</summary>",
        "<summary>🔍 Click to see what FakeData contains</summary>",
    ),
    ("# FakeData genererar:", "# FakeData generates:"),
    (
        "# - Slumpmässiga RGB-bilder (64x64 pixlar)",
        "# - Random RGB images (64x64 pixels)",
    ),
    ("# - Slumpmässiga klasser (0, 1, 2, ...)", "# - Random classes (0, 1, 2, ...)"),
    (
        "# - Samma struktur som riktiga bilddataset",
        "# - Same structure as real image datasets",
    ),
    (
        "# Låt oss skapa en liten FakeData för att se vad den innehåller",
        "# Let's create a small FakeData to see what it contains",
    ),
    ("# Skapa FakeData med 2 klasser", "# Create FakeData with 2 classes"),
    ("# Visa första bilden", "# Show first image"),
    ('print(f"Bildstorlek: {image.size}")', 'print(f"Image size: {image.size}")'),
    ('print(f"Klass: {label}")', 'print(f"Class: {label}")'),
    (
        'print(f"Pixelvärden: {image.getextrema()}")',
        'print(f"Pixel values: {image.getextrema()}")',
    ),
    ("# Visa bilden", "# Show the image"),
    (
        'plt.title(f"FakeData - Klass {label}")',
        'plt.title(f"FakeData - Class {label}")',
    ),
    ("## 🤔 Reflektionsfrågor", "## 🤔 Reflection Questions"),
    (
        "<summary>💭 Vad händer med överfitting när du höjer epochs?</summary>",
        "<summary>💭 What happens with overfitting when you increase epochs?</summary>",
    ),
    (
        "**Svar**: Med fler epochs kan modellen lära sig träningsdata för bra och dåligt generalisera till nya data. Detta kallas överfitting.",
        "**Answer**: With more epochs, the model can learn the training data too well and generalize poorly to new data. This is called overfitting.",
    ),
    (
        "**Experiment**: Kör samma träning men med `--epochs 5` och jämför accuracy på tränings- vs valideringsdata.",
        "**Experiment**: Run the same training but with `--epochs 5` and compare accuracy on training vs validation data.",
    ),
    (
        "<summary>💭 Varför exporterar vi till ONNX (för Pi/edge)?</summary>",
        "<summary>💭 Why do we export to ONNX (for Pi/edge)?</summary>",
    ),
    (
        "**Svar**: ONNX är ett standardformat som fungerar på många plattformar (CPU, GPU, mobil, edge). Det gör modellen portabel och optimerad för inference.",
        "**Answer**: ONNX is a standard format that works on many platforms (CPU, GPU, mobile, edge). It makes the model portable and optimized for inference.",
    ),
    ("**Fördelar**:", "**Benefits**:"),
    ("- Snabbare inference än PyTorch", "- Faster inference than PyTorch"),
    ("- Mindre minnesanvändning", "- Less memory usage"),
    ("- Fungerar på Raspberry Pi", "- Works on Raspberry Pi"),
    ("- Stöd för kvantisering (INT8)", "- Support for quantization (INT8)"),
    ("## 🎯 Ditt eget experiment", "## 🎯 Your own experiment"),
    (
        "**Uppgift**: Träna en modell med andra inställningar och jämför resultaten.",
        "**Task**: Train a model with different settings and compare the results.",
    ),
    ("**Förslag**:", "**Suggestions**:"),
    ("- Öka epochs till 3-5", "- Increase epochs to 3-5"),
    ("- Ändra batch_size till 64 eller 256", "- Change batch_size to 64 or 256"),
    (
        "- Testa med och utan `--no-pretrained`",
        "- Test with and without `--no-pretrained`",
    ),
    ("**Kod att modifiera**:", "**Code to modify**:"),
    ("# Ändra dessa värden:", "# Change these values:"),
    (
        "USE_PRETRAINED = False  # True för förtränade vikter",
        "USE_PRETRAINED = False  # True for pretrained weights",
    ),
    (
        "# TODO: Implementera ditt experiment här",
        "# TODO: Implement your experiment here",
    ),
    (
        "# Ändra värdena nedan och kör träningen",
        "# Change the values below and run the training",
    ),
    (
        'print(f"🧪 Mitt experiment: epochs={EPOCHS}, batch_size={BATCH_SIZE}, pretrained={USE_PRETRAINED}")',
        'print(f"🧪 My experiment: epochs={EPOCHS}, batch_size={BATCH_SIZE}, pretrained={USE_PRETRAINED}")',
    ),
    (
        "# TODO: Kör träningen med dina inställningar",
        "# TODO: Run the training with your settings",
    ),
    ("## 🎉 Sammanfattning", "## 🎉 Summary"),
    ("you have nu lärt dig:", "You have now learned:"),
    (
        "- Vad FakeData är och varför vi använder det",
        "- What FakeData is and why we use it",
    ),
    (
        "- Hur träning fungerar med olika hyperparametrar",
        "- How training works with different hyperparameters",
    ),
    (
        "- Varför ONNX-export är viktigt för edge deployment",
        "- Why ONNX export is important for edge deployment",
    ),
    (
        "**Nästa steg**: Gå till `02_latency_benchmark.ipynb` för att förstå hur vi mäter modellens prestanda.",
        "**Next step**: Go to `02_latency_benchmark.ipynb` to understand how we measure model performance.",
    ),
    ("**Viktiga begrepp**:", "**Key concepts**:"),
    (
        "- **Epochs**: Antal genomgångar av datasetet",
        "- **Epochs**: Number of passes through the dataset",
    ),
    (
        "- **Batch size**: Antal bilder per träningssteg",
        "- **Batch size**: Number of images per training step",
    ),
    (
        "- **Pretrained weights**: Förtränade vikter från ImageNet",
        "- **Pretrained weights**: Pre-trained weights from ImageNet",
    ),
    (
        "- **ONNX**: Standardformat för edge deployment",
        "- **ONNX**: Standard format for edge deployment",
    ),
    # 02 — tail section translations
    (
        "# ⚡ Latensbenchmark - Förstå modellens prestanda",
        "# ⚡ Latency Benchmark - Understand model performance",
    ),
    (
        "## 3️⃣ Latensbenchmark",
        "## 3️⃣ Latency Benchmark",
    ),
    (
        "**Mål**: Förstå hur vi mäter och tolkar modellens latens (svarstid).",
        "**Goal**: Understand how we measure and interpret model latency (response time).",
    ),
    (
        "Measuring how fast the model is på CPU:",
        "Measuring how fast the model is on CPU:",
    ),
    ("I detta notebook kommer vi att:", "In this notebook we will:"),
    (
        "- Förstå vad latens är och varför det är viktigt",
        "- Understand what latency is and why it's important",
    ),
    (
        "- Se hur benchmark fungerar (warmup, runs, providers)",
        "- See how benchmark works (warmup, runs, providers)",
    ),
    (
        "- Tolka resultat (p50, p95, histogram)",
        "- Interpret results (p50, p95, histogram)",
    ),
    ("- Experimentera med olika inställningar", "- Experiment with different settings"),
    (
        "> **💡 Tips**: Latens är avgörande för edge deployment - en modell som är för långsam är inte användbar i verkligheten!",
        "> **💡 Tip**: Latency is critical for edge deployment - a model that's too slow is not usable in real life!",
    ),
    (
        "## 🤔 Vad är latens och varför är det viktigt?",
        "## 🤔 What is latency and why is it important?",
    ),
    (
        "**Latens** = tiden det tar för modellen att göra en förutsägelse (inference time).",
        "**Latency** = the time it takes for the model to make a prediction (inference time).",
    ),
    ("**Varför viktigt för edge**:", "**Why important for edge**:"),
    (
        "- **Realtidsapplikationer** - robotar, autonoma fordon",
        "- **Real-time applications** - robots, autonomous vehicles",
    ),
    (
        "- **Användarupplevelse** - ingen vill vänta 5 sekunder på en bildklassificering",
        "- **User experience** - no one wants to wait 5 seconds for image classification",
    ),
    (
        "- **Resursbegränsningar** - Raspberry Pi har begränsad CPU/memory",
        "- **Resource constraints** - Raspberry Pi has limited CPU/memory",
    ),
    (
        "<summary>🔍 Klicka för att se typiska latensmål</summary>",
        "<summary>🔍 Click to see typical latency targets</summary>",
    ),
    ("**Typiska latensmål**:", "**Typical latency targets**:"),
    ("- **< 10ms**: Realtidsvideo, gaming", "- **< 10ms**: Real-time video, gaming"),
    (
        "- **< 100ms**: Interaktiva applikationer",
        "- **< 100ms**: Interactive applications",
    ),
    (
        "- **< 1000ms**: Batch processing, offline analys",
        "- **< 1000ms**: Batch processing, offline analysis",
    ),
    (
        "**Vår modell**: Förväntar oss ~1-10ms på CPU (bra för edge!)",
        "**Our model**: Expect ~1-10ms on CPU (good for edge!)",
    ),
    ("## 🔧 Hur fungerar benchmark?", "## 🔧 How does benchmark work?"),
    ("**Benchmark-processen**:", "**Benchmark process**:"),
    (
        '1. **Warmup** - kör modellen några gånger för att "värmma upp" (JIT compilation, cache)',
        '1. **Warmup** - run the model a few times to "warm up" (JIT compilation, cache)',
    ),
    (
        "2. **Runs** - mäter latens för många körningar",
        "2. **Runs** - measure latency for many runs",
    ),
    (
        "3. **Statistik** - beräknar p50, p95, mean, std",
        "3. **Statistics** - calculate p50, p95, mean, std",
    ),
    ("**Varför warmup?**", "**Why warmup?**"),
    (
        "- Första körningen är ofta långsam (JIT compilation)",
        "- First run is often slow (JIT compilation)",
    ),
    ("- Cache-värme påverkar prestanda", "- Cache warming affects performance"),
    (
        '- Vi vill mäta "steady state" prestanda',
        '- We want to measure "steady state" performance',
    ),
    ('print("🚀 Kör benchmark...")', 'print("🚀 Running benchmark...")'),
    ('print("📈 Benchmark-resultat:")', 'print("📈 Benchmark results:")'),
    ('print(f"📊 Latens-statistik:")', 'print(f"📊 Latency statistics:")'),
    ('print(f"Antal mätningar: {len(df)}")', 'print(f"Num measurements: {len(df)}")'),
    ("plt.xlabel('Latens (ms)')", "plt.xlabel('Latency (ms)')"),
    ("plt.ylabel('Antal')", "plt.ylabel('Count')"),
    ("plt.title('Latens-distribution')", "plt.title('Latency distribution')"),
    ("plt.ylabel('Latens (ms)')", "plt.ylabel('Latency (ms)')"),
    ("plt.title('Latens Box Plot')", "plt.title('Latency Box Plot')"),
    ('print("❌ Latens CSV missing")', 'print("❌ Latency CSV missing")'),
    ("# Kör benchmark (snabb running)", "# Run benchmark (quick mode)"),
    # 02 — summary & key concepts tail section
    ("## 🎉 Sammanfattning", "## 🎉 Summary"),
    ("you have nu lärt dig:", "You have now learned:"),
    (
        "- Vad latens är och varför det är kritiskt för edge deployment",
        "- What latency is and why it is critical for edge deployment",
    ),
    (
        "- Hur benchmark fungerar (warmup, runs, statistik)",
        "- How the benchmark works (warmup, runs, statistics)",
    ),
    (
        "- Hur man tolkar latens-resultat (p50, p95, varians)",
        "- How to interpret latency results (p50, p95, variance)",
    ),
    (
        "- Varför p95 är viktigare än mean för användarupplevelse",
        "- Why P95 is more important than mean for user experience",
    ),
    (
        "**Nästa steg**: Gå till `03_quantization.ipynb` för att förstå hur kvantisering kan förbättra prestanda.",
        "**Next step**: Go to `03_quantization.ipynb` to understand how quantization can improve performance.",
    ),
    ("**Viktiga begrepp**:", "**Key concepts**:"),
    (
        "- **Latens**: Inference-tid (kritiskt för edge)",
        "- **Latency**: Inference time (critical for edge)",
    ),
    (
        "- **Warmup**: Förbereder modellen för mätning",
        "- **Warm-up**: Prepares the model for measurement",
    ),
    (
        "- **p50/p95**: Percentiler för latens-distribution",
        "- **p50/p95**: Percentiles for the latency distribution",
    ),
    (
        "- **Varians**: Konsistens i prestanda",
        "- **Variance**: Consistency in performance",
    ),
    # 03 — Quantization notebook (body + tail)
    (
        "# ⚡ Kvantisering (INT8) - Komprimera modellen för snabbare inference",
        "# ⚡ Quantization (INT8) - Compress the model for faster inference",
    ),
    (
        "## 4️⃣ Kvantisering (INT8)",
        "## 4️⃣ Quantization (INT8)",
    ),
    (
        "**Mål**: Förstå hur kvantisering fungerar och när det är värt det.",
        "**Goal**: Understand how quantization works and when it is worth it.",
    ),
    (
        "Compressing the model för snabbare inference:",
        "Compressing the model for faster inference:",
    ),
    (
        "- Förstå vad kvantisering är (FP32 → INT8)",
        "- Understand what quantization is (FP32 → INT8)",
    ),
    (
        "- Se hur det påverkar modellstorlek och latens",
        "- See how it affects model size and latency",
    ),
    (
        "- Experimentera med olika kalibreringsstorlekar",
        "- Experiment with different calibration sizes",
    ),
    (
        "- Förstå kompromisser (accuracy vs prestanda)",
        "- Understand the trade-offs (accuracy vs performance)",
    ),
    ("## 🤔 Vad är kvantisering?", "## 🤔 What is quantization?"),
    (
        "**Kvantisering** = konvertera modellen från 32-bit flyttal (FP32) till 8-bit heltal (INT8).",
        "**Quantization** = convert the model from 32-bit floating point (FP32) to 8-bit integers (INT8).",
    ),
    (
        "- **4x mindre modellstorlek** (32 bit → 8 bit)",
        "- **4x smaller model size** (32-bit → 8-bit)",
    ),
    (
        "- **2-4x snabbare inference** (INT8 är snabbare att beräkna)",
        "- **2–4x faster inference** (INT8 is faster to compute)",
    ),
    (
        "- **Mindre minnesanvändning** (viktigt för edge)",
        "- **Lower memory usage** (important for edge)",
    ),
    ("**Kompromisser**:", "**Trade-offs**:"),
    (
        "- **Accuracy-förlust** - modellen kan bli mindre exakt",
        "- **Accuracy loss** — the model can become less accurate",
    ),
    (
        "- **Kalibrering required** - behöver representativ data för att hitta rätt skala",
        "- **Calibration required** — needs representative data to find proper scales",
    ),
    (
        "<summary>🔍 Klicka för att se tekniska detaljer</summary>",
        "<summary>🔍 Click to see technical details</summary>",
    ),
    ("**Teknisk förklaring**:", "**Technical details**:"),
    ("- FP32: 32 bit per vikt (4 bytes)", "- FP32: 32 bits per weight (4 bytes)"),
    ("- INT8: 8 bit per vikt (1 byte)", "- INT8: 8 bits per weight (1 byte)"),
    (
        "- Kvantisering hittar rätt skala för varje vikt",
        "- Quantization finds the right scale for each weight",
    ),
    (
        "- Kalibrering använder representativ data för att optimera skalan",
        "- Calibration uses representative data to optimize scales",
    ),
    (
        "# Först skapar vi en modell att kvantisera",
        "# First create a model to quantize",
    ),
    (
        'print("🚀 Skapar modell för kvantisering...")',
        'print("🚀 Creating a model for quantization...")',
    ),
    (
        'print(f"📦 Ursprunglig modellstorlek: {original_size:.2f} MB")',
        'print(f"📦 Original model size: {original_size:.2f} MB")',
    ),
    ('print("❌ Modell missing")', 'print("❌ Model missing")'),
    (
        "## 🧪 Experimentera med olika kalibreringsstorlekar",
        "## 🧪 Experiment with different calibration sizes",
    ),
    (
        "**Kalibreringsstorlek** = antal bilder som används för att hitta rätt skala för kvantisering.",
        "**Calibration size** = number of images used to find the right quantization scales.",
    ),
    ("**Större kalibrering**:", "**Larger calibration**:"),
    (
        "- ✅ Bättre accuracy (mer representativ data)",
        "- ✅ Better accuracy (more representative data)",
    ),
    ("- ❌ Längre kvantiserings-tid", "- ❌ Longer quantization time"),
    ("- ❌ Mer minne under kvantisering", "- ❌ More memory during quantization"),
    ("**Mindre kalibrering**:", "**Smaller calibration**:"),
    ("- ✅ Snabbare kvantisering", "- ✅ Faster quantization"),
    ("- ✅ Mindre minnesanvändning", "- ✅ Lower memory usage"),
    ("- ❌ Potentiellt sämre accuracy", "- ❌ Potentially worse accuracy"),
    (
        'print("⚡ Test 1: Liten kalibrering (16 bilder)")',
        'print("⚡ Test 1: Small calibration (16 images)")',
    ),
    ("# Visa kvantiseringsresultat", "# Show quantization results"),
    ('print("📊 Kvantiseringsresultat:")', 'print("📊 Quantization results:")'),
    (
        'print("❌ Kvantiseringsrapport missing")',
        'print("❌ Quantization report missing")',
    ),
    ("# Jämför modellstorlekar", "# Compare model sizes"),
    ('print("📦 Modellstorlekar:")', 'print("📦 Model sizes:")'),
    (
        'print(f"  Ursprunglig (FP32): {original_size:.2f} MB")',
        'print(f"  Original (FP32): {original_size:.2f} MB")',
    ),
    (
        'print(f"  Kvantiserad (INT8): {quantized_size:.2f} MB")',
        'print(f"  Quantized (INT8): {quantized_size:.2f} MB")',
    ),
    (
        'print(f"  Komprimering: {original_size/quantized_size:.1f}x")',
        'print(f"  Compression: {original_size/quantized_size:.1f}x")',
    ),
    ('print("❌ Modellfiler missing")', 'print("❌ Model files missing")'),
    (
        'print("🚀 Benchmark ursprunglig modell (FP32)...")',
        'print("🚀 Benchmark original model (FP32)...")',
    ),
    (
        'print("⚡ Benchmark kvantiserad modell (INT8)...")',
        'print("⚡ Benchmark quantized model (INT8)...")',
    ),
    ('print(f"📊 Latens-jämförelse:")', 'print(f"📊 Latency comparison:")'),
    (
        'print(f"  FP32 (ursprunglig): {fp32_mean:.2f} ms")',
        'print(f"  FP32 (original): {fp32_mean:.2f} ms")',
    ),
    (
        'print(f"  FP32: Kunde inte parsa latens")',
        'print(f"  FP32: Could not parse latency")',
    ),
    (
        'print(f"  INT8 (kvantiserad): [kommer efter benchmark]")',
        'print(f"  INT8 (quantized): [after benchmark]")',
    ),
    ('print("❌ Benchmark-rapport missing")', 'print("❌ Benchmark report missing")'),
    (
        "<summary>💭 När är INT8-kvantisering värt det?</summary>",
        "<summary>💭 When is INT8 quantization worth it?</summary>",
    ),
    ("**Svar**: INT8 är värt det när:", "**Answer**: INT8 is worth it when:"),
    (
        "- **Latens är kritisk** - realtidsapplikationer, edge deployment",
        "- **Latency is critical** — real-time applications, edge deployment",
    ),
    (
        "- **Minne är begränsat** - mobil, Raspberry Pi",
        "- **Memory is limited** — mobile, Raspberry Pi",
    ),
    (
        "- **Accuracy-förlusten är acceptabel** - < 1-2% accuracy-förlust är ofta OK",
        "- **Accuracy loss is acceptable** — < 1–2% accuracy drop is often OK",
    ),
    (
        "- **Batch size är liten** - kvantisering fungerar bäst med små batches",
        "- **Batch size is small** — quantization often works best with small batches",
    ),
    ("**När INTE värt det**:", "**When NOT worth it**:"),
    ("- Accuracy är absolut kritisk", "- Accuracy is absolutely critical"),
    ("- you have gott om minne och CPU", "- You have ample memory and CPU"),
    ("- Modellen är redan snabb nog", "- The model is already fast enough"),
    (
        "<summary>💭 Vilka risker finns med kvantisering?</summary>",
        "<summary>💭 What are the risks with quantization?</summary>",
    ),
    ("**Svar**: Huvudrisker:", "**Answer**: Main risks:"),
    (
        "- **Accuracy-förlust** - modellen kan bli mindre exakt",
        "- **Accuracy loss** — the model can become less accurate",
    ),
    (
        "- **Kalibreringsdata** - behöver representativ data för bra kvantisering",
        "- **Calibration data** — needs representative data for good quantization",
    ),
    (
        "- **Edge cases** - extrema värden kan orsaka problem",
        "- **Edge cases** — extreme values can cause issues",
    ),
    (
        "- **Debugging** - kvantiserade modeller är svårare att debugga",
        "- **Debugging** — quantized models are harder to debug",
    ),
    ("**Minskning**:", "**Mitigation**:"),
    ("- Testa noggrant med riktig data", "- Test thoroughly with real data"),
    ("- Använd olika kalibreringsstorlekar", "- Use different calibration sizes"),
    ("- Benchmark både accuracy och latens", "- Benchmark both accuracy and latency"),
    (
        "**Uppgift**: Testa olika kalibreringsstorlekar och jämför resultaten.",
        "**Task**: Test different calibration sizes and compare the results.",
    ),
    (
        "- Testa kalibreringsstorlekar: 8, 16, 32, 64",
        "- Try calibration sizes: 8, 16, 32, 64",
    ),
    ("- Jämför modellstorlek och latens", "- Compare model size and latency"),
    (
        "- Analysera accuracy-förlust (om tillgänglig)",
        "- Analyze accuracy loss (if available)",
    ),
    (
        "# Ändra värdena nedan och kör kvantisering",
        "# Change the values below and run quantization",
    ),
    (
        'print(f"🧪 Mitt experiment: kalibreringsstorlek={CALIB_SIZE}")',
        'print(f"🧪 My experiment: calibration_size={CALIB_SIZE}")',
    ),
    (
        "# TODO: Kör kvantisering med din inställning",
        "# TODO: Run quantization with your setting",
    ),
    (
        "- Vad kvantisering är (FP32 → INT8) och varför det är viktigt",
        "- What quantization is (FP32 → INT8) and why it matters",
    ),
    (
        "- Hur kalibreringsstorlek påverkar resultatet",
        "- How calibration size affects the result",
    ),
    (
        "- Kompromisser mellan accuracy och prestanda",
        "- Trade-offs between accuracy and performance",
    ),
    (
        "- När kvantisering är värt det vs när det inte är det",
        "- When quantization is worth it vs when it is not",
    ),
    (
        "**Nästa steg**: Gå till `04_evaluate_and_verify.ipynb` för att förstå automatiska checks och kvittogenerering.",
        "**Next**: Open `04_evaluate_and_verify.ipynb` to understand automated checks and receipt generation.",
    ),
    (
        "- **Kvantisering**: FP32 → INT8 för snabbare inference",
        "- **Quantization**: FP32 → INT8 for faster inference",
    ),
    (
        "- **Kalibrering**: Representativ data för att hitta rätt skala",
        "- **Calibration**: Representative data to find the right scales",
    ),
    (
        "- **Komprimering**: 4x mindre modellstorlek",
        "- **Compression**: 4x smaller model size",
    ),
    ("- **Speedup**: 2-4x snabbare inference", "- **Speedup**: 2–4x faster inference"),
    # 04 — Evaluation & Verification (body + tail)
    (
        "## 🤔 Vad är utvärdering och varför behöver vi det?",
        "## 🤔 What is evaluation and why do we need it?",
    ),
    (
        "## 5️⃣ Utvärdering & Verifiering",
        "## 5️⃣ Evaluation & Verification",
    ),
    (
        "**Utvärdering** = testa modellen på data den inte har sett under träning.",
        "**Evaluation** = test the model on data it has not seen during training.",
    ),
    ("**Vad vi mäter**:", "**What we measure**:"),
    (
        "- **Accuracy** - hur många förutsägelser som är rätta",
        "- **Accuracy** — how many predictions are correct",
    ),
    (
        "- **Confusion matrix** - detaljerad breakdown av rätta/felaktiga förutsägelser",
        "- **Confusion matrix** — detailed breakdown of correct/incorrect predictions",
    ),
    (
        "- **Per-class performance** - hur bra modellen är på varje klass",
        "- **Per-class performance** — how well the model performs for each class",
    ),
    ("**Varför viktigt**:", "**Why important**:"),
    (
        "- **Validering** - säkerställer att modellen faktiskt fungerar",
        "- **Validation** — ensures the model actually works",
    ),
    (
        "- **Debugging** - visar vilka klasser som är svåra",
        "- **Debugging** — shows which classes are difficult",
    ),
    (
        "- **Jämförelse** - kan jämföra olika modeller/inställningar",
        "- **Comparison** — compare different models/settings",
    ),
    (
        "<summary>🔍 Klicka för att se vad en confusion matrix visar</summary>",
        "<summary>🔍 Click to see what a confusion matrix shows</summary>",
    ),
    ("# Kör utvärdering på vår modell", "# Run evaluation on our model"),
    ('print("🔍 Kör utvärdering...")', 'print("🔍 Running evaluation...")'),
    (
        "# Använd modellen från föregående notebooks (eller skapa en snabb)",
        "# Use the model from previous notebooks (or create a quick one)",
    ),
    (
        "# Kör utvärdering med begränsat antal samples (snabbare)",
        "# Run evaluation with a limited number of samples (faster)",
    ),
    ("# Visa utvärderingsresultat", "# Show evaluation results"),
    ('print("📊 Utvärderingsresultat:")', 'print("📊 Evaluation results:")'),
    (
        'print("❌ Utvärderingsrapport missing")',
        'print("❌ Evaluation report missing")',
    ),
    ("# Visa träningsgrafer om de finns", "# Show training curves if available"),
    ('print("📈 Träningsgrafer:")', 'print("📈 Training curves:")'),
    (
        'print("⚠️ Träningsgrafer missing – kör träningen först.")',
        'print("⚠️ Training curves missing – run training first.")',
    ),
    ("## 🔍 Automatisk verifiering", "## 🔍 Automatic verification"),
    (
        "**Verifiering** = automatiska checks som säkerställer att lektionen fungerar korrekt.",
        "**Verification** = automated checks ensuring the lesson works correctly.",
    ),
    ("**Vad kontrolleras**:", "**What is checked**:"),
    (
        "- **Artefakter finns** - alla nödvändiga filer är skapade",
        "- **Artifacts exist** — all required files are created",
    ),
    (
        "- **Benchmark fungerar** - latens-data är giltig",
        "- **Benchmark works** — latency data is valid",
    ),
    (
        "- **Kvantisering fungerar** - kvantiserad modell är skapad",
        "- **Quantization works** — quantized model is created",
    ),
    (
        "- **Utvärdering fungerar** - confusion matrix och accuracy är tillgänglig",
        "- **Evaluation works** — confusion matrix and accuracy are available",
    ),
    (
        "**Resultat**: `progress/receipt.json` med PASS/FAIL status",
        "**Result**: `progress/receipt.json` with PASS/FAIL status",
    ),
    ("# Kör automatisk verifiering", "# Run automatic verification"),
    (
        'print("🔍 Kör automatisk verifiering...")',
        'print("🔍 Running automatic verification...")',
    ),
    (
        "📋 Verifieringskvitto:",
        "📋 Verification receipt:",
    ),
    (
        "Kontroller:",
        "Checks:",
    ),
    (
        "# Visa kvitto",
        "# Show receipt",
    ),
    (
        'print("\nKontroller:")',
        'print("\nChecks:")',
    ),
    ("# Analysera kvittot i detalj", "# Analyze the receipt in detail"),
    ('print("📋 Detaljerad kvitto-analys:")', 'print("📋 Detailed receipt analysis:")'),
    ('print("\n🔍 Kontroller:")', 'print("\n🔍 Checks:")'),
    ('print("\n📁 Genererade filer:")', 'print("\n📁 Generated files:")'),
    ('print("❌ Kvitto missing")', 'print("❌ Receipt missing")'),
    (
        "<summary>💭 Vilka mål verifieras av vår automatiska check?</summary>",
        "<summary>💭 Which goals are verified by our automatic check?</summary>",
    ),
    ("**Svar**: Vår verifiering kontrollerar:", "**Answer**: Our verification checks:"),
    (
        "- **Teknisk funktionalitet** - alla steg körs utan fel",
        "- **Technical functionality** — all steps run without errors",
    ),
    (
        "- **Artefakt-generering** - nödvändiga filer skapas",
        "- **Artifact generation** — required files are created",
    ),
    (
        "- **Data-integritet** - rapporter är giltiga och parseable",
        "- **Data integrity** — reports are valid and parseable",
    ),
    (
        "- **Pipeline-integration** - alla komponenter fungerar tillsammans",
        "- **Pipeline integration** — all components work together",
    ),
    ("**Vad som INTE verifieras**:", "**What is NOT verified**:"),
    (
        "- Accuracy-kvalitet (bara att utvärdering körs)",
        "- Accuracy quality (only that evaluation runs)",
    ),
    (
        "- Latens-mål (bara att benchmark körs)",
        "- Latency targets (only that benchmark runs)",
    ),
    (
        "- Produktionsredo (bara att pipeline fungerar)",
        "- Production readiness (only that the pipeline works)",
    ),
    (
        '<summary>💭 Vad missing för "produktion"?</summary>',
        '<summary>💭 What is missing for "production"?</summary>',
    ),
    ("**Svar**: För produktion behöver vi:", "**Answer**: For production we need:"),
    ("- **Riktig data** - inte FakeData", "- **Real data** — not FakeData"),
    (
        "- **Accuracy-mål** - specifika krav på precision/recall",
        "- **Accuracy targets** — specific precision/recall requirements",
    ),
    (
        "- **Latens-mål** - SLA-krav på inference-tid",
        "- **Latency targets** — SLA requirements on inference time",
    ),
    (
        "- **Robusthet** - hantering av edge cases och fel",
        "- **Robustness** — handling of edge cases and errors",
    ),
    (
        "- **Monitoring** - kontinuerlig övervakning av prestanda",
        "- **Monitoring** — continuous monitoring of performance",
    ),
    (
        "- **A/B-testing** - jämförelse av olika modeller",
        "- **A/B testing** — comparison of different models",
    ),
    (
        "- **Rollback** - möjlighet att gå tillbaka till tidigare version",
        "- **Rollback** — ability to revert to previous versions",
    ),
    (
        "**Uppgift**: Kör verifiering på olika modeller och jämför kvittona.",
        "**Task**: Run verification on different models and compare receipts.",
    ),
    (
        "- Träna modeller med olika inställningar",
        "- Train models with different settings",
    ),
    ("- Kör verifiering på varje modell", "- Run verification on each model"),
    (
        "- Jämför kvittona och se vilka som passerar/failar",
        "- Compare receipts and see which pass/fail",
    ),
    (
        "- Analysera vilka checks som är mest kritiska",
        "- Analyze which checks are most critical",
    ),
    (
        "# Träna olika modeller och kör verifiering",
        "# Train different models and run verification",
    ),
    (
        'print("🧪 Mitt experiment: Jämför olika modeller")',
        'print("🧪 My experiment: Compare different models")',
    ),
    (
        "# TODO: Implementera loop som tränar och verifierar varje modell",
        "# TODO: Implement a loop that trains and verifies each model",
    ),
    # 00 — remaining Swedish print statements
    ('print("\\n📈 Träningsgrafer:")', 'print("\\n📈 Training curves:")'),
    (
        'print("\\n⚠️ Träningsgrafer missing – run träningen först.")',
        'print("\\n⚠️ Training curves missing – run training first.")',
    ),
    (
        'print("\\nℹ️ INT8-kvantisering kan fallera på vissa miljöer. I denna lektion är **FP32** godkänt; verify accepterar fallback.")',
        'print("\\nℹ️ INT8 quantization may fail on some environments. In this lesson **FP32** is accepted; verify accepts fallback.")',
    ),
    # 00 — miscellaneous headings and phrases from the user's list
    ("# Kontrollera att modellen skapades", "# Check that the model was created"),
    ("# Visa träningsgrafer", "# Show training curves"),
    ("# Kör utvärdering", "# Run evaluation"),
    (
        "# Kör verifiering och generera kvitto",
        "# Run verification and generate receipt",
    ),
    ("Done!… Nästa steg", "Done!… Next steps"),
    ("Verifieringskvitto", "Verification receipt"),
    ("Visa kvitto", "Show receipt"),
    ("Kontroller", "Checks"),
    ("Latensbenchmark", "Latency Benchmark"),
    ("Kvantisering (INT8)", "Quantization (INT8)"),
    ("Utvärdering & Verifiering", "Evaluation & Verification"),
    (
        'print("\\n⚠️ Confusion matrix missing – run utvärderingen först.")',
        'print("\\n⚠️ Confusion matrix missing – run evaluation first.")',
    ),
    # 02 — remaining Swedish print
    (
        'print(f"🧪 Mitt experiment: warmup={WARMUP_RUNS}, runs={BENCHMARK_RUNS}")',
        'print(f"🧪 My experiment: warmup={WARMUP_RUNS}, runs={BENCHMARK_RUNS}")',
    ),
    # 04 — ensure Swedish prints replaced
    ('print("\\n🔍 Kontroller:")', 'print("\\n🔍 Checks:")'),
    ('print("\\n📁 Genererade filer:")', 'print("\\n📁 Generated files:")'),
]


def replace_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def process_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        src = "".join(cell.get("source", []))
        new_src = replace_text(src)
        if new_src != src:
            cell["source"] = [line for line in new_src.splitlines(keepends=True)]
            changed = True
    if changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: fix_language_notebooks.py <notebook.ipynb|directory> [more ...]")
        return 2
    any_changed = False
    targets: list[Path] = []
    for arg in argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"[WARN] Not found: {p}")
            continue
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.ipynb")))
        else:
            targets.append(p)

    if not targets:
        print("[WARN] No notebooks found to process")
        return 0

    for nb_path in targets:
        changed = process_notebook(nb_path)
        print(f"[OK] {nb_path} – {'updated' if changed else 'no changes'}")
        any_changed = any_changed or changed
    return 0 if any_changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
