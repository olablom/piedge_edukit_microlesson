#!/usr/bin/env python3
"""
Quick fix for Swedish text in notebooks - simple replacements
"""

import pathlib
import re


def fix_notebook(notebook_path):
    """Fix Swedish text in a single notebook."""
    print(f"Processing {notebook_path}...")

    # Read notebook
    with open(notebook_path, encoding="utf-8") as f:
        content = f.read()

    # Simple replacements
    replacements = [
        (r"måste kallas före", "must be called before"),
        (r"Använd `model\.train\(\)` under", "Use `model.train()` during"),
        (r"under Training", "during training"),
        (r"CPU\(\)", "cpu()"),
        (r"på edge", "on edge"),
        (r"MÅSTE följa", "MUST follow"),
        (r"detta exakt", "this exactly"),
        (r"Viktigt", "Important"),
        (r"Om normalisering ändras", "If normalization changes"),
        (r"i träning", "in training"),
        (r"uppdatera edge-pipeline", "update edge-pipeline"),
        (r"och exportera ny ONNX", "and export new ONNX"),
        (r"du senare vill", "you later want"),
        (r"normalisera \(t\.ex\.", "normalize (e.g."),
        (r"gör det i både", "do it in both"),
        (r"och inferens och bumpa", "and inference and bump"),
        (r"Störst effekt", "Biggest effect"),
        (r"konvergenshastighet/stabilitet", "convergence speed/stability"),
        (r"generaliseringsbeteende", "generalization behavior"),
        (r"För hög", "For high"),
        (r"divergens", "divergence"),
        (r"för stor batch", "for large batch"),
        (r"eller för många epoker", "or for many epochs"),
        (r"överanpassning på liten", "overfitting on small"),
        (r"ger portabilitet", "gives portability"),
        (r"språk/ramverks-agnostiskt", "language/framework-agnostic"),
        (r"Faster/mer förutsägbar", "Faster/more predictable"),
        (r"nödvändigt för", "necessary for"),
        (r"där PyTorch inte", "where PyTorch is not"),
        (r"optimalt", "optimal"),
        (r"Utan `eval\(\)`", "Without `eval()`"),
        (r"fångas träningsbeteende", "captures training behavior"),
        (r"dropout aktiv", "dropout active"),
        (r"BN-statistik fel", "BN statistics wrong"),
        (r"i grafen", "in the graph"),
        (r"icke-deterministisk", "non-deterministic"),
        (r"sämre inferens", "worse inference"),
        (r"Understanding what happens", "Understanding what happens"),
        (r"understand How Training works", "understand how training works"),
        (r"och experiments with", "and experiment with"),
        (r"in detta notebook kommer we att", "in this notebook we will"),
        (r"understand What FakeData is", "understand what FakeData is"),
        (r"och why we use It", "and why we use it"),
        (r"Se How dataset-pipeline", "See how dataset-pipeline"),
        (r"experiments with different", "experiment with different"),
        (r"understand why we export", "understand why we export"),
        (r"Run cellerna in ordning", "Run cells in order"),
        (r"och läs förklaringarna", "and read the explanations"),
        (r"experiments gärna with värdena", "experiment freely with the values"),
        (r"What is FakeData och why", "What is FakeData and why"),
        (r"use we It", "use it"),
        (r"FakeData is syntetiska", "FakeData is synthetic"),
        (r"that PyTorch genererar", "that PyTorch generates"),
        (r"It is perfekt for", "It is perfect for"),
        (r"no nedladdning av", "no downloading of"),
        (r"stora dataset", "large dataset"),
        (r"Reproducerbarhet", "Reproducibility"),
        (r"same data varje gång", "same data every time"),
        (r"Undervisning", "Education"),
        (r"fokus on algoritmer", "focus on algorithms"),
        (r"inte datahantering", "not data management"),
        (r"Klicka for att se", "Click to see"),
        (r"What FakeData innehåller", "what FakeData contains"),
        (r"FakeData genererar", "FakeData generates"),
        (r"Slumpmässiga RGB-images", "Random RGB images"),
        (r"64x64 pixlar", "64x64 pixels"),
        (r"Slumpmässiga klasser", "Random classes"),
        (r"same struktur that", "same structure as"),
        (r"riktiga bilddataset", "real image datasets"),
        (r"Låt oss skapa", "Let us create"),
        (r"en liten FakeData", "a small FakeData"),
        (r"för att se vad den", "to see what it"),
        (r"innehåller", "contains"),
        (r"Skapa FakeData med", "Create FakeData with"),
        (r"Visa första bilden", "Show first image"),
        (r"Bildstorlek", "Image size"),
        (r"Klass", "Class"),
        (r"Pixelvärden", "Pixel values"),
        (r"Visa bilden", "Show image"),
        (r"FakeData - Klass", "FakeData - Class"),
        (r"Experimentera med Träning", "Experiment with Training"),
        (r"Nu ska vi träna", "Now we will train"),
        (r"en modell och se", "a model and see"),
        (r"hur olika inställningar", "how different settings"),
        (r"påverkar resultatet", "affect the result"),
        (r"antal genomgångar av", "number of passes through"),
        (r"antal bilder per", "number of images per"),
        (r"börja från noll", "start from scratch"),
        (r"förtränade vikter", "pretrained weights"),
        (r"Snabb träning", "Quick training"),
        (r"ingen pretrained", "no pretrained"),
        (r"Visa träningsresultat", "Show training results"),
        (r"från Experiment", "from Experiment"),
        (r"Resultaten visas ovan", "Results are shown above"),
        (r"i träningsloopen", "in the training loop"),
        (r"Du kan jämföra", "You can compare"),
        (r"accuracy och loss", "accuracy and loss"),
        (r"mellan olika experiment", "between different experiments"),
        (r"Vad händer med", "What happens with"),
        (r"överfitting när du", "overfitting when you"),
        (r"höjer epochs", "increase epochs"),
        (r"Med fler epochs", "With more epochs"),
        (r"kan modellen lära sig", "can the model learn"),
        (r"träningsdata för bra", "training data too well"),
        (r"dåligt generalisera", "poorly generalize"),
        (r"till nya data", "to new data"),
        (r"Detta kallas", "This is called"),
        (r"Kör samma träning", "Run the same training"),
        (r"men med", "but with"),
        (r"och jämför accuracy", "and compare accuracy"),
        (r"på tränings- vs", "on training vs"),
        (r"valideringsdata", "validation data"),
        (r"Varför exporterar vi", "Why do we export"),
        (r"till ONNX \(för", "to ONNX (for"),
        (r"ONNX är ett", "ONNX is a"),
        (r"standardformat som", "standard format that"),
        (r"fungerar på många", "works on many"),
        (r"plattformar \(CPU", "platforms (CPU"),
        (r"mobil, edge\)", "mobile, edge)"),
        (r"Det gör modellen", "It makes the model"),
        (r"portabel och optimerad", "portable and optimized"),
        (r"Snabbare inference än", "Faster inference than"),
        (r"Mindre minnesusening", "Less memory usage"),
        (r"Fungerar på", "Works on"),
        (r"Stöd för kvantisering", "Support for quantization"),
        (r"Your own experiment", "Your own experiment"),
        (r"Träna en modell", "Train a model"),
        (r"andra inställningar", "other settings"),
        (r"och jämför resultaten", "and compare results"),
        (r"Öka epochs till", "Increase epochs to"),
        (r"Ändra batch_size till", "Change batch_size to"),
        (r"Testa med och utan", "Test with and without"),
        (r"Ändra dessa värden", "Change these values"),
        (r"Träningen körs", "Training runs"),
        (r"automatiskt i nästa", "automatically in next"),
        (r"Implementera ditt experiment", "Implement your experiment"),
        (r"Ändra värdena nedan", "Change the values below"),
        (r"och run träningen", "and run training"),
        (r"Mitt experiment", "My experiment"),
        (r"Create new data loader", "Create new data loader"),
        (r"with different hyperparameters", "with different hyperparameters"),
        (r"Create new model and optimizer", "Create new model and optimizer"),
        (r"Train for specified epochs", "Train for specified epochs"),
        (r"Training with batch_size", "Training with batch_size"),
        (r"My experiment completed", "My experiment completed"),
        (r"you have now learned", "you have now learned"),
        (r"What FakeData is och", "What FakeData is and"),
        (r"why we use It", "why we use it"),
        (r"How Training works", "How training works"),
        (r"with different hyperparameters", "with different hyperparameters"),
        (r"why ONNX export is", "why ONNX export is"),
        (r"important for edge deployment", "important for edge deployment"),
        (r"Next step", "Next step"),
        (r"Go to", "Go to"),
        (r"för att understand", "to understand"),
        (r"How we measure", "how we measure"),
        (r"model\'s performance", "model's performance"),
        (r"Important concepts", "Important concepts"),
        (r"number of passes av", "number of passes through"),
        (r"number of images per", "number of images per"),
        (r"training step", "training step"),
        (r"Pretrained weights", "Pretrained weights"),
        (r"från ImageNet", "from ImageNet"),
        (r"standard format for", "standard format for"),
        (r"edge deployment", "edge deployment"),
    ]

    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    if content != original_content:
        with open(notebook_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[FIXED] {notebook_path}")
        return True
    else:
        print(f"[OK] No changes needed for {notebook_path}")
        return False


def main():
    """Fix all notebooks in the project."""
    # Find all .ipynb files (excluding .bak files)
    notebook_files = []
    for pattern in ["notebooks/*.ipynb", "*.ipynb"]:
        notebook_files.extend(pathlib.Path(".").glob(pattern))

    # Filter out .bak files
    notebook_files = [f for f in notebook_files if not f.name.endswith(".bak")]

    if not notebook_files:
        print("No notebooks found")
        return

    print(f"Found {len(notebook_files)} notebooks to process")

    fixed_count = 0
    for notebook_file in notebook_files:
        if fix_notebook(notebook_file):
            fixed_count += 1

    print(f"\n[SUMMARY] Fixed {fixed_count} out of {len(notebook_files)} notebooks")


if __name__ == "__main__":
    main()
