#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# benchmark.py - Latency benchmarking for PiEdge EduKit

import time
import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import sys
import platform
import psutil
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

try:
    # Prefer absolute imports so the file can be executed both as a module and as a script
    from piedge_edukit.preprocess import (
        PreprocessConfig,
        validate_preprocessing_consistency,
    )
    from piedge_edukit.labels import LabelManager, validate_labels_integrity
except ImportError:  # Allow running via file path without installation
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))
    from piedge_edukit.preprocess import (
        PreprocessConfig,
        validate_preprocessing_consistency,
    )
    from piedge_edukit.labels import LabelManager, validate_labels_integrity


def ensure_dir(p: str) -> Path:
    """Ensure directory exists, create if needed."""
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


class LatencyBenchmark:
    """Latency benchmarking with deterministic methodology."""

    def __init__(
        self,
        model_path: str,
        data_dir: str = None,
        output_dir: str = "reports",
        use_fakedata: bool = False,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = ensure_dir(output_dir)
        self.use_fakedata = use_fakedata

        # Benchmark parameters (Smoke Test defaults)
        self.warmup_runs = 1
        self.benchmark_runs = 3
        self.batch_size = 1  # Single image inference

        # Initialize components
        self.preprocess_config = PreprocessConfig()
        self.label_manager = LabelManager()

        # System info
        self.system_info = self._get_system_info()

        # Results storage
        self.results = []

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for reporting."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "onnxruntime_version": ort.__version__,
            "piedge_edukit_version": self._get_package_version(),
        }

        # Try to get CPU info
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpu_info = f.read()
                    if "Raspberry Pi" in cpu_info:
                        info["device"] = "Raspberry Pi"
                    else:
                        info["device"] = "PC/Laptop"
            else:
                info["device"] = "PC/Laptop"
        except:
            info["device"] = "Unknown"

        # Get CPU governor (Linux only)
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    info["cpu_governor"] = result.stdout.strip()
                else:
                    info["cpu_governor"] = "Unknown"
            else:
                info["cpu_governor"] = "N/A"
        except:
            info["cpu_governor"] = "Unknown"

        return info

    def _get_package_version(self) -> str:
        """Get PiEdge EduKit package version."""
        try:
            from importlib.metadata import version, PackageNotFoundError

            return version("piedge-edukit")
        except PackageNotFoundError:
            return "Unknown"

    def _load_model(self, providers: List[str] = None) -> ort.InferenceSession:
        """Load ONNX model with specified providers (default: CPU only)."""
        try:
            # Default to CPU provider (golden path)
            if providers is None:
                providers = ["CPUExecutionProvider"]

            # Try to use specified providers, fallback to CPU
            try:
                session = ort.InferenceSession(
                    str(self.model_path), providers=providers
                )
            except Exception:
                print(f"[WARNING] Failed to use {providers}, falling back to CPU")
                session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )

            print(f"[OK] Model loaded successfully")
            print(f"  Providers: {session.get_providers()}")
            # Cache input metadata for downstream logic (fake data, feeds)
            self.input_name = session.get_inputs()[0].name
            self.input_shape = session.get_inputs()[0].shape
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output shape: {session.get_outputs()[0].shape}")

            return session

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _generate_fake_test_data(self) -> List[np.ndarray]:
        """Generate fake test data matching the model's input rank (2D or 4D)."""
        print("[INFO] Generating fake test data for benchmarking")

        num_samples = 50  # small smoke set
        input_shape = getattr(self, "input_shape", [None, 64])
        rank = len(input_shape)

        test_data: List[np.ndarray] = []
        if rank == 2:
            # [N, F] typical for MLP (sklearn → ONNX)
            feat = input_shape[1] if input_shape[1] not in (None, "None") else 64
            X = (np.random.rand(num_samples, int(feat)).astype(np.float32) - 0.5) * 2.0
            test_data = [X[i] for i in range(num_samples)]
            print(f"[OK] Generated {num_samples} fake test rows")
        elif rank == 4:
            # [N, C, H, W] typical for CNNs
            C = int(input_shape[1] or 1)
            H = int(input_shape[2] or 64)
            W = int(input_shape[3] or 64)
            X = (np.random.rand(num_samples, C, H, W).astype(np.float32) - 0.5) * 2.0
            test_data = [X[i] for i in range(num_samples)]
            print(f"[OK] Generated {num_samples} fake test images")
        else:
            # Fallback: flatten everything but batch
            feat = int(
                np.prod([d for d in input_shape[1:] if d not in (None, "None")] or [64])
            )
            X = (np.random.rand(num_samples, feat).astype(np.float32) - 0.5) * 2.0
            test_data = [X[i] for i in range(num_samples)]
            print(f"[OK] Generated {num_samples} fake samples as flat vectors")

        return test_data

    def _prepare_test_data(self) -> List[np.ndarray]:
        """Prepare test data by loading and preprocessing all validation images."""

        if self.use_fakedata:
            return self._generate_fake_test_data()

        # Validate preprocessing and labels
        if not validate_preprocessing_consistency():
            raise RuntimeError("Preprocessing validation failed")

        if not validate_labels_integrity():
            raise RuntimeError("Labels validation failed")

        # Find validation images - handle both flat structure and train/val structure
        test_images = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        if (self.data_dir / "val").exists():
            # Train/val structure - use val for testing
            val_dir = self.data_dir / "val"
            for class_dir in val_dir.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in extensions:
                            test_images.append(img_path)
        else:
            # Flat structure
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in extensions:
                            test_images.append(img_path)

        if not test_images:
            raise ValueError(f"No test images found in {self.data_dir}")

        print(f"Found {len(test_images)} test images")

        # Preprocess all images
        preprocessed_images = []
        transform = self.preprocess_config.get_transform(is_training=False)

        for img_path in tqdm(test_images, desc="Preprocessing images"):
            try:
                from PIL import Image

                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)
                preprocessed_images.append(tensor.numpy())
            except Exception as e:
                print(f"Warning: Failed to preprocess {img_path}: {e}")

        if not preprocessed_images:
            raise ValueError("No images could be preprocessed")

        print(f"Preprocessed {len(preprocessed_images)} images")
        return preprocessed_images

    def _run_warmup(self, session: ort.InferenceSession, test_data: List[np.ndarray]):
        """Run warmup iterations."""
        print(f"Running {self.warmup_runs} warmup iterations...")

        for i in tqdm(range(self.warmup_runs), desc="Warmup"):
            # Select random image
            img = test_data[np.random.randint(0, len(test_data))]
            img_batch = img[np.newaxis, ...]  # Add batch dimension

            # Run inference
            _ = session.run(None, {self.input_name: img_batch})

        print("[OK] Warmup completed")

    def _benchmark_latency(
        self, session: ort.InferenceSession, test_data: List[np.ndarray]
    ) -> List[float]:
        """Benchmark inference latency."""
        print(f"Running {self.benchmark_runs} benchmark iterations...")

        latencies = []

        for i in tqdm(range(self.benchmark_runs), desc="Benchmarking"):
            # Select random image
            img = test_data[np.random.randint(0, len(test_data))]
            img_batch = img[np.newaxis, ...]  # Add batch dimension

            # Measure inference time
            start_time = time.perf_counter()
            _ = session.run(None, {self.input_name: img_batch})
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        return latencies

    def _calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        latencies = np.array(latencies)

        stats = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        }

        return stats

    def _save_results(self, latencies: List[float], stats: Dict[str, float]):
        """Save benchmark results."""

        # Save detailed results
        results_df = pd.DataFrame(
            {"run": range(len(latencies)), "latency_ms": latencies}
        )

        csv_path = self.output_dir / "latency.csv"
        results_df.to_csv(csv_path, index=False)

        # Save summary
        summary_path = self.output_dir / "latency_summary.txt"
        with open(summary_path, "w") as f:
            f.write("PiEdge EduKit - Latency Benchmark Results\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"Version: {self.system_info.get('piedge_edukit_version', 'Unknown')}\n"
            )
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Version Information:\n")
            f.write(f"  Python: {self.system_info['python_version']}\n")
            f.write(f"  ONNX Runtime: {self.system_info['onnxruntime_version']}\n")
            f.write(f"  Platform: {self.system_info['platform']}\n")
            f.write(f"  Device: {self.system_info['device']}\n")
            f.write("\n")

            f.write("System Information:\n")
            for key, value in self.system_info.items():
                if key not in [
                    "python_version",
                    "onnxruntime_version",
                    "platform",
                    "device",
                ]:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("Benchmark Configuration:\n")
            f.write(f"  Model: {self.model_path.name}\n")
            f.write(f"  Warmup runs: {self.warmup_runs}\n")
            f.write(f"  Benchmark runs: {self.benchmark_runs}\n")
            f.write(f"  Batch size: {self.batch_size}\n")
            f.write("\n")

            f.write("Latency Statistics (ms):\n")
            f.write(f"  Mean: {stats['mean']:.3f}\n")
            f.write(f"  Std:  {stats['std']:.3f}\n")
            f.write(f"  Min:  {stats['min']:.3f}\n")
            f.write(f"  Max:  {stats['max']:.3f}\n")
            f.write(f"  P50:  {stats['p50']:.3f}\n")
            f.write(f"  P95:  {stats['p95']:.3f}\n")
            f.write(f"  P99:  {stats['p99']:.3f}\n")
            # Explicit machine-readable median for verify scripts
            f.write(f"\nmedian_ms: {stats['p50']:.3f}\n")

        # Save JSON results
        json_results = {
            "system_info": self.system_info,
            "benchmark_config": {
                "model_path": str(self.model_path),
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
                "batch_size": self.batch_size,
            },
            "statistics": stats,
            "raw_latencies": latencies,
        }

        json_path = self.output_dir / "latency_results.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"[OK] Results saved to {self.output_dir}")

    def _create_plot(self, latencies: List[float], stats: Dict[str, float]):
        """Create latency distribution plot."""

        plt.figure(figsize=(12, 8))

        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(latencies, bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(
            stats["mean"],
            color="red",
            linestyle="--",
            label=f"Mean: {stats['mean']:.3f}ms",
        )
        plt.axvline(
            stats["p50"],
            color="green",
            linestyle="--",
            label=f"P50: {stats['p50']:.3f}ms",
        )
        plt.axvline(
            stats["p95"],
            color="orange",
            linestyle="--",
            label=f"P95: {stats['p95']:.3f}ms",
        )
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.title("Latency Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(latencies, vert=True)
        plt.ylabel("Latency (ms)")
        plt.title("Latency Box Plot")
        plt.grid(True, alpha=0.3)

        # Time series
        plt.subplot(2, 2, 3)
        plt.plot(latencies, alpha=0.7)
        plt.axhline(
            stats["mean"],
            color="red",
            linestyle="--",
            label=f"Mean: {stats['mean']:.3f}ms",
        )
        plt.xlabel("Run")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Statistics table
        plt.subplot(2, 2, 4)
        plt.axis("off")
        stats_text = f"""
Statistics (ms):
Mean: {stats["mean"]:.3f}
Std:  {stats["std"]:.3f}
Min:  {stats["min"]:.3f}
Max:  {stats["max"]:.3f}
P50:  {stats["p50"]:.3f}
P95:  {stats["p95"]:.3f}
P99:  {stats["p99"]:.3f}

System: {self.system_info["device"]}
Runs: {len(latencies)}
        """
        plt.text(
            0.1,
            0.5,
            stats_text,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "latency_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Plot saved to {plot_path}")

    def run_benchmark(self, providers: List[str] = None) -> Dict[str, float]:
        """Run complete latency benchmark."""

        print("Starting latency benchmark...")
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_dir}")
        print(f"Output: {self.output_dir}")

        # Load model
        session = self._load_model(providers)

        # Prepare test data
        test_data = self._prepare_test_data()

        # Run warmup
        self._run_warmup(session, test_data)

        # Run benchmark
        latencies = self._benchmark_latency(session, test_data)

        # Calculate statistics
        stats = self._calculate_statistics(latencies)

        # Save results
        self._save_results(latencies, stats)

        # Create plot
        self._create_plot(latencies, stats)

        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Mean latency: {stats['mean']:.3f} ms")
        print(f"P50 latency:  {stats['p50']:.3f} ms")
        print(f"P95 latency:  {stats['p95']:.3f} ms")
        print(f"Std deviation: {stats['std']:.3f} ms")
        print("=" * 50)

        return stats


def main():
    import argparse
    from pathlib import Path
    import shutil

    parser = argparse.ArgumentParser(description="Benchmark ONNX model latency")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--data-path", help="Path to test data (not needed with --fakedata)"
    )
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs (1=Smoke Test, 50=Pretty Demo)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Benchmark runs (3=Smoke Test, 200=Pretty Demo)",
    )
    parser.add_argument(
        "--fakedata", action="store_true", help="Use FakeData instead of real images"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers (e.g., CUDAExecutionProvider,CPUExecutionProvider)",
    )
    parser.add_argument(
        "--save-summary-as",
        help="Optional path to also copy latency_summary.txt to after run",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.fakedata and not args.data_path:
        parser.error("--data-path is required unless --fakedata is used")

    # Friendly error if model is missing
    if not Path(args.model_path).exists():
        print(
            f"[ERROR] Model file not found: {args.model_path}\nTip: run training first to export an ONNX model."
        )
        raise SystemExit(2)

    # Create benchmark
    benchmark = LatencyBenchmark(
        model_path=args.model_path,
        data_dir=args.data_path,
        output_dir=args.output_dir,
        use_fakedata=args.fakedata,
    )

    # Update parameters
    benchmark.warmup_runs = args.warmup
    benchmark.benchmark_runs = args.runs

    # Run benchmark with specified providers
    stats = benchmark.run_benchmark(providers=args.providers)

    # Optionally copy the summary to a specific filename for downstream tooling
    if args.save_summary_as:
        src_summary = Path(args.output_dir) / "latency_summary.txt"
        dst_summary = Path(args.save_summary_as)
        try:
            dst_summary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_summary, dst_summary)
            print(f"[OK] Summary copied to: {dst_summary}")
        except Exception as e:
            print(f"[WARN] Failed to copy summary to {dst_summary}: {e}")

    print(f"\n[OK] Benchmark completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    # Use package import to ensure absolute imports resolve when executed as a script
    from piedge_edukit.benchmark import main as _main

    _main()
