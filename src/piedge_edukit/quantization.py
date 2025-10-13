#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# quantization.py - Static INT8 quantization for PiEdge EduKit

import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from datetime import datetime
from tqdm import tqdm

from .preprocess import PreprocessConfig, validate_preprocessing_consistency
from .labels import LabelManager, validate_labels_integrity


class QuantizationBenchmark:
    """Static INT8 quantization and comparison."""

    def __init__(
        self,
        model_path: str,
        data_dir: str = None,
        output_dir: str = "reports",
        use_fakedata: bool = False,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_fakedata = use_fakedata

        # Quantization parameters
        self.warmup_runs = 50
        self.benchmark_runs = 200
        self.calibration_size = 100

        # Initialize components
        self.preprocess_config = PreprocessConfig()
        self.label_manager = LabelManager()

        # Results storage
        self.fp32_results = {}
        self.int8_results = {}
        self.comparison_results = {}

    def _generate_fake_calibration_data(self) -> List[np.ndarray]:
        """Generate fake calibration data for quantization."""
        import torch
        from torchvision import datasets, transforms

        print("[INFO] Generating fake calibration data for quantization")

        # Create transform
        transform = self.preprocess_config.get_transform(is_training=False)

        # Generate fake data
        fake_data = datasets.FakeData(
            size=self.calibration_size,
            image_size=(3, 64, 64),
            num_classes=2,
            transform=transform,
        )

        # Convert to numpy arrays
        calibration_data = []
        for i in range(len(fake_data)):
            image, _ = fake_data[i]
            calibration_data.append(image.numpy())

        print(f"[OK] Generated {len(calibration_data)} fake calibration images")
        return calibration_data

    def _prepare_calibration_data(self) -> List[np.ndarray]:
        """Prepare calibration data for quantization."""

        if self.use_fakedata:
            return self._generate_fake_calibration_data()

        # Find calibration images
        calibration_images = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in extensions:
                        calibration_images.append(img_path)

        if not calibration_images:
            raise ValueError(f"No calibration images found in {self.data_dir}")

        print(f"Found {len(calibration_images)} calibration images")

        # Preprocess images
        preprocessed_images = []
        transform = self.preprocess_config.get_transform(is_training=False)

        for img_path in tqdm(calibration_images, desc="Preprocessing calibration data"):
            try:
                from PIL import Image

                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)
                preprocessed_images.append(tensor.numpy())
            except Exception as e:
                print(f"Warning: Failed to preprocess {img_path}: {e}")

        if not preprocessed_images:
            raise ValueError("No images could be preprocessed")

        print(f"Preprocessed {len(preprocessed_images)} calibration images")
        return preprocessed_images

    def _create_calibration_dataset(
        self, calibration_data: List[np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """Create calibration dataset for quantization."""

        calibration_dataset = []

        for img in calibration_data:
            # Add batch dimension
            img_batch = img[np.newaxis, ...]
            calibration_dataset.append({"input": img_batch})

        return calibration_dataset

    def _quantize_model(self, calibration_dataset: List[Dict[str, np.ndarray]]) -> str:
        """Quantize FP32 model to INT8."""

        print("Quantizing model to INT8...")

        # Load original model
        model = onnx.load(str(self.model_path))

        # Create quantized model path
        quantized_path = self.model_path.parent / "model_static.onnx"

        try:
            # Perform static quantization
            from onnxruntime.quantization import quantize_static, QuantType

            quantize_static(
                str(self.model_path),
                str(quantized_path),
                calibration_dataset,
                quant_format=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                reduce_range=False,
                op_types_to_quantize=["Conv", "MatMul", "Add", "Mul"],
            )

            print(f"[OK] Model quantized successfully to {quantized_path}")
            return str(quantized_path)

        except Exception as e:
            print(f"[ERROR] Quantization failed: {e}")
            print("This may be due to unsupported operations or ONNX Runtime version.")
            print("Continuing with FP32 model only.")
            return None

    def _benchmark_model(
        self, model_path: str, test_data: List[np.ndarray], model_type: str
    ) -> Dict[str, float]:
        """Benchmark model latency."""

        print(f"Benchmarking {model_type} model...")

        # Load model
        try:
            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type} model: {e}")

        # Warmup
        for i in tqdm(range(self.warmup_runs), desc=f"Warmup {model_type}"):
            img = test_data[np.random.randint(0, len(test_data))]
            img_batch = img[np.newaxis, ...]
            _ = session.run(None, {"input": img_batch})

        # Benchmark
        latencies = []
        for i in tqdm(range(self.benchmark_runs), desc=f"Benchmark {model_type}"):
            img = test_data[np.random.randint(0, len(test_data))]
            img_batch = img[np.newaxis, ...]

            start_time = time.perf_counter()
            _ = session.run(None, {"input": img_batch})
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "raw_latencies": latencies.tolist(),
        }

        return stats

    def _compare_accuracy(
        self, fp32_path: str, int8_path: str, test_data: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compare accuracy between FP32 and INT8 models."""

        print("Comparing model accuracy...")

        # Load models
        fp32_session = ort.InferenceSession(
            fp32_path, providers=["CPUExecutionProvider"]
        )
        int8_session = ort.InferenceSession(
            int8_path, providers=["CPUExecutionProvider"]
        )

        # Test on subset of data
        test_subset = test_data[: min(100, len(test_data))]

        fp32_predictions = []
        int8_predictions = []

        for img in tqdm(test_subset, desc="Accuracy comparison"):
            img_batch = img[np.newaxis, ...]

            # FP32 prediction
            fp32_output = fp32_session.run(None, {"input": img_batch})
            fp32_pred = np.argmax(fp32_output[0], axis=1)[0]
            fp32_predictions.append(fp32_pred)

            # INT8 prediction
            int8_output = int8_session.run(None, {"input": img_batch})
            int8_pred = np.argmax(int8_output[0], axis=1)[0]
            int8_predictions.append(int8_pred)

        # Calculate accuracy metrics
        fp32_preds = np.array(fp32_predictions)
        int8_preds = np.array(int8_predictions)

        agreement = np.mean(fp32_preds == int8_preds)

        # Calculate output differences
        output_diffs = []
        for img in test_subset:
            img_batch = img[np.newaxis, ...]

            fp32_output = fp32_session.run(None, {"input": img_batch})
            int8_output = int8_session.run(None, {"input": img_batch})

            diff = np.mean(np.abs(fp32_output[0] - int8_output[0]))
            output_diffs.append(diff)

        accuracy_metrics = {
            "prediction_agreement": agreement,
            "mean_output_diff": np.mean(output_diffs),
            "max_output_diff": np.max(output_diffs),
            "test_samples": len(test_subset),
        }

        return accuracy_metrics

    def _get_model_info(self, model_path: str) -> Dict[str, any]:
        """Get model information."""

        info = {
            "file_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
            "path": str(model_path),
        }

        try:
            # Load model
            model = onnx.load(model_path)

            # Get model info
            info["opset_version"] = model.opset_import[0].version

            # Count quantized nodes
            quantized_nodes = 0
            for node in model.graph.node:
                if node.op_type in ["Conv", "MatMul", "Add", "Mul"]:
                    quantized_nodes += 1

            info["quantized_nodes"] = quantized_nodes

        except Exception as e:
            info["error"] = str(e)

        return info

    def _save_comparison_results(self):
        """Save quantization comparison results."""

        # Create comparison DataFrame
        comparison_data = []

        for model_type, results in [
            ("FP32", self.fp32_results),
            ("INT8", self.int8_results),
        ]:
            if results is not None:
                comparison_data.append(
                    {
                        "Model": model_type,
                        "File_Size_MB": results["model_info"]["file_size_mb"],
                        "Mean_Latency_ms": results["latency"]["mean"],
                        "P50_Latency_ms": results["latency"]["p50"],
                        "P95_Latency_ms": results["latency"]["p95"],
                        "Std_Latency_ms": results["latency"]["std"],
                    }
                )
            else:
                comparison_data.append(
                    {
                        "Model": model_type,
                        "File_Size_MB": "N/A",
                        "Mean_Latency_ms": "N/A",
                        "P50_Latency_ms": "N/A",
                        "P95_Latency_ms": "N/A",
                        "Std_Latency_ms": "N/A",
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        # Save CSV to reports directory for consistency
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / "quantization_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)

        # Save detailed results
        detailed_results = {
            "fp32_results": self.fp32_results,
            "int8_results": self.int8_results,
            "comparison_results": self.comparison_results,
        }

        # Save detailed results to reports directory
        json_path = reports_dir / "quantization_results.json"
        with open(json_path, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save summary text to reports directory
        summary_path = reports_dir / "quantization_summary.txt"
        with open(summary_path, "w") as f:
            f.write("PiEdge EduKit - Quantization Comparison Results\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Version: {self._get_package_version()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Model Comparison:\n")
            f.write(
                f"  FP32 Size: {self.fp32_results['model_info']['file_size_mb']:.2f} MB\n"
            )
            if self.int8_results is not None:
                f.write(
                    f"  INT8 Size: {self.int8_results['model_info']['file_size_mb']:.2f} MB\n"
                )
            else:
                f.write("  INT8 Size: N/A (quantization failed)\n")
            if self.int8_results is not None:
                f.write(
                    f"  Size Reduction: {self.comparison_results['summary']['size_reduction_percent']:.1f}%\n\n"
                )
            else:
                f.write("  Size Reduction: N/A (quantization failed)\n\n")

            f.write("Latency Comparison:\n")
            f.write(f"  FP32 Mean: {self.fp32_results['latency']['mean']:.3f} ms\n")
            if self.int8_results is not None:
                f.write(f"  INT8 Mean: {self.int8_results['latency']['mean']:.3f} ms\n")
            else:
                f.write("  INT8 Mean: N/A (quantization failed)\n")
            if self.int8_results is not None:
                f.write(
                    f"  Speedup: {self.comparison_results['summary']['speedup_factor']:.2f}x\n\n"
                )
            else:
                f.write("  Speedup: N/A (quantization failed)\n\n")

            if self.int8_results is not None:
                f.write("Accuracy Comparison:\n")
                f.write(
                    f"  Prediction Agreement: {self.comparison_results['accuracy']['prediction_agreement']:.3f}\n"
                )
                f.write(
                    f"  Mean Output Diff: {self.comparison_results['accuracy']['mean_output_diff']:.6f}\n"
                )
                f.write(
                    f"  Test Samples: {self.comparison_results['accuracy']['test_samples']}\n"
                )
            else:
                f.write("Accuracy Comparison: N/A (quantization failed)\n")

        print(f"[OK] Comparison results saved to {reports_dir}")

    def _get_package_version(self) -> str:
        """Get PiEdge EduKit package version."""
        try:
            from importlib.metadata import version, PackageNotFoundError

            return version("piedge-edukit")
        except PackageNotFoundError:
            return "Unknown"

    def _create_comparison_plot(self):
        """Create quantization comparison plots."""

        if self.int8_results is None:
            # Create fallback plot for failed quantization
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes.text(
                0.5,
                0.5,
                "Quantization Failed\nFP32 model only",
                ha="center",
                va="center",
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 1)
            axes.set_title("Quantization Status")
            axes.axis("off")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Size comparison
            axes[0, 0].bar(
                ["FP32", "INT8"],
                [
                    self.fp32_results["model_info"]["file_size_mb"],
                    self.int8_results["model_info"]["file_size_mb"],
                ],
                color=["blue", "red"],
                alpha=0.7,
            )
            axes[0, 0].set_ylabel("File Size (MB)")
            axes[0, 0].set_title("Model Size Comparison")
            axes[0, 0].grid(True, alpha=0.3)

            # Latency comparison
            latency_data = [
                self.fp32_results["latency"]["mean"],
                self.int8_results["latency"]["mean"],
            ]
            axes[0, 1].bar(
                ["FP32", "INT8"], latency_data, color=["blue", "red"], alpha=0.7
            )
            axes[0, 1].set_ylabel("Mean Latency (ms)")
            axes[0, 1].set_title("Latency Comparison")
            axes[0, 1].grid(True, alpha=0.3)

            # Latency distribution comparison
            fp32_latencies = self.fp32_results["raw_latencies"]
            int8_latencies = self.int8_results["raw_latencies"]

            axes[1, 0].hist(
                fp32_latencies, bins=30, alpha=0.7, label="FP32", color="blue"
            )
            axes[1, 0].hist(
                int8_latencies, bins=30, alpha=0.7, label="INT8", color="red"
            )
            axes[1, 0].set_xlabel("Latency (ms)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Latency Distribution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Summary metrics
            axes[1, 1].axis("off")
            summary_text = f"""
Quantization Summary:

Size Reduction: {self.comparison_results["summary"]["size_reduction_percent"]:.1f}%
Speedup: {self.comparison_results["summary"]["speedup_factor"]:.2f}x
Accuracy Agreement: {self.comparison_results["summary"]["accuracy_agreement"]:.3f}

FP32 Model:
  Size: {self.fp32_results["model_info"]["file_size_mb"]:.2f} MB
  Mean Latency: {self.fp32_results["latency"]["mean"]:.3f} ms

INT8 Model:
  Size: {self.int8_results["model_info"]["file_size_mb"]:.2f} MB
  Mean Latency: {self.int8_results["latency"]["mean"]:.3f} ms
        """
            axes[1, 1].text(
                0.1,
                0.5,
                summary_text,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )

        plt.tight_layout()

        # Save plot to reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        plot_path = reports_dir / "quantization_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Comparison plot saved to {plot_path}")

    def run_quantization_benchmark(self) -> Dict[str, any]:
        """Run complete quantization benchmark."""

        print("Starting quantization benchmark...")
        print(f"FP32 Model: {self.model_path}")
        print(f"Data: {self.data_dir}")
        print(f"Output: {self.output_dir}")

        # Validate preprocessing and labels
        if not validate_preprocessing_consistency():
            raise RuntimeError("Preprocessing validation failed")

        if not validate_labels_integrity():
            raise RuntimeError("Labels validation failed")

        # Prepare calibration data
        calibration_data = self._prepare_calibration_data()
        calibration_dataset = self._create_calibration_dataset(calibration_data)

        # Quantize model
        quantized_path = self._quantize_model(calibration_dataset)

        # Benchmark FP32 model
        print("\nBenchmarking FP32 model...")
        self.fp32_results = {
            "latency": self._benchmark_model(
                str(self.model_path), calibration_data, "FP32"
            ),
            "model_info": self._get_model_info(str(self.model_path)),
        }
        self.fp32_results["raw_latencies"] = self._benchmark_model(
            str(self.model_path), calibration_data, "FP32"
        )["raw_latencies"]

        # Benchmark INT8 model (if quantization succeeded)
        if quantized_path:
            print("\nBenchmarking INT8 model...")
            self.int8_results = {
                "latency": self._benchmark_model(
                    quantized_path, calibration_data, "INT8"
                ),
                "model_info": self._get_model_info(quantized_path),
            }
            self.int8_results["raw_latencies"] = self._benchmark_model(
                quantized_path, calibration_data, "INT8"
            )["raw_latencies"]

            # Compare accuracy
            print("\nComparing accuracy...")
            self.comparison_results = {
                "accuracy": self._compare_accuracy(
                    str(self.model_path), quantized_path, calibration_data
                )
            }
        else:
            print("\nSkipping INT8 benchmarking (quantization failed)")
            self.int8_results = None
            self.comparison_results = None

        # Calculate summary (if quantization succeeded)
        if self.int8_results and self.comparison_results:
            self.comparison_results["summary"] = {
                "size_reduction_percent": (
                    1
                    - self.int8_results["model_info"]["file_size_mb"]
                    / self.fp32_results["model_info"]["file_size_mb"]
                )
                * 100,
                "speedup_factor": self.fp32_results["latency"]["mean"]
                / self.int8_results["latency"]["mean"],
                "accuracy_agreement": self.comparison_results["accuracy"][
                    "prediction_agreement"
                ],
            }
        else:
            # Create fallback summary for failed quantization
            self.comparison_results = {
                "summary": {
                    "quantization_status": "FAILED",
                    "error_message": "INT8 quantization failed - continuing with FP32 only",
                    "fp32_latency_ms": self.fp32_results["latency"]["mean"],
                    "fp32_model_size_mb": self.fp32_results["model_info"][
                        "file_size_mb"
                    ],
                }
            }

        # Save results
        self._save_comparison_results()
        self._create_comparison_plot()

        # Print summary
        print("\n" + "=" * 60)
        print("QUANTIZATION BENCHMARK RESULTS")
        print("=" * 60)
        if self.int8_results is not None:
            print(
                f"Size Reduction: {self.comparison_results['summary']['size_reduction_percent']:.1f}%"
            )
            print(
                f"Speedup: {self.comparison_results['summary']['speedup_factor']:.2f}x"
            )
            print(
                f"Accuracy Agreement: {self.comparison_results['summary']['accuracy_agreement']:.3f}"
            )
        else:
            print("Quantization Status: FAILED")
            print(f"Error: {self.comparison_results['summary']['error_message']}")
            print(
                f"FP32 Latency: {self.comparison_results['summary']['fp32_latency_ms']:.3f} ms"
            )
            print(
                f"FP32 Size: {self.comparison_results['summary']['fp32_model_size_mb']:.2f} MB"
            )
        print("=" * 60)

        return self.comparison_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quantize and benchmark ONNX model")
    parser.add_argument("--model-path", required=True, help="Path to FP32 ONNX model")
    parser.add_argument(
        "--data-path", help="Path to calibration data (not needed with --fakedata)"
    )
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark runs")
    parser.add_argument(
        "--fakedata", action="store_true", help="Use FakeData for calibration"
    )
    parser.add_argument(
        "--calib-size", type=int, default=100, help="Calibration dataset size"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.fakedata and not args.data_path:
        parser.error("--data-path is required unless --fakedata is used")

    # Create quantization benchmark
    benchmark = QuantizationBenchmark(
        model_path=args.model_path,
        data_dir=args.data_path,
        output_dir=args.output_dir,
        use_fakedata=args.fakedata,
    )

    # Update parameters
    benchmark.calibration_size = args.calib_size
    benchmark.warmup_runs = args.warmup
    benchmark.benchmark_runs = args.runs

    # Run benchmark
    results = benchmark.run_quantization_benchmark()

    print(f"\n[OK] Quantization benchmark completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
