#!/usr/bin/env python3
# file: src/piedge_edukit/evaluate_accuracy.py

import argparse
import json
import numpy as np
import onnxruntime as ort
import os
import sys
import textwrap


def run_onnx(model_path: str, x: np.ndarray, input_name: str = "input") -> np.ndarray:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session.run(None, {input_name: x})[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare FP32 vs INT8 outputs")
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--input-npz", default="artifacts/calib.npz")
    ap.add_argument("--npz-key", default="input")
    ap.add_argument("--report", default="reports/accuracy.json")
    ap.add_argument("--summary", default="reports/accuracy_summary.txt")
    ap.add_argument("--max-mae", type=float, default=0.02, help="Fail if MAE > this")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--feat", type=int, default=32)
    args = ap.parse_args()

    # Load inputs
    if os.path.exists(args.input_npz):
        data = np.load(args.input_npz)
        x = data[args.npz_key].astype(np.float32)
        if x.ndim == 2 and x.shape[0] > args.batch:
            x = x[: args.batch]
    else:
        x = np.random.randn(args.batch, args.feat).astype(np.float32)

    y32 = run_onnx(args.fp32, x)
    y8 = run_onnx(args.int8, x)

    diff = y32 - y8
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    mx = float(np.max(np.abs(diff)))
    # Cosine similarity per row then mean
    num = np.sum(y32 * y8, axis=1)
    den = np.linalg.norm(y32, axis=1) * np.linalg.norm(y8, axis=1) + 1e-12
    cossim = float(np.mean(num / den))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    out = {
        "inputs": {"batch": int(x.shape[0]), "feat": int(x.shape[1])},
        "metrics": {"mae": mae, "mse": mse, "max_abs": mx, "cosine_sim": cossim},
        "thresholds": {"max_mae": args.max_mae},
        "pass": mae <= args.max_mae,
        "models": {"fp32": args.fp32, "int8": args.int8},
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = textwrap.dedent(
        f"""
        ACCURACY COMPARISON
        ===================
        Inputs:  x.shape = {tuple(x.shape)}
        MAE:     {mae:.6f}
        MSE:     {mse:.6f}
        MAX:     {mx:.6f}
        CosSim:  {cossim:.6f}
        Threshold (max_mae): {args.max_mae:.6f}
        RESULT:  {"PASS" if out["pass"] else "FAIL"}
        """
    ).strip()
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
