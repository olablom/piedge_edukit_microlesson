#!/usr/bin/env python3
# file: src/piedge_edukit/prepare_model.py

import argparse
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, in_dim: int = 256, hidden: int = 2048, out_dim: int = 10
    ) -> None:
        super().__init__()
        # Deeper MLP to increase compute and make INT8 optimizations visible
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-fp32", required=True)
    p.add_argument("--calib-npz", default="artifacts/calib.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-dim", type=int, default=256)
    p.add_argument("--out-dim", type=int, default=10)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--calib-samples", type=int, default=128)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = MLP(args.in_dim, args.hidden, args.out_dim).eval()
    dummy = torch.randn(1, args.in_dim)

    # Export ONNX with stable names
    torch.onnx.export(
        model,
        dummy,
        args.out_fp32,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
    )

    # Create calibration NPZ (2D) matching ONNX input name 'input'
    calib = np.random.randn(args.calib_samples, args.in_dim).astype(np.float32)
    np.savez(args.calib_npz, input=calib)


if __name__ == "__main__":
    main()
