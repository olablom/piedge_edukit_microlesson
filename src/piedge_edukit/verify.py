#!/usr/bin/env python3
# file: src/piedge_edukit/verify.py

import argparse, json, re
from pathlib import Path


def _try_parse_text(text: str) -> float | None:
    patterns = [
        r"(?im)^\s*Mean\s+latency\s*:\s*([-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?)\s*ms\b",
        r"(?im)^\s*Mean\s+latency\s*\(ms\)\s*:\s*([-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def _try_parse_json(p: Path) -> float | None:
    alt_json = p.with_suffix(".json")
    if alt_json.exists():
        try:
            j = json.loads(alt_json.read_text(encoding="utf-8", errors="ignore"))
            for k in ("mean_ms", "mean_latency_ms", "mean", "mean_latency"):
                if k in j:
                    return float(j[k])
        except Exception:
            return None
    return None


def read_mean_ms_from_summary(path: str) -> float:
    p = Path(path)

    # Candidate files to try (given, generic, typical copies)
    candidates: list[Path] = [p]
    generic = p.with_name("latency_summary.txt")
    fp32_copy = p.with_name("latency_summary_fp32.txt")
    int8_copy = p.with_name("latency_summary_int8.txt")
    for c in (generic, fp32_copy, int8_copy):
        if c not in candidates:
            candidates.append(c)

    # Try all candidates: parse text first, then JSON fallback
    last_head = None
    for cand in candidates:
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="ignore")
            val = _try_parse_text(text)
            if val is not None:
                return val
            json_val = _try_parse_json(cand)
            if json_val is not None:
                return json_val
            last_head = "\n".join(text.splitlines()[:10])

    # If nothing matched, raise with context from the original file
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:10])
    else:
        head = last_head or "<file not found>"
    raise ValueError(
        f"Could not parse mean latency from {path}\n--- first lines ---\n{head}\n-------------------"
    )


def load_accuracy(acc_json_path: str):
    d = json.loads(Path(acc_json_path).read_text(encoding="utf-8", errors="ignore"))
    # Tålig mot olika nycklar
    mae = (
        d.get("mae")
        or d.get("MAE")
        or d.get("mean_absolute_error")
        or d.get("mae_mean")
    )
    passed = d.get("pass") if "pass" in d else d.get("passed")
    thresh = d.get("threshold") or d.get("max_mae") or d.get("max_mae_threshold")
    if mae is None:
        # Om rapporten är en lista av metrics, ta första som har mae
        if isinstance(d, list):
            for item in d:
                if isinstance(item, dict) and ("mae" in item or "MAE" in item):
                    mae = item.get("mae", item.get("MAE"))
                    break
    return (
        float(mae),
        (bool(passed) if passed is not None else None),
        (float(thresh) if thresh is not None else None),
        d,
    )


def write_receipt(path: str, status: str, metrics: dict, details: dict):
    out = {
        "status": status,
        "metrics": metrics,
        "details": details,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    # Deterministic behavior for any sampling logic in future extensions
    try:
        import numpy as _np

        _np.random.seed(42)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-fp32", required=True)
    ap.add_argument("--lat-int8", required=True)
    ap.add_argument("--acc-json", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument(
        "--progress", required=False
    )  # alias/duplikat – vi skriver till båda om angivet
    ap.add_argument("--min-speedup-pct", type=float, default=-10.0)
    args = ap.parse_args()

    try:
        fp32_ms = read_mean_ms_from_summary(args.lat_fp32)
    except Exception as e:
        raise SystemExit(
            f"Missing or unparsable FP32 latency summary.\nReason: {e}\n"
            "Tip: Run the benchmark steps (FP32) in run_lesson.sh or the all-in-one notebook to generate reports/latency_summary_fp32.txt/.json."
        )

    try:
        int8_ms = read_mean_ms_from_summary(args.lat_int8)
    except Exception as e:
        raise SystemExit(
            f"Missing or unparsable INT8 latency summary.\nReason: {e}\n"
            "Tip: Run the quantization + benchmark steps (INT8) to generate reports/latency_summary_int8.txt/.json."
        )

    # speedup_pct: positivt när INT8 är snabbare
    speedup_pct = (fp32_ms - int8_ms) / fp32_ms * 100.0 if fp32_ms > 0 else 0.0

    mae, acc_pass, thresh, acc_raw = load_accuracy(args.acc_json)
    if acc_pass is None:
        # Om evaluate-rapporten inte satte bool, avgör via tröskel om vi har den
        acc_pass = True if (thresh is None or mae <= thresh) else False

    ok = (acc_pass is True) and (speedup_pct >= args.min_speedup_pct)

    status = "PASS" if ok else "FAIL"
    metrics = {
        "fp32_mean_ms": fp32_ms,
        "int8_mean_ms": int8_ms,
        "speedup_pct": speedup_pct,
        "mae": mae,
        "max_mae_threshold": thresh,
        "accuracy_pass": bool(acc_pass),
        "min_speedup_pct": args.min_speedup_pct,
    }
    details = {
        "latency_summaries": {
            "fp32": args.lat_fp32,
            "int8": args.lat_int8,
        },
        "accuracy_json": args.acc_json,
        "raw_accuracy": acc_raw,
    }

    # Add soft guidance when accuracy seems too low for typical preprocessing
    if (
        metrics.get("mae") is not None
        and metrics["mae"] > (metrics.get("max_mae_threshold") or 0.02)
        and metrics["mae"] > 0.4
    ):
        details.setdefault("notes", "")
        details["notes"] += (
            "Warning: High MAE — verify that preprocessing in evaluation matches training (resize, normalization)."
        )

    write_receipt(args.receipt, status, metrics, details)
    # skriv även till --progress om det särskiljs
    if args.progress and args.progress != args.receipt:
        write_receipt(args.progress, status, metrics, details)

    print(f"VERIFY RESULT: {status}")
    print(f"  FP32 mean: {fp32_ms:.6f} ms")
    print(f"  INT8 mean: {int8_ms:.6f} ms")
    print(
        f"  Speedup:   {speedup_pct:.2f} %  (min required: {args.min_speedup_pct:.2f} %)"
    )
    print(f"  MAE:       {mae:.6f}  (threshold: {thresh})")
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
