#!/usr/bin/env python3
# filename: verify.py
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRESS_DIR = ROOT / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(p: Path):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def main() -> int:
    checks = []
    metrics = {}

    # 1) Model exists
    model_path = ROOT / "models" / "model.onnx"
    ok_model = model_path.exists() and model_path.stat().st_size > 0
    checks.append(
        {
            "name": "onnx_model_exists",
            "ok": ok_model,
            "reason": "models/model.onnx present and non-empty"
            if ok_model
            else "models/model.onnx missing or empty",
        }
    )

    # 2) Latency results (optional but recommended)
    lat_path = ROOT / "reports" / "latency_results.json"
    lat = read_json(lat_path)
    if lat and isinstance(lat, dict):
        # normalize keys commonly produced by your scripts
        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            if k in lat:
                metrics[k] = float(lat[k])
        # fallback: extract from nested statistics { mean, p50, p95, p99 }
        stats = lat.get("statistics")
        if isinstance(stats, dict):
            mapping = {
                "mean": "mean_ms",
                "p50": "p50_ms",
                "p95": "p95_ms",
                "p99": "p99_ms",
            }
            for src, dst in mapping.items():
                if src in stats:
                    metrics[dst] = float(stats[src])
        ok_lat = all(k in metrics for k in ("p50_ms", "p95_ms"))
        checks.append(
            {
                "name": "latency_report_present",
                "ok": ok_lat,
                "reason": "reports/latency_results.json with p50_ms & p95_ms"
                if ok_lat
                else "latency report missing required keys",
            }
        )
    else:
        checks.append(
            {
                "name": "latency_report_present",
                "ok": False,
                "reason": "reports/latency_results.json missing",
            }
        )

    # 3) Quantization summary (optional; passes even if absent)
    q_path = ROOT / "reports" / "quantization_summary.txt"
    ok_q = q_path.exists()
    checks.append(
        {
            "name": "quantization_attempted",
            "ok": ok_q,
            "reason": "quantization summary present"
            if ok_q
            else "quantization summary not found (accepted on Windows)",
        }
    )

    # 4) Confusion matrix (optional image)
    cm_path = ROOT / "reports" / "confusion_matrix.png"
    ok_cm = cm_path.exists()
    checks.append(
        {
            "name": "confusion_matrix_present",
            "ok": ok_cm,
            "reason": "confusion matrix image present"
            if ok_cm
            else "confusion matrix not found",
        }
    )

    # PASS logic: must have model + latency metrics; others optional
    is_pass = all(
        c["ok"]
        for c in checks
        if c["name"] in ("onnx_model_exists", "latency_report_present")
    )

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pass": bool(is_pass),
        "checks": checks,
        "metrics": metrics,
        "notes": "Lesson verification. Fails if no ONNX model or latency metrics.",
    }

    # Write receipt (always)
    (PROGRESS_DIR / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )

    # Append/update lesson progress (last run snapshot)
    lp_path = PROGRESS_DIR / "lesson_progress.json"
    progress = read_json(lp_path) or {}
    progress["last_run"] = {"timestamp": receipt["timestamp"], "pass": receipt["pass"]}
    progress["last_checks"] = checks
    progress["last_metrics"] = metrics
    lp_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    # Exit code reflects pass/fail (still wrote JSON above)
    return 0 if is_pass else 2


if __name__ == "__main__":
    sys.exit(main())
