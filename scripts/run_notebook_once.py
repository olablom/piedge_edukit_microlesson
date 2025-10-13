#!/usr/bin/env python3
# filename: scripts/run_notebook_once.py
"""
Execute a single Jupyter notebook using nbclient with Windows event loop policy when needed.
Usage:
  python scripts/run_notebook_once.py notebooks/01_training_and_export.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: run_notebook_once.py <notebook.ipynb>")
        return 2

    nb_path = Path(argv[1])
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found: {nb_path}")
        return 1

    # Windows selector policy to avoid ZMQ warnings
    try:
        import os

        if os.name == "nt":
            import asyncio as aio

            aio.set_event_loop_policy(aio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        allow_errors=False,
        raise_on_iopub_timeout=True,
    )
    client.execute()
    print(f"[OK] {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
