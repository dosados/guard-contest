#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (log label, script path relative to repo root)
_JOBS: list[tuple[str, str]] = [
    ("xgb_grid_search", "training/xgb_grid_search.py"),
    ("cat_grid_search", "training/cat_grid_search.py"),
    ("xgb_train", "training/main.py"),
    ("cat_train", "training/cat_main.py"),
    ("research_xgb", "research/main.py"),
    ("research_cat", "research/cat_main.py"),
    ("submission_xgb", "submission/main.py"),
    ("submission_cat", "submission/cat_submission.py"),
]


def main() -> int:
    failures: list[str] = []
    for name, rel_script in _JOBS:
        script_path = _PROJECT_ROOT / rel_script
        if not script_path.is_file():
            print(
                f"[ERROR] {name}: script not found: {script_path}",
                flush=True,
            )
            failures.append(name)
            continue

        cmd = [sys.executable, str(script_path)]
        stamp = datetime.now().isoformat(timespec="seconds")
        print(f"\n{'=' * 60}\n[{stamp}] START {name}\n  {' '.join(cmd)}\n{'=' * 60}", flush=True)

        try:
            proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
        except OSError as exc:
            print(f"[ERROR] {name}: failed to start process: {exc}", flush=True)
            failures.append(name)
            continue

        if proc.returncode != 0:
            print(
                f"[WARN] {name}: exited with code {proc.returncode}, continuing to next job.",
                flush=True,
            )
            failures.append(name)
        else:
            print(f"[OK] {name}: finished successfully.", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    if failures:
        print(f"Summary: failed jobs: {', '.join(failures)}", flush=True)
        return 1
    print("Summary: all jobs finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
