#!/usr/bin/env python3
"""
Запускает по очереди длинные задачи обучения/исследования.
При ошибке (ненулевой exit code) пишет предупреждение и переходит к следующей программе.
Запуск из корня репозитория: python scripts/run_training_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (имя для лога, путь к скрипту относительно корня репозитория)
_JOBS: list[tuple[str, str]] = [
    ("xgb_grid_search", "training/xgb_grid_search.py"),
    ("research_main", "research/main.py"),
    ("xgb_train", "training/main.py"),
]


def main() -> int:
    failures: list[str] = []
    for name, rel_script in _JOBS:
        script_path = _PROJECT_ROOT / rel_script
        if not script_path.is_file():
            print(
                f"[ERROR] {name}: файл не найден: {script_path}",
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
            print(f"[ERROR] {name}: не удалось запустить процесс: {exc}", flush=True)
            failures.append(name)
            continue

        if proc.returncode != 0:
            print(
                f"[WARN] {name}: завершился с кодом {proc.returncode}, переходим к следующей задаче.",
                flush=True,
            )
            failures.append(name)
        else:
            print(f"[OK] {name}: успешно завершён.", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    if failures:
        print(f"Итог: с ошибкой: {', '.join(failures)}", flush=True)
        return 1
    print("Итог: все задачи завершились успешно.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
