#!/usr/bin/env python3
"""
Последовательно запускает:
  training/catboost_train.py
  training/main.py
  training/xgb_grid_search.py

Вывод идёт в консоль (stdout/stderr дочерних процессов не перехватываются).
Рабочая директория — корень репозитория, чтобы пути к данным и артефактам совпадали с ручным запуском.
При ненулевом коде выхода или сбое запуска переходим к следующей программе.
Запуск из корня репозитория: python scripts/run_catboost_main_xgb_grid.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_JOBS: list[tuple[str, str, list[str]]] = [
    ("catboost_train", "training/catboost_train.py", []),
    ("training_main", "training/main.py", ["--xgb-config", "best"]),
    ("xgb_grid_search", "training/xgb_grid_search.py", []),
]


def main() -> int:
    failures: list[str] = []
    for name, rel_script, args in _JOBS:
        script_path = _PROJECT_ROOT / rel_script
        if not script_path.is_file():
            print(
                f"[ERROR] {name}: файл не найден: {script_path}",
                flush=True,
            )
            failures.append(name)
            continue

        cmd = [sys.executable, str(script_path), *args]
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
