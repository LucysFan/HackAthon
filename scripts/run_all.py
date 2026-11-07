"""Полный конвейер одной командой (порядок этапов 1–6 ТЗ).

Запуск:
    python scripts/run_all.py                 # цели → тесты → обучение → оценка → дашборд → отчёт
    python scripts/run_all.py --skip-train    # без обучения (использовать готовый чекпоинт)
    python scripts/run_all.py --quick         # быстрый прогон (смоук) для проверки конвейера

Скрипт только вызывает этапы, вся вычислительная работа — в модулях hdoe/ и
scripts/*.py (п. 5.1.4: каждый шаг запускается и отдельной командой).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import _path  # noqa: F401
from hdoe import config as C

PY = sys.executable
ROOT = C.PATHS.root


def run(cmd: list[str], title: str) -> None:
    print("\n" + "=" * 78)
    print(f"→ {title}")
    print("  " + " ".join(cmd))
    print("=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"этап завершился с кодом {r.returncode}: {title}")
    print(f"✓ {title} ({time.time() - t0:.1f} с)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None, help="чекпоинт сети для этапа оценки")
    args = ap.parse_args()
    C.PATHS.ensure()

    if not args.eval_only:
        run([PY, "-m", "hdoe.targets"], "этап 0: объектные изображения (data/targets)")
        run([PY, "-m", "hdoe.corpus"], "этап 0: корпус объектов (data/corpus.npy)")

    if not args.skip_train and not args.eval_only:
        # Обучение: дистилляция учительских масок GS в U-Net. Сквозное
        # дообучение физики (стадия 2) в этом рецепте отключено (--steps 0):
        # эксперименты показали, что оно ухудшает метрики. Физика работает
        # внутри сети на инференсе (K шагов, см. hdoe/netphys.py).
        cmd = [PY, "scripts/train_gpu.py", "--out", C.PATHS.out,
               "--width", "48", "--depth", "5", "--distill-steps", "1600",
               "--steps", "0", "--geom-only", "--teachers", "6",
               "--gs-iters", "250", "--gs-restarts", "2",
               "--val-every", "200", "--snap-every", "200",
               "--phys-steps", "2048", "--phys-gain", "0.35"]
        if args.quick:
            cmd += ["--smoke"]
        run(cmd, "этап 3: обучение сети (дистилляция учительских масок GS)")

    # фаза = один прямой проход сети с физикой внутри (без внешней доводки)
    ev = [PY, "scripts/evaluate.py", "--out", C.PATHS.out, "--pipeline", "netphys",
          "--gs-iters", "250", "--gs-restarts", "3", "--gs-refine", "100"]
    if args.quick:
        ev = [PY, "scripts/evaluate.py", "--out", C.PATHS.out, "--pipeline", "netphys",
              "--gs-iters", "80", "--gs-restarts", "1", "--gs-refine", "0"]
    if args.ckpt:
        ev += ["--ckpt", args.ckpt]
    run(ev, "этап 4: оценка (сеть с физикой внутри), GS-бейзлайн, DOE.png + metrics.json")
    run([PY, "scripts/sweep_phys.py"], "этап 4б: абляция K — вклад физики внутри сети")

    if not args.skip_tests:
        run([PY, "tests/test_all.py"], "этапы 1–2: верификация (тесты 8.1–8.9)")

    run([PY, "scripts/build_dashboard.py"], "этап 5: сборка бандла дашборда")
    run([PY, "scripts/make_figures.py"], "этап 6: визуализации для отчёта")
    run([PY, "scripts/make_report.py"], "этап 6: отчёт report.pdf")
    print("\nГотово. Артефакты: outputs/, dashboard/ (index.html), figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
