"""Один вход для пересборки всего, что нужно README.

    python scripts/generate_main.py              # метрики + картинки + анимации
    python scripts/generate_main.py --no-splits  # без долгого пересчёта train/val/test
    python scripts/generate_main.py --only docs  # только картинки

Что делает по шагам:

1. ``scripts/eval_splits.py`` — метрики на тренировочном, валидационном и
   тестовом наборах → ``outputs/splits.json``. Долго (один прямой проход
   модели ≈ минута на процессоре, наборов около двадцати), поэтому этот шаг
   можно выключить флагом ``--no-splits``: тогда берётся уже посчитанный файл.
2. ``scripts/make_readme_figures.py`` — схемы задачи, маски, свипа по z,
   подготовки входа, кривых обучения → ``docs/*.png``.
3. ``scripts/make_ui_shots.py`` — кадры вкладок дашборда из того же бандла,
   что видит браузер → ``docs/ui_*.png``.
4. ``scripts/make_readme_gifs.py`` — две короткие анимации → ``docs/*.gif``.
5. Сжатие картинок в WebP (≤1400 px) и удаление исходных PNG: README ссылается
   на лёгкие файлы, репозиторий не пухнет.

Ничего не выдумывается: все числа берутся из ``outputs/metrics.json``,
``outputs/splits.json`` и ``outputs/train_log.json``, картинки строятся из
сданных ``outputs/DOE.png`` и ``data/targets``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
PY = sys.executable


def run(script: str, *args: str, env: dict | None = None) -> float:
    """Запустить шаг отдельным процессом (torch не делит память между шагами)."""
    cmd = [PY, os.path.join(ROOT, "scripts", script), *args]
    print(f"\n▸ {script} {' '.join(args)}".rstrip(), flush=True)
    t0 = time.time()
    e = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if env:
        e.update(env)
    res = subprocess.run(cmd, cwd=ROOT, env=e)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"  шаг завершился с кодом {res.returncode} за {dt:.0f} с", flush=True)
    else:
        print(f"  готово за {dt:.0f} с", flush=True)
    return dt


def to_webp(max_width: int = 1400, quality: int = 88, keep_png: bool = False) -> list[str]:
    """PNG → WebP для README; тяжёлые исходники убираем."""
    from PIL import Image
    made = []
    for p in sorted(glob.glob(os.path.join(DOCS, "*.png"))):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w > max_width:
            im = im.resize((max_width, round(h * max_width / w)), Image.LANCZOS)
        out = p[:-4] + ".webp"
        im.save(out, "WEBP", quality=quality, method=6)
        made.append(out)
        print(f"   {os.path.basename(out)}  {os.path.getsize(out) / 1e3:.0f} КБ  {im.size}")
        if not keep_png:
            os.remove(p)
    return made


def splits_ok() -> dict | None:
    p = os.path.join(ROOT, "outputs", "splits.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:                                        # noqa: BLE001
        return None


def summary() -> None:
    """Короткая сводка по числам, которые попали в README."""
    print("\n—— что получилось ——————————————————————————————")
    mp = os.path.join(ROOT, "outputs", "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        mean = m.get("mean") or {}
        print(f"сдаваемый набор: SSIM {mean.get('ssim', 0):.4f} · "
              f"Pearson {mean.get('pcc', 0):.4f}")
    sp = splits_ok()
    if sp:
        for key, name in (("train", "тренировка"), ("val", "валидация"), ("test", "тест")):
            s = (sp.get(key) or {}).get("summary") or {}
            if s:
                print(f"{name:11s} n={s.get('n', 0):2d}  SSIM {s.get('ssim', 0):.4f}  "
                      f"Pearson {s.get('pcc', 0):.4f}")
    else:
        print("splits.json нет — протокол train/val/test не пересчитан "
              "(уберите --no-splits)")
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DOCS, "*")))
    if files:
        size = sum(os.path.getsize(os.path.join(DOCS, f)) for f in files)
        print(f"\ndocs/: {len(files)} файлов, {size / 1e6:.2f} МБ")
        print("  " + ", ".join(files))


def main() -> int:
    ap = argparse.ArgumentParser(description="Собрать всё для README")
    ap.add_argument("--only", choices=("all", "splits", "docs"), default="all")
    ap.add_argument("--no-splits", action="store_true",
                    help="не пересчитывать train/val/test (взять готовый splits.json)")
    ap.add_argument("--quick", action="store_true", help="меньше четвёрок: быстрая проверка")
    ap.add_argument("--keep-png", action="store_true", help="оставить PNG рядом с WebP")
    args = ap.parse_args()

    os.makedirs(DOCS, exist_ok=True)
    t_all = time.time()

    if args.only in ("all", "splits") and not args.no_splits:
        run("eval_splits.py", *(["--quick"] if args.quick else []))
    elif args.only == "splits":
        print("пропущено: --no-splits")

    if args.only in ("all", "docs"):
        need = [("make_readme_figures.py", ()), ("make_ui_shots.py", ()),
                ("make_readme_gifs.py", ())]
        for script, extra in need:
            run(script, *extra)
        print("\n▸ сжатие картинок")
        to_webp(keep_png=args.keep_png)

    print(f"\nвсего {time.time() - t_all:.0f} с")
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
