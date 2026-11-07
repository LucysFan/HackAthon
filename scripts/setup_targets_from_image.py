"""Подготовка целевых изображений из произвольного PNG (п. 6 ТЗ).

Скрипт приводит исходную картинку к формату задачи:
128×128, 8 бит, режим «L», тёмный фон (объектная полярность), выравнивание
энергии по п. 5.4.1 — и раскладывает её в четыре цели target_1..4.png.

Полярность: фазовая маска формирует светящийся объект на тёмном фоне,
поэтому изображение инвертируется (1 − I). Это стандартная объектная
полярность задачи; на исходной картинке с белым фоном инверсия даёт
PCC(GS) ≈ 0,81 против ≈ 0,41 без неё (см. experiments/).

Запуск:
    python scripts/setup_targets_from_image.py --src data/source/object.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C          # noqa: E402
from hdoe import targets as T         # noqa: E402


def prepare(src: str, invert: bool = True, crop: bool = True,
            border: int = 0) -> np.ndarray:
    """PNG → float [4,N,N] в [0,1] с объектной полярностью и равной энергией."""
    im = Image.open(src).convert("L")
    if crop:
        a = np.asarray(im, dtype=np.float64) / 255.0
        # обрезаем однотонные поля вокруг объекта (белый фон сканирования)
        fg = a < (np.median(a) - 0.08)
        if fg.any():
            ys, xs = np.where(fg)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            pad = int(0.04 * max(y1 - y0, x1 - x0))
            y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
            y1, x1 = min(a.shape[0], y1 + pad), min(a.shape[1], x1 + pad)
            im = im.crop((x0, y0, x1, y1))
    side = min(im.size)
    im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                  (im.width + side) // 2, (im.height + side) // 2))
    im = im.resize((C.N, C.N), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64) / 255.0
    if border > 0:                              # тёмная рамка (поле вне объекта)
        a[:border] = a[-border:] = 0.0
        a[:, :border] = a[:, -border:] = 0.0
    if invert:
        a = 1.0 - a
    a = a - a.min()
    a = a / (a.max() + 1e-12)
    a[a < 0.02] = 0.0
    arr = np.stack([a] * C.N_PLANES)
    arr = T.equalize_energy(arr)
    return arr / arr.max()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "source", "object.png"))
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--border", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    C.PATHS.ensure()
    arr = prepare(args.src, invert=not args.no_invert, crop=not args.no_crop,
                  border=args.border)
    paths = T.save_targets(arr, args.out or C.PATHS.targets)
    for p, t in zip(paths, arr):
        print(f"{os.path.basename(p)}: max {t.max():.2f}, энергия {t.sum():.1f}, "
              f"ненулевых {(t > 0.02).mean() * 100:.1f} %")
    # превью для отчёта
    prev = Image.new("L", (C.N * 4 + 12, C.N), 0)
    for i, t in enumerate(arr):
        prev.paste(Image.fromarray((t * 255).astype(np.uint8), mode="L"), (i * (C.N + 4), 0))
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    prev.save(os.path.join(ROOT, "figures", "targets_preview.png"))
    print("готово:", C.PATHS.targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
