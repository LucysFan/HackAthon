"""Две короткие анимации для README.

``docs/sweep.gif``  — свип по z вокруг второй плоскости: видно, что
                      изображение собирается только на своём расстоянии;
``docs/pipeline.gif`` — путь от исходной картинки до маски и обратно.

Запуск:  python scripts/make_readme_gifs.py
Нужен imageio (+ ffmpeg для mp4, для gif не нужен).
"""
from __future__ import annotations

import io
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C            # noqa: E402
from hdoe import physics as P           # noqa: E402
from hdoe import prepare as PR          # noqa: E402
from hdoe import targets as T           # noqa: E402

DOCS = os.path.join(ROOT, "docs")


def phase_nn() -> np.ndarray:
    v = np.asarray(Image.open(os.path.join(C.PATHS.out, "DOE.png")).convert("L"),
                   dtype=np.uint8)
    return np.asarray(P.decode8bit(v), dtype=np.float64)


def gray_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _font(size: int = 13):
    """Шрифт с кириллицей: берём DejaVu из matplotlib, иначе дефолтный."""
    from PIL import ImageFont
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                         "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(p, size)
    except Exception:                                        # noqa: BLE001
        return ImageFont.load_default()


def caption(img: np.ndarray, text: str, size: int = 2) -> np.ndarray:
    """Увеличить кадр ×size и подписать снизу."""
    rgb = np.stack([img] * 3, axis=-1)
    big = np.repeat(np.repeat(rgb, size, 0), size, 1)
    h, w = big.shape[:2]
    big = np.vstack([big, np.zeros((30, w, 3), np.uint8)])
    im = Image.fromarray(big)
    from PIL import ImageDraw
    ImageDraw.Draw(im).text((9, h + 6), text, fill=(235, 242, 250), font=_font(15))
    return np.asarray(im)


def make_sweep() -> str:
    phi = phase_nn()
    zs = np.linspace(C.ZS[1] * 1e3 - 1.2, C.ZS[1] * 1e3 + 1.2, 17)
    frames = []
    t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
    for z_mm in zs:
        z = z_mm * 1e-3
        if z >= C.Z_CRIT:
            I = P.normalize01(P.propagate(t, z)[0, 0].numpy())
        else:
            I = np.zeros((C.N, C.N))
        frames.append(caption(gray_u8(I), f"z = {z_mm:.2f} мм"))
    # обратный ход — анимация читается как «качели»
    frames = frames + frames[-2:0:-1]
    p = os.path.join(DOCS, "sweep.gif")
    imageio.mimsave(p, frames, duration=0.16, loop=0, palettesize=32)
    print("  ", os.path.relpath(p, ROOT), f"{os.path.getsize(p) / 1e6:.2f} МБ")
    return p


def make_pipeline() -> str:
    src = os.path.join(ROOT, "data", "source", "object.png")
    steps = PR.preview_steps(src, size=C.N)
    targets = T.load_targets()
    phi = phase_nn()
    recon = [P.normalize01(P.propagate(torch.as_tensor(phi, dtype=torch.float64)[None, None],
                                       z)[0, 0].numpy()) for z in C.ZS]
    frames = []
    for name, im in steps:
        frames.append(caption(np.asarray(im.resize((C.N, C.N)), dtype=np.uint8),
                              name))
    for _ in range(4):                     # пауза на маске
        code = np.asarray(Image.open(os.path.join(C.PATHS.out, "DOE.png")).convert("L"))
        frames.append(caption(code, "маска 128×128 · 8 бит"))
    for i, z in enumerate(C.ZS):
        for _ in range(3):
            frames.append(caption(gray_u8(recon[i]),
                                  f"z{i + 1} = {C.ZS_MM[i]:.1f} мм · восстановление"))
        frames.append(caption(gray_u8(targets[i]), f"цель z{i + 1}"))
    p = os.path.join(DOCS, "pipeline.gif")
    imageio.mimsave(p, frames, duration=0.34, loop=0, palettesize=32)
    print("  ", os.path.relpath(p, ROOT), f"{os.path.getsize(p) / 1e6:.2f} МБ")
    return p


def main() -> int:
    os.makedirs(DOCS, exist_ok=True)
    print("анимации:")
    make_sweep()
    make_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
