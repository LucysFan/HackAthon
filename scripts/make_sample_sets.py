"""Пять тестовых комплектов по четыре изображения разных типов.

Зачем: заказчик проверяет решение на своих картинках, но чтобы проверка была
воспроизводимой, в репозитории лежат готовые комплекты. Каждый комплект — это
четыре цели 128×128, 8 бит, grayscale в ``data/sets/<id>/`` и, если попросить,
готовый ``DOE.png`` + ``recon_*.png`` + ``metrics.json`` (прямой проход сети).

Типы изображений подобраны так, чтобы покрыть разные классы сложности:

===============  ====================================================
``shapes``       простые фигуры (круг, квадрат, треугольник, крест)
``letters``      знаки с резкими краями
``gradients``    плавные яркостные переходы
``textures``     мелкая текстура (рябь, штриховка, точки)
``photo``        реальное фото (светлый фон), приведённое hdoe/prepare.py
===============  ====================================================

Запуск:
    python scripts/make_sample_sets.py                 # только комплекты
    python scripts/make_sample_sets.py --infer         # + прогон сетью
    python scripts/make_sample_sets.py --clean         # удалить и собрать заново
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C       # noqa: E402
from hdoe import prepare as PR     # noqa: E402
from hdoe import sets as SS        # noqa: E402
from hdoe import targets as T      # noqa: E402

N = C.N
S = 512                                  # рисуем крупно, потом уменьшаем
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:                              # noqa: BLE001
                continue
    return ImageFont.load_default()


def blank(bg: int = 0) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("L", (S, S), bg)
    return im, ImageDraw.Draw(im)


# ------------------------------------------------------------------ генераторы
def shapes() -> list[np.ndarray]:
    out = []
    specs = ["circle", "square", "triangle", "cross"]
    for kind in specs:
        im, d = blank()
        if kind == "circle":
            d.ellipse((120, 120, 392, 392), fill=255)
            d.ellipse((196, 196, 316, 316), fill=60)
        elif kind == "square":
            d.rectangle((128, 128, 384, 384), fill=255)
            d.rectangle((192, 192, 320, 320), fill=40)
        elif kind == "triangle":
            d.polygon([(256, 96), (416, 408), (96, 408)], fill=255)
            d.polygon([(256, 190), (352, 372), (160, 372)], fill=30)
        else:
            d.rectangle((216, 96, 296, 416), fill=255)
            d.rectangle((96, 216, 416, 296), fill=255)
            d.ellipse((226, 226, 286, 286), fill=20)
        out.append(np.asarray(im, dtype=np.float64) / 255.0)
    return out


def letters() -> list[np.ndarray]:
    out = []
    f = font(300)
    for ch in ["E", "Ж", "7", "Ф"]:
        im, d = blank()
        bbox = d.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((S - w) / 2 - bbox[0], (S - h) / 2 - bbox[1]), ch, font=f, fill=255)
        out.append(np.asarray(im, dtype=np.float64) / 255.0)
    return out


def gradients() -> list[np.ndarray]:
    y, x = np.mgrid[0:S, 0:S] / (S - 1)
    out = [
        x,                                   # линейный по горизонтали
        y,                                   # линейный по вертикали
        np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) * 1.4,   # радиальный
        0.5 + 0.5 * np.sin(6 * np.pi * x) * np.cos(5 * np.pi * y),  # волна
    ]
    return [np.clip(a, 0, 1) for a in out]


def textures() -> list[np.ndarray]:
    rng = np.random.default_rng(C.SEED)
    y, x = np.mgrid[0:S, 0:S] / (S - 1)
    out = [
        0.5 + 0.45 * np.sin(2 * np.pi * 26 * x),                       # штриховка
        0.5 + 0.35 * np.sin(2 * np.pi * 18 * x + 10 * np.sin(2 * np.pi * y)),
        0.5 + 0.4 * np.cos(2 * np.pi * 22 * y) * np.cos(2 * np.pi * 22 * x),
    ]
    dots = np.zeros((S, S))
    yy, xx = np.mgrid[0:S, 0:S]
    for _ in range(70):
        cy, cx = rng.integers(20, S - 20, 2)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        dots = np.maximum(dots, np.clip((8 - r) / 3.0, 0, 1))
    out.append(dots)
    return [np.clip(a, 0, 1) for a in out]


def photo() -> list[np.ndarray]:
    """Одна реальная картинка, четырьмя способами кадрирования.

    Возвращаются **сырые** кропы в исходной полярности (светлый фон, тёмный
    объект) — ровно такими, какими их увидел бы пользователь, загрузивший
    фотографии через дашборд. Инверсию полярности делает только
    ``PR.prepare_batch`` в :func:`build_one`; раньше кропы инвертировались
    здесь (``to_array(..., invert=True)``) и затем ещё раз в ``prepare_batch``,
    то есть дважды. В результате объект становился тёмным на светлом фоне,
    ДОЭ светил фоном вместо объекта, а метрики комплекта падали вдвое-втрое
    против остальных наборов.
    """
    src = os.path.join(ROOT, "data", "source", "object.png")
    base = PR._to_gray(PR.load_image(src))
    w, h = base.size
    side = min(w, h)
    crops = [
        (0, 0, side, side),
        (w - side, 0, w, side),
        (0, h - side, side, h),
        (w - side, h - side, w, h),
    ]
    arr = []
    for box in crops:
        piece = base.crop(box)
        arr.append(PR.to_array(piece, invert=False, crop=False))
    return arr


BUILDERS = dict(shapes=shapes, letters=letters, gradients=gradients,
                textures=textures, photo=photo)


# ------------------------------------------------------------------ сборка
def build_one(kind: str, *, force: bool = False) -> str:
    """Создать комплект ``kind`` (идемпотентно: повторный запуск перезаписывает)."""
    sid = f"sample-{kind}"
    path = os.path.join(SS.SETS_DIR, sid)
    if os.path.isdir(path) and not force:
        return sid
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    stack = np.stack([np.clip(a, 0, 1) for a in BUILDERS[kind]()])[:C.N_PLANES]
    # каждый кадр — отдельный файл-исходник, чтобы работал тот же путь, что и у загрузки
    for i in range(C.N_PLANES):
        Image.fromarray(T.to_u8(stack[i]), mode="L").save(
            os.path.join(path, "source", f"source_{i + 1}.png"))
    arr = PR.prepare_batch([os.path.join(path, "source", f"source_{i + 1}.png")
                            for i in range(C.N_PLANES)])
    PR.save_set(arr, path)
    PR.save_preview(arr, os.path.join(path, "preview.png"))
    import json
    import time
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(id=sid, kind=kind, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       sources=[f"source_{i}.png" for i in range(1, C.N_PLANES + 1)],
                       synthetic=True, sample=True,
                       energy=float(arr.reshape(C.N_PLANES, -1).sum(-1).mean()),
                       nonzero_pct=float((arr > 0.02).mean() * 100.0),
                       size=[C.N, C.N], bit_depth=8), fh, indent=1, ensure_ascii=False)
    return sid


def main() -> int:
    ap = argparse.ArgumentParser(description="Пять тестовых комплектов по 4 изображения")
    ap.add_argument("--infer", action="store_true", help="сразу прогнать сетью")
    ap.add_argument("--steps", type=int, default=0, help="K шагов физики (0 — из чекпоинта)")
    ap.add_argument("--force", action="store_true", help="перезаписать существующие")
    ap.add_argument("--clean", action="store_true", help="удалить sample-* перед сборкой")
    args = ap.parse_args()

    if args.clean:
        for kind in BUILDERS:
            p = os.path.join(SS.SETS_DIR, f"sample-{kind}")
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p)
    ids = []
    for kind in BUILDERS:
        sid = build_one(kind, force=args.force)
        info = SS.info(sid)
        print(f"комплект {sid}: энергия {info['energy']:.1f}, "
              f"ненулевых {info['nonzero_pct']:.1f} %")
        ids.append(sid)

    if args.infer:
        import subprocess
        for sid in ids:
            cmd = [sys.executable, "scripts/infer_set.py", "--set", SS.set_path(sid)]
            if args.steps:
                cmd += ["--steps", str(args.steps)]
            print("→", " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
