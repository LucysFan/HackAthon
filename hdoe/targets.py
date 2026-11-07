"""Объектные изображения (раздел 6 ТЗ).

Четыре контрастных объекта на тёмном фоне: 128×128 px, 8 бит, одноканальные,
PNG. Энергии выравниваются по п. 5.4.1. Если во входном каталоге уже лежат
готовые изображения — они читаются и приводятся к формату (п. 6.3).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from . import config as C

NAMES = ("target_1.png", "target_2.png", "target_3.png", "target_4.png")

_TITLES = (
    "Спираль-2, плотная навивка + зерно",
    "Спираль-3, разворот 40°, зерно σ=2 px",
    "Спираль-3, разворот 80°, крупное зерно",
    "Спираль-3, разворот 120°, мелкое зерно",
)


def _canvas(x, y):
    pad = np.zeros((C.N, C.N), dtype=np.float64)
    return pad


def _disk(t, cx, cy, r, val=1.0, x=None, y=None):
    t[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = val
    return t


def _rect(t, cx, cy, w, h, val=1.0, x=None, y=None):
    t[(np.abs(x - cx) <= w / 2) & (np.abs(y - cy) <= h / 2)] = val
    return t


TARGET_BLUR_SIGMA = 0.0      # цели задаются уже полосно-ограниченными по построению


# --------------------------------------------------------------- семейство целей
# Объектные изображения — «непростые» (раздел 6 ТЗ и указание заказчика):
# не простые геометрические фигуры, а богатые зернистые (спекл-текстуры)
# интерференционные структуры. Все четыре цели принадлежат одному семейству
# спиральных фазово-модулированных текстур с зернистостью и различаются
# числом рукавов, шагом навивки, разворотом и статистикой зерна.
#
# Почему именно такое семейство (см. отчёт, раздел «Пределы достижимости»):
# одиночная фазовая маска на четырёх близких расстояниях переносит только
# полосно-ограниченный класс изображений. Численный эксперимент
# (experiments/texture_feasibility.py, experiments/sigma_sweep.py):
#   • резкие блочные/пиксельные структуры (мозаика, «платы», шахматки) —
#     PCC(GS) ≈ 0,25–0,33, потолок ≈ 0,45;
#   • гладкие фигуры с σ = 4 px (прежний набор) — PCC ≈ 0,58, потолок ≈ 0,70;
#   • спиральные текстуры с зерном (σ_z ≈ 2 px) — PCC ≈ 0,63–0,74, потолок 0,85+;
#   • белый шум / фрактал с σ = 1 px — PCC ≈ 0,25 (принципиально не восстановим).
# То есть «непростое» изображение обязано быть богатым по структуре, но
# полосно-ограниченным по спектру: именно таковы спиральные текстуры с
# зернистостью σ_z ≈ 2 px.


def _grain(seed: float, sigma: float = 2.0, amp: float = 0.4) -> np.ndarray:
    """Зернистость: гауссово-сглаженный белый шум, нормированный в ±1."""
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    g = np.zeros((C.N, C.N))
    g[:] = rng.standard_normal((C.N, C.N))
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(g, sigma)
    return 1.0 + amp * g / (np.abs(g).max() + 1e-9)


def textured_target(seed: float, rot: float = 0.0, arms: int = 3, twist: float = 5.0,
                    p: float = 2.2, grain_sigma: float = 2.0, grain_amp: float = 0.4,
                    r0: float = 56.0) -> np.ndarray:
    """Одна цель: спиральная текстура с зернистостью.

    rot — разворот маски (град), arms — число рукавов, twist — «шаг» навивки,
    p — степень контраста, grain_sigma/grain_amp — статистика зерна.
    """
    n = np.arange(C.N) - C.N // 2
    X, Y = np.meshgrid(n, n, indexing="ij")
    R = np.hypot(X, Y)
    PH = np.arctan2(Y, X)
    ph = PH - np.deg2rad(rot)
    th = ph + R / twist * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th)) ** p * np.exp(-(R / r0) ** 3)
    return np.clip(s * _grain(seed, grain_sigma, grain_amp) * 1.7, 0.0, 1.0)


_TARGET_SPECS = (
    dict(seed=701, rot=0.0, twist=4.8, arms=3),
    dict(seed=702, rot=40.0, twist=5.0, arms=3),
    dict(seed=703, rot=80.0, twist=5.2, arms=3, grain_sigma=2.6, grain_amp=0.45),
    dict(seed=704, rot=120.0, twist=5.4, arms=3, grain_sigma=1.6, grain_amp=0.38),
)


def make_targets(sigma: float = TARGET_BLUR_SIGMA) -> np.ndarray:
    """Детерминированная генерация четырёх целей, float64 [4,128,128] в [0,1].

    Цели — богатые зернистые спиральные текстуры (см. комментарий к семейству
    выше). ``sigma`` оставлен для совместимости: дополнительное сглаживание
    уже сгенерированных целей (>0) применяется только по явному запросу.
    """
    arr = np.stack([textured_target(**sp) for sp in _TARGET_SPECS])
    arr[arr < 0.02] = 0.0                    # «тёмный фон» — чистый ноль
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        arr = np.stack([gaussian_filter(t, sigma) for t in arr])
        arr[arr < 0.02] = 0.0
    arr = equalize_energy(arr)
    peak = arr.max()
    if peak > 1.0:
        arr = arr / peak
    return arr / arr.max()


def make_targets_simple(sigma: float = 4.0) -> np.ndarray:
    """Прежний набор (простые фигуры + σ = 4 px) — для сравнения в отчёте."""
    n = np.arange(C.N) - C.N // 2
    X, Y = np.meshgrid(n, n, indexing="ij")
    R = np.hypot(X, Y)
    PH = np.arctan2(Y, X)
    out = []
    t = _canvas(X, Y)
    ring = (np.abs(R - 30) <= 5.0) & ~((PH > 1.2) & (PH < 2.4))
    t[ring] = 1.0
    _disk(t, -18, 18, 8, 0.85, X, Y)
    _disk(t, 22, -20, 5, 0.6, X, Y)
    t[(np.abs(X) <= 1) & (R < 22)] = 0.9
    out.append(t)
    t = _canvas(X, Y)
    _rect(t, 0, 0, 9, 62, 1.0, X, Y)
    _rect(t, 0, -14, 58, 9, 1.0, X, Y)
    for j in range(-1, 2):
        _rect(t, j * 18, 26, 5, 16, 0.7, X, Y)
    _disk(t, 28, 30, 7, 0.5, X, Y)
    out.append(t)
    arr = np.stack(out + [out[0], out[1]]).astype(np.float64)
    from scipy.ndimage import gaussian_filter
    arr = np.stack([gaussian_filter(a, sigma) for a in arr])
    arr[arr < 0.02] = 0.0
    return equalize_energy(arr) / np.max(arr)


def equalize_energy(t: np.ndarray) -> np.ndarray:
    """П. 5.4.1: t_i ← t_i·Ē/E_i (выравнивание суммарной энергии)."""
    t = np.asarray(t, dtype=np.float64)
    if t.ndim == 3:
        e = t.reshape(t.shape[0], -1).sum(-1)
        return t * (e.mean() / np.maximum(e, 1e-12))[:, None, None]
    e = t.sum()
    return t * (e / max(e, 1e-12))


def to_u8(t: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(t) * 255.0), 0, 255).astype(np.uint8)


def save_targets(t: np.ndarray, out_dir: str | None = None) -> list[str]:
    """Сохранение целей в PNG 8 бит / режим 'L'."""
    out_dir = out_dir or C.PATHS.targets
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    u8 = to_u8(t)
    for name, img in zip(NAMES, u8):
        p = os.path.join(out_dir, name)
        Image.fromarray(img, mode="L").save(p)
        paths.append(p)
    return paths


def load_targets(in_dir: str | None = None) -> np.ndarray:
    """Чтение целей; отсутствующие — генерируются. Возврат: float64 [4,N,N] в [0,1]."""
    in_dir = in_dir or C.PATHS.targets
    imgs = []
    for name in NAMES:
        p = os.path.join(in_dir, name)
        if os.path.exists(p):
            imgs.append(_load_as_L(p))
        else:
            imgs = None
            break
    if imgs is None:
        t = make_targets()
        save_targets(t, in_dir)
        return t
    return np.stack(imgs).astype(np.float64) / 255.0


def _load_as_L(path: str) -> np.ndarray:
    """П. 6.3: приведение к 128×128, режим 'L', без искажения пропорций."""
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    if im.size != (C.N, C.N):
        s = min(C.N / im.size[0], C.N / im.size[1])
        new = (max(1, int(round(im.size[0] * s))), max(1, int(round(im.size[1] * s))))
        im = im.resize(new, Image.LANCZOS)
        canvas = Image.new("L", (C.N, C.N), 0)
        canvas.paste(im, ((C.N - new[0]) // 2, (C.N - new[1]) // 2))
        im = canvas
    return np.asarray(im, dtype=np.uint8)


def load_png_as_float(path: str) -> np.ndarray:
    """Изображение → float64 [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def titles() -> tuple[str, ...]:
    return _TITLES


def main() -> None:
    C.PATHS.ensure()
    t = make_targets()
    paths = save_targets(t)
    for p, e in zip(paths, t.reshape(4, -1).sum(-1)):
        print(f"{os.path.basename(p)}  E={e:.1f}  max={t[paths.index(p)].max():.2f}")


if __name__ == "__main__":
    main()
