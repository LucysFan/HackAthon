"""Корпус объектов и учебные четвёрки (ТЗ, п. 5.4.2).

Корпус: 1200 синтезированных изображений 128×128 (фигуры на тёмном фоне:
кольца, диски, кресты, рамки, штриховые «буквы», спирали, пятна, отрезки).
Учебный пример — упорядоченная четвёрка целей [4,128,128] ∈ [0,1] с
выравниванием суммарной энергии (п. 5.4.1) и аугментацией: общие для всех
четырёх каналов сдвиги/отражения + поканальная фотометрия.

Полный режим (корпус ≥1000) используется по умолчанию. Сокращённый режим
(``mode='fixed'``) — заданные 4 изображения с аугментацией (п. 5.4.2).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C

CORPUS_SIZE = 1200
CORPUS_FILE = "corpus.npy"


# ------------------------------------------------------------------ генераторы
def _grid(np_dtype=np.float64):
    n = np.arange(C.N) - C.N // 2
    Y, X = np.meshgrid(n, n, indexing="ij")
    R = np.hypot(X, Y)
    return X, Y, R


def _ring(X, Y, R, rng, val=1.0):
    r = rng.uniform(8, 48)
    w = rng.uniform(2, 9)
    t = (np.abs(R - r) <= w).astype(float)
    if rng.random() < 0.5:                        # разрыв дуги
        a0 = rng.uniform(-np.pi, np.pi)
        a1 = a0 + rng.uniform(0.5, 2.5)
        ang = np.arctan2(Y, X)
        t[(ang > a0) & (ang < a1)] = 0
    return t * val


def _disk(X, Y, R, rng, val=1.0):
    r = rng.uniform(5, 45)
    t = (R <= r).astype(float)
    if rng.random() < 0.5:                        # вырез
        cx, cy = rng.uniform(-15, 15, size=2)
        t[(X - cx) ** 2 + (Y - cy) ** 2 <= (r * rng.uniform(0.25, 0.6)) ** 2] = 0
    return t * val


def _cross(X, Y, R, rng, val=1.0):
    w = rng.uniform(3, 14)
    lx = rng.uniform(15, 55)
    ly = rng.uniform(15, 55)
    t = (((np.abs(X) <= w) & (np.abs(Y) <= lx)) |
         ((np.abs(Y) <= w) & (np.abs(X) <= ly))).astype(float)
    return t * val


def _frame(X, Y, R, rng, val=1.0):
    w = rng.uniform(4, 14)
    a = rng.uniform(18, 52)
    b = rng.uniform(18, 52)
    t = ((np.abs(X) < a) & (np.abs(Y) < b) &
         ~((np.abs(X) < a - w) & (np.abs(Y) < b - w))).astype(float)
    return t * val


def _strokes(X, Y, R, rng, val=1.0):
    """Штриховая «буква»: 2–4 отрезка/наклонные стойки."""
    t = np.zeros_like(X)
    nseg = rng.integers(2, 5)
    for _ in range(nseg):
        x0, y0 = rng.uniform(-30, 30, size=2)
        ang = rng.uniform(-np.pi, np.pi)
        L = rng.uniform(12, 45)
        w = rng.uniform(2.5, 7)
        dx, dy = np.cos(ang), np.sin(ang)
        px = X - x0
        py = Y - y0
        along = px * dx + py * dy
        perp = -px * dy + py * dx
        t[(np.abs(perp) <= w) & (np.abs(along) <= L / 2)] = val
    return t


def _spiral(X, Y, R, rng, val=1.0):
    t = np.zeros_like(X)
    k = int(rng.integers(12, 46))
    a = rng.uniform(-np.pi, np.pi)
    dr = rng.uniform(0.6, 1.0)
    rr = rng.uniform(1.5, 5.0)
    rad = rng.uniform(3, 14)
    for i in range(k):
        ang = a + i * rng.uniform(0.3, 0.55)
        rad += dr
        if rad > 48:
            break
        cx, cy = rad * np.cos(ang), rad * np.sin(ang)
        t[(X - cx) ** 2 + (Y - cy) ** 2 <= rr ** 2] = val if i % 2 == 0 else val * 0.75
    return t


def _blobs(X, Y, R, rng, val=1.0):
    t = np.zeros_like(X)
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(-35, 35, size=2)
        s = rng.uniform(3, 12)
        t = np.maximum(t, np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2)))
    return t * val


def _polygon(X, Y, R, rng, val=1.0):
    """Многоугольник как пересечение полуплоскостей."""
    m = int(rng.integers(3, 7))
    r = rng.uniform(14, 46)
    ang = np.sort(rng.uniform(-np.pi, np.pi, m))
    t = np.ones_like(X)
    for a in ang:
        t &= (X * np.cos(a) + Y * np.sin(a)) <= r * rng.uniform(0.75, 1.0)
    if rng.random() < 0.4:
        t &= R > r * 0.35
    return t.astype(float) * val


def _halftone(X, Y, R, rng, val=1.0):
    """Решётка точек (дискретный объект)."""
    t = np.zeros_like(X)
    step = rng.uniform(9, 18)
    rad = rng.uniform(1.8, 4.0)
    mask_r = rng.uniform(18, 46)
    shape = rng.integers(0, 2)
    ok = (R < mask_r) if shape == 0 else ((np.abs(X) < mask_r) & (np.abs(Y) < mask_r))
    xs = np.arange(-step * 8, step * 8, step)
    for cx in xs:
        for cy in xs:
            t[((X - cx) ** 2 + (Y - cy) ** 2 <= rad ** 2) & ok] = val
    return t


FAMILIES = (_ring, _disk, _cross, _frame, _strokes, _spiral, _blobs, _polygon, _halftone)


def draw_object(rng) -> np.ndarray:
    """Одно изображение 128×128 ∈ [0,1] (контрастный объект на тёмном фоне)."""
    X, Y, R = _grid()
    t = np.zeros_like(X)
    n_fam = int(rng.integers(1, 3))
    for f in rng.choice(len(FAMILIES), size=n_fam, replace=False):
        t = np.maximum(t, FAMILIES[f](X, Y, R, rng, val=rng.uniform(0.6, 1.0)))
    if rng.random() < 0.35:                        # добавочная мелкая деталь
        f = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
        small = f(X, Y, R, rng, val=1.0)
        t = np.maximum(t, small * 0.65)
    # инверсия (светлый фон) не применяется: объекты должны быть на тёмном фоне
    # лёгкая бинаризация, как у «объектных» изображений
    t = np.clip((t - 0.15) / 0.85, 0, 1)
    t[t < 0.05] = 0.0
    if t.sum() < 60:                               # пустой кадр — недопустим
        return draw_object(rng)
    return t / max(t.max(), 1e-9)


def build_corpus(size: int = CORPUS_SIZE, seed: int = C.SEED) -> np.ndarray:
    """Корпус [size,128,128] uint8."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, C.N, C.N), dtype=np.uint8)
    for i in range(size):
        out[i] = np.clip(np.round(draw_object(rng) * 255), 0, 255).astype(np.uint8)
    return out


def load_corpus(root: str | None = None, size: int = CORPUS_SIZE,
                seed: int = C.SEED) -> np.ndarray:
    """Чтение/создание корпуса (кэш в data/corpus.npy)."""
    import os
    root = root or os.path.join(C.PATHS.root, "data")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, CORPUS_FILE)
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape == (size, C.N, C.N):
            return arr
    arr = build_corpus(size, seed)
    np.save(path, arr)
    return arr


# ------------------------------------------------------- учебные четвёрки
def equalize_energy_t(t: torch.Tensor) -> torch.Tensor:
    """П. 5.4.1: выравнивание суммарной энергии четвёрки целей."""
    e = t.flatten(1).sum(1)
    return t * (e.mean() / torch.clamp(e, min=1e-9))[:, None, None]


def augment(quad: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Аугментация: общие для 4 каналов сдвиг/отражение + поканальная фотометрия."""
    out = quad.clone()
    # общий сдвиг (циклический, чтобы не терять энергию)
    dy, dx = rng.integers(-8, 9, size=2)
    out = torch.roll(out, shifts=(int(dy), int(dx)), dims=(-2, -1))
    if rng.random() < 0.5:
        out = torch.flip(out, dims=(-1,))
    if rng.random() < 0.5:
        out = torch.flip(out, dims=(-2,))
    if rng.random() < 0.5:                       # общий поворот на 90°
        out = torch.rot90(out, k=int(rng.integers(1, 4)), dims=(-2, -1))
    # поканальная фотометрия
    for i in range(out.shape[0]):
        g = float(np.exp(rng.normal(0, 0.12)))
        b = float(np.clip(rng.normal(0, 0.03), -0.1, 0.1))
        out[i] = torch.clamp(out[i] * g + b, 0.0, 1.0)
    return out


class QuadDataset(Dataset):
    """Датасет учебных четвёрок.

    mode='corpus' — четвёрки случайных изображений корпуса;
    mode='fixed'  — сокращённый режим: заданные 4 цели + аугментация.
    """

    def __init__(self, corpus: np.ndarray, length: int = 4096, mode: str = "corpus",
                 fixed: np.ndarray | None = None, seed: int = C.SEED,
                 augment: bool = True):
        self.corpus = corpus
        self.length = length
        self.mode = mode
        self.fixed = fixed
        self.seed = seed
        self.do_augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + idx)
        if self.mode == "fixed" and self.fixed is not None:
            q = torch.as_tensor(self.fixed, dtype=torch.float32)
        else:
            ids = rng.choice(len(self.corpus), size=C.N_PLANES, replace=False)
            q = torch.as_tensor(self.corpus[ids], dtype=torch.float32) / 255.0
        if self.do_augment:
            q = augment(q, rng)
        else:
            q = q.clone()
        return equalize_energy_t(q)


def validation_quads(corpus: np.ndarray, n: int = 4, seed: int = C.SEED + 12345,
                     fixed: np.ndarray | None = None) -> torch.Tensor:
    """Контрольный набор четвёрок (фиксирован, без аугментации) [B,4,N,N]."""
    rng = np.random.default_rng(seed)
    quads = []
    for _ in range(n):
        if fixed is not None:
            ids = rng.choice(len(fixed), size=C.N_PLANES, replace=False)
            src = torch.as_tensor(np.asarray(fixed)[ids], dtype=torch.float32)
        else:
            ids = rng.choice(len(corpus), size=C.N_PLANES, replace=False)
            src = torch.as_tensor(corpus[ids], dtype=torch.float32) / 255.0
        quads.append(equalize_energy_t(src))
    return torch.stack(quads)


def main() -> None:
    a = load_corpus()
    print("corpus", a.shape, a.dtype)
    from PIL import Image
    tile = np.zeros((4 * 128, 8 * 128), np.uint8)
    for k in range(32):
        r, c = divmod(k, 8)
        tile[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = a[k * 7]
    Image.fromarray(tile).save("/tmp/corpus_preview.png")


if __name__ == "__main__":
    main()
