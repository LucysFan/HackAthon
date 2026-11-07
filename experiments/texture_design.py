"""Поиск класса целей: визуально богатые, но физически разрешимые."""
import sys, os, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.ndimage import gaussian_filter, zoom
from hdoe import config as C, polish as POL, gs

torch.set_num_threads(2)
N = C.N
n = np.arange(N) - N // 2
X, Y = np.meshgrid(n, n, indexing="ij")
R = np.hypot(X, Y)
PH = np.arctan2(Y, X)


def perlin(seed, octaves=3, base=4):
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N)); amp = 1.0
    for o in range(octaves):
        s = base * 2 ** o
        acc += amp * zoom(rng.standard_normal((s, s)), N / s, order=3)[:N, :N]
        amp *= 0.5
    return (acc - acc.min()) / (acc.max() - acc.min())


def voronoi(seed, cells=28):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-60, 60, (cells, 2))
    d = np.stack([np.hypot(X - px, Y - py) for px, py in pts])
    idx = d.argmin(0)
    depth = np.sort(d, 0)[1] - np.sort(d, 0)[0]
    v = (idx % 7) / 7.0 * 0.8 + 0.2
    v = v * np.clip(1 - depth / 6.0, 0.55, 1.0)      # тёмные границы ячеек
    return np.clip(v, 0, 1)


def spiral(seed):
    rng = np.random.default_rng(seed)
    arms = 3
    th = PH + R / 5.0 * 2 * np.pi / 3
    s = 0.5 + 0.5 * np.cos(arms * th)
    s = s ** 2.2 * np.exp(-(R / 56.0) ** 3)
    grain = gaussian_filter(rng.standard_normal((N, N)), 2.0)
    s = np.clip(s * (1 + 0.35 * grain / (np.abs(grain).max() + 1e-9)) * 1.5, 0, 1)
    return s


def rings_mod(seed):
    base = perlin(seed, 3, 4)
    t = 0.5 + 0.5 * np.sin(2 * np.pi * (R / 9.0 + 2.2 * base))
    t = t * np.exp(-(R / 58.0) ** 4)
    t = t ** 1.5
    return np.clip(t, 0, 1)


def circuit(seed):
    rng = np.random.default_rng(seed)
    t = np.zeros((N, N))
    for _ in range(26):
        x0, y0 = rng.integers(-50, 50, 2)
        ln = rng.integers(18, 55); wd = rng.choice([3, 4, 5])
        horiz = rng.random() < 0.5
        if horiz:
            m = (np.abs(X - x0) <= ln / 2) & (np.abs(Y - y0) <= wd / 2)
        else:
            m = (np.abs(X - x0) <= wd / 2) & (np.abs(Y - y0) <= ln / 2)
        t[m] = rng.uniform(0.5, 1.0)
    for _ in range(14):
        cx, cy = rng.integers(-45, 45, 2); r = rng.uniform(4, 9)
        t[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = rng.uniform(0.6, 1.0)
    return gaussian_filter(t, 1.2)


def mosaic_letters(seed):
    rng = np.random.default_rng(seed)
    t = np.zeros((N, N))
    # «плитки» с текстурой
    for i in range(-3, 4):
        for j in range(-3, 4):
            if rng.random() < 0.45:
                continue
            cx, cy = i * 18 + rng.integers(-3, 3), j * 18 + rng.integers(-3, 3)
            m = (np.abs(X - cx) <= 7) & (np.abs(Y - cy) <= 7)
            t[m] = rng.uniform(0.35, 1.0)
    g = gaussian_filter(rng.standard_normal((N, N)), 1.8)
    t = t * (1 + 0.3 * g / (np.abs(g).max() + 1e-9))
    return np.clip(t, 0, 1)


def check(tag, imgs, iters=200):
    T = np.stack(imgs).astype(np.float64); T = T / T.max()
    r = gs.gs_solve(T, iters=iters, restarts=1, seed=0, verbose=False)
    q = POL.quantize(np.asarray(r["phi"][0, 0]))
    m = POL.metrics_of(q, T)
    print(f"{tag:24s} GS{iters}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)


check("voronoi", [voronoi(s) for s in (1, 2, 3, 4)])
check("spiral+speckle", [spiral(s) for s in (5, 6, 7, 8)])
check("modulated rings", [rings_mod(s) for s in (9, 10, 11, 12)])
check("circuit", [circuit(s) for s in (13, 14, 15, 16)])
check("mosaic+grain", [mosaic_letters(s) for s in (17, 18, 19, 20)])
