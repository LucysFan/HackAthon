"""Какие классы «непростых» целей физически разрешимы одним ДОЭ (GS 200)."""
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


def fractal(seed, octaves=5, base=8):
    """Фрактальный (перлиновый) шум — зернистый, много масштабов."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N))
    amp = 1.0
    for o in range(octaves):
        s = base * 2 ** o
        g = rng.standard_normal((s, s))
        acc += amp * zoom(g, N / s, order=1)[:N, :N]
        amp *= 0.55
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc


def shapes_with_grain(seed, n_shapes=6):
    rng = np.random.default_rng(seed)
    t = np.zeros((N, N))
    for _ in range(n_shapes):
        cx, cy = rng.integers(-40, 40, 2)
        w, h = rng.integers(12, 45, 2)
        rot = rng.uniform(0, np.pi)
        xr = (X - cx) * np.cos(rot) + (Y - cy) * np.sin(rot)
        yr = -(X - cx) * np.sin(rot) + (Y - cy) * np.cos(rot)
        m = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
        t[m] = rng.uniform(0.45, 1.0)
    # зернистость внутри объектов
    g = gaussian_filter(rng.standard_normal((N, N)), 1.2)
    g = g / (np.abs(g).max() + 1e-9)
    t = t * (0.75 + 0.25 * g) + 0.12 * np.clip(g, 0, None) * t
    return np.clip(t, 0, 1)


def rings_texture(seed):
    rng = np.random.default_rng(seed)
    ph = fractal(seed, 4, 6)
    t = 0.5 + 0.5 * np.sin(6 * ph * np.pi + R / 4.0)
    t = t * np.exp(-(R / 55.0) ** 4)
    return np.clip(t, 0, 1)


def check(tag, imgs):
    T = np.stack(imgs).astype(np.float64)
    T = T / T.max()
    r = gs.gs_solve(T, iters=200, restarts=1, seed=0, verbose=False)
    q = POL.quantize(np.asarray(r["phi"][0, 0]))
    m = POL.metrics_of(q, T)
    print(f"{tag:28s} GS200: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
    return m


check("фрактальный шум (4 сида)", [fractal(s) for s in (1, 2, 3, 4)])
check("фигуры + зерно", [shapes_with_grain(s) for s in (11, 12, 13, 14)])
check("кольца/текстура", [rings_texture(s) for s in (21, 22, 23, 24)])
# смесь: фигуры с зерном, но ограниченная полоса (σ=2)
check("фигуры+зерно (σ=2 размытие)",
      [gaussian_filter(shapes_with_grain(s), 2.0) for s in (11, 12, 13, 14)])
check("фрактал (σ=3 размытие, контраст)",
      [np.clip(gaussian_filter(fractal(s), 3.0) * 1.6 - 0.25, 0, 1) for s in (1, 2, 3, 4)])
