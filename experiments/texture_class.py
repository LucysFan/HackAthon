"""Итоговый класс целей: «богатая зернистая текстура» (band-limited, но сложная)."""
import sys, os, time, warnings
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


def grain(seed, sigma=2.0):
    rng = np.random.default_rng(seed + 9999)
    g = gaussian_filter(rng.standard_normal((N, N)), sigma)
    return g / (np.abs(g).max() + 1e-9)


def tex_spiral(seed):
    arms = 3
    th = PH + R / 5.0 * 2 * np.pi / 3
    s = (0.5 + 0.5 * np.cos(arms * th)) ** 2.2 * np.exp(-(R / 56.0) ** 3)
    return np.clip(s * (1 + 0.40 * grain(seed)) * 1.6, 0, 1)


def tex_rings(seed):
    b = perlin(seed, 3, 4)
    t = (0.5 + 0.5 * np.sin(2 * np.pi * (R / 8.0 + 2.6 * b))) * np.exp(-(R / 58.0) ** 4)
    return np.clip((t ** 1.4) * 1.3 * (1 + 0.25 * grain(seed)), 0, 1)


def tex_vortex(seed):
    """Вихревое поле: накопленная угловая «закрутка» → интерференционная рябь."""
    b = perlin(seed, 3, 5)
    ang = PH + 2.4 * (b - 0.5) + R / 22.0
    t = (0.5 + 0.5 * np.cos(5.0 * ang)) ** 1.6 * np.exp(-(R / 58.0) ** 4)
    return np.clip(t * 1.45 * (1 + 0.30 * grain(seed)), 0, 1)


def tex_zones(seed):
    """Зонная «муаровая» текстура: cos(αr²) с фрактальной модуляцией."""
    b = perlin(seed, 3, 4)
    t = 0.5 + 0.5 * np.cos(0.0055 * R ** 2 + 6.0 * b)
    t = t * np.exp(-(R / 60.0) ** 4)
    return np.clip(t ** 1.5 * 1.5 * (1 + 0.30 * grain(seed)), 0, 1)


def tex_waves(seed):
    """Интерференция двух фрактальных волновых фронтов."""
    a, b = perlin(seed, 3, 4), perlin(seed + 5, 3, 6)
    t = 0.5 + 0.5 * np.cos(9.0 * a + 7.0 * b)
    t = t * np.exp(-(R / 62.0) ** 4) ** 0.5
    return np.clip(t ** 1.3 * 1.5 * (1 + 0.28 * grain(seed, 1.6)), 0, 1)


def evaluate(fn, tag, seeds, restarts=3, gs_iters=300, polish_steps=600, lr=2.5e-2):
    T = np.stack([fn(s) for s in seeds]); T = T / T.max()
    t0 = time.time(); best = None
    for r in range(restarts):
        g = gs.gs_run(T, iters=gs_iters, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        if best is None or m["pcc"] + m["ssim"] > best[0]:
            best = (m["pcc"] + m["ssim"], q)
    res = POL.polish(T, best[1], steps=polish_steps, lr=lr)
    q = POL.quantize(res["phi"]); m = POL.metrics_of(q, T)
    print(f"{tag:12s} PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} "
          f"плоскости {[round(r['pcc'], 3) for r in m['rows']]} ({time.time()-t0:.0f}s)", flush=True)
    return q, m


if __name__ == "__main__":
    for fn, tag, seeds in ((tex_spiral, "spiral", (5, 6, 7, 8)), (tex_rings, "rings", (9, 10, 11, 12)),
                           (tex_vortex, "vortex", (21, 22, 23, 24)), (tex_zones, "zones", (31, 32, 33, 34)),
                           (tex_waves, "waves", (41, 42, 43, 44))):
        evaluate(fn, tag, seeds)
