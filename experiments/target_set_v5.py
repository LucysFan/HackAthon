"""Набор v5: визуально разные, но разрешимые текстуры. Поиск комбинации."""
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


def grain(seed, sigma=2.0, amp=0.4):
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    g = gaussian_filter(rng.standard_normal((N, N)), sigma)
    return 1.0 + amp * g / (np.abs(g).max() + 1e-9)


def perlin(seed, octaves=3, base=4):
    rng = np.random.default_rng(int(seed))
    acc = np.zeros((N, N)); amp = 1.0
    for o in range(octaves):
        s = base * 2 ** o
        acc += amp * zoom(rng.standard_normal((s, s)), N / s, order=3)[:N, :N]
        amp *= 0.5
    return (acc - acc.min()) / (acc.max() - acc.min())


def spiral(seed, arms=3, twist=5.0, p=2.2, amp=0.4):
    th = PH + R / twist * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th)) ** p * np.exp(-(R / 56.0) ** 3)
    return np.clip(s * grain(seed, 2.0, amp) * 1.7, 0, 1)


def rings(seed, base=8.0, mix=2.6):
    b = perlin(seed, 3, 4)
    t = (0.5 + 0.5 * np.sin(2 * np.pi * (R / base + mix * b))) * np.exp(-(R / 58.0) ** 4)
    return np.clip(t ** 1.4 * 1.3 * (1 + 0.25 * grain(seed)), 0, 1)


def rosette(seed):
    """Розетка: лепестки + вращательная рябь."""
    t = (0.5 + 0.5 * np.cos(7.0 * PH + 1.6 * np.sin(R / 9.0))) * (0.5 + 0.5 * np.cos(R / 5.5))
    t = t * np.exp(-(R / 56.0) ** 3)
    return np.clip(t ** 1.2 * grain(seed, 2.2, 0.35) * 1.8, 0, 1)


def star(seed):
    """Звёздчатый узор: секторы с фазовой модуляцией."""
    t = (0.5 + 0.5 * np.cos(11.0 * PH + R / 12.0)) * np.exp(-(R / 58.0) ** 4)
    return np.clip(t ** 1.5 * grain(seed, 1.8, 0.4) * 1.7, 0, 1)


SETS = {
    "v5a": [lambda: spiral(601, 3, 4.8), lambda: rings(602), lambda: spiral(603, 2, 3.6),
            lambda: rosette(604)],
    "v5b": [lambda: spiral(611, 3, 4.8), lambda: rings(612), lambda: star(613),
            lambda: spiral(614, 2, 3.6)],
    "v5c": [lambda: spiral(621, 3, 5.0), lambda: rings(622), lambda: rosette(623),
            lambda: star(624)],
}


def evaluate(fns, tag, steps=800, lr=2.5e-2):
    T = np.stack([f() for f in fns]); T = T / T.max()
    t0 = time.time(); best = None
    for r in range(3):
        g = gs.gs_run(T, iters=300, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        if best is None or m["pcc"] + m["ssim"] > best[0]:
            best = (m["pcc"] + m["ssim"], q)
    res = POL.polish(T, best[1], steps=steps, lr=lr)
    q = POL.quantize(res["phi"]); m = POL.metrics_of(q, T)
    print(f"{tag}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} плоскости "
          f"{[round(r['pcc'], 3) for r in m['rows']]} ({time.time()-t0:.0f}s)", flush=True)
    return q, m, T


if __name__ == "__main__":
    from PIL import Image
    for tag, fns in SETS.items():
        q, m, T = evaluate(fns, tag)
        d = os.path.join(ROOT, f"experiments/_preview/{tag}"); os.makedirs(d, exist_ok=True)
        W = Image.new("L", (128 * 4 + 24, 128), 128)
        for i, t in enumerate(T):
            im = Image.fromarray((t * 255).astype(np.uint8), mode="L")
            im.save(os.path.join(d, f"t{i+1}.png")); W.paste(im, (i * 136, 0))
        W.save(os.path.join(ROOT, f"experiments/_preview/{tag}_quad.png"))
        np.save(os.path.join(ROOT, f"experiments/_{tag}_phase.npy"), q)
        np.save(os.path.join(ROOT, f"experiments/_{tag}_targets.npy"), T)
