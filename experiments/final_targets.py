"""Подбор финального набора целей: богатые, но физически разрешимые текстуры.

Проверяет полный пайплайн (GS multi-restart + доводка) на кандидатах.
"""
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
V = np.load(os.path.join(ROOT, "experiments/_cand.npy")) if os.path.exists(
    os.path.join(ROOT, "experiments/_cand.npy")) else None


def perlin(seed, octaves=3, base=4):
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N)); amp = 1.0
    for _ in range(octaves):
        s = base * 2 ** _ if False else base * (2 ** _)
        acc += amp * zoom(rng.standard_normal((s, s)), N / s, order=3)[:N, :N]
        amp *= 0.5
    return (acc - acc.min()) / (acc.max() - acc.min())


def spiral(seed):
    """Спираль с зернистостью (спекл внутри рукавов)."""
    rng = np.random.default_rng(seed)
    arms = 3
    th = PH + R / 5.0 * 2 * np.pi / 3
    s = (0.5 + 0.5 * np.cos(arms * th)) ** 2.2 * np.exp(-(R / 56.0) ** 3)
    grain = gaussian_filter(rng.standard_normal((N, N)), 2.0)
    return np.clip(s * (1 + 0.40 * grain / (np.abs(grain).max() + 1e-9)) * 1.6, 0, 1)


def rings_mod(seed):
    """Модулированные кольца (интерференционная «рябь» с фрактальной фазой)."""
    b = perlin(seed, 3, 4)
    t = (0.5 + 0.5 * np.sin(2 * np.pi * (R / 8.0 + 2.6 * b))) * np.exp(-(R / 58.0) ** 4)
    return np.clip((t ** 1.4) * 1.25, 0, 1)


def vortex_dots(seed):
    """Решётка «точек-вихрей»: пятна с градиентными сердцевинами."""
    rng = np.random.default_rng(seed)
    t = np.zeros((N, N))
    for _ in range(22):
        cx, cy = rng.integers(-48, 48, 2)
        r0 = rng.uniform(4, 11)
        rr = np.hypot(X - cx, Y - cy)
        t += np.exp(-(rr / r0) ** 2) * rng.uniform(0.5, 1.0)
    t = gaussian_filter(t, 1.5)
    return np.clip(t / t.max(), 0, 1) ** 1.2


def swirl_field(seed):
    """Поле «завихрений»: сумма гауссовых вихрей, фаза → яркость."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N))
    for _ in range(9):
        cx, cy = rng.uniform(-45, 45, 2)
        sgn = rng.choice([-1, 1])
        ang = np.arctan2(Y - cy, X - cx)
        rr = np.hypot(X - cx, Y - cy)
        acc += sgn * ang * np.exp(-(rr / rng.uniform(16, 34)) ** 2)
    t = 0.5 + 0.5 * np.tanh(acc * 1.2)
    return np.clip(t * np.exp(-(R / 62.0) ** 6), 0, 1)


def evaluate(T, tag, restarts=3, gs_iters=300, polish_steps=600, lr=2.5e-2):
    T = np.asarray(T, dtype=np.float64)
    T = T / T.max()
    t0 = time.time()
    best = None
    for r in range(restarts):
        g = gs.gs_run(T, iters=gs_iters, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        sc = m["pcc"] + m["ssim"]
        if best is None or sc > best[0]:
            best = (sc, q, m)
    print(f"  {tag} GS{gs_iters}×{restarts}: PCC {best[2]['pcc']:.4f} SSIM {best[2]['ssim']:.4f}", flush=True)
    res = POL.polish(T, best[1], steps=polish_steps, lr=lr)
    q = POL.quantize(res["phi"])
    m = POL.metrics_of(q, T)
    print(f"  {tag} +polish{polish_steps}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} "
          f"per-plane {[round(r['pcc'], 3) for r in m['rows']]} ({time.time()-t0:.0f}s)", flush=True)
    return q, m


if __name__ == "__main__":
    qa, ma = evaluate([spiral(s) for s in (5, 6, 7, 8)], "spiral")
    qb, mb = evaluate([rings_mod(s) for s in (9, 10, 11, 12)], "rings")
    qc, mc = evaluate([vortex_dots(s) for s in (31, 32, 33, 34)], "vortex")
    qd, md = evaluate([swirl_field(s) for s in (41, 42, 43, 44)], "swirl")
    best = max([(ma["pcc"], "spiral"), (mb["pcc"], "rings"), (mc["pcc"], "vortex"),
                (md["pcc"], "swirl")])
    print("лучший класс:", best, flush=True)
