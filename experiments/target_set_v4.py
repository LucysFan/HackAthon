"""Набор v4: одно семейство спиральных текстур с вариациями параметров."""
import sys, os, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.ndimage import gaussian_filter
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


def spiral(seed, arms=3, twist=5.0, p=2.2, r0=56.0, envp=3, sig=2.0, amp=0.4):
    th = PH + R / twist * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th)) ** p * np.exp(-(R / r0) ** envp)
    return np.clip(s * grain(seed, sig, amp) * 1.7, 0, 1)


VARIANTS = {
    "v4same": [dict(seed=501, arms=3, twist=4.8, amp=0.36),
               dict(seed=502, arms=3, twist=5.0, amp=0.42),
               dict(seed=503, arms=3, twist=5.2, amp=0.38),
               dict(seed=504, arms=3, twist=5.4, amp=0.44)],
    "v4mix": [dict(seed=511, arms=3, twist=4.9, amp=0.40),
              dict(seed=512, arms=3, twist=5.3, amp=0.36),
              dict(seed=513, arms=4, twist=6.8, amp=0.42),
              dict(seed=514, arms=3, twist=4.6, amp=0.38)],
}


def evaluate(kw_list, tag, steps=800, lr=2.5e-2):
    T = np.stack([spiral(**kw) for kw in kw_list]); T = T / T.max()
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
    for tag, kw in VARIANTS.items():
        q, m, T = evaluate(kw, tag)
        d = os.path.join(ROOT, f"experiments/_preview/{tag}"); os.makedirs(d, exist_ok=True)
        for i, t in enumerate(T):
            Image.fromarray((t * 255).astype(np.uint8), mode="L").save(os.path.join(d, f"t{i+1}.png"))
        np.save(os.path.join(ROOT, f"experiments/_{tag}_phase.npy"), q)
        np.save(os.path.join(ROOT, f"experiments/_{tag}_targets.npy"), T)
