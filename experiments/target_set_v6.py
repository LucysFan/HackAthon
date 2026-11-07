"""Набор v6: семейство спиралей, различающихся разворотом, шагом и зерном."""
import sys, os, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.ndimage import gaussian_filter, rotate
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


def spiral_rot(seed, rot=0.0, arms=3, twist=5.0, p=2.2, sig=2.0, amp=0.4, r0=56.0):
    # разворот: та же маска, повёрнутая на rot (спираль не осесимметрична при нечётных рукавах)
    ph = PH - np.deg2rad(rot)
    th = ph + R / twist * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th)) ** p * np.exp(-(R / r0) ** 3)
    return np.clip(s * grain(seed, sig, amp) * 1.7, 0, 1)


CANDS = {
    "v6rot": [dict(seed=701, rot=0, twist=4.8), dict(seed=702, rot=40, twist=5.0),
              dict(seed=703, rot=80, twist=5.2), dict(seed=704, rot=120, twist=5.4)],
    "v6rot2": [dict(seed=711, rot=0, twist=5.0, sig=1.6, amp=0.45),
               dict(seed=712, rot=55, twist=5.0, sig=2.4, amp=0.35),
               dict(seed=713, rot=110, twist=5.0, sig=2.0, amp=0.50),
               dict(seed=714, rot=165, twist=5.0, sig=2.8, amp=0.40)],
}


def evaluate(kw, tag, steps=800, lr=2.5e-2):
    T = np.stack([spiral_rot(**k) for k in kw]); T = T / T.max()
    t0 = time.time(); best = None
    for r in range(3):
        g = gs.gs_run(T, iters=300, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0])); m = POL.metrics_of(q, T)
        if best is None or m["pcc"] + m["ssim"] > best[0]:
            best = (m["pcc"] + m["ssim"], q)
    res = POL.polish(T, best[1], steps=steps, lr=lr)
    q = POL.quantize(res["phi"]); m = POL.metrics_of(q, T)
    print(f"{tag}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} плоскости "
          f"{[round(r['pcc'], 3) for r in m['rows']]} ({time.time()-t0:.0f}s)", flush=True)
    return q, m, T


if __name__ == "__main__":
    from PIL import Image
    for tag, kw in CANDS.items():
        q, m, T = evaluate(kw, tag)
        d = os.path.join(ROOT, f"experiments/_preview/{tag}"); os.makedirs(d, exist_ok=True)
        W = Image.new("L", (128 * 4 + 24, 128), 128)
        for i, t in enumerate(T):
            im = Image.fromarray((t * 255).astype(np.uint8), mode="L")
            im.save(os.path.join(d, f"t{i+1}.png")); W.paste(im, (i * 136, 0))
        W.save(os.path.join(ROOT, f"experiments/_preview/{tag}_quad.png"))
        np.save(os.path.join(ROOT, f"experiments/_{tag}_phase.npy"), q)
        np.save(os.path.join(ROOT, f"experiments/_{tag}_targets.npy"), T)
