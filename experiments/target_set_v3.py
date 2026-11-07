"""Набор v3: 4 разные спиральные текстуры (разные рукава/закрутка/зерно)."""
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
    rng = np.random.default_rng(seed * 7919 + 13)
    g = gaussian_filter(rng.standard_normal((N, N)), sigma)
    return 1.0 + amp * g / (np.abs(g).max() + 1e-9)


def spiral(seed, arms=3, twist=5.0, p=2.2, r0=56.0, envp=3, mod=0.0, sig=2.0, amp=0.4):
    th = PH + R / twist * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th + mod * np.cos(R / 9.0))) ** p
    s = s * np.exp(-(R / r0) ** envp)
    return np.clip(s * grain(seed, sig, amp) * 1.7, 0, 1)


CANDIDATES = {
    "v3a": [dict(seed=101, arms=2, twist=3.6, p=2.0, amp=0.40),
            dict(seed=102, arms=3, twist=5.0, p=2.2, amp=0.40),
            dict(seed=103, arms=4, twist=6.5, p=2.4, amp=0.35),
            dict(seed=104, arms=5, twist=8.0, p=2.6, amp=0.45)],
    "v3b": [dict(seed=201, arms=3, twist=4.4, p=2.0, r0=58.0, amp=0.45),
            dict(seed=202, arms=3, twist=6.0, p=2.6, r0=52.0, amp=0.35),
            dict(seed=203, arms=2, twist=4.0, p=2.2, r0=60.0, amp=0.50),
            dict(seed=204, arms=4, twist=7.5, p=2.4, r0=54.0, amp=0.40)],
}


def evaluate(kw_list, tag, **pol):
    T = np.stack([spiral(**kw) for kw in kw_list]); T = T / T.max()
    t0 = time.time(); best = None
    for r in range(3):
        g = gs.gs_run(T, iters=300, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        if best is None or m["pcc"] + m["ssim"] > best[0]:
            best = (m["pcc"] + m["ssim"], q)
        print(f"  {tag} GS{r}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
    res = POL.polish(T, best[1], steps=pol.get("steps", 700), lr=pol.get("lr", 2.5e-2))
    q = POL.quantize(res["phi"]); m = POL.metrics_of(q, T)
    print(f"{tag}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} плоскости "
          f"{[round(r['pcc'], 3) for r in m['rows']]} ({time.time()-t0:.0f}s)", flush=True)
    return q, m, T


if __name__ == "__main__":
    for tag, kw in CANDIDATES.items():
        q, m, T = evaluate(kw, tag)
        from PIL import Image
        os.makedirs(os.path.join(ROOT, f"experiments/_preview/{tag}"), exist_ok=True)
        for i, t in enumerate(T):
            Image.fromarray((t * 255).astype(np.uint8), mode="L").save(
                os.path.join(ROOT, f"experiments/_preview/{tag}/t{i+1}.png"))
        np.save(os.path.join(ROOT, f"experiments/_{tag}_phase.npy"), q)
        np.save(os.path.join(ROOT, f"experiments/_{tag}_targets.npy"), T)
