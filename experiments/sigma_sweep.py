"""Сколько даёт сглаживание целей: замер разрешимости (GS 200 итераций)."""
import sys, os, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from hdoe import config as C, polish as POL, gs
from hdoe.targets import load_targets

torch.set_num_threads(2)
T0 = load_targets()          # исходные (уже σ=4)
for sig in (4, 8, 10, 12, 14):
    T = np.stack([gaussian_filter(t, sig - 4 if sig > 4 else 0.1, mode="nearest") for t in T0])
    T = T / T.max()
    for seed in (0,):
        r = gs.gs_solve(T, iters=200, restarts=1, seed=seed, verbose=False)
        q = POL.quantize(np.asarray(r["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        print(f"sigma={sig:2d}: GS200 PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
