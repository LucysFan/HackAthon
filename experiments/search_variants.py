"""Поиск лучшей схемы синтеза: GS-варианты, многоплоскостной GS, доводка.

Запуск:  .venv/bin/python experiments/search_variants.py
"""
import sys, time, warnings, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from hdoe import config as C, physics as P, metrics as M, polish as POL, gs
from hdoe.targets import load_targets

torch.set_num_threads(2)
T = load_targets()
Tq = POL._norm4(T)


def score(phi):
    q = POL.quantize(np.asarray(phi))
    m = POL.metrics_of(q, T)
    return q, m


def report(tag, phi, t0):
    q, m = score(phi)
    print(f"{tag:38s} PCC {m['pcc']:.4f}  SSIM {m['ssim']:.4f}  avg {(m['pcc']+m['ssim'])/2:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return q, m


def gs_run_state(phi, targets_amp, zs, iters, beta=1.0, seq="tp"):
    """Модифицированный GS с пере-последовательностью и обратной связью."""
    t = targets_amp
    for it in range(iters):
        order = range(len(zs)) if seq == "tp" else reversed(range(len(zs)))
        err = torch.zeros_like(phi)
        for i in order:
            z = zs[i]
            u = P.propagate_complex(phi, z)[0, 0]
            amp = torch.sqrt(torch.clamp(t[i], min=0))
            u_p = amp * torch.exp(1j * torch.angle(u))
            back = P.propagate_adj(u_p[None, None], z)[0, 0]
            new = torch.angle(back)[None, None]
            err = err + (new - phi)
            phi = phi + beta * (new - phi)
    return phi


t0 = time.time()
targets_amp = Tq[0]

# 1) GS 400 итераций, чередование прямой/обратный порядок (МИФИ-style)
phi = (torch.rand(1, 1, C.N, C.N, dtype=torch.float32) * 2 * np.pi - np.pi) * 0.5
p1 = gs_run_state(phi.clone(), targets_amp, C.ZS, 200, beta=1.0, seq="tp")
report("GS 200 (прямой)", p1, t0)
p2 = gs_run_state(p1, targets_amp, C.ZS, 200, beta=1.0, seq="rev")
report("GS 400 (прям+обратн)", p2, t0)

# 2) GS + доводка
for extra, lr in ((250, 2e-2),):
    r = POL.polish(T, np.asarray(p2[0, 0]), steps=extra, lr=lr)
    q, m = report(f"GS400 + polish {extra}@{lr}", r["phi"], t0)

# 3) продолжение доводки
r2 = POL.polish(T, np.asarray(POL.quantize(np.asarray(r["phi"]))), steps=250, lr=1e-2)
q2, m2 = report("...+polish 250@1e-2 (цикл2)", r2["phi"], t0)
r3 = POL.polish(T, np.asarray(POL.quantize(np.asarray(r2["phi"]))), steps=250, lr=5e-3)
q3, m3 = report("...+polish 250@5e-3 (цикл3)", r3["phi"], t0)
np.save(os.path.join(ROOT, "outputs/_search_best.npy"), q3)
print("сохранено outputs/_search_best.npy", flush=True)
