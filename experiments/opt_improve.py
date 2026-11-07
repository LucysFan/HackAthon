"""Улучшение оптимизатора: SGDR-циклы, штраф фона (support), веса плоскостей."""
import sys, os, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from hdoe import config as C, physics as P, metrics as M, polish as POL, losses as L
from hdoe.targets import load_targets

torch.set_num_threads(2)
T = load_targets(); t = POL._norm4(T)
# «фон» — точки цели ниже 2% от максимума (тёмный фон объекта)
BG = (t < 0.02).float()
SIG = 1.0 - BG


def metrics_q(phi):
    q = POL.quantize(np.asarray(phi))
    m = POL.metrics_of(q, T)
    return q, m


def run(phi0, cycles, steps, lr, support=0.0, tag="", seed=0):
    torch.manual_seed(seed)
    x = torch.nn.Parameter(torch.as_tensor(np.asarray(phi0, dtype=np.float32))[None, None].clone())
    opt = torch.optim.Adam([x], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.03)
    t0 = time.time(); best = (None, -1)
    for cyc in range(cycles):
        for s in range(steps):
            I = torch.cat([P.propagate(x, z) for z in C.ZS], 1)
            Ip = torch.cat([L.normalize_batch(I[:, i:i+1]) for i in range(4)], 1)
            loss = sum((1 - M.pcc_torch(Ip[:, i:i+1], t[:, i:i+1])).mean() for i in range(4)) / 4
            loss = loss + sum((1 - M.ssim_torch(Ip[:, i:i+1], t[:, i:i+1])).mean() for i in range(4)) / 4
            if support > 0:
                # энергия в «тёмной» зоне должна быть мала относительно сигнальной
                e_bg = (Ip * BG).sum(dim=(1, 2, 3)) / BG.sum(dim=(1, 2, 3))
                e_sg = (Ip * SIG).sum(dim=(1, 2, 3)) / SIG.sum(dim=(1, 2, 3))
                loss = loss + support * (e_bg / (e_sg + 1e-8)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        q, m = metrics_q(x.detach()[0, 0])
        sc = m["pcc"] + m["ssim"]
        if sc > best[1]:
            best = (x.detach().clone(), sc, m)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.03)
        print(f"  {tag} цикл {cyc+1}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} (best {best[2]['pcc']:.4f}/{best[2]['ssim']:.4f}) {time.time()-t0:.0f}s", flush=True)
    q, m = metrics_q(best[0][0, 0])
    print(f"{tag}: ИТОГ PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
    return q, m, best[0][0, 0].numpy()


base = np.load(os.path.join(ROOT, "outputs/_search_best.npy"))
print("старт:", POL.metrics_of(base, T)["pcc"], flush=True)
q1, m1, p1 = run(base, cycles=6, steps=500, lr=2e-2, tag="SGDR-6x500")
q2, m2, p2 = run(p1, cycles=6, steps=500, lr=1.5e-2, support=0.02, tag="SGDR+support")
np.save(os.path.join(ROOT, "outputs/_opt_improved.npy"), q2 if (m2["pcc"]+m2["ssim"]) > (m1["pcc"]+m1["ssim"]) else q1)
