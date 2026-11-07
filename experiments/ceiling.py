"""Оценка потолка качества: длинная оптимизация без ограничений на число шагов."""
import sys, time, warnings, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from hdoe import config as C, physics as P, metrics as M, polish as POL, losses as L
from hdoe.targets import load_targets

torch.set_num_threads(2)
T = load_targets(); t = POL._norm4(T)

def run(phi0, steps, lr, lr_end=None, pcc_only=False, tag=""):
    x = torch.nn.Parameter(torch.as_tensor(np.asarray(phi0, dtype=np.float32))[None, None].clone())
    opt = torch.optim.Adam([x], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr_end or lr*0.02)
    t0 = time.time()
    for s in range(steps):
        I = torch.cat([P.propagate(x, z) for z in C.ZS], 1)
        Ip = torch.cat([L.normalize_batch(I[:, i:i+1]) for i in range(4)], 1)
        loss = sum((1 - M.pcc_torch(Ip[:, i:i+1], t[:, i:i+1])).mean() for i in range(4)) / 4
        if not pcc_only:
            loss = loss + sum((1 - M.ssim_torch(Ip[:, i:i+1], t[:, i:i+1])).mean() for i in range(4)) / 4
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (s + 1) % 500 == 0:
            print(f"  {tag} шаг {s+1}: loss {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    q = POL.quantize(np.asarray(x.detach()[0, 0]))
    m = POL.metrics_of(q, T)
    print(f"{tag}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f} avg {(m['pcc']+m['ssim'])/2:.4f} за {time.time()-t0:.0f}s", flush=True)
    return m, np.asarray(x.detach()[0, 0]), q

base = np.load(os.path.join(ROOT, "outputs/_search_best.npy"))
print("старт из _search_best:", POL.metrics_of(base, T)["pcc"], flush=True)
m, ph, q = run(base, 1500, 1e-2, tag="A: 1500@1e-2 (PCC+SSIM)")
m2, ph2, q2 = run(ph, 1500, 5e-3, tag="B: ещё 1500@5e-3")
m3, ph3, q3 = run(ph2, 2000, 2e-3, tag="C: ещё 2000@2e-3")
# покоординатно: только PCC
m4, ph4, q4 = run(q3, 1500, 3e-3, pcc_only=True, tag="D: только PCC 1500@3e-3")
m5, ph5, q5 = run(q3, 1500, 3e-3, tag="E: повтор с SSIM 1500@3e-3")
best = max([(m4["pcc"]+m4["ssim"], q4), (m5["pcc"]+m5["ssim"], q5)], key=lambda v: v[0])
np.save(os.path.join(ROOT, "outputs/_ceiling_best.npy"), best[1])
print("лучшее сохранено в outputs/_ceiling_best.npy", flush=True)
