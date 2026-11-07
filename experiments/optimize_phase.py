"""Поиск предельно достижимой маски для заданной четвёрки целей.

Сравниваются: GS (перезапуски), GS+Adam-доводка физическим лоссом
(без TV / с малым TV), Adam с нуля. Метрики — ровно протокол отчёта
(мин–макс нормировка интенсивности, PCC по 16384, SSIM 11×11 σ=1.5).
"""
import sys, time, warnings
sys.path.insert(0, '/home/user/HackAthon')
import numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G
from hdoe.targets import load_targets

T = load_targets()
q = CO.equalize_energy_t(torch.as_tensor(T, dtype=torch.float32))
Tt = torch.as_tensor(T, dtype=torch.float32)


def report_metrics(phi):
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi, dtype=np.float32)))))
    ph = torch.as_tensor(P.decode8bit(v))[None, None]
    pcc, ssim = [], []
    for i, z in enumerate(C.ZS):
        I = P.propagate(ph, z)[0, 0].numpy()
        I01 = P.normalize01(I)
        pcc.append(M.pcc(T[i], I01)); ssim.append(M.ssim(T[i], I01))
    return pcc, ssim


def show(tag, phi, t0=None):
    pcc, ssim = report_metrics(phi)
    extra = f" ({time.time()-t0:.0f}s)" if t0 else ""
    print(f"{tag:28s} PCC {[round(v,3) for v in pcc]} mean {np.mean(pcc):.4f} | "
          f"SSIM {[round(v,3) for v in ssim]} mean {np.mean(ssim):.4f}{extra}", flush=True)
    return float(np.mean(pcc)), float(np.mean(ssim))


def adam_refine(phi0, steps=500, lr=5e-3, tv=0.0, seed=0):
    torch.manual_seed(seed)
    a0 = torch.as_tensor(np.asarray(phi0, dtype=np.float32))
    a0 = a0.reshape(C.N, C.N)                       # принимаем [.]/[1,N,N]/[1,1,N,N]
    phi = torch.nn.Parameter(a0[None, None].clone())
    opt = torch.optim.Adam([phi], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    for s in range(steps):
        I = torch.cat([P.propagate(phi, z) for z in C.ZS], 1)
        Ip = torch.cat([L.normalize_batch(I[:, i:i + 1]) for i in range(4)], 1)  # [1,4,N,N]
        qq = q[None]
        loss = sum((1 - M.pcc_torch(Ip[:, i:i + 1], qq[:, i:i + 1])).mean() for i in range(4)) / 4
        loss = loss + sum((1 - M.ssim_torch(Ip[:, i:i + 1], qq[:, i:i + 1])).mean() for i in range(4)) / 4
        if tv > 0:
            d1 = (phi[..., 1:, :] - phi[..., :-1, :]).abs().mean()
            d2 = (phi[..., :, 1:] - phi[..., :, :-1]).abs().mean()
            loss = loss + tv * (d1 + d2)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return phi.detach()


print("=== 1. GS (300 кругов × 3 перезапуска) ===")
t0 = time.time()
best = None
for r in range(3):
    res = G.gs_run(T, iters=300, seed=C.SEED + r)
    m = show(f"GS restart {r}", res["phi"][0, 0].numpy())
    if best is None or m[1] > best[0]:
        best = (m[1], res["phi"].clone())
print("best GS SSIM", best[0])
phi_gs = best[1]

print("\n=== 2. GS + Adam-доводка ===")
for tv in (0.0, 1e-4, 1e-3):
    t0 = time.time()
    ph = adam_refine(phi_gs, steps=400, lr=3e-3, tv=tv)
    show(f"GS + adam tv={tv}", ph[0, 0].numpy(), t0)

print("\n=== 3. Adam с нуля ===")
g = torch.Generator().manual_seed(0)
z0 = (torch.rand(1, 1, C.N, C.N, generator=g) * 2 * np.pi - np.pi)
t0 = time.time()
ph = adam_refine(z0, steps=800, lr=1e-2, tv=0.0)
show("from scratch 800", ph[0, 0].numpy(), t0)
t0 = time.time()
ph = adam_refine(ph, steps=400, lr=2e-3, tv=0.0)
show("then 400 more", ph[0, 0].numpy(), t0)
np.save('/home/user/HackAthon/outputs/_opt_phi.npy', ph[0, 0].numpy())
