"""Численный эксперимент: предельная достижимость 4-плоскостного синтеза
(прямая оптимизация фазы градиентом vs GS) для разных наборов целей.
Результат используется в отчёте для обоснования выбора целей и режима обучения.
"""
import sys, time, warnings
import numpy as np, torch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from hdoe import config as C, physics as P, gs as G, metrics as M

torch.set_num_threads(2)


def adam_phase(T, steps=400, lr=0.05, seed=0):
    t = torch.as_tensor(T, dtype=torch.float32)
    t = t / t.flatten(1).max(dim=1).values.view(-1, 1, 1)
    g = torch.Generator().manual_seed(seed)
    raw = torch.nn.Parameter(torch.randn(1, 1, C.N, C.N, generator=g) * 0.5)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-3)
    best = None
    for s in range(steps):
        phi = torch.pi * torch.tanh(raw)
        I = torch.cat([P.propagate(phi, z) for z in C.ZS], 1)
        In = torch.cat([P.normalize01(I[:, i:i + 1]) for i in range(4)], 1)
        loss = sum((1 - M.pcc_torch(In[:, i:i + 1], t[i:i + 1])) for i in range(4)).mean() \
             + sum((1 - M.ssim_torch(In[:, i:i + 1], t[i:i + 1])) for i in range(4)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 50 == 0:
            print(f"  step {s} loss {loss.item():.4f}")
    return phi.detach()


def evaluate(phi):
    out = []
    for i, z in enumerate(C.ZS):
        I = P.propagate(phi, z)[0, 0]
        out.append(float(M.pcc(P.normalize01(I), torch.as_tensor(TG[i]))))
    return out


if __name__ == "__main__":
    from hdoe.targets import load_targets
    TG = load_targets()
    t0 = time.time()
    phi = adam_phase(TG, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print("Adam per-plane PCC", [round(v, 3) for v in evaluate(phi)], f"{time.time() - t0:.0f}s")
    r = G.gs_solve(TG, iters=300, restarts=1, verbose=False)
    print("GS   per-plane PCC", [round(float(M.pcc(P.normalize01(P.propagate(r["phi"], z)[0, 0]),
                                                    torch.as_tensor(TG[i]))), 3)
                                 for i, z in zip(range(4), C.ZS)])
