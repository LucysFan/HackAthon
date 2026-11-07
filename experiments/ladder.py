"""Лестница сложности: достижимая точность 4-плоскостного синтеза в зависимости
от «разнообразия» целей (доля независимых деталей между плоскостями)."""
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from hdoe import config as C, physics as P, metrics as M, gs as G
from hdoe.targets import load_targets
torch.set_num_threads(2)

def blend(shared, distinct, frac):
    """frac = доля «независимой» части (по площади)."""
    out = []
    for d in distinct:
        m = d > 0.05
        out.append(shared * (1 - m) + d * m if False else np.where(m, distinct[len(out)] if False else d, shared))
    return np.stack(out)

def ladder(frac_list):
    T = load_targets()
    shared = T.mean(0); shared = (shared > shared.mean() * 0.5).astype(float)
    # общая база: крупный объект; независимые детали — вырезы/добавки
    base = np.zeros_like(shared)
    n = np.arange(C.N) - C.N // 2
    Y, X = np.meshgrid(n, n, indexing="ij"); R = np.hypot(X, Y)
    base[(R > 12) & (R < 44)] = 1.0
    res = {}
    for frac in frac_list:
        quads = []
        for i in range(4):
            q = base.copy()
            m = (np.abs(X) < frac * 60) & (np.abs(Y - i * 8 + 12) < frac * 45)
            q[m] = (T[i] > 0.05)[m] * 1.0
            q = q / max(q.max(), 1e-9)
            quads.append(q)
        quads = np.stack(quads)
        quads = quads / quads.reshape(4, -1).sum(-1)[:, None, None] * quads[0].size * 0.2
        t0 = __import__("time").time()
        r = G.gs_solve(quads, iters=200, restarts=1, verbose=False)
        pccs = []
        for i, z in enumerate(C.ZS):
            I = P.normalize01(P.propagate(r["phi"], z)[0, 0])
            pccs.append(float(M.pcc(torch.as_tensor(quads[i]), I)))
        res[frac] = (round(float(np.mean(pccs)), 3), round(float(r["score"]), 3))
        print(f"frac={frac}: GS per-plane PCC {[round(p,3) for p in pccs]} mean {res[frac]}")
    return res

if __name__ == "__main__":
    ladder([0.0, 0.15, 0.3, 0.5, 0.8, 1.0])
