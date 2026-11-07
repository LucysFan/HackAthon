"""Многоплоскостной алгоритм Герчберга–Саксона (ТЗ, п. 5.6).

Цикл по плоскостям:
    U = D_z(e^{iφ});  U' = √t_i · U/|U|;  φ ← arg(D_z^{-1}(U'))
≥ 200 кругов, ≥ 3 перезапуска со случайной начальной фазой, выбор лучшего
(по средней PCC после квантования в 8 бит — результат приводится к тому же
протоколу, что и сдаваемый файл).
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import metrics as M
from . import physics as P


def _pcc_t(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def gs_run(targets: np.ndarray, zs=C.ZS, iters: int = 200, seed: int = 0,
           init: torch.Tensor | None = None, verbose: bool = False,
           dtype=torch.float64) -> dict:
    """Один прогон GS. targets: [4,N,N] float в произвольном масштабе."""
    t = torch.as_tensor(np.asarray(targets), dtype=dtype)
    t = t / t.flatten(1).max(dim=1).values.view(-1, 1, 1)      # √t_i — амплитуда цели
    g = torch.Generator().manual_seed(seed)
    if init is None:
        phi = (torch.rand(1, 1, C.N, C.N, generator=g, dtype=dtype) * 2 * torch.pi
               - torch.pi)
    else:
        phi = init.to(dtype).clone()
    hist = []
    for it in range(iters):
        for i, z in enumerate(zs):
            u = P.propagate_complex(phi, z)[0, 0]
            amp = torch.sqrt(torch.clamp(t[i], min=0))
            u_p = amp * torch.exp(1j * torch.angle(u))
            back = P.propagate_adj(u_p[None, None], z)[0, 0]
            phi = torch.angle(back)[None, None]
        if verbose and (it + 1) % 50 == 0:
            hist.append((it + 1, _score(phi, t, zs)))
            print(f"  GS iter {it + 1}: mean PCC {hist[-1][1]:.4f}")
    return dict(phi=phi, score=_score(phi, t, zs), hist=hist, iters=iters, seed=seed)


def _score(phi: torch.Tensor, t: torch.Tensor, zs) -> float:
    vals = []
    for i, z in enumerate(zs):
        I = P.propagate(phi, z)[0, 0]
        vals.append(_pcc_t(P.normalize01(I), t[i]))
    return float(np.mean(vals))


def gs_solve(targets: np.ndarray, zs=C.ZS, iters: int = 200, restarts: int = 3,
             seed: int = C.SEED, verbose: bool = True, refine_iters: int = 0,
             init: torch.Tensor | None = None) -> dict:
    """GS с перезапусками; возвращает лучшую маску и историю по перезапускам.

    ``init`` — необязательная начальная маска (для «тёплого старта»).
    """
    best = None
    runs = []
    for r in range(restarts):
        run = gs_run(targets, zs=zs, iters=iters, seed=seed + r, verbose=verbose,
                     init=(init if r == 0 else None))
        runs.append(dict(seed=seed + r, score=run["score"], iters=iters))
        if best is None or run["score"] > best["score"]:
            best = run
        if verbose:
            print(f"GS restart {r + 1}/{restarts}: mean PCC = {run['score']:.4f}")
    out = dict(phi=best["phi"], score=best["score"], runs=runs,
               iters=iters, restarts=restarts, seed=seed, hist=best["hist"])
    if refine_iters > 0:
        ref = gs_run(targets, zs=zs, iters=refine_iters, seed=seed,
                     init=best["phi"], verbose=verbose)
        out = dict(phi=ref["phi"], score=ref["score"], runs=runs,
                   iters=iters, restarts=restarts, seed=seed, hist=best["hist"],
                   refine=dict(iters=refine_iters, score=ref["score"]))
    return out


def gs_phase_quantized(targets: np.ndarray, **kw) -> tuple[torch.Tensor, dict]:
    """GS → квантованная (8 бит) фаза, как для сдаваемого файла."""
    res = gs_solve(targets, **kw)
    v, phi_q = P.quantize(res["phi"])
    return phi_q, res


def evaluate_phase(phi_q: np.ndarray, targets01: np.ndarray, zs=C.ZS) -> list[dict]:
    """Метрики по протоколу п. 5.5 (из квантованной фазы, мин–макс нормировка)."""
    phi_t = torch.as_tensor(np.asarray(phi_q), dtype=torch.float64)[None, None]
    rows = []
    for i, z in enumerate(zs):
        I = P.propagate(phi_t, z)[0, 0].numpy()
        I01 = P.normalize01(I)
        rows.append(dict(plane=i + 1, z_mm=zs[i] * 1e3,
                         pcc=M.pcc(targets01[i], I01),
                         ssim=M.ssim(targets01[i], I01),
                         pcc_raw=M.pcc(targets01[i], I),
                         ssim_raw=M.ssim(targets01[i], I)))
    return rows
