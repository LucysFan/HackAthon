"""Доводка фазы по физической модели (п. 5.6.1 ТЗ).

Модуль реализует точную, но дорогую часть конвейера: прямое обращение
физической модели (4 плоскости, одно-БПФ-схема Френеля) и градиентную
оптимизацию маски. Используется:

* ``scripts/refine_phase.py`` — отдельная доводка;
* ``hdoe.polish`` — пакетная доводка нескольких стартов (сеть, GS, смесь)
  с выбором лучшего по протоколу отчёта.

Параметризация — «сырая» фаза без ограничения формы: поле зависит от φ
периодически, а после квантования в 8 бит это даёт полный диапазон кодов
0…255 (требование п. 7.2 — наличие в DOE.png значений 0 и 255).
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import losses as L
from . import metrics as M
from . import physics as P


def _norm4(targets: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(C.N_PLANES, C.N, C.N)
    t = t / torch.clamp(t.flatten(1).max(dim=1).values.view(-1, 1, 1), min=1e-9)
    return t[None]


@torch.no_grad()
def metrics_of(phi_q: np.ndarray, targets01: np.ndarray) -> dict:
    """Метрики ровно по протоколу отчёта (из квантованной фазы, мин–макс)."""
    ph = torch.as_tensor(np.asarray(phi_q, dtype=np.float32))
    ph = ph[None, None] if ph.dim() == 2 else ph
    rows = []
    pcc, ssim = [], []
    for i, z in enumerate(C.ZS):
        I = P.propagate(ph, z)[0, 0].numpy()
        I01 = P.normalize01(I)
        a, b = float(M.pcc(targets01[i], I01)), float(M.ssim(targets01[i], I01))
        pcc.append(a)
        ssim.append(b)
        rows.append(dict(plane=i + 1, z_mm=C.ZS_MM[i], pcc=a, ssim=b))
    return dict(pcc=float(np.mean(pcc)), ssim=float(np.mean(ssim)), rows=rows)


def polish(targets01: np.ndarray, phi0: np.ndarray, steps: int = 200, lr: float = 3e-2,
           tv: float = 0.0, quant_noise: bool = False, seed: int = 0,
           refine_ssim: bool = True) -> dict:
    """Доводка фазы. Возвращает dict(phi, hist, loss).

    Критерий — тот же, что в отчёте: (1−PCC) + (1−SSIM) по четырём плоскостям,
    интенсивности нормируются мин–макс (иначе SSIM считается на «сырых»
    значениях с огромными μ и почти не реагирует на структуру — см. отчёт).
    """
    torch.manual_seed(seed)
    t = _norm4(targets01)
    a0 = torch.as_tensor(np.asarray(phi0, dtype=np.float32)).reshape(C.N, C.N)
    x = torch.nn.Parameter(a0[None, None].clone())
    opt = torch.optim.Adam([x], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    hist = []
    for _ in range(steps):
        phi = x
        phi_in = phi
        if quant_noise:
            phi_in = phi + (torch.rand_like(phi) - 0.5) * C.QSTEP
        I = torch.cat([P.propagate(phi_in, z) for z in C.ZS], 1)         # [1,4,N,N]
        Ip = torch.cat([L.normalize_batch(I[:, i:i + 1]) for i in range(C.N_PLANES)], 1)
        loss = sum((1 - M.pcc_torch(Ip[:, i:i + 1], t[:, i:i + 1])).mean()
                   for i in range(C.N_PLANES))
        if refine_ssim:
            loss = loss + sum((1 - M.ssim_torch(Ip[:, i:i + 1], t[:, i:i + 1])).mean()
                              for i in range(C.N_PLANES))
        loss = loss / C.N_PLANES
        if tv > 0:
            d1 = (phi[..., 1:, :] - phi[..., :-1, :]).abs().mean()
            d2 = (phi[..., :, 1:] - phi[..., :, :-1]).abs().mean()
            loss = loss + tv * (d1 + d2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        hist.append(float(loss.detach()))
    phi_out = x.detach()[0, 0].numpy()
    return dict(phi=phi_out, hist=hist, loss=hist[-1] if hist else None)


def quantize(phi: np.ndarray) -> np.ndarray:
    """float-фаза → квантованная 8-битная фаза (п. 4.6)."""
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi, dtype=np.float32)))))
    return np.asarray(P.decode8bit(v))


def best_start(targets01: np.ndarray, cands: dict, steps: int, lr: float,
               restarts: int = 1, verbose: bool = True) -> dict:
    """Доводит каждый старт и возвращает лучший по (PCC + SSIM) после квантования.

    ``restarts`` > 1 добавляет к каждому старту перезапуски со случайным
    сдвигом фазы (разные локальные минимумы), финальный выбор — по метрикам
    квантованной маски, то есть ровно по протоколу отчёта.
    """
    out = {}
    for name, phi in cands.items():
        variants = [(0, phi)]
        if restarts > 1:
            rng = np.random.default_rng(1234)
            for k in range(1, restarts):
                variants.append((k, np.asarray(phi) + rng.uniform(-0.5, 0.5, np.shape(phi))))
        best_v = None
        for k, ph0 in variants:
            r = polish(targets01, ph0, steps=steps, lr=lr, seed=k)
            phq = quantize(r["phi"])
            m = metrics_of(phq, targets01)
            sc = m["pcc"] + m["ssim"]
            if best_v is None or sc > best_v[0]:
                best_v = (sc, phq, m, r["hist"], k)
        out[name] = dict(phi=best_v[1], metrics=best_v[2], hist=best_v[3],
                         restart=best_v[4])
        if verbose:
            print(f"    старт «{name}»: PCC {best_v[2]['pcc']:.4f} "
                  f"SSIM {best_v[2]['ssim']:.4f}", flush=True)
    best = max(out, key=lambda k: out[k]["metrics"]["pcc"] + out[k]["metrics"]["ssim"])
    if verbose:
        print(f"    выбран старт «{best}»", flush=True)
    return dict(best=best, results=out, phi=out[best]["phi"], metrics=out[best]["metrics"])
