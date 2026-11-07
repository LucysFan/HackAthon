"""Доводка фазы по физической модели (ТЗ, п. 5.6.2).

Прямая оптимизация фазы φ (не весов сети) градиентом физического лосса
п. 5.4.3 из заданного начального приближения: выход сети, GS-маска или их
усреднение. Используется как «старт доводки сети»: 300–800 шагов Adam,
lr ≈ 5·10⁻² с косинусным расписанием, шум квантования δ/2 на каждом шаге.

Доводка не заменяет сеть: сеть даёт начальное приближение за один проход,
доводка уточняет маску под конкретную четвёрку целей (тестовая оптимизация).
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import losses as L
from . import physics as P


def refine_phase(targets01: np.ndarray, init_phi: np.ndarray | torch.Tensor,
                 steps: int = 400, lr: float = 5e-2, quant_noise: bool = True,
                 w_tv: float = 1e-3, seed: int = 0, verbose: bool = False) -> dict:
    """Уточнение фазы из начального приближения. Возвращает float-фазу (рад)."""
    torch.manual_seed(seed)
    t = torch.as_tensor(np.asarray(targets01), dtype=torch.float32)
    t = t / torch.clamp(t.flatten(1).max(dim=1).values.view(-1, 1, 1), min=1e-9)
    t = t[None]                                   # [1,4,N,N]
    if isinstance(init_phi, np.ndarray):
        init = np.arctan2(np.sin(init_phi), np.cos(init_phi))
        raw0 = torch.as_tensor(init, dtype=torch.float32)[None, None]
    else:
        raw0 = init_phi.float()[None, None] if init_phi.dim() == 2 else init_phi.float()
    # параметризация φ = π·tanh(u): гладко, без «прыжков» через ±π
    u = torch.nn.Parameter(torch.atanh(torch.clamp(raw0 / np.pi, -0.995, 0.995)))
    opt = torch.optim.Adam([u], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    hist = []
    for s in range(steps):
        phi = np.pi * torch.tanh(u)
        if quant_noise:
            xi = (torch.rand_like(phi) - 0.5) * C.QSTEP
            phi_in = phi + xi
        else:
            phi_in = phi
        I = torch.cat([P.propagate(phi_in, z) for z in C.ZS], 1)   # [1,4,N,N]
        I = torch.cat([L.normalize_batch(I[:, i:i + 1]) for i in range(C.N_PLANES)], 1)
        loss, log = L.total_loss(phi, I, t, w_tv=w_tv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(log)
        if verbose and (s + 1) % 100 == 0:
            print(f"  доводка {s + 1}/{steps}: loss {log['loss']:.4f} pcc {log['pcc']:.3f} "
                  f"ssim {log['ssim']:.3f}")
    phi = (np.pi * torch.tanh(u)).detach()
    return dict(phi=phi[0, 0], hist=hist, loss=hist[-1] if hist else None)


def blend_phases(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    """Круговая интерполяция двух фаз: arg(w·e^{ia} + (1−w)·e^{ib})."""
    za = np.exp(1j * np.asarray(a))
    zb = np.exp(1j * np.asarray(b))
    z = w * za + (1 - w) * zb
    return np.angle(z)
