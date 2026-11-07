"""Функции потерь и метрики обучения (ТЗ, п. 5.4.3, приложение В)."""
from __future__ import annotations

import torch

from . import config as C
from . import metrics as M
from . import physics as P


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Мин–макс нормировка каждой плоскости к [0,1] (дифференцируемо)."""
    dims = (1, 2, 3)
    lo = x.amin(dim=dims, keepdim=True)
    hi = x.amax(dim=dims, keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def plane_losses(inten: torch.Tensor, targets: torch.Tensor, norm: bool = True) -> dict:
    """Поплоскостные метрики [B,4,N,N] и разложение общей потери.

    inten  — интенсивности из модели; targets — цели [B,4,N,N] в [0,1].
    """
    pred = normalize_batch(inten) if norm else inten
    dims = (1, 2, 3)
    b = inten.shape[0]
    pcc = torch.stack([M.pcc_torch(pred[:, i].unsqueeze(1), targets[:, i].unsqueeze(1))
                       for i in range(C.N_PLANES)], 1)                 # [B,4]
    ssim = torch.stack([M.ssim_torch(pred[:, i].unsqueeze(1), targets[:, i].unsqueeze(1))
                        for i in range(C.N_PLANES)], 1)                # [B,4]
    mse = torch.stack([M.scale_invariant_mse(pred[:, i].flatten(1), targets[:, i].flatten(1))
                       for i in range(C.N_PLANES)], 1)                 # [B,4]
    return dict(pcc=pcc, ssim=ssim, mse=mse, pred=pred, n=b, dims=dims)


def total_loss(phi: torch.Tensor, inten: torch.Tensor, targets: torch.Tensor,
               w_pcc: float = 1.0, w_ssim: float = 1.0, w_mse: float = 0.1,
               w_tv: float = 1e-3, tv: bool = True) -> tuple[torch.Tensor, dict]:
    """L = Σ_i[(1−PCC_i) + (1−SSIM_i) + 0,1·L_mse_i] + 1e-3·TV(φ)."""
    parts = plane_losses(inten, targets)
    per_plane = (1 - parts["pcc"]) + (1 - parts["ssim"]) + w_mse * parts["mse"]
    loss = per_plane.mean()
    tv_val = None
    if tv and w_tv > 0:
        tv_val = P.phase_tv(phi).mean()
        loss = loss + w_tv * tv_val
    log = dict(loss=float(loss.detach()),
               pcc=float(parts["pcc"].detach().mean()),
               ssim=float(parts["ssim"].detach().mean()),
               mse=float(parts["mse"].detach().mean()),
               tv=float(tv_val.detach()) if tv_val is not None else 0.0)
    return loss, log
