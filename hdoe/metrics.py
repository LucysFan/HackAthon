"""Метрики: PCC, SSIM (окно 11×11, гауссово, σ=1,5), локальная SSIM-карта, TV.

Формулы — приложение В. Реализация самодостаточна (torch/numpy), при этом
численно совпадает с ``skimage.metrics.structural_similarity`` при
``gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


# ------------------------------------------------------------------ ядро SSIM
def _gauss_kernel(win: int = C.SSIM_WIN, sigma: float = C.SSIM_SIGMA, dtype=torch.float64):
    r = torch.arange(win, dtype=dtype) - (win - 1) / 2.0
    k = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k.view(1, 1, win, 1)


def _filter(x: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
    """Разделимое гауссово сглаживание без изменения размера (replicate-pad)."""
    pad = (kern.shape[-2] - 1) // 2
    y = F.pad(x, (0, 0, pad, pad), mode="reflect")
    y = F.conv2d(y, kern.to(x.dtype))
    y = F.pad(y, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, kern.to(x.dtype).transpose(-1, -2))
    return y


def _tt(x):
    """numpy → torch (float64), torch — без изменений."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Поэлементная карта SSIM для пары [B,1,H,W] (или [H,W]) в [0,1]."""
    x, y = _tt(x), _tt(y)
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None, None]
        y = y[None, None]
    kern = _gauss_kernel(dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
    ux = _filter(x, kern)
    uy = _filter(y, kern)
    uxx = _filter(x * x, kern)
    uyy = _filter(y * y, kern)
    uxy = _filter(x * y, kern)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    c1, c2 = C.SSIM_C1, C.SSIM_C2
    m = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return m[0, 0] if squeeze else m


def _crop(m: torch.Tensor, pad: int | None = None) -> torch.Tensor:
    """Обрезка краевой зоны карты (как в skimage при gaussian_weights=True)."""
    if pad is None:
        pad = (C.SSIM_WIN - 1) // 2
    if pad <= 0:
        return m
    return m[..., pad:-pad, pad:-pad]


def ssim(x, y, crop: bool = True) -> float:
    """Скалярный SSIM: среднее по карте (краевая зона 5 px исключена, как в skimage)."""
    m = ssim_map(*map(_tt, (x, y)))
    if crop:
        m = _crop(m)
    return float(m.mean())


def ssim_torch(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean",
               crop: bool = True) -> torch.Tensor:
    """Дифференцируемый SSIM по батчу [B,1,H,W] → [B].

    Краевая зона (5 px) исключается так же, как в скалярном ``ssim`` и в
    ``skimage``: критерий обучения совпадает с отчётной метрикой.
    """
    m = ssim_map(x, y)
    if crop:
        m = _crop(m)
    dims = (1, 2, 3)
    _ = reduction
    return m.mean(dim=dims)


# ------------------------------------------------------------------------ PCC
def pcc(x, y) -> float:
    """Коэффициент корреляции Пирсона по развёрнутым векторам (В2)."""
    if isinstance(x, torch.Tensor):
        a = x.detach().flatten().double()
        b = y.detach().flatten().double()
    else:
        a = torch.as_tensor(np.asarray(x), dtype=torch.float64).flatten()
        b = torch.as_tensor(np.asarray(y), dtype=torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def pcc_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Дифференцируемый PCC по батчу [B,...] → [B]."""
    a = x.flatten(1)
    b = y.flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


# ------------------------------------------------------ масштабно-инвариантная MSE
def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """В1: L_mse = mean((s*p - t)²), s = ⟨p,t⟩/⟨p,p⟩, по 16384 элементам."""
    p = pred.flatten(1)
    t = target.flatten(1)
    s = (p * t).sum(1) / ((p * p).sum(1) + 1e-12)
    return ((s[:, None] * p - t) ** 2).mean(1)


# ------------------------------------------------------------------ полный протокол
def evaluate_pair(target01: np.ndarray, recon01: np.ndarray) -> dict:
    """Метрики одной пары (цель, реконструкция) в [0,1] по 128×128."""
    return dict(pcc=pcc(target01, recon01), ssim=ssim(target01, recon01))


def roi_stats(img01: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    """Статистика ROI (x0, y0, x1, y1) для дашборда."""
    x0, y0, x1, y1 = box
    roi = np.asarray(img01, dtype=np.float64)[y0:y1, x0:x1]
    total = float(np.abs(np.asarray(img01, dtype=np.float64)).sum())
    return dict(mean=float(roi.mean()), std=float(roi.std()),
                min=float(roi.min()), max=float(roi.max()),
                energy_frac=float(roi.sum() / (total + 1e-12)))
