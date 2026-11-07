"""Физические и вычислительные параметры комплекса (ТЗ, разделы 4 и приложение Б).

Все величины — в СИ (метры, радианы, пиксели), кроме явно помеченных ``_mm``.
Модуль не зависит от Torch: используется и в build-скриптах, и в отчёте.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 4.7, 2.8 --
LAM: float = 532e-9                 # длина волны, м
DX: float = 10e-6                   # шаг пиксела ПВМС, м
N: int = 128                        # размер маски / изображений, px
ZS_MM: tuple[float, float, float, float] = (24.1, 25.3, 26.5, 27.7)
ZS: tuple[float, ...] = tuple(z * 1e-3 for z in ZS_MM)
N_PLANES: int = len(ZS)

K: float = 2.0 * np.pi / LAM                       # 1.18105e7 1/м
Z_CRIT: float = N * DX ** 2 / LAM                  # 24.0602 мм
QSTEP: float = 2.0 * np.pi / 255.0                 # шаг квантования 8 бит, рад
QSTEP_SIGMA: float = QSTEP / np.sqrt(12.0)         # 0.00711 рад

# параметры зоны дашборда (п. 12.4 A1, приложение Б)
ZSTACK_Z0: float = 23.0e-3
ZSTACK_Z1: float = 28.6e-3
ZSTACK_STEP: float = 0.02e-3

# SSIM (п. 5.4.3, В3)
SSIM_WIN: int = 11
SSIM_SIGMA: float = 1.5
SSIM_C1: float = 0.01 ** 2
SSIM_C2: float = 0.03 ** 2

# воспроизводимость
SEED: int = 2026

# прочее
PAGE_MM = tuple(N * (LAM * z / (N * DX)) * 1e3 for z in ZS)   # физический размер страницы, мм


def out_pixel(z: float) -> float:
    """Шаг выходной сетки Δx' = λz/(NΔx), м."""
    return LAM * z / (N * DX)


OUT_DX = tuple(out_pixel(z) for z in ZS)
SCALE_S = tuple(dx / DX for dx in OUT_DX)          # s_i = Δx'_i/Δx


def chirp_fringe_fraction(z: float) -> float:
    """4.7: набег фазы чирпа на пиксель у края выходной сетки, в долях π."""
    # приращение фазы чирпа на пиксель у края: dφ/dx·Δx/π = 2πx_maxΔx/(λz)/π,
    # x_max = (N/2)Δx  ⇒  NΔx²/(λz) = z_crit/z  (для z₁ даёт 0,99835 — таблица 4.7)
    return (N * DX ** 2) / (LAM * z)


def z_margin(z: float) -> float:
    """Запас до границы применимости одно-БПФ-схемы, доли единицы."""
    return z / Z_CRIT - 1.0


def zgrid(z0: float = ZSTACK_Z0, z1: float = ZSTACK_Z1, step: float = ZSTACK_STEP) -> np.ndarray:
    """Сетка z для z-стопки дашборда (23,0 … 28,6 мм с шагом 0,02 мм)."""
    n = int(round((z1 - z0) / step)) + 1
    return z0 + step * np.arange(n, dtype=np.float64)


@dataclass(frozen=True)
class Paths:
    """Раскладка каталогов проекта."""
    root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @property
    def targets(self) -> str:
        return os.path.join(self.root, "data", "targets")

    @property
    def out(self) -> str:
        return os.path.join(self.root, "outputs")

    @property
    def ckpt(self) -> str:
        return os.path.join(self.root, "outputs", "checkpoints")

    @property
    def figures(self) -> str:
        return os.path.join(self.root, "figures")

    @property
    def dashboard(self) -> str:
        return os.path.join(self.root, "dashboard")

    def ensure(self) -> "Paths":
        for p in (self.targets, self.out, self.ckpt, self.figures,
                  os.path.join(self.dashboard, "data")):
            os.makedirs(p, exist_ok=True)
        return self


PATHS = Paths()

# человекочитаемые подписи
PLANE_LABELS = [f"z{i + 1} = {z:.1f} мм" for i, z in enumerate(ZS_MM)]


def control_table() -> list[dict]:
    """Таблица п. 4.7 для отчёта / дашборда (панель «Физика»)."""
    rows = []
    for i, z in enumerate(ZS):
        rows.append(dict(
            plane=i + 1,
            z_mm=ZS_MM[i],
            dx_out_um=OUT_DX[i] * 1e6,
            page_mm=PAGE_MM[i],
            s_i=SCALE_S[i],
            chirp_frac_pi=chirp_fringe_fraction(z),
            margin_pct=z_margin(z) * 100.0,
            margin_min_pct=z_margin(min(ZS)) * 100.0,
            fresnel_number=(DX * N) ** 2 / (LAM * z),
        ))
    return rows


def summary() -> dict:
    """Сводка констант для manifest.json / шапки дашборда."""
    return dict(
        lam=LAM, dx=DX, N=N, zs=list(ZS), zs_mm=list(ZS_MM),
        z_crit=Z_CRIT, z_crit_mm=Z_CRIT * 1e3,
        k=K, qstep=QSTEP, qstep_deg=float(np.degrees(QSTEP)),
        qstep_sigma=QSTEP_SIGMA, seed=SEED,
        out_dx=list(OUT_DX), page_mm=list(PAGE_MM), s=list(SCALE_S),
        max_diffraction_angle=float(LAM / (2 * DX)),
        fresnel_number_z1=float((DX * N) ** 2 / (LAM * ZS[0])),
        zstack=dict(z0=ZSTACK_Z0, z1=ZSTACK_Z1, step=ZSTACK_STEP),
    )
