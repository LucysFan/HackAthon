"""Нейросетевой синтез фазовой маски (ТЗ, п. 5.3).

Архитектура: U-Net c skip-связями (``segmentation_models_pytorch``), энкодер
resnet18 без предобученных весов (п. 5.3.1). Вход — упорядоченная четвёрка
целевых изображений [B,4,128,128] в [0,1] (канал i ↔ z_i), голова — 2 канала
(c, s), фаза φ = atan2(s, c) ∈ (−π, π] (п. 5.3.3).

Сквозной пайплайн ``FullPipeline`` (п. 5.3.6) не содержит разрывов
дифференцирования: все операции (включая квантование-шум и пропагацию)
дифференцируемы.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from . import physics as P


# --------------------------------------------------------------------- U-Net
def _norm_layer(kind: str = "group"):
    if kind == "instance":
        return lambda ch: nn.InstanceNorm2d(ch, affine=True)
    if kind == "batch":
        return lambda ch: nn.BatchNorm2d(ch)
    return lambda ch: nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        N = _norm_layer(norm)
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), N(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), N(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetUNet(nn.Module):
    """U-Net с энкодером resnet18 и батч-независимой нормализацией.

    Энкодер — torchvision resnet18 (слои conv1…layer4, без ImageNet-весов),
    декодер — симметричный с skip-связями; голова — 2 канала (c, s).
    """

    def __init__(self, in_ch: int = C.N_PLANES, out_ch: int = 2, width: int = 32,
                 depth: int = 4, norm: str = "group", drop: float = 0.0,
                 stem_stride: int = 2):
        super().__init__()
        from torchvision.models.resnet import resnet18
        W = width
        chs = [W, W * 2, W * 4, W * 8, W * 8][:depth]
        self.stem_stride = int(stem_stride)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 7, stride=self.stem_stride, padding=3, bias=False),
            _norm_layer(norm)(chs[0]), nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        cin = chs[0]
        for c in chs:
            self.blocks.append(_DoubleConv(cin, c, norm))
            self.pools.append(nn.MaxPool2d(2))
            cin = c
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.decs.append(_DoubleConv(chs[i - 1] * 2, chs[i - 1], norm))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        # Голова инициализируется малым случайным значением, а не нулём:
        # при c = s = 0 формула φ = atan2(s/r, c/r), r = √(c²+s²)+1e-8 даёт
        # расходимость градиента (1/r). Случайный старт |(c,s)| ~ 0,02 — устойчив.
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.normal_(self.head.bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = []
        y = self.stem(x)
        for blk, pool in zip(self.blocks, self.pools):
            y = blk(y)
            feats.append(y)
            y = pool(y)
        y = self.drop(feats[-1])
        for j, (up, dec) in enumerate(zip(self.ups, self.decs)):
            skip = feats[-2 - j]
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = dec(torch.cat([y, skip], dim=1))
        if y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return self.head(y)


# ------------------------------------------------------------- сеть → фаза
class PhaseNet(nn.Module):
    """U-Net + голова (c, s) + шум квантования (п. 5.3.3–5.3.4)."""

    def __init__(self, backbone: str = "unet", width: int = 32, depth: int = 4,
                 norm: str = "group", noise: bool = True, drop: float = 0.0,
                 stem_stride: int = 2):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "unet":
            self.net = ResNetUNet(out_ch=2, width=width, depth=depth, norm=norm, drop=drop,
                                  stem_stride=stem_stride)
        elif backbone == "smp":
            import segmentation_models_pytorch as smp
            self.net = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                                in_channels=C.N_PLANES, classes=2)
        else:
            raise ValueError(backbone)
        self.noise = noise
        self.last_phi_raw: torch.Tensor | None = None

    def forward(self, targets: torch.Tensor, noise: bool | None = None) -> torch.Tensor:
        """targets: [B,4,N,N] ∈ [0,1] → φ: [B,1,N,N] ∈ (−π, π]."""
        if targets.dim() == 3:
            targets = targets[None]
        cs = self.net(targets)
        c, s = cs[:, :1], cs[:, 1:2]
        r = torch.sqrt(c * c + s * s) + 1e-8
        phi = torch.atan2(s / r, c / r)
        self.last_phi_raw = phi
        use_noise = self.noise if noise is None else noise
        if use_noise and self.training:
            xi = (torch.rand_like(phi) - 0.5) * C.QSTEP
            phi = phi + xi
            # шум не должен выводить фазу за границы диапазона
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
        return phi


class UnrolledRefine(nn.Module):
    """K дифференцируемых шагов физической доводки (развёрнутый GS, п. 5.6.2).

    φ_{k+1} = φ_k + g·wrap( arg(D_z⁻¹(√t_i·D_z(e^{iφ_k})/|D_z|)) − φ_k )

    Шаг — классическое обновление Герчберга–Саксона по каждой плоскости,
    записанное в инкрементальной форме (снимается скачок 2π). В отличие от
    покадрового GS, все шаги дифференцируемы по φ, поэтому градиент
    физического лосса проходит сквозь них и «затягивает» выход сети в
    физически разрешимое решение. g — обучаемый коэффициент доверия.
    """

    def __init__(self, steps: int = 0, zs=C.ZS, gain: float = 1.0, learnable: bool = True):
        super().__init__()
        self.steps = int(steps)
        self.zs = tuple(zs)
        g0 = min(max(gain, 1e-3), 1 - 1e-3)
        raw = math.log(g0 / (1 - g0))
        self.gain = nn.Parameter(torch.tensor(raw)) if learnable else None
        if not learnable:
            self._gain_val = gain

    def g(self) -> torch.Tensor:
        if self.gain is not None:
            return torch.sigmoid(self.gain)
        return torch.tensor(self._gain_val)

    def forward(self, phi: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.steps <= 0:
            return phi
        t = targets
        if t.dim() == 3:
            t = t[None]
        mx = t.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-9
        amp_all = torch.sqrt(torch.clamp(t / mx, min=0.0))
        g = self.g()
        for _ in range(self.steps):
            for i, z in enumerate(self.zs):
                u = P.propagate_complex(phi, z)
                u_t = amp_all[:, i:i + 1] * torch.exp(1j * torch.angle(u))
                back = P.propagate_adj(u_t, z)
                phi_new = torch.angle(back)
                d = torch.remainder(phi_new - phi + math.pi, 2 * math.pi) - math.pi
                phi = phi + g * d
        return phi


class _Wrap(nn.Module):
    """Обёртка, гарантирующая форму [B,1,N,N] для произвольного входа пайплайна."""

    def __init__(self, net: PhaseNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        if x.dim() == 3:
            x = x[None]
        return self.net(x)


# ------------------------------------------------------------ сквозной пайплайн
class FullPipeline(nn.Module):
    """targets [B,4,N,N] → (I [B,4,N,N], φ [B,1,N,N]) — п. 5.3.6.

    Дифференцирование сквозное: цели → сеть → φ (+шум при обучении) →
    одно-БПФ-Френель на четырёх z_i → интенсивности.
    """

    def __init__(self, net: PhaseNet, zs=C.ZS, unroll: int = 0, gain: float = 1.0):
        super().__init__()
        self.wrap = _Wrap(net)
        self.zs = tuple(zs)
        self.unroll = UnrolledRefine(steps=unroll, zs=zs, gain=gain) if unroll > 0 else None

    @property
    def net_(self) -> PhaseNet:
        return self.wrap.net

    def forward(self, targets: torch.Tensor, noise: bool | None = None):
        phi = self.wrap(targets)
        if self.unroll is not None:
            phi = self.unroll(phi, targets)
        if noise is not None:
            # явное управление шумом (eval-инференс: noise=False)
            self.net_.noise = noise
        inten = []
        for z in self.zs:
            inten.append(P.propagate(phi, z))
        I = torch.cat(inten, dim=1)
        self.net_.noise = True
        return I, phi


def phase_to_pair(phi: torch.Tensor):
    """Фаза → (cos, sin) — для сохранения/визуализации."""
    return torch.cos(phi), torch.sin(phi)


def build_net(cfg: dict | None = None) -> PhaseNet:
    cfg = cfg or {}
    return PhaseNet(backbone=cfg.get("backbone", "unet"),
                    width=int(cfg.get("width", 32)),
                    depth=int(cfg.get("depth", 4)),
                    norm=cfg.get("norm", "group"),
                    noise=True,
                    drop=float(cfg.get("drop", 0.0)),
                    stem_stride=int(cfg.get("stem_stride", 2)))
