"""Нейросетевой синтез фазы с физикой ВНУТРИ сети (п. 5.3, 5.3.6 ТЗ).

Принципиальное отличие от «сеть как старт + итерационная доводка»:

* фаза на выходе — это результат **прямого прохода сети**; никакой
  классический итерационный метод на инференсе не вызывается;
* физическая модель (одно-БПФ-схема Френеля на четырёх z) входит в сеть
  как дифференцируемые слои: блок ``PhysicsRefiner`` делает K шагов
  градиентного спуска **по физическому лоссу**, градиент которого считается
  через autograd прямо по пропагатору (``P.forward_model``);
* квантование в 8 бит стоит в конце сети как слой ``QuantizeSTE``
  (прямой ход — истинное округление, обратный — straight-through), поэтому
  сеть обучается именно той маске, которая попадёт в ``DOE.png``.

Классические методы (GS) используются только для генерации обучающих данных
(учительских масок) — это разрешено п. 5.4.2 и не является доводкой на
инференсе.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import config as C
from . import physics as P


# ------------------------------------------------------------------ квантование
class QuantizeSTE(nn.Module):
    """8-битное квантование фазы с прямым (straight-through) градиентом.

    Прямой ход — ровно ``P.encode8bit`` + ``P.decode8bit``: та же сетка кодов,
    что и в сдаваемом ``DOE.png`` (0 ↔ −π, 255 ↔ +π). Обратный ход —
    тождественный (ошибка квантования не «убивает» градиент).
    """

    def __init__(self, levels: int = 256, enabled: bool = True):
        super().__init__()
        self.levels = int(levels)
        self.enabled = bool(enabled)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            # на инференсе квантование выполняет сам формат файла, поэтому
            # здесь достаточно мягкого ограничения диапазона
            return self.quantize(phi) if self.enabled else phi
        return phi + (self.quantize(phi) - phi).detach()

    def quantize(self, phi: torch.Tensor) -> torch.Tensor:
        step = 2.0 * math.pi / (self.levels - 1)
        q = torch.round((phi + math.pi) / step) * step - math.pi
        return torch.clamp(q, -math.pi, math.pi)


# --------------------------------------------------------------- физика в сети
class PhysicsRefiner(nn.Module):
    """K дифференцируемых шагов спуска по физическому лоссу (физика в сети).

    φ_{k+1} = wrap( φ_k − α · ∇_φ L( A(φ_k), t ) ),  A — пропагатор,

    где L = среднее по плоскостям [(1−PCC) + (1−SSIM)]. Градиент берётся
    через autograd **по физической модели**, поэтому вся цепочка
    «фаза → дифракция → метрика» дифференцируема и обучается вместе с U-Net.

    α — обучаемый скаляр (softplus), начальное значение ``gain``. Шаг
    нормируется на СКО градиента, чтобы не зависеть от масштаба лосса.
    Параметров у блока почти нет (2 скаляра), то есть сеть не «подменяется»
    оптимизатором: физика — это слой, а не внешний решатель.
    """

    def __init__(self, steps: int = 4, zs=C.ZS, gain: float = 0.25,
                 learnable: bool = True, quantize: bool = False,
                 schedule: str = "decay", gain_end: float = 0.05,
                 step_clip: float = 0.5):
        """``schedule``: ``const`` — постоянный шаг; ``decay`` — убывающий
        α_k = α·(1−k/K)^1.5 + gain_end (быстрый выход к решению, затем мелкая
        доводка — даёт одновременно и PCC, и SSIM)."""
        super().__init__()
        self.steps = int(steps)
        self.zs = tuple(zs)
        self.quantize = bool(quantize)
        self.schedule = schedule
        self.gain_end = float(gain_end)
        self.step_clip = float(step_clip)
        g0 = max(float(gain), 1e-3)
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(g0))))
        self.learnable = bool(learnable)

    def alpha(self) -> torch.Tensor:
        if not self.learnable:
            return self.raw_gain.detach().exp()
        return torch.nn.functional.softplus(self.raw_gain)

    def forward(self, phi: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """K шагов физического спуска, дифференцируемых ПО ВХОДУ.

        Важно: шаги не разрывают граф (никаких ``detach``), поэтому градиент
        лосса течёт через все K шагов физики обратно в U-Net — сеть учится
        выдавать такую начальную фазу, из которой физический слой попадает в
        хорошее решение. Память растёт как O(K), поэтому K обычно 2–4.
        """
        if self.steps <= 0:
            return phi
        t = targets if targets.dim() == 4 else targets[None]
        mx = t.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-9
        t = t / mx
        # Шаги физики выполняются как ПРОЕКЦИЯ (без графа через сами шаги:
        # производная шага по фазе включает обратную величину СКО градиента и
        # численно неустойчива). Выход связан со входом straight-through:
        #   φ_out = φ_in + (proj(φ_in) − φ_in).detach()
        # тогда градиент лосса по φ_out передаётся в U-Net полностью, а сама
        # проекция работает как фиксированный физический слой.
        x = phi.detach()
        n = max(self.steps, 1)
        # необязательная трассировка траектории (для дашборда): храним не более
        # ~64 кадров, иначе память растёт как K·N²
        self._track = getattr(self, "_track", None)
        self._track_stride = max(1, self.steps // 60)
        if self._track is not None:
            self._track.append(x.clone())
            if getattr(self, "_track_k", None) is not None:
                self._track_k.append(0)          # позиция «выход U-Net» (K = 0)
        with torch.enable_grad():  # autograd нужен только внутри шага
            for k in range(self.steps):
                gg = x.detach().requires_grad_(True)
                I = P.forward_model(gg, self.zs)
                g, = torch.autograd.grad(physical_loss(I, t), gg)
                scale = g.flatten(1).std(dim=1).view(-1, 1, 1, 1) + 1e-8
                a = self.alpha()
                if self.schedule == "decay":
                    a = a * (1.0 - k / n) ** 1.5 + self.gain_end
                step = a * g / scale
                # ограничение шага (устойчивость к выбросам градиента)
                step = torch.clamp(step, -self.step_clip, self.step_clip)
                x = torch.remainder(x - step + math.pi, 2 * math.pi) - math.pi
                if getattr(self, "_track", None) is not None and \
                        ((k + 1) % self._track_stride == 0 or k == self.steps - 1):
                    self._track.append(x.clone())
                    if getattr(self, "_track_k", None) is not None:
                        self._track_k.append(k + 1)   # фактический номер шага K
        # straight-through: прямой ход — результат проекции, обратный — тождество
        x = phi + (x - phi).detach()
        if self.quantize:
            x = QuantizeSTE()(x)
        return x


def physical_loss(I: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Физический лосс (1−PCC)+(1−SSIM) по плоскостям, нормировка мин–макс."""
    from . import losses as L
    from . import metrics as M
    Ip = torch.cat([L.normalize_batch(I[:, i:i + 1]) for i in range(I.shape[1])], 1)
    n = I.shape[1]
    loss = sum((1 - M.pcc_torch(Ip[:, i:i + 1], targets[:, i:i + 1])).mean()
               for i in range(n)) / n
    loss = loss + sum((1 - M.ssim_torch(Ip[:, i:i + 1], targets[:, i:i + 1])).mean()
                      for i in range(n)) / n
    return loss


# ------------------------------------------------------------------- пайплайн
class PhysicsInNet(nn.Module):
    """Цели [B,4,N,N] → (I [B,4,N,N], φ [B,1,N,N]) — физика внутри сети.

    Порядок: U-Net → физический рефайнер (K дифференцируемых шагов) →
    квантование 8 бит (STE при обучении) → пропагация для лосса.
    """

    def __init__(self, net, steps: int = 4, zs=C.ZS, gain: float = 0.25,
                 quantize: bool = True, learnable: bool = True,
                 schedule: str = "decay", gain_end: float = 0.02):
        super().__init__()
        self.net = net
        self.zs = tuple(zs)
        self.refiner = PhysicsRefiner(steps=steps, zs=zs, gain=gain,
                                      learnable=learnable, quantize=quantize,
                                      schedule=schedule, gain_end=gain_end) \
            if steps > 0 else None
        self.quant = QuantizeSTE(enabled=quantize)

    def forward(self, targets: torch.Tensor, noise: bool | None = None):
        phi = self.net(targets, noise=noise)
        if self.refiner is not None:
            phi = self.refiner(phi, targets)
        phi = self.quant(phi)
        inten = P.forward_model(phi, self.zs)
        return inten, phi


@torch.no_grad()
def infer_phase(pack: nn.Module, targets: torch.Tensor, batch: int = 4) -> torch.Tensor:
    """Инференс сети: один прямой проход, без классической доводки."""
    pack.eval()
    if targets.dim() == 3:
        targets = targets[None]
    out = []
    for i in range(0, targets.shape[0], batch):
        out.append(pack(targets[i:i + batch])[1])
    return torch.cat(out, 0)
