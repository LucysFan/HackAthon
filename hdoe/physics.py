"""Пропагатор: дифракция Френеля, одно-БПФ-схема (ТЗ, п. 4.3, приложение А).

Порядок операций строго по п. 4.3.3:
    1) входная сетка x_m = (m - N/2)Δx
    2) Q_in = exp(iπ(x²+y²)/(λz))
    3) G = U₀·Q_in
    4) F = fftshift(fft2(ifftshift(G)))
    5) Δx' = λz/(NΔx)
    6) U_z = e^{ikz}/(iλz)·Δx²·exp(iπ(x'²+y'²)/(λz))·F
    7) I_z = |U_z|²

Каскадная пропагация запрещена (п. 4.3.5): каждое z считается от маски.
"""
from __future__ import annotations

import warnings

import cmath
import math

import torch

from . import config as C


# --------------------------------------------------------------------------- сетки
def _index(n: int = C.N, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=dtype) - n // 2


def _grid(z: float, device=None, dtype=torch.float64):
    """Входная и выходная координатные сетки для расстояния z (м)."""
    n = _index(device=device, dtype=dtype)
    x = n * C.DX
    X, Y = torch.meshgrid(x, x, indexing="ij")
    dxo = C.LAM * z / (C.N * C.DX)
    xo = n * dxo
    Xo, Yo = torch.meshgrid(xo, xo, indexing="ij")
    return (X, Y), (Xo, Yo, dxo)


class _ChirpCache:
    """Кэш квадратичных множителей: они не зависят от батча и от фазы."""

    def __init__(self) -> None:
        self._d: dict = {}

    def get(self, z: float, device, dtype):
        """``dtype`` — желаемый комплексный тип результата (complex64/128).

        Сетки считаются в действительном типе той же разрядности: на GPU
        комплексные чирпы в float32 занимают в 2 раза меньше памяти, а
        приведение выполняется один раз при создании кэша, а не на каждом
        вызове пропагатора.
        """
        cdtype = dtype if dtype in (torch.complex64, torch.complex128) else _as_complex_dtype(
            torch.empty(0, dtype=dtype))
        rtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        key = (round(z, 12), str(device), str(cdtype))
        if key not in self._d:
            (X, Y), (Xo, Yo, dxo) = _grid(z, device=device, dtype=rtype)
            q_in = torch.exp(1j * torch.pi / (C.LAM * z) * (X ** 2 + Y ** 2)).to(cdtype)
            q_out = torch.exp(1j * torch.pi / (C.LAM * z) * (Xo ** 2 + Yo ** 2)).to(cdtype)
            pref = cmath.exp(1j * C.K * z) / (1j * C.LAM * z) * (C.DX ** 2)
            self._d[key] = (q_in, q_out, pref)
        return self._d[key]

    def clear(self) -> None:
        self._d.clear()


_CHIRP = _ChirpCache()


def _as_complex_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.complex64 if t.dtype in (torch.float16, torch.float32) else torch.complex128


def _check_z(z: float, strict: bool = True) -> None:
    """П. 4.5.2 и 8.6: граница применимости и предупреждение при запасе < 1 %."""
    if z < C.Z_CRIT:
        msg = (f"z={z * 1e3:.4f} мм ниже границы применимости одно-БПФ-схемы "
               f"z_crit={C.Z_CRIT * 1e3:.4f} мм")
        if strict:
            raise AssertionError(msg)
        warnings.warn(msg)
    elif z < 1.01 * C.Z_CRIT:
        warnings.warn(
            f"ВНИМАНИЕ: запас по границе применимости для z={z * 1e3:.2f} мм "
            f"составляет {(z / C.Z_CRIT - 1) * 100:.2f} % (< 1 %)", stacklevel=3)


# ------------------------------------------------------------------- основной ход
def propagate_complex(phi: torch.Tensor, z: float, n: int | None = None) -> torch.Tensor:
    """Комплексное поле на расстоянии z. phi: [B,1,N,N] в радианах (float).

    ``n`` — необязательный размер сетки (для тестов gradcheck на 8×8;
    по умолчанию N = 128 из конфигурации).
    """
    _check_z(z)
    if phi.dim() == 2:
        phi = phi[None, None]
    elif phi.dim() == 3:
        phi = phi[None]
    cdtype = _as_complex_dtype(phi)
    if n is not None and n != C.N:
        i = _index(n, device=phi.device, dtype=torch.float64)
        X, Y = torch.meshgrid(i * C.DX, i * C.DX, indexing="ij")
        dxo = C.LAM * z / (n * C.DX)
        Xo, Yo = torch.meshgrid(i * dxo, i * dxo, indexing="ij")
        q_in = torch.exp(1j * torch.pi / (C.LAM * z) * (X ** 2 + Y ** 2))
        q_out = torch.exp(1j * torch.pi / (C.LAM * z) * (Xo ** 2 + Yo ** 2))
        pref = cmath.exp(1j * C.K * z) / (1j * C.LAM * z) * (C.DX ** 2)
        q_in, q_out = q_in.to(cdtype), q_out.to(cdtype)
        u0 = torch.exp(1j * phi.to(cdtype))
        g = u0 * q_in
        f = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g, dim=(-2, -1))), dim=(-2, -1))
        return pref * q_out * f
    q_in, q_out, pref = _CHIRP.get(z, phi.device, torch.float64)
    q_in = q_in.to(cdtype)
    q_out = q_out.to(cdtype)
    u0 = torch.exp(1j * phi.to(cdtype))
    g = u0 * q_in
    f = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g, dim=(-2, -1))), dim=(-2, -1))
    return pref * q_out * f


# ---------------------------------------------------------------- быстрый путь
def _fast_chirps(z: float, device, dtype):
    """Кэш чирпов сразу в рабочем комплексном dtype: на GPU это в 2 раза меньше
    памяти и одна операция приведения вместо двух на каждый вызов."""
    return _CHIRP.get(z, device, dtype)


def propagate_fast(phi: torch.Tensor, z: float) -> torch.Tensor:
    """Одно-БПФ-Френель без лишних приведений типов (GPU-путь).

    Отличие от ``propagate`` — только в отсутствии повторных ``.to(complex)``:
    чирпы кэшируются в том же комплексном типе, что и поле. Численно
    идентично ``propagate`` (проверяется тестом).
    """
    _check_z(z)
    if phi.dim() == 2:
        phi = phi[None, None]
    elif phi.dim() == 3:
        phi = phi[None]
    cdtype = _as_complex_dtype(phi)
    q_in, q_out, pref = _fast_chirps(z, phi.device, cdtype)
    g = torch.exp(1j * phi.to(cdtype)) * q_in
    f = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g, dim=(-2, -1))),
                           dim=(-2, -1))
    return (pref * q_out * f).abs() ** 2


def forward_model(phi: torch.Tensor, zs=None, fast: bool = True) -> torch.Tensor:
    """[B,1,N,N] фаза → [B,P,N,N] интенсивности на всех z (P = len(zs))."""
    zs = tuple(C.ZS if zs is None else zs)
    fn = propagate_fast if fast else propagate
    return torch.cat([fn(phi, z) for z in zs], dim=1)


def forward_and_grad(phi: torch.Tensor, zs=None, targets: torch.Tensor | None = None,
                     loss_fn=None) -> tuple:
    """Физический прямой ход + градиент по фазе.

    Если задан ``loss_fn`` (функция (I_planes, targets) → скаляр), градиент
    вычисляется через autograd по физической модели — это и есть
    «физика внутри сети» (differentiable model-based layer).
    """
    I = forward_model(phi, zs)
    if loss_fn is None:
        return I, None
    loss = loss_fn(I, targets)
    g, = torch.autograd.grad(loss, phi, create_graph=False)
    return I, g


def propagate(phi: torch.Tensor, z: float, n: int | None = None) -> torch.Tensor:
    """Интенсивность на расстоянии z: |U_z|². Форма входа и выхода совпадает."""
    return propagate_complex(phi, z, n=n).abs() ** 2


def propagate_adj(u: torch.Tensor, z: float) -> torch.Tensor:
    """Сопряжённый оператор: U_z → U₀ (сопряжённые чирпы, обратное БПФ).

    A^H(U) = N²·conj(pref)·F⁻¹(conj(Q_out)·U)·conj(Q_in). Множитель N² следует из
    сопряжённости неортонормированного БПФ (FFTᴴ = N²·FFT⁻¹); он обеспечивает
    тождество ⟨A G, U⟩ = ⟨G, Aᴴ U⟩ (проверяется тестом).
    """
    q_in, q_out, pref = _CHIRP.get(z, u.device, torch.float64)
    cdtype = torch.complex64 if u.dtype in (torch.complex64, torch.float32) else torch.complex128
    a = torch.conj(q_out.to(cdtype)) * u
    out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(a, dim=(-2, -1))),
                             dim=(-2, -1))
    scale = (C.N ** 2) * torch.conj(torch.tensor(pref, dtype=cdtype))
    return out * scale * torch.conj(q_in.to(cdtype))


def propagate_np(phi, z: float, dtype=torch.float64) -> "np.ndarray":  # noqa: A002
    """Утилита для numpy-кода (build-скрипты, GS, оценка). Вход/выход — numpy."""
    import numpy as np
    t = torch.as_tensor(np.asarray(phi), dtype=dtype)
    return propagate(t, z).numpy()


# ------------------------------------------------------- альтернативные схемы (8.5)
def angular_spectrum(phi: torch.Tensor, z: float, upsample: int = 1,
                     band_limit: bool = True) -> torch.Tensor:
    """Метод углового спектра (только для кросс-верификации, п. 4.3.6 и 8.5).

    Реализован с ограничением полосы (band-limited AS, Matsushima & Shimobaba,
    Opt. Express 17(22) 2009): передаточная функция обнуляется вне
    f_lim = 1/(λ·√((2Δx_eff·z)² + 1)), где Δx_eff — шаг сетки (с учётом
    апсемплинга). Без ограничения полосы на расстояниях порядка z_crit
    численная H(f) недооцифрована и результат искажается алиасингом.
    Возвращает интенсивность на сетке входа (после даунсемплинга с усреднением).
    """
    if phi.dim() == 2:
        phi = phi[None, None]
    elif phi.dim() == 3:
        phi = phi[None]
    n = C.N * upsample
    dxs = C.DX / upsample
    u0 = torch.exp(1j * phi.to(torch.complex128))
    if upsample > 1:
        # апсемплинг полем: нулевое дополнение спектра (точная интерполяция
        # для полосно-ограниченного поля; билинейная интерполяция гасила бы
        # амплитуду на быстрых чирпах)
        m = upsample
        spec = torch.fft.fftshift(torch.fft.fft2(u0), dim=(-2, -1))     # центрировано
        big = torch.zeros(u0.shape[0], u0.shape[1], n, n, dtype=spec.dtype)
        o = (n - C.N) // 2
        big[..., o:o + C.N, o:o + C.N] = spec
        u0 = torch.fft.ifft2(torch.fft.ifftshift(big, dim=(-2, -1))) * (m * m)
    i = torch.arange(n, dtype=torch.float64) - n // 2
    fx = i / (n * dxs)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")
    arg = 1.0 - (C.LAM * FX) ** 2 - (C.LAM * FY) ** 2
    h = torch.exp(1j * C.K * z * torch.sqrt(torch.clamp(arg, min=0.0).to(torch.complex128)))
    if band_limit:
        f_lim = 1.0 / (C.LAM * math.sqrt((2.0 * dxs * z) ** 2 + 1.0))
        h = torch.where(torch.abs(FX) <= f_lim, h, torch.zeros_like(h))
        h = torch.where(torch.abs(FY) <= f_lim, h, torch.zeros_like(h))
    spec = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(u0, dim=(-2, -1))),
                              dim=(-2, -1)) * h
    u = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1))),
                           dim=(-2, -1))
    if upsample > 1:
        # усредняем ИНТЕНСИВНОСТЬ (усреднение комплексных амплитуд гасило бы
        # быстро осциллирующие подпиксельные компоненты)
        return torch.nn.functional.avg_pool2d(u.abs() ** 2, upsample)
    return u.abs() ** 2


def propagate_any(phi: torch.Tensor, z: float, upsample_as: int = 1):
    """Интенсивность с автоматическим выбором схемы по границе применимости.

    П. 4.5.3 ТЗ: при z < z_crit одно-БПФ-схема неприменима и заменяется на
    метод углового спектра (band-limited). Используется ТОЛЬКО для
    демонстрационных z-стопок (23,0–28,6 мм в п.12.4), где часть кадров лежит
    ниже z_crit; все итоговые реконструкции считаются одно-БПФ-схемой на z_i.

    Возвращает (I, method), где method ∈ {"fresnel1fft", "angular_spectrum"}.
    """
    if z >= C.Z_CRIT:
        return propagate(phi, z), "fresnel1fft"
    return angular_spectrum(phi, z, upsample=upsample_as), "angular_spectrum"


def fresnel_two_step(phi: torch.Tensor, z: float) -> torch.Tensor:
    """Двух-БПФ схема (только для кросс-верификации): полная свёртка через FFT."""
    if phi.dim() == 2:
        phi = phi[None, None]
    u0 = torch.exp(1j * phi.to(torch.complex128))
    c = C.DX * (C.N - 1) * 2
    m = 2 * C.N
    x = (torch.arange(m, dtype=torch.float64) - m // 2) * c / m
    X, Y = torch.meshgrid(x, x, indexing="ij")
    h = torch.exp(1j * torch.pi / (C.LAM * z) * (X ** 2 + Y ** 2))
    u = torch.zeros(1, 1, m, m, dtype=torch.complex128)
    u[0, 0, m // 2 - C.N // 2:m // 2 + C.N // 2, m // 2 - C.N // 2:m // 2 + C.N // 2] = u0
    hf = torch.fft.fft2(torch.fft.ifftshift(h))
    out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(u) * hf))
    sl = slice(m // 2 - C.N // 2, m // 2 + C.N // 2)
    return (out[0, 0, sl, sl].abs() ** 2)


# ------------------------------------------------------------------ квантование
def encode8bit(phi):
    """Фаза → код 0..255 (п. 4.6.1). Работает с numpy и torch."""
    import numpy as np
    if isinstance(phi, torch.Tensor):
        v = torch.clamp(torch.round(255.0 * (phi + torch.pi) / (2 * torch.pi)), 0, 255)
        return v.to(torch.uint8)
    v = np.clip(np.round(255.0 * (np.asarray(phi) + np.pi) / (2 * np.pi)), 0, 255)
    return v.astype(np.uint8)


def decode8bit(v):
    """Код → фаза (п. 4.6.2)."""
    import numpy as np
    if isinstance(v, torch.Tensor):
        return -torch.pi + v.float() * (2 * torch.pi / 255.0)
    return -np.pi + np.asarray(v, dtype=np.float32) * (2 * np.pi / 255.0)


def quantize(phi):
    """Полный цикл: float-фаза → (коды, квантованная фаза)."""
    v = encode8bit(phi)
    return v, decode8bit(v)


def phase_tv(phi: torch.Tensor) -> torch.Tensor:
    """В4: TV(φ) — сумма модулей разностей фазы по соседним пикселям."""
    dims = tuple(range(1, phi.dim()))
    dx = (phi[..., 1:, :] - phi[..., :-1, :]).abs().sum(dim=dims)
    dy = (phi[..., :, 1:] - phi[..., :, :-1]).abs().sum(dim=dims)
    return dx + dy


def normalize01(img):
    """Мин–макс нормировка к [0,1] (п. 5.5.2)."""
    import numpy as np
    if isinstance(img, torch.Tensor):
        lo = img.amin(dim=tuple(range(1, img.dim())), keepdim=True)
        hi = img.amax(dim=tuple(range(1, img.dim())), keepdim=True)
        return (img - lo) / (hi - lo + 1e-12)
    img = np.asarray(img, dtype=np.float64)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)


def to_u8(img01) -> "np.ndarray":
    """[0,1] → uint8."""
    import numpy as np
    a = np.asarray(img01, dtype=np.float64)
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
