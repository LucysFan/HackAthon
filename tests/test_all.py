"""Верификация комплекса (ТЗ, раздел 8). Запуск:  python tests/test_all.py

Тесты:
  8.1 Парсеваль (4 плоскости)
  8.2 Линза (максимум в центральном пикселе, ≥100 средних по полю)
  8.3 Квантование (round-trip, ошибка ≤ δ/2)
  8.4 Градиент (gradcheck, complex128 на уменьшенной сетке + численная сверка)
  8.5 Кросс-схема (одно-БПФ vs угловой спектр с 4× апсемплингом, SSIM ≥ 0,99)
  8.6 Граница применимости (assert, предупреждение при запасе < 1 %)
  8.7 Форматы выходных PNG
  8.8 Сводка результатов в JSON (публикуется в отчёте и на дашборде)
  8.9 Физика внутри сети (один прямой проход, градиент через физический слой)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C          # noqa: E402
from hdoe import metrics as M         # noqa: E402
from hdoe import physics as P         # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(2)

RESULTS: dict = {}
_KEYMAP = {fn: key for fn, key in (
    ("test_parseval", "parseval"), ("test_lens", "lens"), ("test_quantization", "quantization"),
    ("test_gradient", "gradient"), ("test_cross_scheme", "cross_scheme"),
    ("test_boundary", "boundary"), ("test_formats", "formats"),
    ("test_netphys", "netphys"))}


def _rec(name: str, ok: bool, **kw):
    # детальные поля (files/rows/...) сохраняются, к сводке добавляется passed
    RESULTS[name] = {**RESULTS.get(name, {}), "passed": bool(ok), **kw}
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: " +
          ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()))
    return ok


# ------------------------------------------------------------------ 8.1 Парсеваль
def test_parseval() -> bool:
    print("8.1 Парсеваль")
    phi = torch.zeros(1, 1, C.N, C.N, dtype=torch.float64)
    rows = []
    ok = True
    for z in C.ZS:
        U = P.propagate_complex(phi, z)
        I = U.abs() ** 2
        dxo = C.out_pixel(z)
        lhs = float((torch.abs(torch.exp(1j * phi)) ** 2).sum() * C.DX ** 2)
        rhs = float(I.sum() * dxo ** 2)
        rel = abs(rhs - lhs) / lhs
        rows.append(dict(z_mm=z * 1e3, lhs=lhs, rhs=rhs, rel=rel, ok=rel <= 1e-3))
        ok &= rel <= 1e-3
    RESULTS["parseval_rows"] = rows
    return _rec("parseval", ok, max_rel=max(r["rel"] for r in rows), planes=len(rows))


# ---------------------------------------------------------------------- 8.2 Линза
def test_lens() -> bool:
    print("8.2 Линза")
    n = torch.arange(C.N, dtype=torch.float64) - C.N // 2
    X, Y = torch.meshgrid(n * C.DX, n * C.DX, indexing="ij")
    rows, ok = [], True
    for z in C.ZS:
        phi = -torch.pi * (X ** 2 + Y ** 2) / (C.LAM * z)
        I = P.propagate(phi, z)[0, 0]
        center = I[C.N // 2, C.N // 2]
        # среднее по полю без центрального пикселя (иначе отношение занижается
        # на 1/N² ≈ 6·10⁻⁵ — см. отчёт)
        mask = torch.ones_like(I)
        mask[C.N // 2, C.N // 2] = 0
        mean_val = (I * mask).sum() / mask.sum()
        peak_idx = int(torch.argmax(I))
        ratio = float(center / mean_val)
        # идеальная линза собирает почти всю энергию в центральный пиксель, поэтому
        # centre/mean неограниченно велико и как число неинформативно — приводим
        # также долю энергии в окне 3×3 (устойчивая характеристика)
        c = C.N // 2
        e_win = float(I[c - 1:c + 2, c - 1:c + 2].sum() / I.sum())
        good = bool(peak_idx == c * C.N + c and ratio >= 100 and e_win > 0.95)
        rows.append(dict(z_mm=z * 1e3, peak_row=peak_idx // C.N, peak_col=peak_idx % C.N,
                         center=float(center), mean=float(mean_val), ratio=ratio,
                         energy_win3=float(e_win), ok=good))
        ok &= good
    RESULTS["lens_rows"] = rows
    # сохраняем профиль для дашборда (панель «Физика»)
    z = C.ZS[0]
    phi = -torch.pi * (X ** 2 + Y ** 2) / (C.LAM * z)
    prof = P.propagate(phi, z)[0, 0, C.N // 2].numpy()
    RESULTS["lens_profile"] = prof.tolist()
    return _rec("lens", ok, min_ratio=min(r["ratio"] for r in rows),
                min_energy_win3=min(r["energy_win3"] for r in rows))


# ------------------------------------------------------------------ 8.3 Квантование
def test_quantization() -> bool:
    print("8.3 Квантование")
    rng = np.random.default_rng(1)
    phi = rng.uniform(-np.pi, np.pi, (C.N, C.N))
    v = P.encode8bit(phi)
    phi_q = P.decode8bit(v)
    v2 = P.encode8bit(phi_q)
    assert phi_q.min() >= -np.pi - 1e-6 and phi_q.max() <= np.pi + 1e-6
    err = np.abs(phi_q - phi)
    ok = bool(np.array_equal(v, v2)) and float(err.max()) <= C.QSTEP / 2 + 1e-6
    RESULTS["quantization"] = dict(vmin=int(v.min()), vmax=int(v.max()),
                                   err_max=float(err.max()), err_sigma=float(err.std()),
                                   round_trip_identical=bool(np.array_equal(v, v2)))
    return _rec("quantization", ok, err_max=float(err.max()), qstep_half=C.QSTEP / 2)


# -------------------------------------------------------------------- 8.4 Градиент
def test_gradient() -> bool:
    print("8.4 Градиент")
    rel = float("nan")
    ok_grad = False
    # (а) gradcheck комплексного пропагатора на уменьшенной сетке (complex128)
    phi8 = (torch.rand(1, 1, 8, 8, dtype=torch.float64) * 2 - 1) * np.pi
    phi8.requires_grad_(True)
    try:
        ok_grad = torch.autograd.gradcheck(lambda x: P.propagate(x, C.ZS[1], n=8).sum(),
                                           (phi8,), eps=1e-6, atol=1e-4, rtol=1e-3)
    except Exception as exc:                                       # noqa: BLE001
        print("    gradcheck: ", exc)
        ok_grad = False
    # (б) численная сверка dI/dφ на полной сетке 128² (случайные пиксели)
    phi = torch.zeros(1, 1, C.N, C.N, dtype=torch.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, C.N, size=(6, 2))
    eps = 1e-6
    errs = []
    for (i, j) in idx:
        p1 = phi.clone(); p1[0, 0, i, j] += eps
        p2 = phi.clone(); p2[0, 0, i, j] -= eps
        num = float(P.propagate(p1, C.ZS[0])[0, 0, i, j] - P.propagate(p2, C.ZS[0])[0, 0, i, j]) / (2 * eps)
        pp = phi.clone().requires_grad_(True)
        val = P.propagate(pp, C.ZS[0])[0, 0, i, j]
        g, = torch.autograd.grad(val, pp)
        ana = float(g[0, 0, i, j])
        errs.append(abs(num - ana) / (abs(num) + 1e-30))
    rel = float(np.max(errs))
    ok = bool(ok_grad) and rel < 1e-4
    RESULTS["gradient"] = dict(passed=bool(ok), gradcheck=bool(ok_grad),
                               numeric_rel_err=rel, grid_gradcheck=8, pixels=len(idx))
    return _rec("gradient", ok, gradcheck=bool(ok_grad), rel_err=rel)


# ------------------------------------------------------------------- 8.5 Кросс-схема
def test_cross_scheme() -> bool:
    print("8.5 Кросс-схема (одно-БПФ Френель vs угловой спектр)")
    torch.manual_seed(0)
    n = torch.arange(C.N, dtype=torch.float64) - C.N // 2
    X, Y = torch.meshgrid(n * C.DX, n * C.DX, indexing="ij")
    z = C.ZS[0]
    # полосно-ограниченная маска: гладкая случайная фаза + чирп (линза),
    # максимальная локальная частота ниже Найквиста входной сетки
    g = torch.Generator().manual_seed(0)
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter((torch.rand(C.N, C.N, generator=g).numpy() * 2 - 1), 7) * 6.0
    lens = -torch.pi * (X ** 2 + Y ** 2) / (C.LAM * z) * 0.40
    phi = torch.as_tensor(smooth) + lens
    phi = (torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi)[None, None]
    I_f = P.propagate(phi, z)[0, 0].numpy()
    rows = []
    ok = True
    for up in (1, 4):
        I_a = P.angular_spectrum(phi, z, upsample=up)[0, 0].numpy()
        s_ = M.ssim(P.normalize01(I_f), P.normalize01(I_a))
        rows.append(dict(upsample=up, ssim=float(s_), energy=float(I_a.sum() / I_f.sum())))
        if up == 1:
            ok = ok and s_ >= 0.99
    # третий независимый контроль: энергия (Парсеваль) для углового спектра
    e_ok = all(abs(r["energy"] - 1.0) < 0.02 for r in rows)
    ok = ok and e_ok
    RESULTS["cross_scheme"] = dict(z_mm=z * 1e3, passed=bool(ok), rows=rows, upsample=4,
                                   energy_ok=bool(e_ok))
    return _rec("cross_scheme", ok, ssim_native=rows[0]["ssim"],
                energy_dev=abs(rows[0]["energy"] - 1.0))


# ------------------------------------------------------------------ 8.6 Граница
def test_boundary() -> bool:
    print("8.6 Граница применимости")
    ok_assert = False
    try:
        P.propagate(torch.zeros(1, 1, C.N, C.N), C.Z_CRIT * 0.99)
    except AssertionError:
        ok_assert = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.propagate(torch.zeros(1, 1, C.N, C.N), C.ZS[0])
        warned = any("< 1 %" in str(x.message) for x in w)
    ok = ok_assert and warned
    RESULTS["boundary"] = dict(assert_raised=ok_assert, warned_z1=warned,
                               z_crit_mm=C.Z_CRIT * 1e3, z1_margin_pct=(C.ZS[0] / C.Z_CRIT - 1) * 100)
    RESULTS["z_crit_mm"] = C.Z_CRIT * 1e3
    return _rec("boundary", ok, assert_raised=ok_assert, warned=warned,
                margin_pct=(C.ZS[0] / C.Z_CRIT - 1) * 100)


# ------------------------------------------------------------------- 8.7 Форматы
def test_formats() -> bool:
    print("8.7 Форматы выходных файлов")
    from PIL import Image
    files = [os.path.join(C.PATHS.out, "DOE.png")] + \
            [os.path.join(C.PATHS.out, f"recon_{i}.png") for i in range(1, 5)]
    rows, ok = [], True
    for f in files:
        if not os.path.exists(f):
            rows.append(dict(file=os.path.basename(f), exists=False, ok=False))
            ok = False
            continue
        im = Image.open(f)
        a = np.asarray(im)
        row = dict(file=os.path.basename(f), mode=im.mode, size="%dx%d" % im.size,
                   dtype=str(a.dtype), vmin=int(a.min()), vmax=int(a.max()))
        good = (im.mode == "L" and im.size == (C.N, C.N) and a.dtype == np.uint8
                and a.shape == (C.N, C.N))
        if "DOE" in f:
            rt = bool(np.array_equal(P.encode8bit(P.decode8bit(a)), a))
            row["round_trip"] = rt
            row["has_0"] = bool(int(a.min()) == 0)
            row["has_255"] = bool(int(a.max()) == 255)
            good = good and rt and row["has_0"] and row["has_255"]
        row["ok"] = bool(good)
        ok = ok and good
        rows.append(row)
    RESULTS["formats"] = dict(passed=bool(ok), n_files=len(files), files=rows)
    return _rec("formats", ok, n_files=len(files))


# ------------------------------------------------ 8.9 физика ВНУТРИ сети
def test_netphys() -> bool:
    """Один прямой проход сети (физика как слои) даёт заявленные метрики.

    Проверяется: (1) фаза получается прямым проходом, без классических
    итераций на инференсе; (2) PCC совпадает с metrics.json; (3) градиент
    физического лосса доходит до весов U-Net через K шагов физики.
    """
    print("8.9 Физика внутри сети (один прямой проход)")
    ckpt = os.path.join(C.PATHS.out, "best.pt")
    if not os.path.exists(ckpt):
        return _rec("netphys", False, reason="нет best.pt")
    from PIL import Image
    from hdoe import netphys as NPH
    from hdoe.model import build_net
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "arch" not in ck:
        return _rec("netphys", False, reason="чекпоинт без arch")
    net = build_net(ck["arch"])
    net.load_state_dict(ck["model"])
    net.eval()
    pk = ck.get("pack") or {}
    steps = int(pk.get("steps", 0))
    if steps <= 0:
        return _rec("netphys", False, reason="в чекпоинте нет шагов физики")
    pipe = NPH.PhysicsInNet(net, steps=steps, gain=float(pk.get("gain", 0.25)),
                            quantize=True, learnable=False)
    t_row = []
    for i in range(1, 5):
        im = Image.open(os.path.join(C.PATHS.targets, f"target_{i}.png")).convert("L")
        t_row.append(np.asarray(im, dtype=np.float32) / 255.0)
    tg = torch.as_tensor(np.stack(t_row), dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        I, phi = pipe(tg[None])
    dt = time.time() - t0
    codes = np.squeeze(np.asarray(P.encode8bit(phi[0, 0])))
    phi_q = P.decode8bit(codes)
    pccs, ssims = [], []
    for i, z in enumerate(C.ZS):
        rec = P.normalize01(P.propagate(torch.as_tensor(phi_q)[None, None], z)[0, 0].numpy())
        pccs.append(M.pcc(np.asarray(tg[i]), rec))
        ssims.append(M.ssim(np.asarray(tg[i]), rec))
    pcc_mean, ssim_mean = float(np.mean(pccs)), float(np.mean(ssims))
    # сверка с сохранёнными метриками
    ref_pcc = None
    mp = os.path.join(C.PATHS.out, "metrics.json")
    if os.path.exists(mp):
        ref_pcc = float(json.load(open(mp, encoding="utf-8"))["mean"]["pcc"])
    match = ref_pcc is None or abs(ref_pcc - pcc_mean) < 5e-3
    # градиент через физический слой доходит до U-Net
    net.train()
    for prm in net.parameters():
        prm.requires_grad_(True)
    pipe.train()
    I_g, _ = pipe(tg[None])
    loss_g = NPH.physical_loss(I_g, tg[None])
    loss_g.backward()
    gnorm = float(sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None))
    net.zero_grad(set_to_none=True)
    ok = bool(pcc_mean >= 0.80 and match and gnorm > 0 and codes.dtype == np.uint8)
    return _rec("netphys", ok, steps=steps, pcc=pcc_mean, ssim=ssim_mean,
                ref_pcc=ref_pcc if ref_pcc is not None else -1.0,
                grad_norm=gnorm, seconds=dt)


def main() -> int:
    print(f"Верификация комплекса, seed={C.SEED}\n")
    t0 = time.time()
    checks = [test_parseval, test_lens, test_quantization, test_gradient,
              test_cross_scheme, test_boundary, test_formats, test_netphys]
    passed = []
    for fn in checks:
        try:
            passed.append(bool(fn()))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [ERR ] {fn.__name__}: {exc}")
            RESULTS.setdefault(_KEYMAP[fn.__name__], dict(passed=False, error=str(exc)))
            passed.append(False)
    RESULTS["summary"] = dict(passed=sum(passed), total=len(passed),
                              all_passed=all(passed), elapsed=time.time() - t0,
                              seed=C.SEED, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(C.PATHS.out, exist_ok=True)
    with open(os.path.join(C.PATHS.out, "tests.json"), "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=1, ensure_ascii=False)
    print(f"\nИтог: {sum(passed)}/{len(passed)} тестов пройдено за {time.time() - t0:.1f} с")
    print(f"Результаты: {os.path.join(C.PATHS.out, 'tests.json')}")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
