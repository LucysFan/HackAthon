"""Сборка статического бандла данных для дашборда (ТЗ, п. 12.7, приложение Г).

Запуск:  python scripts/build_dashboard.py [--quick]

Собирает из тех же артефактов, что и сдаваемые файлы (outputs/DOE.png,
recon_*.png, metrics.json, tests.json, лог обучения), бинарные массивы и
manifest.json в dashboard/data/. Числа дашборда совпадают с metrics.json
(п. 12.7.2).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import warnings

import numpy as np
import torch

import _path  # noqa: F401
from hdoe import config as C
from hdoe import gs as G
from hdoe import metrics as M
from hdoe import physics as P
from hdoe import targets as T

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(2)

DATA = os.path.join(C.PATHS.dashboard, "data")


# ------------------------------------------------------------------- хелперы
def write_u8(path: str, arr: np.ndarray) -> int:
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    a.tofile(path)
    return a.nbytes


def write_f32(path: str, arr: np.ndarray) -> int:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    a.tofile(path)
    return a.nbytes


def load_phase(out_dir: str, tag: str = "nn") -> np.ndarray:
    """Квантованная фаза, прочитанная из сданного PNG (воспроизводимость)."""
    from PIL import Image
    name = "DOE.png" if tag == "nn" else "DOE_gs.png"
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    v = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return P.decode8bit(v)


def bit_depth_stack(phi: np.ndarray, levels=(8, 6, 4, 2), z: float = C.ZS[1]) -> np.ndarray:
    """A8: деградация битности (8→6→4→2) — предпросчитанные реконструкции."""
    out = []
    phi_t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
    for b in levels:
        nlev = 2 ** b
        # квантование в b бит по той же шкале [−π, π]
        q = np.round((phi + np.pi) / (2 * np.pi) * (nlev - 1)) / (nlev - 1) * 2 * np.pi - np.pi
        I, _m = P.propagate_any(torch.as_tensor(q, dtype=torch.float64)[None, None], z)
        I = I[0, 0].numpy()
        out.append(P.to_u8(P.normalize01(I)))
    _ = phi_t
    return np.stack(out)


def build_stack(phi_q: np.ndarray, zs: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Z-стопка интенсивностей (мин–макс нормировка каждого кадра)."""
    phi_t = torch.as_tensor(phi_q, dtype=torch.float64)[None, None]
    frames = []
    for z in zs:
        # ниже z_crit одно-БПФ-схема неприменима (п. 4.5.3) → угловой спектр;
        # для кадров z ≥ z_crit используется штатная одно-БПФ-схема
        I, _ = P.propagate_any(phi_t, float(z))
        frames.append(P.to_u8(P.normalize01(I[0, 0].numpy())))
    return np.stack(frames)


def ssim_maps(targets: np.ndarray, phi_q: np.ndarray) -> np.ndarray:
    """Локальные SSIM-карты (4 плоскости) для NN и GS."""
    phi_t = torch.as_tensor(phi_q, dtype=torch.float64)[None, None]
    maps = []
    for i, z in enumerate(C.ZS):
        I = P.normalize01(P.propagate(phi_t, z)[0, 0].numpy())
        maps.append(np.clip(M.ssim_map(targets[i], I).numpy(), -1, 1))
    return np.stack(maps)


def power_curve(phi_q: np.ndarray, x: int, y: int, zs: np.ndarray) -> np.ndarray:
    """z-зонд: кривая I(z) для кликнутого пиксела по всей стопке."""
    phi_t = torch.as_tensor(phi_q, dtype=torch.float64)[None, None]
    vals = []
    for z in zs:
        I, _m = P.propagate_any(phi_t, float(z))
        vals.append(float(I[0, 0, y, x]))
    return np.array(vals, dtype=np.float32)


def train_snapshots(out_dir: str) -> tuple[np.ndarray, list]:
    """Снапшоты маски обучения + лог."""
    snap_dir = os.path.join(out_dir, "train_masks")
    files = sorted(glob.glob(os.path.join(snap_dir, "mask_*.npy")))
    masks, steps = [], []
    for f in files:
        masks.append(np.load(f))
        m = re.search(r"mask_(\d+)", os.path.basename(f))
        steps.append(int(m.group(1)) if m else 0)
    if not masks:
        return np.zeros((0, C.N, C.N), np.uint8), []
    return np.stack(masks).astype(np.uint8), steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="каталог outputs с артефактами")
    ap.add_argument("--quick", action="store_true", help="уменьшенная z-стопка (для отладки)")
    ap.add_argument("--train-dir", default=None, help="каталог с train_log.json и train_masks/")
    ap.add_argument("--skip-gs", action="store_true")
    args = ap.parse_args()
    C.PATHS.ensure()
    os.makedirs(DATA, exist_ok=True)
    out_dir = args.out or C.PATHS.out
    train_dir = args.train_dir or out_dir
    sizes = {}

    # ------------------------------------------------------------------ цели
    targets = T.load_targets()
    sizes["targets.u8.bin"] = write_u8(os.path.join(DATA, "targets.u8.bin"),
                                       T.to_u8(targets))
    # метрики и тесты
    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    tests_path = os.path.join(out_dir, "tests.json")
    if not os.path.exists(tests_path):
        tests_path = os.path.join(C.PATHS.out, "tests.json")
    tests = json.load(open(tests_path)) if os.path.exists(tests_path) else {}

    # ----------------------------------------------------------- фазы NN/GS
    phases, doe_codes, recons, ssims = {}, {}, {}, {}
    for tag in ("nn", "gs"):
        phi = load_phase(out_dir, tag)
        if phi is None:
            continue
        phases[tag] = phi
        codes = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi)))).astype(np.uint8)
        doe_codes[tag] = codes
        sizes[f"phase_{tag}.f32.bin"] = write_f32(os.path.join(DATA, f"phase_{tag}.f32.bin"),
                                                  phi.astype(np.float32))
        sizes[f"doe_{tag}.u8.bin"] = write_u8(os.path.join(DATA, f"doe_{tag}.u8.bin"), codes)
        phi_t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
        recons[tag] = np.stack([P.normalize01(P.propagate(phi_t, z)[0, 0].numpy())
                                for z in C.ZS]).astype(np.float32)
        sizes[f"recon_{tag}.f32.bin"] = write_f32(os.path.join(DATA, f"recon_{tag}.f32.bin"),
                                                  recons[tag])
        ssims[tag] = ssim_maps(targets, phi)
        sizes[f"ssim_map_{tag}.f32.bin"] = write_f32(os.path.join(DATA, f"ssim_map_{tag}.f32.bin"),
                                                     ssims[tag].astype(np.float32))

    if "nn" not in phases:
        print("ОШИБКА: нет outputs/DOE.png — сначала запустите scripts/evaluate.py")
        return 2
    if "gs" not in phases and not args.skip_gs:
        print("GS-маска отсутствует: считаю GS с параметрами по умолчанию…")
        res = G.gs_solve(targets, iters=300, restarts=3, seed=C.SEED, verbose=True)
        codes = np.squeeze(np.asarray(P.encode8bit(res["phi"]))).astype(np.uint8)
        from PIL import Image
        Image.fromarray(codes, mode="L").save(os.path.join(out_dir, "DOE_gs.png"))
        phi = P.decode8bit(codes)
        phases["gs"] = phi
        sizes["phase_gs.f32.bin"] = write_f32(os.path.join(DATA, "phase_gs.f32.bin"),
                                              phi.astype(np.float32))
        sizes["doe_gs.u8.bin"] = write_u8(os.path.join(DATA, "doe_gs.u8.bin"), codes)
        phi_t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
        recons["gs"] = np.stack([P.normalize01(P.propagate(phi_t, z)[0, 0].numpy())
                                 for z in C.ZS]).astype(np.float32)
        sizes["recon_gs.f32.bin"] = write_f32(os.path.join(DATA, "recon_gs.f32.bin"), recons["gs"])
        ssims["gs"] = ssim_maps(targets, phi)
        sizes["ssim_map_gs.f32.bin"] = write_f32(os.path.join(DATA, "ssim_map_gs.f32.bin"),
                                                 ssims["gs"].astype(np.float32))

    # ------------------------------------------------------------- z-стопки
    zs = C.zgrid()
    if args.quick:
        zs = zs[::10]
    sizes["zgrid.f32.bin"] = write_f32(os.path.join(DATA, "zgrid.f32.bin"), zs)
    z_method = ["fresnel1fft" if float(z) >= C.Z_CRIT else "angular_spectrum" for z in zs]
    for tag, phi in phases.items():
        t0 = time.time()
        stack = build_stack(phi, zs)
        name = f"zstack_{tag}.u8.bin"
        sizes[name] = write_u8(os.path.join(DATA, name), stack)
        print(f"  zstack_{tag}: {stack.shape[0]} кадров за {time.time() - t0:.1f} с")
        # кэш профиля I(z) для нескольких пикселей (z-зонд, п. 12.5.4)
        pix = [(64, 64), (32, 32), (96, 96), (16, 96), (48, 80)]
        curves = np.stack([power_curve(phi, x, y, zs) for (y, x) in pix]).astype(np.float32)
        sizes[f"zprobe_{tag}.f32.bin"] = write_f32(os.path.join(DATA, f"zprobe_{tag}.f32.bin"),
                                                   curves)

    # ------------------------------------------------ деградация битности (A8)
    bd = {}
    for tag, phi in phases.items():
        arr = np.stack([bit_depth_stack(phi, (b,), C.ZS[1])[0] for b in (8, 6, 4, 2)])
        bd[tag] = arr
        sizes[f"bitdepth_{tag}.u8.bin"] = write_u8(os.path.join(DATA, f"bitdepth_{tag}.u8.bin"),
                                                   arr)

    # --------------------------------------------------- обучение (панель 7)
    masks, steps = train_snapshots(train_dir)
    if len(masks) == 0:
        masks = np.stack([doe_codes["nn"]]).astype(np.uint8)
        steps = [0]
    sizes["train_masks.u8.bin"] = write_u8(os.path.join(DATA, "train_masks.u8.bin"), masks)
    log_path = os.path.join(train_dir, "train_log.json")
    train_log = json.load(open(log_path)) if os.path.exists(log_path) else []
    json.dump(train_log, open(os.path.join(DATA, "train_log.json"), "w"), indent=1)
    json.dump(dict(steps=steps), open(os.path.join(DATA, "train_steps.json"), "w"))
    # реконструкции на снапшотах (для A3) — по самой дешёвой плоскости
    snap_rec = []
    for m in masks[:24]:
        phi_s = P.decode8bit(m)
        I = P.propagate(torch.as_tensor(phi_s, dtype=torch.float64)[None, None],
                        C.ZS[1])[0, 0].numpy()
        snap_rec.append(P.to_u8(P.normalize01(I)))
    sizes["train_recon.u8.bin"] = write_u8(os.path.join(DATA, "train_recon.u8.bin"),
                                           np.stack(snap_rec))

    # ---------------------------------------------------------------- тесты
    if not tests:
        tests = {"summary": dict(passed=0, total=0, all_passed=False)}
    json.dump(tests, open(os.path.join(DATA, "tests.json"), "w"), indent=1, ensure_ascii=False)

    # --------------------------------------------- метрики по файлам (сверка)
    file_metrics = {}
    for tag in phases:
        rows = []
        for i, z in enumerate(C.ZS):
            r01 = recons[tag][i]
            rows.append(dict(plane=i + 1, z_mm=C.ZS_MM[i],
                             pcc=M.pcc(targets[i], r01), ssim=M.ssim(targets[i], r01)))
        file_metrics[tag] = rows

    # ------------------------------------------------------------- manifest
    manifest = dict(
        schema="hdoe/dashboard/1",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        constants=C.summary(),
        control_table=C.control_table(),
        planes=dict(labels=C.PLANE_LABELS, zs=list(C.ZS), zs_mm=list(C.ZS_MM),
                    dx_out=list(C.OUT_DX), page_mm=list(C.PAGE_MM), s=list(C.SCALE_S)),
        layout=dict(n=C.N, targets="targets.u8.bin",
                    shapes=dict(targets=[4, C.N, C.N], zstack=[len(zs), C.N, C.N],
                                ssim_map=[4, C.N, C.N], train_masks=[len(masks), C.N, C.N],
                                train_recon=[len(snap_rec), C.N, C.N],
                                bitdepth=[4, C.N, C.N], zprobe=[5, len(zs)])),
        files=sizes,
        metrics=metrics,
        file_metrics=file_metrics,
        tests=tests,
        train=dict(log=train_log, steps=steps, masks=len(masks)),
        variants=list(phases.keys()),
        protocol=metrics.get("protocol"),
        seed=C.SEED,
        versions=versions(),
        zprobe_pixels=[[64, 64], [32, 32], [96, 96], [16, 96], [48, 80]],
        z_method=dict(frames=z_method, z_crit=C.Z_CRIT,
                      note=("кадры z ≥ z_crit — одно-БПФ-схема Френеля; кадры z < z_crit "
                            "(п. 4.5.3) — band-limited угловой спектр")),
        bit_levels=[8, 6, 4, 2],
    )
    json.dump(manifest, open(os.path.join(DATA, "manifest.json"), "w"), indent=1,
              ensure_ascii=False)
    # ---------------- data_bundle.js: офлайн-режим (открытие двойным кликом)
    js_size = write_js_bundle(manifest)
    total = sum(sizes.values())
    print(f"\nбандл собран: {len(sizes) + 4} файлов, {total / 1e6:.2f} МБ сырых данных; "
          f"data_bundle.js {js_size / 1e6:.2f} МБ "
          f"(лимит ТЗ 12 МБ: {'OK' if js_size < 12e6 else 'ПРЕВЫШЕН'})")
    return 0


def write_js_bundle(manifest: dict) -> int:
    """Упаковка массивов в data_bundle.js (gzip + base64) для работы по file://."""
    import base64
    import gzip

    def pack(path: str, key: str = "") -> str:
        raw = open(path, "rb").read()
        # z-стопки — самый большой объём: храним с дельта-кодированием по времени
        # (соседние кадры похожи) и максимальным сжатием
        if key.startswith("zstack") and len(raw) > 4 * 128 * 128:
            a = np.frombuffer(raw, dtype=np.uint8)
            n = 128 * 128
            d = a.reshape(-1, n).copy()
            d[1:] = (a.reshape(-1, n)[1:].astype(np.int16)
                     - a.reshape(-1, n)[:-1].astype(np.int16)).astype(np.uint8)
            raw = d.reshape(-1).tobytes()
        return base64.b64encode(gzip.compress(raw, 9)).decode("ascii")

    entries = ["targets", "phase_nn", "phase_gs", "doe_nn", "doe_gs",
               "zstack_nn", "zstack_gs", "recon_nn", "recon_gs",
               "ssim_map_nn", "ssim_map_gs", "zgrid", "train_masks", "train_recon",
               "bitdepth_nn", "bitdepth_gs", "zprobe_nn", "zprobe_gs"]
    parts = []
    for key in entries:
        for cand in (f"{key}.u8.bin", f"{key}.f32.bin"):
            path = os.path.join(DATA, cand)
            if os.path.exists(path):
                parts.append(f'{key}:"{pack(path, key)}"')
                break
    jsons = {}
    for fname, key in (("train_log.json", "train_log"), ("train_steps.json", "train_steps"),
                       ("tests.json", "tests")):
        path = os.path.join(DATA, fname)
        if os.path.exists(path):
            jsons[key] = json.load(open(path))
    js = ("// Автоматически собранный бандл данных для офлайн-режима (gzip + base64).\n"
          "// Сформирован scripts/build_dashboard.py; не редактировать вручную.\n"
          "window.HDOE_BUNDLE={\n"
          "manifest:" + json.dumps(manifest, ensure_ascii=False) + ",\n"
          "b64:{\n" + ",\n".join(parts) + "\n},\n"
          "json:" + json.dumps(jsons, ensure_ascii=False) + "\n};\n")
    path = os.path.join(C.PATHS.dashboard, "data_bundle.js")
    with open(path, "w") as fh:
        fh.write(js)
    return len(js)


def versions() -> dict:
    import torch as _t
    import numpy as _np
    v = dict(python=sys.version.split()[0], numpy=_np.__version__, torch=_t.__version__)
    try:
        import PIL
        v["pillow"] = PIL.__version__
    except Exception:                                             # noqa: BLE001
        pass
    try:
        import skimage
        v["scikit-image"] = skimage.__version__
    except Exception:                                             # noqa: BLE001
        pass
    try:
        import segmentation_models_pytorch as smp
        v["segmentation-models-pytorch"] = smp.__version__
    except Exception:                                             # noqa: BLE001
        pass
    return v


if __name__ == "__main__":
    sys.exit(main())
