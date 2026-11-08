"""Данные выбранного комплекта для дашборда.

Глобальный ``data_bundle.js`` относится к сдаваемому набору. Пользовательский
комплект получает собственный небольшой payload: цели, маска, реконструкции,
метрики и настоящая z-стопка по сохранённому DOE.png. Пока маска не рассчитана,
возвращаются только цели — старый демо-результат нельзя выдавать за новый.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

from . import config as C
from . import metrics as M
from . import physics as P
from . import prepare as PR
from . import sets as SS

SCHEMA = "hdoe/set-view/1"


def _signature(set_id: str) -> str:
    h = hashlib.sha256(SCHEMA.encode())
    base = SS.set_path(set_id)
    for name in ("meta.json", "DOE.png", "metrics.json",
                 *(f"target_{i}.png" for i in range(1, 5)),
                 *(f"recon_{i}.png" for i in range(1, 5))):
        p = os.path.join(base, name)
        st = os.stat(p) if os.path.isfile(p) else None
        h.update(f"{name}:{st.st_mtime_ns if st else 0}:{st.st_size if st else 0}".encode())
    return h.hexdigest()[:20]


def _pack(a: np.ndarray, delta: bool = False) -> str:
    a = np.ascontiguousarray(a)
    if delta:
        a = a.copy()
        a[1:] = (a[1:].astype(np.int16) - a[:-1].astype(np.int16)).astype(np.uint8)
    return base64.b64encode(gzip.compress(a.tobytes(), 6)).decode("ascii")


def _gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def normalize_metrics(metrics: dict | None) -> tuple[dict, list[dict]]:
    """Один формат для metrics.json инференса и манифеста сдаваемого набора."""
    metrics = dict(metrics or {})
    rows = []
    for i, r in enumerate(metrics.get("planes") or []):
        rows.append(dict(r, plane=i + 1, z_mm=r.get("z_mm", C.ZS_MM[i]),
                         pcc=r.get("pcc", r.get("pearson")),
                         pearson=r.get("pearson", r.get("pcc"))))
    mean = dict(metrics.get("mean") or {})
    if rows:
        for key, aliases in (("ssim", ("ssim",)), ("pcc", ("pcc", "pearson")),
                             ("ssim_raw", ("ssim_raw",)), ("pearson_raw", ("pearson_raw",))):
            if mean.get(key) is None:
                vals = [next((r[a] for a in aliases if r.get(a) is not None), None) for r in rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean[key] = float(np.mean(vals))
    if mean.get("pcc") is None:
        mean["pcc"] = mean.get("pearson")
    if mean.get("pearson") is None:
        mean["pearson"] = mean.get("pcc")
    metrics["mean"] = mean
    return metrics, rows


def build_set_view(set_id: str, cache_dir: str) -> dict:
    if not SS.valid_id(set_id) or not SS.has_targets(set_id):
        raise ValueError("нет такого комплекта")
    torch.set_num_threads(min(2, torch.get_num_threads() or 2))
    signature = _signature(set_id)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"view_{set_id}_{signature}.json")
    if os.path.isfile(cache):
        try:
            with open(cache) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    base = SS.set_path(set_id)
    targets = SS.load_targets(set_id)
    target_u8 = np.asarray(targets * 255 + 0.5, dtype=np.uint8)
    arrays = {"targets": _pack(target_u8)}
    empty_f32 = np.empty(0, dtype=np.float32)
    for key in ("phase_nn", "recon_nn", "ssim_map_nn", "zprobe_nn"):
        arrays[key] = _pack(empty_f32)
    for key in ("doe_nn", "zstack_nn", "bitdepth_nn"):
        arrays[key] = _pack(np.empty(0, dtype=np.uint8))

    # Реальные кадры подготовки первого исходника, а не демо-картинки.
    source_dir = os.path.join(base, "source")
    sources = sorted(os.listdir(source_dir)) if os.path.isdir(source_dir) else []
    source_names = []
    arrays["source_steps"] = _pack(np.empty(0, dtype=np.uint8))
    if sources:
        try:
            steps = PR.preview_steps(os.path.join(source_dir, sources[0]), size=C.N)
            arrays["source_steps"] = _pack(np.stack([np.asarray(im, dtype=np.uint8)
                                                       for _, im in steps]))
            source_names = [name for name, _ in steps]
        except (OSError, ValueError):
            pass

    # Для загруженного, но ещё не рассчитанного комплекта никаких чужих масок.
    zs = C.zgrid(step=0.05e-3)
    arrays["zgrid"] = _pack(zs.astype(np.float32))
    doe_path = os.path.join(base, "DOE.png")
    met_path = os.path.join(base, "metrics.json")
    ready = os.path.isfile(doe_path) and os.path.isfile(met_path) and all(
        os.path.isfile(os.path.join(base, f"recon_{i}.png")) for i in range(1, 5))
    metrics, rows = {}, []
    if ready:
        with open(met_path) as fh:
            metrics, rows = normalize_metrics(json.load(fh))
        codes = _gray(doe_path)
        phi = np.asarray(P.decode8bit(codes), dtype=np.float64)
        recon = np.stack([_gray(os.path.join(base, f"recon_{i}.png"))
                          for i in range(1, 5)]).astype(np.float32) / 255.0
        ssim_maps = np.stack([np.clip(M.ssim_map(targets[i], recon[i]).numpy(), -1, 1)
                              for i in range(C.N_PLANES)])
        arrays.update(doe_nn=_pack(codes), phase_nn=_pack(phi.astype(np.float32)),
                      recon_nn=_pack(recon), ssim_map_nn=_pack(ssim_maps.astype(np.float32)))
        phi_t = torch.as_tensor(phi)[None, None]
        frames, probe = [], [[] for _ in range(5)]
        pixels = ((64, 64), (32, 32), (96, 96), (16, 96), (48, 80))
        with torch.no_grad():
            for z in zs:
                intensity, _ = P.propagate_any(phi_t, float(z))
                intensity = intensity[0, 0].numpy()
                frames.append(P.to_u8(P.normalize01(intensity)))
                for curve, (y, x) in zip(probe, pixels):
                    curve.append(float(intensity[y, x]))
            bits = []
            for level in (8, 6, 4, 2):
                nlev = 2 ** level
                q = np.round((phi + np.pi) / (2 * np.pi) * (nlev - 1)) / (nlev - 1) * 2 * np.pi - np.pi
                intensity, _ = P.propagate_any(torch.as_tensor(q)[None, None], C.ZS[1])
                bits.append(P.to_u8(P.normalize01(intensity[0, 0].numpy())))
        arrays.update(zstack_nn=_pack(np.stack(frames), delta=True),
                      zprobe_nn=_pack(np.asarray(probe, dtype=np.float32)),
                      bitdepth_nn=_pack(np.stack(bits)))
    out = dict(ok=True, schema=SCHEMA, id=set_id, signature=signature,
               has_result=ready, metrics=metrics, file_metrics=dict(nn=rows),
               source_names=source_names, b64=arrays,
               note="Дифракция рассчитана по DOE.png этого комплекта" if ready else
                    "Цели загружены. Нажмите «прогнать сеть», чтобы получить маску и дифракцию.")
    # Не оставляем пользователю растущий кэш из десятков полных z-стопов.
    for name in os.listdir(cache_dir):
        if name.startswith(f"view_{set_id}_") and name != os.path.basename(cache):
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass
    tmp = cache + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    os.replace(tmp, cache)
    return out
