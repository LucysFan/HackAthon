"""Комплекты тестовых изображений: хранение, подготовка, список.

Комплект — это каталог ``data/sets/<id>/`` с четырьмя целями ``target_1..4.png``
(128×128, 8 бит, grayscale). Рядом лежат исходники (``source/``) и результаты
инференса (``DOE.png``, ``recon_1..4.png``, ``metrics.json``).

Идентификатор комплекта — метка времени, поэтому комплекты не перетирают друг
друга и легко сортируются. Сдаваемый набор ``data/targets`` остаётся отдельным
и не удаляется: на него опираются отчёт, тесты и README.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time

import numpy as np
from PIL import Image

from . import config as C
from . import prepare as PR

SETS_DIR = os.path.join(C.PATHS.root, "data", "sets")
# метка времени (загрузка) либо sample-<тип> (готовый комплект из репозитория)
_ID_RE = re.compile(r"^(\d{8}-\d{6}(-\d+)?|sample-[a-z0-9]+)$")


def sets_dir() -> str:
    os.makedirs(SETS_DIR, exist_ok=True)
    return SETS_DIR


def new_set_id() -> str:
    base = time.strftime("%Y%m%d-%H%M%S")
    sid, k = base, 1
    while os.path.exists(os.path.join(SETS_DIR, sid)):
        k += 1
        sid = f"{base}-{k}"
    return sid


DEFAULT_ID = "default"


def valid_id(sid: str) -> bool:
    return bool(_ID_RE.match(sid or "")) or sid == DEFAULT_ID


def set_path(sid: str) -> str:
    if not valid_id(sid):
        raise ValueError(f"неверный идентификатор комплекта: {sid!r}")
    return os.path.join(SETS_DIR, sid)


def targets_of(sid: str) -> str:
    return set_path(sid)


def has_targets(sid: str) -> bool:
    return all(os.path.exists(os.path.join(set_path(sid), f"target_{i}.png"))
               for i in range(1, 5))


# ------------------------------------------------------------------ запись
def save_uploads(sid: str, files: list) -> list[str]:
    """Сохранить исходники комплекта (пути/байты) и вернуть их имена."""
    src = os.path.join(set_path(sid), "source")
    os.makedirs(src, exist_ok=True)
    names = []
    for i, f in enumerate(files, 1):
        im = PR.load_image(f)
        ext = ".png"
        name = f"source_{i}{ext}"
        im.save(os.path.join(src, name))
        names.append(name)
    return names


def build_targets(sid: str, *, invert: bool = True, crop: bool = True,
                  border: int = 0, single: bool = False) -> np.ndarray:
    """Собрать четыре цели комплекта из сохранённых исходников."""
    src_dir = os.path.join(set_path(sid), "source")
    files = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
    files = [os.path.join(src_dir, f) for f in files
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp",
                                                   ".bmp", ".tif", ".tiff")]
    if not files:
        raise ValueError("в комплекте нет исходных изображений")
    items = files[:1] * C.N_PLANES if single else files
    arr = PR.prepare_batch(items, invert=invert, crop=crop, border=border)
    PR.save_set(arr, set_path(sid))
    PR.save_preview(arr, os.path.join(set_path(sid), "preview.png"))
    meta = dict(id=sid, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                sources=[os.path.basename(f) for f in files],
                invert=bool(invert), crop=bool(crop), border=int(border),
                single=bool(single),
                energy=float(arr.reshape(C.N_PLANES, -1).sum(-1).mean()),
                nonzero_pct=float((arr > 0.02).mean() * 100.0),
                size=[C.N, C.N], bit_depth=8)
    with open(os.path.join(set_path(sid), "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    return arr


def create_from_files(files: list, *, invert: bool = True, crop: bool = True,
                      single: bool = False) -> str:
    """Новый комплект из загруженных файлов. Возвращает id."""
    sets_dir()
    sid = new_set_id()
    os.makedirs(set_path(sid), exist_ok=True)
    save_uploads(sid, files)
    build_targets(sid, invert=invert, crop=crop, single=single)
    return sid


def create_from_array(arr: np.ndarray, *, source_name: str = "array") -> str:
    """Комплект из готового массива [4,N,N] (например, данными из hdoe.targets)."""
    sets_dir()
    sid = new_set_id()
    os.makedirs(set_path(sid), exist_ok=True)
    os.makedirs(os.path.join(set_path(sid), "source"), exist_ok=True)
    for i in range(1, 5):
        Image.fromarray((np.clip(arr[i - 1], 0, 1) * 255).astype(np.uint8),
                        mode="L").save(
            os.path.join(set_path(sid), "source", f"source_{i}.png"))
    PR.save_set(np.asarray(arr, dtype=np.float64), set_path(sid))
    PR.save_preview(np.asarray(arr, dtype=np.float64),
                    os.path.join(set_path(sid), "preview.png"))
    with open(os.path.join(set_path(sid), "meta.json"), "w") as fh:
        json.dump(dict(id=sid, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       sources=[source_name], synthetic=True), fh,
                  indent=1, ensure_ascii=False)
    return sid


def delete(sid: str) -> bool:
    p = set_path(sid)
    if os.path.isdir(p):
        shutil.rmtree(p)
        return True
    return False


# ------------------------------------------------------------------ чтение
def _read_json(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return None


def _mean(met: dict | None) -> dict | None:
    """Средние метрики в едином виде: ssim/pearson (у сдаваемого набора — pcc)."""
    if not met:
        return None
    m = met.get("mean") or {}
    if not m:
        return None
    return dict(ssim=m.get("ssim"), pearson=m.get("pearson", m.get("pcc")),
                ssim_raw=m.get("ssim_raw"), steps=met.get("steps"))


def info(sid: str) -> dict:
    p = set_path(sid)
    meta = _read_json(os.path.join(p, "meta.json")) or {}
    met = _read_json(os.path.join(p, "metrics.json"))
    st = lambda f: (os.stat(os.path.join(p, f)).st_mtime          # noqa: E731
                    if os.path.exists(os.path.join(p, f)) else None)
    thumbs = sorted(f for f in os.listdir(p)
                    if re.match(r"^target_\d\.png$", f)) if os.path.isdir(p) else []
    return dict(id=sid, has_targets=has_targets(sid), created=meta.get("created"),
                sources=meta.get("sources") or [],
                synthetic=bool(meta.get("synthetic")),
                energy=meta.get("energy"), nonzero_pct=meta.get("nonzero_pct"),
                preview=os.path.exists(os.path.join(p, "preview.png")),
                targets=thumbs,
                has_doe=os.path.exists(os.path.join(p, "DOE.png")),
                has_recon=os.path.exists(os.path.join(p, "recon_1.png")),
                has_metrics=met is not None,
                mtime=st("metrics.json") or st("meta.json") or 0,
                mean=_mean(met),
                steps=(met or {}).get("steps"))


def all_sets() -> list[dict]:
    if not os.path.isdir(SETS_DIR):
        return []
    out = [info(s) for s in sorted(os.listdir(SETS_DIR)) if valid_id(s)]
    # готовые комплекты — наверх, дальше свежие загрузки
    out.sort(key=lambda r: (not r["id"].startswith("sample-"), r["id"]), reverse=False)
    return out


def load_targets(sid: str) -> np.ndarray:
    """[4,N,N] float64 в [0,1] (как в hdoe.targets.load_targets)."""
    imgs = []
    for i in range(1, 5):
        im = Image.open(os.path.join(set_path(sid), f"target_{i}.png")).convert("L")
        imgs.append(np.asarray(im, dtype=np.float64) / 255.0)
    return np.stack(imgs)


def ensure_default_set() -> str:
    """Комплект из сдаваемого набора data/targets (создаётся один раз)."""
    from . import targets as T
    for s in all_sets():
        if not s["synthetic"] and s["has_targets"] and s["sources"] and \
                "targets.png" in " ".join(s["sources"]):
            return s["id"]
    arr = T.load_targets()
    return create_from_array(arr, source_name="data/targets")
