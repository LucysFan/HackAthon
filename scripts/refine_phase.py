"""Доводка фазы под физическую модель (ТЗ, п. 5.6.2) — отдельный запуск.

Запуск:
    python scripts/refine_phase.py --init nn+gs --steps 300 --lr 2e-2

Скрипт берёт маску сети (outputs/DOE.png, уже квантованную), маску GS
(outputs/DOE_gs.png, при отсутствии — считает) и уточняет фазу прямой
оптимизацией физического лосса (hdoe.refine). Лучшее приближение
сохраняется как outputs/DOE_refined.png и подхватывается evaluate.py
через --ckpt-refined / отчёт.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import _path  # noqa: F401
from hdoe import config as C
from hdoe import gs as G
from hdoe import physics as P
from hdoe import refine as R
from hdoe import targets as T

torch.set_num_threads(2)


def load_png_phase(path):
    from PIL import Image
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def save_png_phase(path, phi):
    from PIL import Image
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi, dtype=np.float32)))))
    Image.fromarray(v.astype(np.uint8), mode="L").save(path)


def score(phi, targets):
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi, dtype=np.float32)))))
    phi_q = P.decode8bit(v)
    from hdoe import gs as G2
    s = G2._score(torch.as_tensor(phi_q)[None, None],
                  torch.as_tensor(targets, dtype=torch.float64), C.ZS)
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or C.PATHS.out
    targets = T.load_targets()

    v_nn = load_png_phase(os.path.join(out_dir, "DOE.png"))
    if v_nn is None:
        print("нет outputs/DOE.png — сначала запустите scripts/evaluate.py")
        return 2
    v_gs = load_png_phase(os.path.join(out_dir, "DOE_gs.png"))
    phi_nn = P.decode8bit(v_nn)
    if v_gs is None:
        print("маска GS не найдена — считаю GS (200 кругов × 3 перезапуска)…")
        res = G.gs_solve(targets, iters=200, restarts=3, seed=C.SEED, verbose=True)
        phi_gs = res["phi"][0, 0].numpy()
        save_png_phase(os.path.join(out_dir, "DOE_gs.png"), phi_gs)
    else:
        phi_gs = P.decode8bit(v_gs)

    cands = [("nn", phi_nn), ("gs", phi_gs), ("blend", R.blend_phases(phi_nn, phi_gs, 0.5))]
    rows = []
    best = None
    for name, init in cands:
        t0 = time.time()
        out = R.refine_phase(targets, init, steps=args.steps, lr=args.lr, verbose=False)
        ph = out["phi"].numpy()
        sc = score(ph, targets)
        rows.append(dict(init=name, score=sc, elapsed=time.time() - t0))
        print(f"  старт {name}: прокси PCC {sc:.4f} ({time.time() - t0:.0f} с)", flush=True)
        if best is None or sc > best[0]:
            best = (sc, name, ph)
    save_png_phase(os.path.join(out_dir, "DOE_refined.png"), best[2])
    json.dump(dict(best_init=best[1], best_score=best[0], steps=args.steps, lr=args.lr,
                   candidates=rows, seed=C.SEED),
              open(os.path.join(out_dir, "refine.json"), "w", encoding="utf-8"), indent=1)
    print(f"лучший старт: {best[1]} (PCC-прокси {best[0]:.4f}) → outputs/DOE_refined.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
