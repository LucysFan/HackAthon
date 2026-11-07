"""Свип числа шагов физики ВНУТРИ сети: K → (PCC, SSIM), плюс абляция.

Запуск:  python scripts/sweep_phys.py            (пишет outputs/phys_sweep.json)

Скрипт ничего не дообучает: берётся сохранённый чекпоинт ``outputs/best.pt``
(веса U-Net) и один и тот же вход (четвёрка целей). Меняется только число
шагов дифференцируемого физического слоя K и расписание шага α. Так
показывается вклад физики, которая находится внутри сети: при K = 0 это
«сеть как есть», при K = 384 — штатный режим одного прямого прохода.
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
from hdoe import metrics as M
from hdoe import netphys as NP
from hdoe import physics as P
from hdoe import targets as T
from hdoe.model import build_net

# (K, α, расписание) — точки кривой «сколько физики внутри сети»
GRID = [(0, 0.35, "none"), (2, 0.35, "decay"), (4, 0.35, "decay"), (8, 0.35, "decay"),
        (16, 0.35, "decay"), (48, 0.35, "decay"), (96, 0.35, "decay"),
        (192, 0.35, "decay"), (384, 0.35, "decay"), (768, 0.35, "decay"),
        (1024, 0.35, "decay"), (2048, 0.35, "decay")]


def measure(phi_nn: np.ndarray, targets: np.ndarray) -> tuple[float, float, list]:
    """Метрики по квантованной 8-битной фазе (как в сдаваемом DOE.png)."""
    codes = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi_nn))))
    phi_q = P.decode8bit(codes)
    pccs, ssims = [], []
    for i, z in enumerate(C.ZS):
        rec = P.normalize01(P.propagate(torch.as_tensor(phi_q)[None, None], z)[0, 0].numpy())
        pccs.append(float(M.pcc(targets[i], rec)))
        ssims.append(float(M.ssim(targets[i], rec)))
    return float(np.mean(pccs)), float(np.mean(ssims)), pccs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(C.PATHS.out, "best.pt"))
    ap.add_argument("--out", default=os.path.join(C.PATHS.out, "phys_sweep.json"))
    args = ap.parse_args()

    targets = T.load_targets()
    q = torch.as_tensor(targets, dtype=torch.float32)[None]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck["arch"])
    net.load_state_dict(ck["model"])
    net.eval()
    with torch.no_grad():
        phi0 = net(q)[0, 0].numpy()
    p0, s0, _ = measure(phi0, targets)
    print(f"сеть без физики: PCC {p0:.4f} SSIM {s0:.4f}")

    rows = []
    for (k, gain, sched) in GRID:
        if k == 0:
            p, s, per = p0, s0, None
            dt = 0.0
        else:
            t0 = time.time()
            ref = NP.PhysicsRefiner(steps=k, gain=gain, learnable=False,
                                    schedule=("decay" if sched == "decay" else "const"),
                                    gain_end=0.02)
            with torch.enable_grad():
                phi = ref(torch.as_tensor(phi0)[None, None], q)
            dt = time.time() - t0
            p, s, per = measure(phi[0, 0].numpy(), targets)
        rows.append(dict(K=k, gain=gain, schedule=sched, pcc=p, ssim=s,
                         seconds=dt, per_plane=per))
        print(f"  K={k:4d} α={gain:.2f} {sched:5s}: PCC {p:.4f} SSIM {s:.4f} "
              f"({dt:.1f} с)")

    data = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=C.SEED,
                checkpoint=os.path.relpath(args.ckpt, C.PATHS.root),
                network_only=dict(pcc=p0, ssim=s0), rows=rows)
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    print(f"сохранено: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
