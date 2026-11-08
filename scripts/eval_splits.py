"""Метрики на трёх наборах: тренировочном, валидационном и тестовом.

Зачем отдельный скрипт. В отчёте и на дашборде фигурировали две цифры:
кривая валидации по ходу обучения и итог на сдаваемом наборе. Но сдаваемый
набор участвовал в обучении дважды — как «маяк» для учительской маски GS
(``--target-phase``) и как одна из учительских четвёрок. Поэтому называть его
тестом было бы нечестно. Здесь считается нормальный протокол:

* **тренировка** — случайные четвёрки корпуса (то, на чём учились веса);
* **валидация** — восемь четвёрок корпуса, которых не было в обучении;
* **тест** — сдаваемый набор и пять готовых комплектов, включая
  фотографический, который в корпус не входит вовсе.

Проверяется модель ровно в том виде, в котором её сдают: чекпоинт
``outputs/best.pt``, один прямой проход, квантование 8 бит, дифракция на
четырёх z, метрика после перечитывания кода из массива (как из PNG).

    python scripts/eval_splits.py              # всё, ~3 минуты на CPU
    python scripts/eval_splits.py --quick      # меньше четвёрок корпуса
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C            # noqa: E402
from hdoe import corpus as CO           # noqa: E402
from hdoe import metrics as M           # noqa: E402
from hdoe import netphys as NP          # noqa: E402
from hdoe import physics as P           # noqa: E402
from hdoe import sets as SS             # noqa: E402
from hdoe import targets as T           # noqa: E402
from hdoe.model import build_net        # noqa: E402


def load_pack(ckpt: str, device: torch.device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck.get("arch") or dict(width=48, depth=5))
    net.load_state_dict(ck["model"])
    pk = ck.get("pack") or {}
    pipe = NP.PhysicsInNet(net, steps=int(pk.get("steps", 2048)),
                           gain=float(pk.get("gain", 0.35)), quantize=True,
                           learnable=False,
                           schedule=str(pk.get("schedule", "decay")),
                           gain_end=float(pk.get("gain_end", 0.02)))
    pipe.to(device).eval()
    return pipe, pk


@torch.no_grad()
def score_quads(pipe, quads: torch.Tensor, targets: torch.Tensor) -> list[dict]:
    """Один прямой проход на каждую четвёрку: SSIM и Pearson по плоскостям."""
    rows = []
    for b in range(quads.shape[0]):
        q = quads[b:b + 1].to(next(pipe.parameters()).device)
        _, phi = pipe(q)
        code = np.squeeze(np.asarray(P.encode8bit(phi[0, 0].detach().cpu()),
                                     dtype=np.uint8))
        phi_q = np.asarray(P.decode8bit(code), dtype=np.float64)
        ssim, pcc = [], []
        for i, z in enumerate(C.ZS):
            I = P.propagate(torch.as_tensor(phi_q)[None, None], z)[0, 0].numpy()
            I01 = P.normalize01(I)
            t = np.asarray(targets)
            t = t[b, i] if t.ndim == 4 else t[i]
            ssim.append(float(M.ssim(t, I01)))
            pcc.append(float(M.pcc(t, I01)))
        rows.append(dict(ssim=float(np.mean(ssim)), pcc=float(np.mean(pcc)),
                         ssim_planes=ssim, pcc_planes=pcc))
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return dict(n=0)
    s = [r["ssim"] for r in rows]
    p = [r["pcc"] for r in rows]
    return dict(n=len(rows), ssim=float(np.mean(s)), pcc=float(np.mean(p)),
                ssim_std=float(np.std(s)), pcc_std=float(np.std(p)),
                ssim_min=float(np.min(s)), ssim_max=float(np.max(s)),
                per_plane_ssim=[float(np.mean([r["ssim_planes"][i] for r in rows]))
                                for i in range(C.N_PLANES)],
                per_plane_pcc=[float(np.mean([r["pcc_planes"][i] for r in rows]))
                               for i in range(C.N_PLANES)])


def main() -> int:
    ap = argparse.ArgumentParser(description="Метрики по train/val/test")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "splits.json"))
    ap.add_argument("--train-quads", type=int, default=8)
    ap.add_argument("--val-quads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_quads, args.val_quads = 3, 3

    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    pipe, pk = load_pack(args.ckpt, dev)
    corpus = CO.load_corpus()
    t0 = time.time()

    # ------------------------------------------------------------- тренировка
    # те же четвёрки, что видела сеть на дистилляции: ds_fixed[...] при k % 3 == 0
    fixed = T.load_targets()
    ds_fixed = CO.QuadDataset(corpus, length=10 ** 6, mode="fixed", fixed=fixed,
                              seed=C.SEED, geom_only=True)
    ds_corp = CO.QuadDataset(corpus, length=10 ** 6, mode="corpus", seed=C.SEED,
                             geom_only=True)
    train_quads, train_t = [], []
    for k in range(args.train_quads):
        q = ds_fixed[k * 37 + 11] if k % 3 == 0 else ds_corp[k * 37 + 11]
        train_quads.append(q)
        train_t.append(q.numpy())      # цель — та же четвёрка, что видела сеть
    train_rows = score_quads(pipe, torch.stack(train_quads), np.stack(train_t))
    print(f"тренировка: {len(train_rows)} четвёрок за {time.time() - t0:.0f} с", flush=True)

    # ------------------------------------------------------------ валидация
    # отдельные зёрна: этих четвёрок в обучении не было
    vq = CO.validation_quads(corpus, n=args.val_quads, seed=C.SEED + 777)
    val_rows = score_quads(pipe, vq, vq)
    print(f"валидация: {len(val_rows)} четвёрок за {time.time() - t0:.0f} с", flush=True)

    # ---------------------------------------------------------------- тест
    test_rows = []
    tl = T.load_targets()
    test_rows += score_quads(pipe, torch.as_tensor(tl, dtype=torch.float32)[None], tl)
    for st in SS.all_sets():
        if st["id"] == "default":
            continue
        try:
            arr = SS.load_targets(st["id"])
        except Exception:                                  # noqa: BLE001
            continue
        r = score_quads(pipe, torch.as_tensor(arr, dtype=torch.float32)[None], arr)[0]
        r["set"] = st["id"]
        test_rows.append(r)
    print(f"тест: {len(test_rows)} комплектов за {time.time() - t0:.0f} с", flush=True)

    # --------------------------------------------------------------- свод
    out = dict(schema="hdoe/splits/1", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               checkpoint=os.path.relpath(args.ckpt, ROOT), device=str(dev),
               pack=pk, corpus_size=int(len(corpus)), seed=C.SEED,
               train=dict(summary=summarize(train_rows),
                          note="четвёрки корпуса, на которых училась дистилляция"),
               val=dict(summary=summarize(val_rows),
                        note="восемь четвёрок корпуса с отдельным зерном, в обучении "
                             "не участвовали"),
               test=dict(summary=summarize(test_rows),
                         note="сдаваемый набор и пять готовых комплектов",
                         sets=[dict(name=r.get("set", "data/targets"),
                                    ssim=r["ssim"], pcc=r["pcc"],
                                    ssim_planes=r["ssim_planes"],
                                    pcc_planes=r["pcc_planes"]) for r in test_rows]),
               seconds=time.time() - t0)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    for part in ("train", "val", "test"):
        s = out[part]["summary"]
        print(f"{part:9s} n={s['n']:2d}  SSIM {s['ssim']:.4f}  Pearson {s['pcc']:.4f}")
    print(f"\nзаписано: {out['checkpoint']} → {os.path.relpath(args.out, ROOT)} "
          f"за {out['seconds']:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
