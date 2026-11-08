"""Инференс на произвольном комплекте изображений (для вкладки «Тест»).

Запуск:
    python scripts/infer_set.py --set data/sets/20260101-120000 --steps 384
    python scripts/infer_set.py --targets data/targets --steps 2048 --out outputs

Что делает: читает четыре цели комплекта (128×128, grayscale, 8 бит), получает
фазу ОДНИМ прямым проходом сети с физикой внутри, квантует её в 8 бит, считает
дифракцию Френеля в одно-БПФ-схеме на четырёх z и сохраняет

    DOE.png          — маска 128×128, 8 бит, grayscale (формат сдачи)
    recon_1..4.png   — восстановленные изображения
    metrics.json     — SSIM и Pearson по плоскостям и средние

Классические итеративные методы не используются: только прямой проход.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C           # noqa: E402
from hdoe import metrics as M          # noqa: E402
from hdoe import netphys as NP         # noqa: E402
from hdoe import physics as P          # noqa: E402
from hdoe.model import build_net       # noqa: E402


def pick_device(kind: str = "auto") -> torch.device:
    if kind != "auto":
        return torch.device(kind)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_set(path: str) -> np.ndarray:
    """Четыре цели комплекта → float64 [4,N,N] в [0,1]."""
    imgs = []
    for i in range(1, 5):
        p = os.path.join(path, f"target_{i}.png")
        if not os.path.exists(p):
            raise SystemExit(f"нет файла {p}")
        im = Image.open(p).convert("L").resize((C.N, C.N), Image.LANCZOS)
        imgs.append(np.asarray(im, dtype=np.float64) / 255.0)
    return np.stack(imgs)


def save_gray(path: str, arr: np.ndarray) -> None:
    a = np.clip(np.asarray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(a, mode="L").save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_dir", default=None, help="каталог комплекта")
    ap.add_argument("--targets", default=None, help="каталог с target_1..4.png")
    ap.add_argument("--ckpt", default=os.path.join(C.PATHS.out, "best.pt"))
    ap.add_argument("--steps", type=int, default=0, help="K шагов физики (0 — из чекпоинта)")
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--out", default=None, help="куда писать артефакты (по умолчанию — каталог комплекта)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = args.set_dir or args.targets or C.PATHS.targets
    out = args.out or src
    os.makedirs(out, exist_ok=True)
    targets = load_set(src)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"нет чекпоинта {args.ckpt} — сначала обучите сеть "
                         f"(python scripts/train_gpu.py)")
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net = build_net(ck.get("arch") or dict(width=48, depth=5))
    net.load_state_dict(ck["model"])
    net.eval()
    dev = pick_device(args.device)
    net.to(dev)
    pk = ck.get("pack") or {}
    steps = args.steps or int(pk.get("steps", 384))
    gain = args.gain or float(pk.get("gain", 0.35))

    pipe = NP.PhysicsInNet(net, steps=steps, gain=gain, quantize=True, learnable=False,
                           schedule=str(pk.get("schedule", "decay")),
                           gain_end=float(pk.get("gain_end", 0.02)))
    pipe.to(dev)
    pipe.eval()
    print(f"веса: {args.ckpt}; устройство: {dev}; K={steps}", flush=True)
    t0 = time.time()
    if pipe.refiner is not None:
        every = max(1, steps // 20)
        def progress(k, total, loss):
            if k % every == 0 or k == total:
                print(f"физика: {k}/{total} · loss {loss:.4f} · {time.time() - t0:.1f} с", flush=True)
        pipe.refiner.on_step = progress
    with torch.no_grad():
        _, phi = pipe(torch.as_tensor(targets, dtype=torch.float32)[None].to(dev))
    secs = time.time() - t0
    phi_np = phi[0, 0].detach().float().cpu().numpy()

    codes = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi_np)), dtype=np.uint8))
    doe_path = os.path.join(out, "DOE.png")
    Image.fromarray(codes, mode="L").save(doe_path)
    # читаем обратно: метрики считаются именно по тому файлу, который сдаётся
    codes = np.asarray(Image.open(doe_path).convert("L"), dtype=np.uint8)
    phi_q = np.asarray(P.decode8bit(codes), dtype=np.float64)

    rows: list[dict] = []
    for i, z in enumerate(C.ZS):
        I = P.propagate(torch.as_tensor(phi_q)[None, None], z)[0, 0].numpy()
        I01 = P.normalize01(I)
        save_gray(os.path.join(out, f"recon_{i + 1}.png"), I01)
        rows.append(dict(plane=i + 1, z_mm=C.ZS_MM[i],
                         ssim=float(M.ssim(targets[i], I01)),
                         pearson=float(M.pcc(targets[i], I01)),
                         ssim_raw=float(M.ssim(targets[i], I)),
                         pearson_raw=float(M.pcc(targets[i], I))))

    summary = dict(
        dataset=os.path.relpath(src, ROOT), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        checkpoint=os.path.relpath(args.ckpt, ROOT), steps=int(steps), gain=float(gain),
        schedule=str(pk.get("schedule", "decay")), seconds=secs,
        device=str(dev), seed=C.SEED, planes=rows,
        mean=dict(ssim=float(np.mean([r["ssim"] for r in rows])),
                  pearson=float(np.mean([r["pearson"] for r in rows]))),
        doe=dict(path=os.path.relpath(doe_path, ROOT), size=[C.N, C.N],
                 bit_depth=8, mode="L", codes_min=int(codes.min()), codes_max=int(codes.max())),
        pipeline="U-Net → K шагов физики внутри сети → квантование 8 бит → реконструкции",
    )
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    if not args.quiet:
        for r in rows:
            print(f"z={r['z_mm']:.1f} мм  SSIM {r['ssim']:.4f}  Pearson {r['pearson']:.4f}")
        print(f"средние: SSIM {summary['mean']['ssim']:.4f}  "
              f"Pearson {summary['mean']['pearson']:.4f}  ({secs:.1f} с, K={steps})")
        print(f"файлы: {out}/DOE.png, recon_1..4.png, metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
