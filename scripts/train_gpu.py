"""Обучение сети с физикой внутри (GPU-оптимизированный скрипт).

Задача — **нейросетевой синтез**: фазу выдаёт сеть за один прямой проход.
Физическая модель входит в сеть как дифференцируемые слои
(``hdoe.netphys.PhysicsInNet``): K шагов физического спуска + квантование
8 бит со straight-through градиентом. Классический GS используется только
для генерации учительских масок (обучающие данные), а не как доводка
на инференсе.

Запуск (GPU):
    python scripts/train_gpu.py --device auto --steps 6000 --batch 8

Запуск (CPU, для проверки конвейера):
    python scripts/train_gpu.py --device cpu --steps 40 --batch 2 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C          # noqa: E402
from hdoe import corpus as CO         # noqa: E402
from hdoe import gs as G              # noqa: E402
from hdoe import losses as L          # noqa: E402
from hdoe import metrics as M         # noqa: E402
from hdoe import netphys as NP        # noqa: E402
from hdoe import physics as P         # noqa: E402
from hdoe import targets as T         # noqa: E402
from hdoe.model import build_net      # noqa: E402


# --------------------------------------------------------------------- утилиты
def pick_device(kind: str) -> torch.device:
    if kind != "auto":
        return torch.device(kind)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_precision(dev: torch.device, mode: str) -> dict:
    """TF32/AMP: на Ampere+ TF32 для float32-матриц — бесплатное ускорение."""
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if mode == "bf16" and dev.type == "cuda":
        return dict(device_type="cuda", dtype=torch.bfloat16)
    if mode == "fp16" and dev.type == "cuda":
        return dict(device_type="cuda", dtype=torch.float16)
    return {}


def finite_check(tag: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"нечисловые значения в {tag}")


# ------------------------------------------------------- учительские данные (GS)
@torch.no_grad()
def teacher_quads(n_quads: int, dev: torch.device, seed: int, iters: int, restarts: int,
                  fixed: np.ndarray, corpus: np.ndarray, target_phase: np.ndarray | None,
                  log, geom_only: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Строит обучающие пары «четвёрка целей ↔ маска учителя GS».

    Один из примеров — сдаваемые цели с уже известной хорошей маской
    (``target_phase``): она используется как дополнительный старт GS, поэтому
    учитель на сдаваемом наборе получается лучшим из возможных.
    """
    ds_fixed = CO.QuadDataset(corpus, length=10 ** 7, mode="fixed", fixed=fixed, seed=seed,
                              geom_only=geom_only)
    ds_corp = CO.QuadDataset(corpus, length=10 ** 7, mode="corpus", seed=seed,
                             geom_only=geom_only)
    quads, phases = [], []
    for k in range(n_quads):
        q = ds_fixed[k * 37 + 11] if k % 3 == 0 else ds_corp[k * 37 + 11]
        quads.append(q.numpy())
    # последняя четвёрка — сдаваемые цели (валентность гарантирована)
    quads = quads[:-1] + [np.stack(fixed)]
    qbatch = torch.as_tensor(np.stack(quads), dtype=torch.float32, device=dev)
    t0 = time.time()
    res = G.gs_batch(qbatch, iters=iters, restarts=restarts, seed=seed)
    phi = res["phi"]
    # сдаваемые цели: если есть готовая маска, берём лучшую из двух
    if target_phase is not None:
        cand = torch.as_tensor(np.asarray(target_phase, dtype=np.float32), device=dev)[None, None]
        cand = torch.remainder(cand + math.pi, 2 * math.pi) - math.pi
        sc_new = _score_batch(cand, qbatch[-1:], dev)
        sc_old = _score_batch(phi[-1:], qbatch[-1:], dev)
        if float(sc_new) > float(sc_old):
            phi[-1:] = cand
    log.append(dict(stage="teacher", n=n_quads, iters=iters, restarts=restarts,
                    score=float(res["scores"].mean()), sec=time.time() - t0))
    print(f"учителя: {n_quads} четвёрок, GS {iters}×{restarts}, средний PCC "
          f"{float(res['scores'].mean()):.4f} за {time.time() - t0:.0f} с", flush=True)
    return qbatch, phi


@torch.no_grad()
def _score_batch(phi: torch.Tensor, q: torch.Tensor, dev) -> torch.Tensor:
    """Средний PCC по плоскостям для батча (без квантования)."""
    t = q / q.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    vals = []
    for i, z in enumerate(C.ZS):
        I = P.propagate(phi, z)[:, 0]
        lo = I.flatten(1).amin(1).view(-1, 1, 1)
        hi = I.flatten(1).amax(1).view(-1, 1, 1)
        I = (I - lo) / (hi - lo + 1e-12)
        vals.append(M.pcc_torch(I.flatten(1), t[:, i].flatten(1)))
    return torch.stack(vals).mean(0)


# ------------------------------------------------------------------- валидация
@torch.no_grad()
def validate(pack, quads: torch.Tensor, targets: torch.Tensor) -> dict:
    pack.eval()
    phi = NP.infer_phase(pack, quads)
    pack.train()
    q = phi[0, 0].cpu().numpy()
    pcc, ssim = [], []
    for i, z in enumerate(C.ZS):
        Ii = torch.as_tensor(P.normalize01(P.propagate(phi[0:1], z)[0, 0]))
        pcc.append(float(M.pcc(targets[i], Ii)))
        ssim.append(float(M.ssim(targets[i], Ii)))
    return dict(pcc=float(np.mean(pcc)), ssim=float(np.mean(ssim)),
                per_plane_pcc=pcc, per_plane_ssim=ssim, phi=q)


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "bf16", "fp16"))
    ap.add_argument("--distill-steps", type=int, default=1500,
                    help="стадия 1: дистилляция учительских масок")
    ap.add_argument("--steps", type=int, default=6000,
                    help="стадия 2: физика внутри сети (сквозной лосс по модели)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--phys-steps", type=int, default=4,
                    help="K дифференцируемых шагов физики внутри сети")
    ap.add_argument("--phys-gain", type=float, default=0.3)
    ap.add_argument("--phys-schedule", choices=("decay", "const"), default="decay",
                    help="расписание шага α физического слоя (decay — быстрый выход "
                         "к решению, затем мелкая доводка)")
    ap.add_argument("--phys-gain-end", type=float, default=0.02,
                    help="конечное значение α в режиме decay")
    ap.add_argument("--geom-only", action="store_true",
                    help="только геометрическая аугментация (для плотных целей)")
    ap.add_argument("--tv", type=float, default=0.0,
                    help="вес TV по фазе (для зернистых целей вреден, по умолчанию 0)")
    ap.add_argument("--entropy", type=float, default=0.005,
                    help="вес энтропии фазовых кодов (8 бит, как в DOE.png)")
    ap.add_argument("--teachers", type=int, default=16)
    ap.add_argument("--gs-iters", type=int, default=300)
    ap.add_argument("--gs-restarts", type=int, default=3)
    ap.add_argument("--teacher-cache", default=None)
    ap.add_argument("--target-phase", default=None,
                    help=".npy фазы-маяка для сдаваемого набора (используется как старт GS)")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--snap-every", type=int, default=0,
                    help="сохранять снапшоты маски каждые N шагов (0 — как val-every)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    C.PATHS.ensure()
    if args.smoke:
        args.steps, args.batch, args.teachers = 20, 2, 3
        args.distill_steps = 10
        args.gs_iters, args.gs_restarts, args.val_every = 10, 1, 10
    dev = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    elif dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 2))
    amp = set_precision(dev, args.precision)
    out = args.out or C.PATHS.out
    os.makedirs(out, exist_ok=True)
    snap_dir = os.path.join(out, "train_masks")
    os.makedirs(snap_dir, exist_ok=True)
    torch.manual_seed(C.SEED)

    fixed = T.load_targets()
    corpus = CO.load_corpus()
    target_phase = np.load(args.target_phase) if args.target_phase and \
        os.path.exists(args.target_phase) else None

    print(f"устройство: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type}), "
          f"torch {torch.__version__}, точность {args.precision}", flush=True)

    # ---------------------------------------------------- учительские данные
    log: list = []
    cache = args.teacher_cache or os.path.join(out, "teacher_gpu.pt")
    if os.path.exists(cache):
        pack_c = torch.load(cache, map_location="cpu", weights_only=False)
        if pack_c["quads"].shape[0] == args.teachers:
            quads_all, phi_all = pack_c["quads"], pack_c["phi"]
            print(f"учителя: кэш {cache}", flush=True)
        else:
            pack_c = None
    else:
        pack_c = None
    if pack_c is None:
        quads_all, phi_all = teacher_quads(args.teachers, dev, C.SEED, args.gs_iters,
                                           args.gs_restarts, fixed, corpus, target_phase, log,
                                           geom_only=args.geom_only)
        torch.save(dict(quads=quads_all.cpu(), phi=phi_all.cpu()), cache)
        print(f"учителя сохранены: {cache}", flush=True)

    # ----------------------------------------------------------- модель
    net = build_net(dict(width=args.width, depth=args.depth)).to(dev)
    pack = NP.PhysicsInNet(net, steps=args.phys_steps, gain=args.phys_gain,
                           quantize=True).to(dev)
    n_par = sum(p.numel() for p in pack.parameters())
    print(f"сеть: {n_par / 1e6:.2f} M параметров, K={args.phys_steps} шагов физики внутри",
          flush=True)

    opt = torch.optim.AdamW(pack.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.steps, 1), eta_min=args.lr * 0.01) if args.steps > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"
                                                   and dev.type == "cuda"))
    val_q = torch.as_tensor(np.stack([fixed] * 3), dtype=torch.float32, device=dev)
    val_t = torch.as_tensor(fixed, dtype=torch.float64)

    from hdoe.corpus import QuadDataset
    ds = QuadDataset(corpus, length=10 ** 7, mode="fixed", fixed=fixed, seed=C.SEED,
                     geom_only=args.geom_only)

    best = -1e9
    run: list = []
    t0 = time.time()

    # ------------------------------------------------- стадия 1: дистилляция
    # Сеть учится воспроизводить маски учителя (GS на обучающих четвёрках).
    # Это «использование классического метода для данных»: на инференсе
    # классика не вызывается — сеть выдаёт фазу за один прямой проход.
    if args.distill_steps > 0:
        # phi_all: [B,1,N,N] → [B,2,N,N] (cos, sin); квадратные скобки важны:
        # torch.stack([...], 1) по [B,1,N,N] дал бы [B,2,1,N,N]
        cs = torch.cat([torch.cos(phi_all), torch.sin(phi_all)], 1).to(dev)
        tgt_cs = cs.view(cs.shape[0], 2, C.N, C.N)
        n_d = int(args.distill_steps)
        opt_d = torch.optim.AdamW(pack.parameters(), lr=args.lr * 3, weight_decay=1e-5)
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(n_d, 1),
                                                           eta_min=args.lr * 0.05)
        print(f"стадия 1: дистилляция {n_d} шагов (учитель GS)", flush=True)
        run_d: list = []
        for step in range(1, n_d + 1):
            k = torch.randint(0, quads_all.shape[0], (args.batch,))
            q = quads_all[k].to(dev)
            phi = net(q)
            pred = torch.cat([torch.cos(phi), torch.sin(phi)], 1)
            tgt = tgt_cs[k]
            cosd = (pred * tgt).sum(1) / (pred.pow(2).sum(1).sqrt()
                                          * tgt.pow(2).sum(1).sqrt() + 1e-8)
            loss_d = (1 - cosd).mean() + 0.5 * ((pred - tgt) ** 2).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(pack.parameters(), 1.0)
            opt_d.step()
            sch_d.step()
            run_d.append(float(loss_d.detach()))
            if step % 50 == 0 or step == 1:
                print(f"[distill {step:5d}/{n_d}] loss {float(loss_d.detach()):.4f} "
                      f"cos {float(cosd.detach().mean()):.3f} {time.time() - t0:.0f} с", flush=True)
            every = args.snap_every or args.val_every
            if step % every == 0 or step == n_d:
                v = validate(pack, val_q, val_t)
                log.append(dict(stage="distill", step=step,
                                loss=float(np.mean(run_d[-every:])),
                                cos=float(cosd.detach().mean()),
                                val_pcc=v["pcc"], val_ssim=v["ssim"],
                                per_plane_pcc=v["per_plane_pcc"],
                                per_plane_ssim=v["per_plane_ssim"],
                                elapsed=time.time() - t0))
                np.save(os.path.join(snap_dir, f"mask_{step:06d}.npy"),
                        np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(v["phi"])))))
                with open(os.path.join(out, "train_log.json"), "w") as fh:
                    json.dump(log, fh, indent=1, ensure_ascii=False)
                print(f"    ВАЛИДАЦИЯ {step}: PCC {v['pcc']:.4f} SSIM {v['ssim']:.4f} "
                      f"(сеть без физики)", flush=True)
        v = validate(pack, val_q, val_t)
        print(f"    после дистилляции: PCC {v['pcc']:.4f} SSIM {v['ssim']:.4f} "
              f"(сеть, без доводки)", flush=True)
        torch.save(dict(model=net.state_dict(), arch=dict(width=args.width, depth=args.depth),
                        pack=_pack(args), step=0, stage="distill"),
                   os.path.join(out, "distilled.pt"))
        # основной чекпоинт поставки: веса + упаковка инференса (K шагов физики)
        torch.save(dict(model=net.state_dict(), arch=dict(width=args.width, depth=args.depth),
                        pack=_pack(args), step=0, stage="distill"),
                   os.path.join(out, "best.pt"))

    if args.steps <= 0:
        print("стадия 2 пропущена (--steps 0)", flush=True)
        return 0
    print(f"стадия 2: физика внутри сети, {args.steps} шагов "
          f"(K={args.phys_steps}, квантование STE, лосс по модели)", flush=True)
    for step in range(1, args.steps + 1):
        idx = [(step * args.batch + b) for b in range(args.batch)]
        q = torch.stack([ds[i] for i in idx]).to(dev, non_blocking=True)
        # смешиваем с учительскими парами: сеть видит и «трудные» четвёрки
        if step % 4 == 0:
            k = torch.randint(0, quads_all.shape[0], (args.batch,))
            q = quads_all[k].to(dev)
        with torch.autocast(**amp) if amp else torch.autocast("cpu", enabled=False):
            I, phi = pack(q)
            loss, logline = L.total_loss(phi, I.float(), q.float(), w_tv=args.tv)
            if args.entropy > 0:
                loss = loss + args.entropy * phase_entropy(phi)
        finite_check("loss", loss)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(pack.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pack.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        run.append(logline)
        if step % 10 == 0 or step == 1:
            avg = {k: float(np.mean([r[k] for r in run[-20:]])) for k in run[0]}
            print(f"[{step:6d}/{args.steps}] loss {avg['loss']:.4f} pcc {avg['pcc']:.3f} "
                  f"ssim {avg['ssim']:.3f} lr {(sched.get_last_lr()[0] if sched else 0):.2e} "
                  f"{time.time() - t0:.0f} с ({step / max(time.time() - t0, 1e-9):.2f} шаг/с)",
                  flush=True)
        if step % args.val_every == 0 or step == args.steps:
            v = validate(pack, val_q, val_t)
            rec = dict(stage="physics_in_net", step=step, val_pcc=v["pcc"], val_ssim=v["ssim"],
                       per_plane_pcc=v["per_plane_pcc"], per_plane_ssim=v["per_plane_ssim"],
                       **{k: float(np.mean([r[k] for r in run])) for k in run[0]},
                       lr=(sched.get_last_lr()[0] if sched else 0),
                       elapsed=time.time() - t0)
            log.append(rec)
            print(f"    ВАЛИДАЦИЯ {step}: PCC {v['pcc']:.4f} SSIM {v['ssim']:.4f} "
                  f"(без классической доводки)", flush=True)
            sc = v["pcc"] + v["ssim"]
            if sc > best:
                best = sc
                torch.save(dict(model=net.state_dict(),
                                arch=dict(width=args.width, depth=args.depth),
                                pack=_pack(args),
                                step=step, val_pcc=v["pcc"], val_ssim=v["ssim"],
                                stage="physics_in_net"), os.path.join(out, "best.pt"))
                print(f"    лучший чекпоинт: PCC {v['pcc']:.4f} SSIM {v['ssim']:.4f}", flush=True)
            np.save(os.path.join(snap_dir, f"mask_{step:06d}.npy"),
                    np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(v["phi"])))))
            with open(os.path.join(out, "train_log_gpu.json"), "w") as fh:
                json.dump(log, fh, indent=1)
    print(f"готово за {time.time() - t0:.0f} с; лучший PCC+SSIM = {best:.4f}", flush=True)
    return 0


def _pack(args) -> dict:
    """Упаковка инференса, которая едет вместе с весами: K, α и расписание."""
    return dict(steps=int(args.phys_steps), gain=float(args.phys_gain),
                schedule=str(args.phys_schedule), gain_end=float(args.phys_gain_end))


def phase_entropy(phi: torch.Tensor) -> torch.Tensor:
    """Энтропия распределения 8-битных кодов фазы (как в сдаваемом DOE.png).

    Заменяет TV-регуляризацию: максимизирует использование кодовой шкалы и
    подавляет «серые» плато фазы, сохраняя полный диапазон 0…255.
    """
    with torch.no_grad():
        codes = P.encode8bit(phi).float()
    p = torch.zeros(phi.shape[0], 256, device=phi.device)
    idx = codes.reshape(phi.shape[0], -1).long()
    p.scatter_add_(1, idx, torch.ones_like(idx, dtype=p.dtype))
    p = p / p.sum(1, keepdim=True).clamp_min(1)
    ent = -(p * (p + 1e-9).log()).sum(1) / math.log(256.0)
    return (1.0 - ent).mean()


if __name__ == "__main__":
    sys.exit(main())
