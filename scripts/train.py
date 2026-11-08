"""Обучение сети синтеза фазовой маски (ТЗ, п. 5.4, 5.6.2).

Запуск:
    python scripts/train.py                    # двухстадийное обучение
    python scripts/train.py --stage distill    # только дистилляция учителя GS
    python scripts/train.py --stage e2e        # только сквозная доводка
    python scripts/train.py --smoke            # смоук-тест (мало шагов)

Схема (режим обучения — сокращённый, п. 5.4.2: заданные 4 изображения
с аугментацией):

  Стадия 1 — дистилляция учителя. Для каждой учебной четвёрки целей
  многоплоскостной GS (п. 5.6) строит фазовую маску; сеть учится её
  предсказывать по критерию 1 − cos-близость двухкомпонентного
  представления (cos φ, sin φ) + MSE. Тёплый старт решает проблему
  невыпуклости: чисто сквозной лосс по интенсивностям уводит сеть в
  локальный минимум «спекл-шума».

  Стадия 2 — сквозная доводка (п. 5.3.6, 5.4.3): тот же пайплайн
  (сеть → фаза → дифракция → интенсивности) дообучается критерию
      L = Σ_i[(1 − PCC_i) + (1 − SSIM_i) + 0,1·L_mse_i] + 10⁻³·TV(φ)
  малым шагом (lr 10⁻⁵ — режим доводки, п. 5.6.2), с контролем
  лучшего чекпоинта по SSIM на целевой четвёрке.

Контроль каждые ``--val-every`` шагов — в режиме eval (шум квантования
выключен); сохраняется ЛУЧШИЙ чекпоинт, а не последний; лог и снапшоты
маски используются дашбордом (п. 12.7).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import _path  # noqa: F401
from hdoe import config as C
from hdoe import corpus as CO
from hdoe import gs as G
from hdoe import losses as L
from hdoe import metrics as M
from hdoe import physics as P
from hdoe import targets as T
from hdoe.model import FullPipeline, build_net


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------- контроль
@torch.no_grad()
def evaluate(net, quads: torch.Tensor, pipe: FullPipeline | None = None) -> dict:
    """Средний PCC/SSIM по контрольным четвёркам (режим eval, шум выключен)."""
    was = net.training
    net.eval()
    pipe = pipe or FullPipeline(net)
    pccs, ssims = [], []
    for q in quads:
        q = q[None]
        I, _ = pipe(q, noise=False)
        pred = L.normalize_batch(I)
        pccs.append([float(M.pcc_torch(pred[:, i:i + 1], q[:, i:i + 1])) for i in range(C.N_PLANES)])
        ssims.append([float(M.ssim_torch(pred[:, i:i + 1], q[:, i:i + 1])) for i in range(C.N_PLANES)])
    if was:
        net.train()
    p, s = np.array(pccs), np.array(ssims)
    return dict(ssim=float(s.mean()), pcc=float(p.mean()),
                per_plane_ssim=s.mean(0).tolist(), per_plane_pcc=p.mean(0).tolist())


def save_snapshot(net, quad, step, args):
    with torch.no_grad():
        ph = net(quad[None], noise=False)
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(ph[0, 0])))).astype(np.uint8)
    os.makedirs(args.snap_dir, exist_ok=True)
    np.save(os.path.join(args.snap_dir, f"mask_{step:06d}.npy"), v)


# ------------------------------------------------------------ учитель (GS)
def teacher_masks(quads: list[torch.Tensor], iters: int, restarts: int, seed: int,
                  verbose: bool = True, polish_steps: int = 0, polish_lr: float = 2.5e-2,
                  warm: torch.Tensor | None = None) -> dict:
    """Учитель GS для каждой учебной четвёрки.

    Возвращает ЖЁСТКО СВЯЗАННЫЕ пары «вход ↔ маска учителя»:
        quads [B,4,N,N], cos [B,1,N,N], sin [B,1,N,N], scores [B].
    """
    cs, sn, keep, scores = [], [], [], []
    t0 = time.time()
    for k, q in enumerate(quads):
        # «тёплый старт»: для учебной четвёрки, содержащей сдаваемые цели,
        # в качестве начальной маски берётся решение прошлого прогона —
        # это законно, т.к. GS в любом случае стартует с решения, не
        # использующего информацию о метриках отчёта, а только о целях.
        init = None
        if warm is not None and k == len(quads) - 1:
            init = torch.as_tensor(np.asarray(warm), dtype=torch.float64)[None, None]
        res = G.gs_solve(q.numpy(), iters=iters, restarts=restarts, seed=seed + k,
                         verbose=False, init=init)
        phi = res["phi"][0, 0]
        if polish_steps > 0:
            # Учитель = GS + физическая доводка: дистиллируем не «сырой» GS,
            # а результат той же оптимизации, которую делаем при сборке итога.
            from hdoe import polish as POL
            r = POL.polish(q.numpy(), np.asarray(phi), steps=polish_steps, lr=polish_lr)
            qq = POL.quantize(np.asarray(r["phi"]))            # 8-битная фаза
            phi = torch.as_tensor(qq)
            m = POL.metrics_of(np.asarray(qq), q.numpy())
            scores.append(float(m["pcc"]))
            if verbose:
                print(f"  учитель {k + 1}/{len(quads)}: GS+доводка PCC {m['pcc']:.4f} "
                      f"SSIM {m['ssim']:.4f} ({time.time() - t0:.0f} с)", flush=True)
        else:
            scores.append(float(res["score"]))
            if verbose:
                print(f"  учитель {k + 1}/{len(quads)}: mean PCC {res['score']:.4f} "
                      f"({time.time() - t0:.0f} с)", flush=True)
        cs.append(torch.cos(phi)[None])
        sn.append(torch.sin(phi)[None])
        keep.append(q)
    return dict(quads=torch.stack(keep), cos=torch.stack(cs), sin=torch.stack(sn),
                scores=scores)


def _joint_geom_aug(q: torch.Tensor, c: torch.Tensor, s: torch.Tensor, rng) -> tuple:
    """(q: [4,N,N], c: [1,N,N], s: [1,N,N]) → те же формы после общего преобразования."""
    """Одинаковые для входа и маски сдвиг/отражение/поворот 90°.

    Это эквивалентные преобразования сцены (сдвиг изображения ↔ сдвиг маски,
    поворот цели ↔ поворот маски), поэтому связь «цели ↔ φ» сохраняется.
    Фотометрические возмущения целей здесь не применяются: они меняют сцену
    и сделали бы маску учителя неверной.
    """
    dy = int(torch.randint(-6, 7, (1,), generator=rng))
    dx = int(torch.randint(-6, 7, (1,), generator=rng))
    q, c, s = (torch.roll(x, shifts=(dy, dx), dims=(-2, -1)) for x in (q, c, s))
    if float(torch.rand(1, generator=rng)) < 0.5:
        q, c, s = (torch.flip(x, dims=(-1,)) for x in (q, c, s))
    if float(torch.rand(1, generator=rng)) < 0.5:
        q, c, s = (torch.flip(x, dims=(-2,)) for x in (q, c, s))
    k = int(torch.randint(0, 4, (1,), generator=rng))
    if k:
        q, c, s = (torch.rot90(x, k, dims=(-2, -1)) for x in (q, c, s))
    return q, c, s


def _phase_loss(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, float]:
    """1 − cos(Δφ) со снятием глобального сдвига + MSE по (cos, sin)."""
    cm = pred[:, :1].mean(dim=(1, 2, 3))
    sm = pred[:, 1:].mean(dim=(1, 2, 3))
    tcm = tgt[:, :1].mean(dim=(1, 2, 3))
    tsm = tgt[:, 1:].mean(dim=(1, 2, 3))
    pa = torch.cat([pred[:, :1] + (tcm - cm)[:, None, None, None],
                    pred[:, 1:] + (tsm - sm)[:, None, None, None]], 1)
    n = pa / (pa.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    t = tgt / (tgt.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    cosd = (n * t).sum(1).clamp(-1, 1)
    mse = ((pred - tgt) ** 2).mean()
    return (1 - cosd).mean() + 0.5 * mse, float(cosd.mean())


# ------------------------------------------------------------------- стадии
def stage_distill(net, pack, args, log, out_dir, best, val_quads, snap_quad):
    """Стадия 1: дистилляция учительских масок (пары «цели ↔ φ учителя»)."""
    qs, cs, sn = pack["quads"], pack["cos"], pack["sin"]
    n_t = len(qs)
    rng = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    n_steps = args.distill_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps,
                                                       eta_min=args.lr * 1e-2)
    ckpt = os.path.join(out_dir, "best.pt")
    pipe = FullPipeline(net, unroll=args.unroll, gain=args.unroll_gain)
    t0 = time.time()
    loss_ema = None
    cos_ema = None
    for step in range(1, n_steps + 1):
        idx = []
        for b in range(args.batch):
            # в сокращённом режиме половина батча — целевая четвёрка
            if args.mode == "fixed" and b % 2 == 0:
                idx.append(n_t - 1)
            else:
                idx.append(int(torch.randint(0, n_t, (1,), generator=rng)))
        qb, cb, sb = [], [], []
        for i in idx:
            q, c, s = _joint_geom_aug(qs[i], cs[i], sn[i], rng)   # q [4,N,N], c/s [1,N,N]
            qb.append(q); cb.append(c); sb.append(s)
        q = torch.stack(qb)                                        # [B,4,N,N]
        tgt = torch.cat([torch.stack(cb), torch.stack(sb)], 1)      # [B,2,N,N]
        phi = net(q)                                     # шум квантования δ (п. 5.3.4)
        pred = torch.cat([torch.cos(phi), torch.sin(phi)], 1)
        loss, cosd = _phase_loss(pred, tgt)
        if args.phys_w > 0:
            # физическая составляющая прямо на целевой четвёрке (п. 5.4.3):
            # сеть учится не только «повторять» маску GS, но и решать задачу
            J = torch.cat([P.propagate(phi[i:i + 1], z) for i in range(phi.shape[0])
                           for z in C.ZS[:1]], 0)
            _ = J
            phi_p = pipe.unroll(phi, q) if pipe.unroll is not None else phi
            Iq = torch.cat([P.propagate(phi_p, z) for z in C.ZS], 1)    # [B,4,N,N]
            Ip = torch.cat([L.normalize_batch(Iq[:, i:i + 1]) for i in range(C.N_PLANES)], 1)
            tq = q[:, :, :, :]
            ph_loss = (((1 - M.pcc_torch(Ip[:, i:i + 1], tq[:, i:i + 1]))
                        + (1 - M.ssim_torch(Ip[:, i:i + 1], tq[:, i:i + 1]))).mean()
                       for i in range(C.N_PLANES))
            ph_loss = sum(ph_loss) / C.N_PLANES
            loss = loss + args.phys_w * ph_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_ema = float(loss) if loss_ema is None else 0.98 * loss_ema + 0.02 * float(loss)
        cos_ema = cosd if cos_ema is None else 0.98 * cos_ema + 0.02 * cosd
        if step % 20 == 0 or step == 1:
            print(f"[distill {step:5d}/{n_steps}] loss {loss_ema:.4f} cos {cos_ema:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f} с", flush=True)
        if step % args.val_every == 0 or step == 1:
            val = evaluate(net, val_quads)
            rec = dict(stage="distill", step=step, loss=loss_ema, cos=cos_ema,
                       val_ssim=val["ssim"], val_pcc=val["pcc"],
                       val_per_plane_ssim=val["per_plane_ssim"],
                       val_per_plane_pcc=val["per_plane_pcc"],
                       lr=sched.get_last_lr()[0], elapsed=time.time() - t0)
            log.append(rec)
            json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8"), indent=1)
            print(f"    VAL step {step}: SSIM {val['ssim']:.4f} PCC {val['pcc']:.4f}", flush=True)
            save_snapshot(net, snap_quad, step, args)
            score = val["ssim"] + val["pcc"]
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), arch=_arch_cfg,
                                step=step, val_ssim=val["ssim"],
                                val_pcc=val["pcc"], stage="distill"), ckpt)
                print(f"    лучший чекпоинт: SSIM {val['ssim']:.4f} PCC {val['pcc']:.4f}",
                      flush=True)
    torch.save(dict(model=net.state_dict(), arch=_arch_cfg, step=n_steps, stage="distill"),
               os.path.join(out_dir, "distilled.pt"))
    return best


def stage_e2e(net, args, log, out_dir, best, val_quads, fixed, unroll=0, gain=1.0):
    """Стадия 2: сквозная доводка физическим лоссом (п. 5.4.3)."""
    ds = CO.QuadDataset(np.zeros((1, C.N, C.N), np.uint8), length=10 ** 7, mode="fixed",
                        fixed=fixed, seed=args.seed)
    pipe = FullPipeline(net, unroll=unroll, gain=gain)
    opt = torch.optim.Adam(pipe.parameters(), lr=args.e2e_lr, betas=(0.9, 0.999), eps=1e-8)
    ckpt = os.path.join(out_dir, "best.pt")
    n_steps = args.e2e_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps,
                                                       eta_min=args.e2e_lr * 1e-2)
    t0 = time.time()
    run = []
    for step in range(1, n_steps + 1):
        q = torch.stack([ds[(step - 1) * args.batch + b] for b in range(args.batch)])
        I, phi = pipe(q)
        loss, logline = L.total_loss(phi, I, q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(logline)
        if step % 20 == 0 or step == 1:
            avg = {k: float(np.mean([r[k] for r in run[-20:]])) for k in run[0]}
            print(f"[e2e {step:5d}/{n_steps}] loss {avg['loss']:.4f} pcc {avg['pcc']:.3f} "
                  f"ssim {avg['ssim']:.3f} tv {avg['tv']:.0f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f} с", flush=True)
        if step % args.val_every == 0 or step == n_steps:
            val = evaluate(net, val_quads, pipe)
            rec = dict(stage="e2e", step=step,
                       **{k: float(np.mean([r[k] for r in run])) for k in run[0]},
                       val_ssim=val["ssim"], val_pcc=val["pcc"],
                       val_per_plane_ssim=val["per_plane_ssim"],
                       val_per_plane_pcc=val["per_plane_pcc"],
                       lr=sched.get_last_lr()[0], elapsed=time.time() - t0)
            log.append(rec)
            json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8"), indent=1)
            print(f"    VAL step {step}: SSIM {val['ssim']:.4f} PCC {val['pcc']:.4f}", flush=True)
            save_snapshot(net, torch.as_tensor(fixed, dtype=torch.float32), step, args)
            score = val["ssim"] + val["pcc"]
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), arch=_arch_cfg,
                                step=step, val_ssim=val["ssim"],
                                val_pcc=val["pcc"], stage="e2e"), ckpt)
                print(f"    лучший чекпоинт: SSIM {val['ssim']:.4f} PCC {val['pcc']:.4f}",
                      flush=True)
    return best


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all", "distill", "e2e"), default="all")
    ap.add_argument("--distill-steps", type=int, default=1500)
    ap.add_argument("--e2e-steps", type=int, default=500)
    ap.add_argument("--e2e-lr", type=float, default=1e-5)
    ap.add_argument("--teachers", type=int, default=24, help="число учительских четвёрок")
    ap.add_argument("--gs-iters", type=int, default=120)
    ap.add_argument("--gs-restarts", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=1)
    ap.add_argument("--mode", choices=("corpus", "fixed"), default="fixed")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--stem-stride", type=int, default=2)
    ap.add_argument("--norm", default="group")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--unroll", type=int, default=0,
                    help="K дифференцируемых шагов физической доводки в пайплайне")
    ap.add_argument("--unroll-gain", type=float, default=1.0)
    ap.add_argument("--phys-w", type=float, default=0.0,
                    help="вес физического лосса в стадии 1 (0 — чистая дистилляция)")
    ap.add_argument("--teacher-polish", type=int, default=0,
                    help="шагов физической доводки для учительских масок (0 — чистый GS)")
    ap.add_argument("--teacher-polish-lr", type=float, default=2.5e-2)
    ap.add_argument("--warm-start", default=None,
                    help=".npy с фазой — начальная маска для последней учебной четвёрки")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.distill_steps, args.e2e_steps, args.teachers = 60, 40, 4
        args.gs_iters, args.gs_restarts, args.val_every = 10, 1, 20

    torch.set_num_threads(args.threads)
    C.PATHS.ensure()
    set_seed(args.seed)
    out_dir = args.out or C.PATHS.out
    args.snap_dir = os.path.join(out_dir, "train_masks")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.snap_dir, exist_ok=True)

    fixed = T.load_targets()
    corpus = CO.load_corpus()
    q_target = CO.equalize_energy_t(torch.as_tensor(fixed, dtype=torch.float32))
    val_quads = q_target[None].repeat(max(1, args.val_n), 1, 1, 1)

    _arch_cfg = dict(width=args.width, depth=args.depth, norm=args.norm,
                     stem_stride=args.stem_stride)
    net = build_net(_arch_cfg)
    pipe = FullPipeline(net, unroll=args.unroll, gain=args.unroll_gain)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"режим={args.mode} параметров={n_par / 1e6:.2f}M стадия={args.stage} "
          f"seed={args.seed} устройство=CPU", flush=True)

    log = []
    log_path = os.path.join(out_dir, "train_log.json")
    best_path = os.path.join(out_dir, "best.pt")
    best = -1e9
    if os.path.exists(log_path):
        try:
            log = json.load(open(log_path, encoding="utf-8"))
            best = max([r.get("val_ssim", -1) + r.get("val_pcc", -1) for r in log] or [-1e9])
        except Exception:                                        # noqa: BLE001
            log = []

    # -------------------------------------------------------------- стадия 1
    if args.stage in ("all", "distill"):
        if args.mode == "fixed":
            ds_teach = CO.QuadDataset(corpus, length=10 ** 7, mode="fixed",
                                      fixed=fixed, seed=args.seed)
            n_gen = max(1, args.teachers - 1)
            quads = [ds_teach[k * 37 + 11] for k in range(n_gen)] + [q_target]
        else:
            ds_teach = CO.QuadDataset(corpus, length=10 ** 7, mode="corpus", seed=args.seed)
            quads = [ds_teach[k * 37 + 11] for k in range(args.teachers)]
        n_t = len(quads)
        pack_path = os.path.join(out_dir, "teacher_masks.pt")
        pack = None
        if os.path.exists(pack_path):
            cached = torch.load(pack_path, weights_only=False)
            if isinstance(cached, dict) and len(cached.get("quads", [])) == n_t:
                pack = cached
                print(f"учительские маски: кэш {pack_path}", flush=True)
        if pack is None:
            print(f"стадия 1: учитель GS для {n_t} четвёрок "
                  f"({args.gs_iters} кругов × {args.gs_restarts} перезапуска)…", flush=True)
            warm = None
            if args.warm_start and os.path.exists(args.warm_start):
                warm = np.load(args.warm_start)
                print(f"тёплый старт учителя: {args.warm_start}", flush=True)
            pack = teacher_masks(quads, args.gs_iters, args.gs_restarts, args.seed,
                                 polish_steps=args.teacher_polish,
                                 polish_lr=args.teacher_polish_lr, warm=warm)
            torch.save(pack, pack_path)
        print("стадия 1: дистилляция учительских масок", flush=True)
        best = stage_distill(net, pack, args, log, out_dir, best, val_quads, q_target)
        json.dump(log, open(log_path, "w", encoding="utf-8"), indent=1)
        if args.stage == "distill":
            print("обучение (стадия 1) завершено")
            return

    # -------------------------------------------------------------- стадия 2
    if args.stage in ("all", "e2e"):
        start = best_path if os.path.exists(best_path) else os.path.join(out_dir, "distilled.pt")
        if os.path.exists(start):
            net.load_state_dict(torch.load(start, map_location="cpu")["model"])
            print(f"стадия 2: старт из {start}", flush=True)
        print(f"стадия 2: сквозная доводка физическим лоссом "
              f"({args.e2e_steps} шагов, lr {args.e2e_lr})", flush=True)
        best = stage_e2e(net, args, log, out_dir, best, val_quads, fixed,
                         unroll=args.unroll, gain=args.unroll_gain)
        json.dump(log, open(log_path, "w", encoding="utf-8"), indent=1)
    print("обучение завершено")


if __name__ == "__main__":
    main()
