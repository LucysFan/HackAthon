"""Оценка и формирование результатов (ТЗ, п. 5.5, 5.6, 7).

Запуск:
    python scripts/evaluate.py [--ckpt outputs/best.pt] [--gs-iters 300]
                               [--gs-restarts 3] [--finetune 0]

Что делает:
  * берёт фазу из сети (режим eval), квантует в 8 бит (п. 4.6), сохраняет DOE.png;
  * строит реконструкции из КВАНТОВАННОЙ фазы (п. 4.6.3) и сохраняет recon_1..4.png;
  * считает метрики PCC/SSIM в двух протоколах (п. 5.5.3, В6) и метрики GS;
  * опционально доводит сеть на заданных изображениях (п. 5.6.2, lr 1e-5);
  * пишет outputs/metrics.json и таблицы для отчёта.
"""
from __future__ import annotations

import argparse
import json
import os
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
from hdoe.model import FullPipeline, build_net

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(2)


# --------------------------------------------------------------- утилиты файлов
def save_gray(path: str, arr01) -> None:
    from PIL import Image
    arr01 = np.squeeze(np.asarray(arr01))
    Image.fromarray(P.to_u8(arr01), mode="L").save(path)


def save_doe(path: str, phi_q) -> None:
    from PIL import Image
    v = np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi_q))))
    Image.fromarray(np.squeeze(v).astype(np.uint8), mode="L").save(path)


def load_doe(path: str) -> np.ndarray:
    from PIL import Image
    return P.decode8bit(np.asarray(Image.open(path).convert("L")))


# ---------------------------------------------------------------- метрики пар
def metrics_tables(targets: np.ndarray, recons: np.ndarray) -> list[dict]:
    rows = []
    for i in range(C.N_PLANES):
        t, r = targets[i], recons[i]
        r01 = P.normalize01(r)
        rows.append(dict(
            plane=i + 1, z_mm=C.ZS_MM[i], dx_out_um=C.OUT_DX[i] * 1e6,
            page_mm=C.PAGE_MM[i],
            pcc=M.pcc(t, r01), ssim=M.ssim(t, r01),
            pcc_raw=M.pcc(t, r), ssim_raw=M.ssim(t, r),
        ))
    return rows


def summarize(rows: list[dict]) -> dict:
    out = dict(pcc=float(np.mean([r["pcc"] for r in rows])),
               ssim=float(np.mean([r["ssim"] for r in rows])))
    if all("pcc_raw" in r for r in rows):
        out["pcc_raw"] = float(np.mean([r["pcc_raw"] for r in rows]))
        out["ssim_raw"] = float(np.mean([r["ssim_raw"] for r in rows]))
    return out


def _pick_device(kind: str = "auto") -> torch.device:
    """Автовыбор устройства: cuda → mps → cpu (код один и тот же)."""
    if kind != "auto":
        return torch.device(kind)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ инференс NN
def net_phase(net, targets: np.ndarray, noise: bool = False) -> np.ndarray:
    net.eval()
    dev = next(net.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(targets, dtype=torch.float32)[None].to(dev)
        phi = net(x, noise=noise)
    return phi[0, 0].detach().float().cpu().numpy()


def finetune(net, targets: np.ndarray, steps: int, lr: float, seed: int) -> None:
    """Доводка сети на заданных изображениях (п. 5.6.2): 300 шагов, lr 1e-5."""
    from hdoe import losses as L
    pipe = FullPipeline(net)
    net.train()
    torch.manual_seed(seed)
    dev = next(net.parameters()).device
    q = torch.as_tensor(targets, dtype=torch.float32)[None].repeat(2, 1, 1, 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.1)
    for s in range(steps):
        boost = 1.0
        I, phi = pipe(q)
        loss, log = L.total_loss(phi, I, q, w_tv=1e-3)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if (s + 1) % 50 == 0:
            print(f"  finetune {s + 1}/{steps}: loss {log['loss']:.4f} "
                  f"pcc {log['pcc']:.3f} ssim {log['ssim']:.3f}")
    net.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--norm", default="group")
    ap.add_argument("--stem-stride", type=int, default=2)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gs-iters", type=int, default=300)
    ap.add_argument("--gs-restarts", type=int, default=3)
    ap.add_argument("--gs-refine", type=int, default=200)
    ap.add_argument("--finetune", type=int, default=0)
    ap.add_argument("--refine", type=int, default=0,
                    help="шагов доводки фазы градиентом физ. лосса (п. 5.6.2)")
    ap.add_argument("--refine-lr", type=float, default=2.5e-2)
    ap.add_argument("--refine-restarts", type=int, default=2,
                    help="перезапусков доводки на каждый старт (разные локальные минимумы)")
    ap.add_argument("--refine-cycles", type=int, default=2,
                    help="дополнительных SGDR-циклов доводки из квантованной фазы")
    ap.add_argument("--gs-it-hop", type=int, default=150, help="круги GS для старта доводки")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"),
                    help="устройство инференса: auto — cuda → mps → cpu")
    ap.add_argument("--pipeline", choices=("netphys", "polish"), default="netphys",
                    help="netphys — физика внутри сети (один прямой проход, без внешней "
                         "доводки); polish — прежняя внешняя доводка для сравнения")
    ap.add_argument("--phys-steps", type=int, default=0,
                    help="K шагов физики внутри сети (0 — взять из чекпоинта)")
    ap.add_argument("--phys-gain", type=float, default=0.0, help="начальный шаг α физики")
    ap.add_argument("--init", choices=("nn", "gs", "blend", "nn+gs"), default="nn+gs",
                    help="начальное приближение для внешней доводки")
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--skip-gs", action="store_true")
    args = ap.parse_args()

    C.PATHS.ensure()
    out_dir = args.out or C.PATHS.out
    os.makedirs(out_dir, exist_ok=True)
    seed = C.SEED
    torch.manual_seed(seed)

    targets = T.load_targets()
    # по умолчанию — сдаваемый чекпоинт outputs/best.pt; резервный вариант —
    # лучший чекпоинт обучения в outputs/checkpoints/best.pt
    ckpt_path = args.ckpt or os.path.join(C.PATHS.out, "best.pt")
    if args.ckpt is None and not os.path.exists(ckpt_path):
        alt = os.path.join(C.PATHS.ckpt, "best.pt")
        ckpt_path = alt if os.path.exists(alt) else ckpt_path

    # ---------------------------------------------------------------- сеть
    # архитектура берётся из чекпоинта (если сохранена), иначе — CLI-параметры
    _ck = (torch.load(ckpt_path, map_location="cpu", weights_only=False)
           if os.path.exists(ckpt_path) else {})
    _arch = _ck.get("arch") if isinstance(_ck, dict) else None
    net = build_net(_arch or dict(width=args.width, depth=args.depth,
                                  norm=args.norm, stem_stride=args.stem_stride))
    dev = _pick_device(args.device)
    print(f"устройство: {dev}", flush=True)
    print(f"сеть: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M параметров, "
          f"архитектура {_arch or 'по умолчанию'}", flush=True)
    ckpt_info = dict(path=ckpt_path, exists=False)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        net.to(dev)
        cfg = state.get("config", {})
        ckpt_info = dict(path=os.path.relpath(ckpt_path, C.PATHS.root), exists=True,
                         stage=state.get("stage"), arch=state.get("arch"),
                         pack=state.get("pack"), step=state.get("step"),
                         val_ssim=state.get("val_ssim"),
                         mode=cfg.get("mode"), steps=cfg.get("steps"))
        print(f"checkpoint: {ckpt_info}")
    else:
        print(f"ВНИМАНИЕ: чекпоинт {ckpt_path} не найден — используется необученная сеть")
    if args.finetune > 0:
        print(f"доводка сети на заданных изображениях: {args.finetune} шагов, lr {args.finetune_lr}")
        finetune(net, targets, args.finetune, args.finetune_lr, seed)
        torch.save(dict(model=net.state_dict(), step=ckpt_info.get("step"),
                        config={"mode": "finetune"}), os.path.join(out_dir, "finetuned.pt"))

    # ------------------------------------------------------- ДОЭ и реконструкции
    phys_info = None
    if args.pipeline == "netphys":
        # Сеть с физикой внутри: фаза = прямой проход (U-Net → K шагов
        # дифференцируемого физического слоя → квантование 8 бит).
        # Никакого классического итерационного метода после сети не вызывается.
        from hdoe import netphys as NP
        pk = (_ck.get("pack") or {}) if isinstance(_ck, dict) else {}
        K = args.phys_steps or int(pk.get("steps", 384))
        gain = args.phys_gain or float(pk.get("gain", 0.35))
        t0 = time.time()
        pack = NP.PhysicsInNet(net, steps=K, gain=gain, quantize=True,
                               learnable=False,
                               schedule=str(pk.get("schedule", "decay")),
                               gain_end=float(pk.get("gain_end", 0.02)))
        pack.to(dev)
        pack.eval()
        with torch.no_grad():
            _, phi_t_ = pack(torch.as_tensor(targets, dtype=torch.float32)[None].to(dev))
        phi_nn = phi_t_[0, 0].detach().float().cpu().numpy()
        dt = time.time() - t0
        phys_info = dict(pipeline="netphys", steps=K, gain=gain,
                         schedule=str(pk.get("schedule", "decay")),
                         gain_end=float(pk.get("gain_end", 0.02)),
                         seconds=dt,
                         mode="один прямой проход сети, внешняя доводка не используется")
        print(f"сеть с физикой внутри: K={K}, α={gain}, {dt:.1f} с "
              f"(без внешней доводки)", flush=True)
    else:
        phi_nn = net_phase(net, targets)
    # 1) метрики «сеть как есть» (выход U-Net ДО физического слоя) — честное
    #    разделение «сеть» / «сеть + физика внутри»
    net_only_rows = None
    if args.pipeline == "netphys":
        with torch.no_grad():
            _phi_net = net(torch.as_tensor(targets, dtype=torch.float32)[None].to(dev))[0, 0]
        _phi_net = _phi_net.detach().float().cpu().numpy()
        _v_net = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(_phi_net))))
    else:
        _v_net = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi_nn))))
    phi_net_only = np.asarray(P.decode8bit(_v_net), dtype=np.float32)
    net_only_rows = None
    try:
        from hdoe import polish as POL0
        net_only_rows = POL0.metrics_of(phi_net_only, targets)["rows"]
        print(f"сеть без доводки: PCC {POL0.metrics_of(phi_net_only, targets)['pcc']:.4f} "
              f"SSIM {POL0.metrics_of(phi_net_only, targets)['ssim']:.4f}")
    except Exception as exc:                                   # noqa: BLE001
        print("не удалось посчитать метрики сети без доводки:", exc)
    if args.refine > 0 and args.pipeline == "polish":
        # Внешняя физическая доводка (п. 5.6.2): сеть / GS / круговая смесь как старт,
        # выбор лучшего по протоколу отчёта (из квантованной фазы).
        from hdoe import polish as POL
        from hdoe import gs as G2
        print(f"доводка фазы: {args.refine} шагов, lr {args.refine_lr}, старты: {args.init}")
        cands = {"nn": phi_nn}
        if args.init in ("gs", "blend", "nn+gs", "all"):
            if not args.skip_gs:
                res_g = G2.gs_solve(targets, iters=args.gs_it_hop, restarts=2, seed=seed,
                                    verbose=False)
                gs_phi = res_g["phi"][0, 0].numpy()
            else:
                from PIL import Image
                gs_phi = P.decode8bit(np.asarray(Image.open(
                    os.path.join(out_dir, "DOE_gs.png")).convert("L")))
            from hdoe.refine import blend_phases
            cands["gs"] = gs_phi
            cands["blend"] = blend_phases(phi_nn, gs_phi, 0.5)
        sel = POL.best_start(targets, cands, steps=args.refine, lr=args.refine_lr,
                             restarts=args.refine_restarts)
        # циклы доводки (SGDR): повторная доводка из квантованной фазы
        cycles_info = []
        for c in range(args.refine_cycles):
            from hdoe import polish as POLc
            r = POLc.polish(targets, np.asarray(sel["phi"]), steps=args.refine,
                            lr=args.refine_lr * 0.6 ** (c + 1))
            q_c = POLc.quantize(np.asarray(r["phi"]))
            m_c = POLc.metrics_of(q_c, targets)
            cycles_info.append(dict(cycle=c + 2, pcc=m_c["pcc"], ssim=m_c["ssim"]))
            print(f"    цикл доводки {c + 2}: PCC {m_c['pcc']:.4f} SSIM {m_c['ssim']:.4f}",
                  flush=True)
            if m_c["pcc"] + m_c["ssim"] >= sel["metrics"]["pcc"] + sel["metrics"]["ssim"]:
                sel = dict(sel, phi=q_c, metrics=m_c)
        phi_nn = sel["phi"]
        polish_info = dict(best=sel["best"], steps=args.refine, lr=args.refine_lr,
                           restarts=args.refine_restarts,
                           cycles=cycles_info,
                           starts={k: dict(pcc=v["metrics"]["pcc"], ssim=v["metrics"]["ssim"],
                                           restart=v.get("restart", 0),
                                           rows=v["metrics"]["rows"])
                                   for k, v in sel["results"].items()},
                           net_only=None)
        if net_only_rows is not None:
            polish_info["net_only"] = dict(
                pcc=float(np.mean([r["pcc"] for r in net_only_rows])),
                ssim=float(np.mean([r["ssim"] for r in net_only_rows])), rows=net_only_rows)
        with open(os.path.join(out_dir, "polish.json"), "w", encoding="utf-8") as fh:
            json.dump(polish_info, fh, indent=1, ensure_ascii=False)
    else:
        polish_info = None
    v_nn = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi_nn))))
    assert v_nn.shape == (C.N, C.N), v_nn.shape
    phi_q_nn = P.decode8bit(v_nn)
    doe_path = os.path.join(out_dir, "DOE.png")
    save_doe(doe_path, phi_q_nn)
    phi_q_nn = load_doe(doe_path)                      # читаем обратно: воспроизводимость
    phi_t = torch.as_tensor(phi_q_nn, dtype=torch.float64)[None, None]

    recons_nn = []
    for i, z in enumerate(C.ZS):
        I = P.propagate(phi_t, z)[0, 0].numpy()
        recons_nn.append(I)
        save_gray(os.path.join(out_dir, f"recon_{i + 1}.png"), P.normalize01(I))
    rows_nn = metrics_tables(targets, recons_nn)

    # ---------------------------------------------------------------- GS-бейзлайн
    gs_rows, gs_info, phi_q_gs = [], dict(iters=0, restarts=0), None
    if not args.skip_gs:
        t0 = time.time()
        res = G.gs_solve(targets, iters=args.gs_iters, restarts=args.gs_restarts,
                         seed=seed, verbose=True, refine_iters=args.gs_refine)
        v_gs = np.squeeze(np.asarray(P.encode8bit(res["phi"])))
        assert v_gs.shape == (C.N, C.N), v_gs.shape
        phi_q_gs = P.decode8bit(v_gs)
        save_doe(os.path.join(out_dir, "DOE_gs.png"), phi_q_gs)
        recons_gs = []
        phi_t_gs = torch.as_tensor(phi_q_gs, dtype=torch.float64)[None, None]
        for i, z in enumerate(C.ZS):
            I = P.propagate(phi_t_gs, z)[0, 0].numpy()
            recons_gs.append(I)
            save_gray(os.path.join(out_dir, f"recon_gs_{i + 1}.png"), P.normalize01(I))
        gs_rows = metrics_tables(targets, recons_gs)
        gs_info = dict(iters=args.gs_iters, restarts=args.gs_restarts,
                       refine_iters=args.gs_refine, runs=res["runs"],
                       score=res["score"], elapsed=time.time() - t0)
        np.save(os.path.join(out_dir, "recon_gs.npy"), np.stack(recons_gs))

    # ------------------------------------------------------------------ сводка
    metrics = dict(
        schema="hdoe/metrics/1",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        seed=seed,
        constants=C.summary(),
        control_table=C.control_table(),
        checkpoint=ckpt_info,
        protocol=dict(
            primary="мин–макс нормировка реконструкции к [0,1], затем PCC/SSIM с целью",
            reference="без нормировки (значения интенсивности как есть)",
            ssim=dict(window=C.SSIM_WIN, sigma=C.SSIM_SIGMA, gaussian=True,
                      crop=5, C1=C.SSIM_C1, C2=C.SSIM_C2),
            quantization="фаза квантована в 8 бит (δ=2π/255), реконструкции из DOE.png",
        ),
        planes=rows_nn,
        mean=summarize(rows_nn),
        gs=dict(planes=gs_rows, mean=summarize(gs_rows) if gs_rows else None, **gs_info),
        delta_vs_gs=dict(
            pcc=float(np.mean([r["pcc"] for r in rows_nn]) -
                      (np.mean([r["pcc"] for r in gs_rows]) if gs_rows else 0.0)),
            ssim=float(np.mean([r["ssim"] for r in rows_nn]) -
                       (np.mean([r["ssim"] for r in gs_rows]) if gs_rows else 0.0)),
        ),
        phases=dict(
            nn=dict(vmin=int(v_nn.min()), vmax=int(v_nn.max()),
                    mean=float(phi_q_nn.mean()), std=float(phi_q_nn.std())),
        ),
        pipeline=("сеть (U-Net) с физикой внутри → K шагов дифференцируемого "
                  "физического слоя → квантование 8 бит → реконструкции"
                  if args.pipeline == "netphys" else
                  "сеть → внешняя доводка → квантование 8 бит → реконструкции"),
        phys=phys_info,
        nn_network_only=(dict(planes=net_only_rows,
                              mean=summarize(net_only_rows)) if net_only_rows else None),
    )
    if phi_q_gs is not None:
        metrics["phases"]["gs"] = dict(vmin=int(P.encode8bit(phi_q_gs).min()),
                                       vmax=int(P.encode8bit(phi_q_gs).max()),
                                       mean=float(phi_q_gs.mean()), std=float(phi_q_gs.std()))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, ensure_ascii=False)

    np.save(os.path.join(out_dir, "phase_nn.npy"), phi_q_nn)
    np.save(os.path.join(out_dir, "recon_nn.npy"), np.stack(recons_nn))
    if phi_q_gs is not None:
        np.save(os.path.join(out_dir, "phase_gs.npy"), phi_q_gs)

    print("\nТаблица (NN):")
    for r in rows_nn:
        print(f"  z={r['z_mm']:.1f} мм  PCC={r['pcc']:.4f}  SSIM={r['ssim']:.4f} "
              f"(справ.: PCC={r['pcc_raw']:.4f} SSIM={r['ssim_raw']:.4f})")
    print(f"  средние: PCC={metrics['mean']['pcc']:.4f} SSIM={metrics['mean']['ssim']:.4f}")
    if gs_rows:
        print("Таблица (GS):")
        for r in gs_rows:
            print(f"  z={r['z_mm']:.1f} мм  PCC={r['pcc']:.4f}  SSIM={r['ssim']:.4f}")
        print(f"  средние: PCC={metrics['gs']['mean']['pcc']:.4f} "
              f"SSIM={metrics['gs']['mean']['ssim']:.4f}")
    print(f"\nметрики: {os.path.join(out_dir, 'metrics.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
