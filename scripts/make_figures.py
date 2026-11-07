"""Визуализации для отчёта и дашборда (matplotlib, без интерактива).

Запуск:  python scripts/make_figures.py

Создаёт figures/*.png:
  targets.png    — четыре объекта;
  doe.png        — фаза NN/GS (циклическая палитра) + 8-битное представление;
  recon.png      — цели / реконструкции NN / реконструкции GS по плоскостям;
  training.png   — кривые обучения (loss, val PCC, val SSIM);
  zstack.png     — кадры z-свипа вокруг плоскостей;
  ssim_maps.png  — локальные SSIM-карты;
  degradation.png— деградация битности 8→6→4→2.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import _path  # noqa: F401
from hdoe import config as C                           # noqa: E402
from hdoe import metrics as M                          # noqa: E402
from hdoe import physics as P                          # noqa: E402
from hdoe import targets as T                          # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "#0d1424", "axes.facecolor": "#0d1424",
    "savefig.facecolor": "#0d1424", "text.color": "#e6edf8",
    "axes.labelcolor": "#c8d6ea", "xtick.color": "#8ea1c2", "ytick.color": "#8ea1c2",
    "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#22304d",
})

CYCLIC = LinearSegmentedColormap.from_list(
    "cyclic", ["#f0f0f0", "#5b8ff9", "#0b1f66", "#00c2a8", "#b8f36b",
               "#f7d154", "#e8642c", "#b3195c", "#f0f0f0"])

FIG = C.PATHS.figures


def save(fig, name: str, dpi: int = 170):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ", os.path.basename(p))


def load_phase(out_dir: str, tag: str):
    from PIL import Image
    name = "DOE.png" if tag == "nn" else "DOE_gs.png"
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        return None, None
    v = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    return P.decode8bit(v), v


def main() -> int:
    out_dir = C.PATHS.out
    targets = T.load_targets()
    metrics = json.load(open(os.path.join(out_dir, "metrics.json"))) \
        if os.path.exists(os.path.join(out_dir, "metrics.json")) else {}
    phi_nn, v_nn = load_phase(out_dir, "nn")
    phi_gs, v_gs = load_phase(out_dir, "gs")

    # ------------------------------------------------------------- цели
    fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.6))
    for i in range(4):
        ax[i].imshow(targets[i], cmap="gray", vmin=0, vmax=1)
        ax[i].set_title(f"T{i + 1}: z = {C.ZS_MM[i]:.1f} мм", fontsize=9, color="#cfe0f7")
        ax[i].axis("off")
    fig.suptitle("Объектные изображения 128×128 (энергии выровнены)", color="#e6edf8", fontsize=10)
    save(fig, "targets.png")

    # -------------------------------------------------------------- ДОЭ
    if phi_nn is not None:
        fig, ax = plt.subplots(2, 2, figsize=(6.4, 5.4))
        for j, (ph, v, name) in enumerate(((phi_nn, v_nn, "NN"), (phi_gs, v_gs, "GS"))):
            if ph is None:
                continue
            im = ax[0, j].imshow(ph, cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
            ax[0, j].set_title(f"φ ({name}), рад", color="#cfe0f7", fontsize=9)
            ax[0, j].axis("off")
            plt.colorbar(im, ax=ax[0, j], fraction=0.046, ticks=[-np.pi, 0, np.pi])
            ax[1, j].imshow(v, cmap="gray", vmin=0, vmax=255)
            ax[1, j].set_title(f"8 бит ({name}): 0 ↔ −π, 255 ↔ +π", color="#cfe0f7", fontsize=9)
            ax[1, j].axis("off")
        fig.suptitle("Фазовые маски ДОЭ", color="#e6edf8")
        save(fig, "doe.png")

        # гистограмма фаз
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.6))
        ax[0].hist(phi_nn.ravel(), bins=80, color="#4dd6c1", alpha=0.85)
        ax[0].set_title("распределение фазы NN, рад"); ax[0].set_xlabel("φ, рад")
        ax[1].hist(v_nn.ravel(), bins=256, color="#7aa2ff")
        ax[1].set_title("распределение кодов 0…255"); ax[1].set_xlabel("код")
        save(fig, "phase_hist.png")

    # ------------------------------------------------------ реконструкции
    rows, labels = [], []
    for tag, phi in (("nn", phi_nn), ("gs", phi_gs)):
        if phi is None:
            continue
        phi_t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
        rr = [P.normalize01(P.propagate(phi_t, z)[0, 0].numpy()) for z in C.ZS]
        rows.append(rr)
        labels.append(tag.upper())
    if rows:
        fig, ax = plt.subplots(1 + len(rows), 4, figsize=(9.2, 2.4 * (1 + len(rows))))
        for i in range(4):
            ax[0, i].imshow(targets[i], cmap="gray", vmin=0, vmax=1)
            ax[0, i].set_title(f"цель T{i + 1}", fontsize=9, color="#cfe0f7")
            ax[0, i].axis("off")
        for r, (rr, lab) in enumerate(zip(rows, labels), start=1):
            for i in range(4):
                ax[r, i].imshow(rr[i], cmap="gray", vmin=0, vmax=1)
                plane = metrics.get("planes", [{}] * 4)[i]
                m = plane if lab == "NN" else (metrics.get("gs", {}).get("planes", [{}] * 4)[i])
                ax[r, i].set_title(f"{lab} z{i + 1}: PCC {m.get('pcc', float('nan')):.3f}, "
                                   f"SSIM {m.get('ssim', float('nan')):.3f}",
                                   fontsize=7.6, color="#cfe0f7")
                ax[r, i].axis("off")
        fig.suptitle("Реконструкции из квантованной маски (мин–макс нормировка)",
                     color="#e6edf8")
        save(fig, "recon.png")

        # SSIM-карты
        fig, ax = plt.subplots(2, 4, figsize=(9.4, 4.6))
        for r, (tag, phi) in enumerate((("NN", phi_nn), ("GS", phi_gs))):
            if phi is None:
                continue
            phi_t = torch.as_tensor(phi, dtype=torch.float64)[None, None]
            for i, z in enumerate(C.ZS):
                I = P.normalize01(P.propagate(phi_t, z)[0, 0].numpy())
                sm = np.clip(M.ssim_map(targets[i], I).numpy(), -0.2, 1)
                im = ax[r, i].imshow(sm, cmap="inferno", vmin=-0.2, vmax=1)
                ax[r, i].set_title(f"{tag} SSIM-карта, z{i + 1}", fontsize=8.5, color="#cfe0f7")
                ax[r, i].axis("off")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        save(fig, "ssim_maps.png")

        # деградация битности
        fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.6))
        for j, b in enumerate((8, 6, 4, 2)):
            nlev = 2 ** b
            q = np.round((phi_nn + np.pi) / (2 * np.pi) * (nlev - 1)) / (nlev - 1) * 2 * np.pi - np.pi
            I = P.propagate(torch.as_tensor(q, dtype=torch.float64)[None, None], C.ZS[1])[0, 0].numpy()
            I01 = P.normalize01(I)
            ax[j].imshow(I01, cmap="gray", vmin=0, vmax=1)
            ax[j].set_title(f"{b} бит: PCC {M.pcc(targets[1], I01):.3f}, "
                            f"SSIM {M.ssim(targets[1], I01):.3f}", fontsize=8, color="#cfe0f7")
            ax[j].axis("off")
        fig.suptitle("Деградация битности фазы (z₂ = 25,3 мм)", color="#e6edf8")
        save(fig, "degradation.png")

    # -------------------------------------------------------- z-свип
    if phi_nn is not None:
        phi_t = torch.as_tensor(phi_nn, dtype=torch.float64)[None, None]
        fig, ax = plt.subplots(2, 5, figsize=(9.6, 4.2))
        zc = C.ZS[1] * 1e3
        dz = np.linspace(-1.2, 1.2, 5)
        for j, d in enumerate(dz):
            z = (zc + d) * 1e-3
            I = P.normalize01(P.propagate(phi_t, z)[0, 0].numpy()) if z >= C.Z_CRIT else None
            ax[0, j].imshow(I if I is not None else np.zeros((C.N, C.N)),
                            cmap="gray", vmin=0, vmax=1)
            ax[0, j].set_title(f"z = {zc + d:.2f} мм", fontsize=8, color="#cfe0f7")
            ax[0, j].axis("off")
        for j, i in enumerate(range(4)):
            I = P.normalize01(P.propagate(phi_t, C.ZS[i])[0, 0].numpy())
            ax[1, j].imshow(I, cmap="gray", vmin=0, vmax=1)
            ax[1, j].set_title(f"плоскость z{i + 1} = {C.ZS_MM[i]:.1f} мм", fontsize=8,
                               color="#cfe0f7")
            ax[1, j].axis("off")
        ax[1, 4].axis("off")
        save(fig, "zstack.png")

    # ------------------------------------------------------- обучение
    log_path = os.path.join(out_dir, "train_log.json")
    if os.path.exists(log_path):
        log = json.load(open(log_path))
        fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.8))
        st = [r["step"] for r in log]
        ax[0].plot(st, [r.get("loss", np.nan) for r in log], color="#4dd6c1", label="loss")
        ax[0].set_yscale("log"); ax[0].set_xlabel("шаг"); ax[0].set_title("лосс (log)")
        ax[0].legend(fontsize=8)
        ax[1].plot(st, [r.get("val_pcc", np.nan) for r in log], color="#7aa2ff", label="val PCC")
        ax[1].plot(st, [r.get("val_ssim", np.nan) for r in log], color="#ffa94d", label="val SSIM")
        ax[1].set_xlabel("шаг"); ax[1].set_title("контрольные метрики (eval)")
        ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
        for a in ax:
            a.grid(alpha=0.15)
        save(fig, "training.png")
    print("фигуры готовы:", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
