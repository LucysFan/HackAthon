"""Картинки для README: компактные, светлые, читаемые в тёмной теме GitHub.

Запуск:  python scripts/make_readme_figures.py

Делает шесть файлов в ``docs/``:

``task.png``      что подаётся и что получается: четыре цели → маска → четыре
                  восстановления, с числами на каждой плоскости;
``splits.png``    метрики на тренировке / валидации / тесте (столбики + разброс);
``training.png``  ход обучения: лосс и контрольные метрики;
``masks.png``     маска сети и маска GS рядом, плюс гистограмма кодов;
``zstack.png``    свип по z: как изображение собирается и рассыпается;
``explain.png``   путь входной картинки: исходник, яркость, кроп, цель;
``corpus.png``    обучающая выборка: девять семейств и живые кадры корпуса.

Все подписи по-русски, размеры — под ширину README (~1100 px).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C             # noqa: E402
from hdoe import metrics as M            # noqa: E402
from hdoe import physics as P            # noqa: E402
from hdoe import prepare as PR           # noqa: E402
from hdoe import targets as T            # noqa: E402

DOCS = os.path.join(ROOT, "docs")
plt.rcParams.update({
    "figure.autolayout": False,
    "savefig.pad_inches": 0.14,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": "#1b2430", "axes.labelcolor": "#3a4658", "axes.edgecolor": "#c8d2e0",
    "xtick.color": "#5a6b80", "ytick.color": "#5a6b80",
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "font.family": "DejaVu Sans",
})
CYCLIC = LinearSegmentedColormap.from_list(
    "cyclic", ["#f2f2f2", "#5b8ff9", "#12235e", "#00bfa5", "#b8f36b",
               "#f7d154", "#e8642c", "#b3195c", "#f2f2f2"])
ACC, ACC2 = "#0f9b8e", "#3f6fd8"


def save(fig, name: str, dpi: int = 150) -> str:
    os.makedirs(DOCS, exist_ok=True)
    p = os.path.join(DOCS, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  ", os.path.relpath(p, ROOT))
    return p


def phase_from_png(path: str) -> np.ndarray:
    from PIL import Image
    v = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return np.asarray(P.decode8bit(v), dtype=np.float64), v


def metrics() -> dict:
    p = os.path.join(ROOT, "outputs", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def splits() -> dict:
    p = os.path.join(ROOT, "outputs", "splits.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------------------ 1. задача
def fig_task(targets: np.ndarray, phi: np.ndarray, code: np.ndarray, met: dict):
    """Верхний ряд — что подаётся; нижний — что получается из одной маски."""
    planes = met.get("planes") or [{}] * C.N_PLANES
    fig, ax = plt.subplots(2, 4, figsize=(10.6, 6.6),
                           gridspec_kw=dict(hspace=0.32, wspace=0.06))
    for i in range(4):
        ax[0, i].imshow(targets[i], cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(f"вход {i + 1}\nz = {C.ZS_MM[i]:.1f} мм", fontsize=10)
        ax[0, i].axis("off")
        I = P.normalize01(P.propagate(torch.as_tensor(phi, dtype=torch.float64)[None, None],
                                      C.ZS[i])[0, 0].numpy())
        ax[1, i].imshow(I, cmap="gray", vmin=0, vmax=1)
        m = planes[i]
        ax[1, i].set_title(f"выход {i + 1} из маски\nSSIM {m.get('ssim', float('nan')):.3f}"
                           f" · Pearson {m.get('pcc', float('nan')):.3f}", fontsize=10)
        ax[1, i].axis("off")
    mean = met.get("mean") or {}
    fig.suptitle("Четыре изображения → один файл 128×128 (8 бит) → четыре изображения "
                 f"на четырёх расстояниях\nсреднее SSIM {mean.get('ssim', 0):.3f} · "
                 f"Pearson {mean.get('pcc', 0):.3f}", fontsize=11.5)
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)

    # отдельная полоса: маска в двух видах
    fig2, ax2 = plt.subplots(1, 3, figsize=(9.4, 3.1))
    ax2[0].imshow(targets[0], cmap="gray", vmin=0, vmax=1)
    ax2[0].set_title("вход: изображение 128×128", fontsize=10); ax2[0].axis("off")
    im = ax2[1].imshow(phi, cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
    ax2[1].set_title("фаза маски, рад", fontsize=10); ax2[1].axis("off")
    plt.colorbar(im, ax=ax2[1], fraction=0.046, ticks=[-np.pi, 0, np.pi])
    ax2[2].imshow(code, cmap="gray", vmin=0, vmax=255)
    ax2[2].set_title("сдаваемый файл: коды 0…255", fontsize=10); ax2[2].axis("off")
    fig2.tight_layout()
    save(fig2, "doe_file.png")
    return save(fig, "task.png")


# ----------------------------------------------------------------- 2. сплиты
def fig_splits(sp: dict):
    if not sp:
        return None
    names = [("train", "тренировка"), ("val", "валидация"), ("test", "тест")]
    ssim = [sp[k]["summary"]["ssim"] for k, _ in names]
    pcc = [sp[k]["summary"]["pcc"] for k, _ in names]
    ns = [sp[k]["summary"]["n"] for k, _ in names]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    x = np.arange(len(names))
    for a, vals, title, color in ((ax[0], ssim, "SSIM (основная метрика)", ACC),
                                  (ax[1], pcc, "Pearson (дополнительная)", ACC2)):
        bars = a.bar(x, vals, width=0.55, color=color, alpha=0.9)
        for b, v, n in zip(bars, vals, ns):
            a.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
                   fontsize=9.5, fontweight="bold")
            a.text(b.get_x() + b.get_width() / 2, 0.03, f"n = {n}", ha="center",
                   fontsize=8, color="white")
        a.set_xticks(x, [t for _, t in names])
        a.set_ylim(0, 1)
        a.set_title(title)
        a.grid(axis="y", alpha=0.25)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Один и тот же прямой проход модели на трёх наборах", fontsize=11)
    fig.tight_layout()
    return save(fig, "splits.png")


# --------------------------------------------------------------- 3. обучение
def fig_training(sp: dict):
    """Лосс и контрольные метрики по ходу дистилляции + итог на тесте."""
    p = os.path.join(ROOT, "outputs", "train_log.json")
    if not os.path.exists(p):
        return None
    log = json.load(open(p))
    st = [r["step"] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.4),
                           gridspec_kw=dict(wspace=0.22))
    ax[0].plot(st, [r["loss"] for r in log], color=ACC, marker="o", ms=3.5, lw=1.8)
    ax[0].set_title("лосс дистилляции")
    ax[0].set_xlabel("шаг")
    ax[0].grid(alpha=0.25)

    ax[1].plot(st, [r["val_ssim"] for r in log], color=ACC, marker="o", ms=3.5,
               lw=1.8, label="контроль SSIM")
    ax[1].plot(st, [r["val_pcc"] for r in log], color=ACC2, marker="s", ms=3.5,
               lw=1.8, label="контроль Pearson")
    if sp and sp.get("test"):
        t = sp["test"]["summary"]
        ax[1].axhline(t["ssim"], ls="--", lw=1.2, color=ACC, alpha=0.9,
                      label=f"тест SSIM {t['ssim']:.3f}")
        ax[1].axhline(t["pcc"], ls=":", lw=1.2, color=ACC2, alpha=0.9,
                      label=f"тест Pearson {t['pcc']:.3f}")
    ax[1].set_title("контроль во время обучения и итог на тесте")
    ax[1].set_xlabel("шаг")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=7.4, loc="upper left", framealpha=0.95, ncol=1)
    ax[1].grid(alpha=0.25)
    for a in ax:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.set_axisbelow(True)
    ax[1].text(0.01, 0.03,
               "во время обучения физический слой — 4 шага,\n"
               "итоговые метрики — с K = 2048 шагов",
               transform=ax[1].transAxes, fontsize=7.4, color="#5a6b80", va="bottom")
    fig.suptitle("Обучение: дистилляция учительских масок, 1600 шагов", fontsize=11)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98)
    return save(fig, "training.png")


# ------------------------------------------------------------------- 4. маски
def fig_masks(phi_nn, code_nn, phi_gs, code_gs):
    """Маска в трёх видах плюс распределение кодов. Заголовок — один на фигуру."""
    fig = plt.figure(figsize=(9.8, 5.8))
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.16)
    for j, (ph, cd, name) in enumerate(((phi_nn, code_nn, "сеть"),
                                        (phi_gs, code_gs, "Герчберг–Саксон"))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(ph, cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{name}: фаза, рад", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045, ticks=[-3, 0, 3], pad=0.02)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(cd, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}: файл 8 бит", fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[:, 2])
    bins = np.arange(0, 257, 8)
    ax.hist(code_nn.ravel(), bins=bins, color=ACC, histtype="step", lw=1.6, label="сеть")
    ax.hist(code_gs.ravel(), bins=bins, color=ACC2, histtype="step", lw=1.2, label="GS")
    ax.set_ylim(0, 400)
    ax.set_title("распределение фазовых кодов", fontsize=10)
    ax.set_xlabel("код 0…255")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle("Маска, которую сдают: 128×128, 8 бит, grayscale", fontsize=11.5)
    fig.subplots_adjust(top=0.9, bottom=0.06, left=0.02, right=0.97)
    return save(fig, "masks.png")


# -------------------------------------------------------------------- 5. z
def fig_zstack(phi):
    fig, ax = plt.subplots(1, 5, figsize=(11.4, 2.8))
    zc = C.ZS_MM[1]
    for j, d in enumerate(np.linspace(-1.0, 1.0, 5)):
        z = (zc + d) * 1e-3
        if z < C.Z_CRIT:
            ax[j].imshow(np.zeros((C.N, C.N)), cmap="gray")
        else:
            I = P.normalize01(P.propagate(torch.as_tensor(phi, dtype=torch.float64)[None, None],
                                          z)[0, 0].numpy())
            ax[j].imshow(I, cmap="gray", vmin=0, vmax=1)
        ax[j].set_title(f"z = {zc + d:.1f} мм", fontsize=9)
        ax[j].axis("off")
    fig.suptitle("Дифракция Френеля: изображение живёт только на своём расстоянии",
                 fontsize=11, y=1.06)
    return save(fig, "zstack.png")


# ------------------------------------------------------------------ 6. вход
def fig_explain():
    src = os.path.join(ROOT, "data", "source", "object.png")
    if not os.path.exists(src):
        return None
    steps = PR.preview_steps(src, size=C.N)
    fig, ax = plt.subplots(1, len(steps), figsize=(11.2, 3.0))
    for a, (name, im) in zip(ax, steps):
        a.imshow(np.asarray(im.resize((C.N, C.N))), cmap="gray")
        a.set_title(name, fontsize=10)
        a.axis("off")
    fig.suptitle("Любая картинка на входе приводится к формату задачи сама", fontsize=11, y=1.04)
    fig.tight_layout()
    return save(fig, "explain.png")


def fig_corpus():
    """Что именно лежит в обучающей выборке: девять семейств и живые кадры корпуса.

    Верхняя половина — по одному случайному образцу каждого семейства
    (``hdoe.corpus.FAMILIES``), то есть весь алфавит пятен, из которого
    собираются четвёрки. Нижняя — 32 кадра, сгенерированные ровно так же, как
    ``build_corpus``: одна-две фигуры плюс мелкая деталь на тёмном фоне.
    Реальный ``data/corpus.npy`` не нужен: генератор детерминирован по зерну,
    поэтому картинка воспроизводится на чистом репозитории.
    """
    from hdoe import corpus as CO
    rng = np.random.default_rng(C.SEED + 5)
    X, Y, R = CO._grid()
    N = C.N
    cols = 8
    names = ["кольцо", "диск", "крест", "рамка", "штрихи",
             "спираль точек", "пятна", "многоугольник", "сетка точек"]
    tile = np.zeros((len(CO.FAMILIES) * N, cols * N), np.uint8)
    for r, fam in enumerate(CO.FAMILIES):
        for c in range(cols):
            t = fam(X, Y, R, rng, val=1.0)
            t = np.clip((t - 0.15) / 0.85, 0, 1)
            t[t < 0.05] = 0.0
            tile[r * N:(r + 1) * N, c * N:(c + 1) * N] = np.round(t * 255).astype(np.uint8)

    rng2 = np.random.default_rng(C.SEED)
    live = np.zeros((4 * N, cols * N), np.uint8)
    for k in range(32):
        rr, cc = divmod(k, cols)
        live[rr * N:(rr + 1) * N, cc * N:(cc + 1) * N] = np.round(CO.draw_object(rng2) * 255).astype(np.uint8)

    fig = plt.figure(figsize=(7.9, 12.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[9, 4], hspace=0.10)
    a0 = fig.add_subplot(gs[0])
    a0.imshow(tile, cmap="gray", vmin=0, vmax=255)
    for r in range(len(CO.FAMILIES)):
        a0.axhline(r * N - 0.5, color="#7f8fa6", lw=0.6, alpha=0.7)
        a0.text(-6, r * N + N / 2, names[r], ha="right", va="center",
                fontsize=8, color="#3a4658")
    a0.set_xlim(-84, cols * N - 0.5)
    a0.set_title("Девять семейств объектов, из которых собирается корпус "
                 "(по 8 случайных образцов)", fontsize=10, loc="left")
    a0.axis("off")

    a1 = fig.add_subplot(gs[1])
    a1.imshow(live, cmap="gray", vmin=0, vmax=255)
    a1.set_title("32 кадра обучающего корпуса — ровно как их видит сеть: "
                 "1–2 фигуры + мелкая деталь, тёмный фон, заполнение ~19 %",
                 fontsize=10, loc="left")
    a1.axis("off")
    return save(fig, "corpus.png")


def main() -> int:
    targets = T.load_targets()
    met = metrics()
    sp = splits()
    out_dir = C.PATHS.out
    phi_nn, code_nn = phase_from_png(os.path.join(out_dir, "DOE.png"))
    gs_path = os.path.join(out_dir, "DOE_gs.png")
    phi_gs, code_gs = phase_from_png(gs_path) if os.path.exists(gs_path) else (phi_nn, code_nn)
    print("картинки для README:")
    fig_task(targets, phi_nn, code_nn, met)
    fig_splits(sp)
    fig_training(sp)
    fig_masks(phi_nn, code_nn, phi_gs, code_gs)
    fig_zstack(phi_nn)
    fig_explain()
    fig_corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())
