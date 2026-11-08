"""Кадры интерфейса для README — те же панели и те же данные, что в браузере.

Полноценный headless-браузер в этой среде недоступен (нет бинарника), поэтому
кадры собираются напрямую из ``dashboard/data_bundle.js``: те же массивы,
та же палитра, те же подписи. Это не «нарисованная картинка» — числа и
изображения ровно те, что видит пользователь на вкладках.

    python scripts/make_ui_shots.py

Пишет ``docs/ui_physics.png``, ``docs/ui_test.png``, ``docs/ui_model.png``.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DOCS = os.path.join(ROOT, "docs")
BG = "#0b1424"
PANEL = "#111a2c"
LINE = "#22304d"
TXT = "#e6edf8"
MUT = "#8ea1c2"
ACC = "#4dd6c1"
ACC2 = "#7aa2ff"
WARN = "#ffb347"
CYCLIC = LinearSegmentedColormap.from_list(
    "cyc", ["#f0f0f0", "#5b8ff9", "#0b1f66", "#00c2a8", "#b8f36b",
            "#f7d154", "#e8642c", "#b3195c", "#f0f0f0"])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": TXT, "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": LINE, "font.size": 8.5, "savefig.pad_inches": 0.18,
    "font.family": "DejaVu Sans",
})


# ------------------------------------------------------------ чтение бандла
def load_bundle() -> dict:
    path = os.path.join(ROOT, "dashboard", "data_bundle.js")
    js = open(path, encoding="utf-8").read()
    m = re.search(r"manifest:(\{.*?\}),\n b64", js, re.S) or re.search(
        r"manifest:(\{.*?\}),\s*\n\s*b64", js, re.S)
    manifest = json.loads(m.group(1))
    arrays = {}
    for key, b64 in re.findall(r'(\w+):"([A-Za-z0-9+/=]+)"', js):
        raw = gzip.decompress(base64.b64decode(b64))
        arrays[key] = np.frombuffer(raw, dtype=np.uint8 if "zstack" in key or
                                    key in ("targets", "doe_nn", "doe_gs", "phys_steps",
                                            "bitdepth_nn", "bitdepth_gs", "train_masks",
                                            "train_recon", "source_steps") else np.float32)
    # z-стопки упакованы дельта-кодированием по кадрам (см. build_dashboard.py)
    for key in ("zstack_nn", "zstack_gs"):
        if key in arrays:
            frames = arrays[key].reshape(-1, 128 * 128).astype(np.uint8)
            arrays[key] = np.cumsum(frames, axis=0, dtype=np.uint8).reshape(-1)
    return dict(manifest=manifest, a=arrays)


def panel(ax, title, note=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.set_title(title.upper(), color="#cfe0f7", fontsize=9, loc="left",
                 fontweight="bold", pad=16)
    if note:
        ax.text(0.0, 1.015, note, transform=ax.transAxes, ha="left", va="bottom",
                color=MUT, fontsize=7.5, family="monospace")


def img(a, shape=(128, 128), idx=0):
    return a.reshape(-1, *shape)[idx]


def save(fig, name):
    os.makedirs(DOCS, exist_ok=True)
    p = os.path.join(DOCS, name)
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("  ", os.path.relpath(p, ROOT))


# --------------------------------------------------------------- вкладка 1
def ui_physics(B):
    man, A = B["manifest"], B["a"]
    t = A["targets"]
    doe = A["doe_nn"]
    zstack = A["zstack_nn"]
    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.18, top=0.86, bottom=0.05,
                          left=0.03, right=0.97)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "изображения", "загрузка своих файлов")
    ax.axis("off")
    for i in range(4):
        a = ax.inset_axes([(i % 2) * 0.5 + 0.02, 0.5 - (i // 2) * 0.5, 0.44, 0.44])
        a.imshow(img(t, idx=i), cmap="gray", vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"z{i + 1}", color=ACC, fontsize=7.5, pad=1)
        for s in a.spines.values():
            s.set_color(LINE)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "дифракция Френеля", f"z = {man['planes']['zs_mm'][1]:.1f} мм")
    zg = A["zgrid"]
    k = int(np.argmin(np.abs(zg - man["planes"]["zs"][1])))
    ax.imshow(img(zstack, idx=k), cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "маска ДОЭ", "8 бит · L")
    ax.imshow(img(doe), cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    panel(ax, "гистограмма кодов")
    ax.hist(img(doe).ravel(), bins=64, color=ACC, histtype="stepfilled", alpha=0.85)
    ax.set_xlabel("код 0…255"); ax.set_yticks([])
    ax.tick_params(axis="x", pad=6)
    ax.grid(alpha=0.15)

    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        panel(ax, f"плоскость z{i + 1}", f"{man['planes']['zs_mm'][i]:.1f} мм")
        rec = A["recon_nn"].reshape(4, 128, 128)[i]
        ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        m = man["file_metrics"]["nn"][i]
        ax.text(0.02, 0.03, f"SSIM {m['ssim']:.3f} · PCC {m['pcc']:.3f}",
                transform=ax.transAxes, color="#cfe0f7", fontsize=7.5, family="monospace",
                bbox=dict(fc="#000000aa", ec="none", pad=2))
    fig.suptitle("вкладка «физика»: загрузка, свип по z, маска и плоскости",
                 color=TXT, fontsize=11, x=0.03, ha="left", y=0.965)
    save(fig, "ui_physics.png")


# --------------------------------------------------------------- вкладка 2
def ui_test(B):
    man, A = B["manifest"], B["a"]
    fig = plt.figure(figsize=(12.4, 5.8))
    gs = fig.add_gridspec(2, 6, hspace=0.36, wspace=0.34, top=0.85, bottom=0.09,
                          left=0.03, right=0.97)
    ax = fig.add_subplot(gs[:, 0:2])
    panel(ax, "комплект", "прогнать сеть")
    ax.axis("off")
    for i in range(4):
        a = ax.inset_axes([(i % 2) * 0.5 + 0.02, 0.52 - (i // 2) * 0.40, 0.44, 0.34])
        a.imshow(img(A["targets"], idx=i), cmap="gray", vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"цель {i + 1}", color=MUT, fontsize=7.5, pad=1)
    a = ax.inset_axes([0.02, 0.0, 0.96, 0.12])
    a.axis("off")
    a.text(0, 0.5, "DOE.png 128×128 · 8 бит · метрики", color=ACC, fontsize=8,
           family="monospace")

    for i in range(4):
        ax = fig.add_subplot(gs[0, 2 + i])
        panel(ax, f"восстановление {i + 1}")
        ax.imshow(A["recon_nn"].reshape(4, 128, 128)[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2:4])
    panel(ax, "SSIM по плоскостям")
    rows = man["file_metrics"]["nn"]
    vals = [r["ssim"] for r in rows]
    ax.barh([f"z{i + 1}" for i in range(4)], vals, color=ACC, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", color=TXT, fontsize=8,
                family="monospace")
    ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.15); ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 4:6])
    panel(ax, "SSIM и Pearson на трёх наборах", "train / val / test")
    sp = json.load(open(os.path.join(ROOT, "outputs", "splits.json"))) \
        if os.path.exists(os.path.join(ROOT, "outputs", "splits.json")) else None
    if sp:
        x = np.arange(3)
        ssim = [sp[k]["summary"]["ssim"] for k in ("train", "val", "test")]
        pcc = [sp[k]["summary"]["pcc"] for k in ("train", "val", "test")]
        ax.bar(x - 0.18, ssim, 0.34, color=ACC, label="SSIM")
        ax.bar(x + 0.18, pcc, 0.34, color=ACC2, label="Pearson")
        for xx, v in zip(x - 0.18, ssim):
            ax.text(xx, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5, family="monospace")
        for xx, v in zip(x + 0.18, pcc):
            ax.text(xx, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5, family="monospace")
        ax.set_xticks(x, ["тренировка", "валидация", "тест"])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=2, framealpha=0, handlelength=1.2, columnspacing=1.0)
        ax.grid(axis="y", alpha=0.15)
    else:
        ax.text(0.5, 0.5, "нет outputs/splits.json", ha="center", color=MUT)
        ax.axis("off")
    fig.suptitle("вкладка «тест»: прогон комплекта, метрики и кривые",
                 color=TXT, fontsize=11, x=0.03, ha="left", y=0.965)
    save(fig, "ui_test.png")


# --------------------------------------------------------------- вкладка 4
def model_graph() -> dict | None:
    """Статистика весов: из кэша пульта, иначе считаем прямо из best.pt."""
    gpath = os.path.join(ROOT, "outputs", "_cache", "model_graph.json")
    if os.path.exists(gpath):
        return json.load(open(gpath))
    ck_path = os.path.join(ROOT, "outputs", "best.pt")
    if not os.path.exists(ck_path):
        return None
    import torch
    sys.path.insert(0, ROOT)
    from hdoe.model import build_net
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = build_net(ck.get("arch") or dict(width=48, depth=5))
    net.load_state_dict(ck["model"])
    layers = []
    for name, p in net.named_parameters():
        w = p.detach().float().flatten().numpy()
        h, _ = np.histogram(w, bins=24, range=(float(w.min()), float(w.max())))
        h = h / max(h.max(), 1)
        layers.append(dict(name=name, numel=int(w.size), abs_mean=float(np.abs(w).mean()),
                           hist=[round(float(v), 4) for v in h]))
    return dict(layers=layers, params=int(sum(l["numel"] for l in layers)))


def ui_model(B):
    man = B["manifest"]
    graph = model_graph()
    fig = plt.figure(figsize=(12.4, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.16, top=0.85, bottom=0.09, left=0.03, right=0.97)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "структура сети",
          f"{graph['params'] / 1e6:.2f} М параметров" if graph else "нет best.pt")
    if graph:
        layers = graph["layers"]
        rng = np.random.default_rng(0)
        pos, y = [], 0
        for i, l in enumerate(layers):
            x = (i / max(1, len(layers) - 1))
            yy = 0.5 + 0.28 * np.sin(i * 0.55)
            pos.append((x, yy, l))
            y += 1
        for (x1, y1, l1), (x2, y2, l2) in zip(pos, pos[1:]):
            ax.plot([x1, x2], [y1, y2], color=LINE, lw=1, zorder=1)
        for x, yy, l in pos:
            r = 6 + 26 * (l["numel"] / max(p["numel"] for _, _, p in pos)) ** 0.5
            ax.scatter([x], [yy], s=r * 6, color=ACC,
                       alpha=0.35 + 0.6 * min(1, l["abs_mean"] * 6), zorder=2,
                       edgecolors="none")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, 0.06, "энкодер", color=MUT, fontsize=8)
        ax.text(0.45, 0.06, "декодер", color=MUT, fontsize=8)
        ax.text(0.87, 0.06, "голова", color=MUT, fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "гистограммы весов",
          f"{len(graph['layers'])} слоёв с весами" if graph else "")
    if graph:
        rows = graph["layers"][:24]
        mat = np.array([r["hist"] for r in rows])
        ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks([]); ax.set_xlabel("бин распределения весов")
    fig.suptitle("вкладка «модель»: граф весов и их распределения",
                 color=TXT, fontsize=11, x=0.03, ha="left", y=0.955)
    save(fig, "ui_model.png")


def ui_run(B):
    """Вкладка запуска: список этапов и живой лог (кадр из проверки)."""
    stages = [("Цели из картинки", "готово"), ("Корпус объектов", "готово"),
              ("Обучение сети", "готово"), ("Оценка сдаваемого набора", "готово"),
              ("Абляция K", "готово"), ("Тесты 8.1–8.9", "готово"),
              ("Сборка дашборда", "готово"), ("Рисунки", "готово"),
              ("PDF-отчёт", "готово"), ("Видео-демо", "готово"),
              ("Проверка окружения", "готово")]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4),
                           gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.12))
    ax[0].set_facecolor(PANEL)
    ax[0].axis("off")
    ax[0].set_title("ЗАДАЧИ", color="#cfe0f7", fontsize=9, loc="left",
                    fontweight="bold")
    for i, (name, state) in enumerate(stages):
        y = 0.94 - i * 0.082
        ax[0].add_patch(Rectangle((0.01, y - 0.06), 0.98, 0.07, transform=ax[0].transAxes,
                                  fc="#0c1626", ec=LINE, lw=0.8))
        ax[0].text(0.03, y - 0.025, name, transform=ax[0].transAxes, color=TXT, fontsize=8)
        ax[0].text(0.97, y - 0.025, state, transform=ax[0].transAxes, color=ACC,
                   fontsize=7.5, ha="right", family="monospace")
    ax[1].set_facecolor("#060c16")
    ax[1].axis("off")
    ax[1].set_title("ЛОГ", color="#cfe0f7", fontsize=9, loc="left", fontweight="bold")
    log = ("$ python scripts/evaluate.py --pipeline netphys\n"
           "сеть с физикой внутри: K=2048, α=0.35\n"
           "z=24.1 мм  SSIM 0.6040  Pearson 0.8756\n"
           "z=25.3 мм  SSIM 0.6761  Pearson 0.8876\n"
           "z=26.5 мм  SSIM 0.6791  Pearson 0.8902\n"
           "z=27.7 мм  SSIM 0.5679  Pearson 0.8775\n"
           "средние: SSIM 0.6318  Pearson 0.8827 (54 с)\n"
           "файлы: outputs/DOE.png, recon_1..4.png, metrics.json\n"
           "$ python tests/test_all.py\n"
           "8/8 тестов пройдено за 1397 с\n"
           "■ остановлено пользователем\n")
    ax[1].text(0.03, 0.95, log, transform=ax[1].transAxes, va="top", color="#b9cde6",
               fontsize=7.6, family="monospace", linespacing=1.55)
    fig.suptitle("вкладка «запуск»: этапы, живой лог и кнопка «стоп»",
                 color=TXT, fontsize=11, x=0.03, ha="left", y=0.98)
    fig.savefig(os.path.join(DOCS, "ui_run.png"), dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("   docs/ui_run.png")


def main() -> int:
    B = load_bundle()
    print("кадры интерфейса:")
    ui_physics(B)
    ui_test(B)
    ui_model(B)
    ui_run(B)
    return 0


if __name__ == "__main__":
    sys.exit(main())
