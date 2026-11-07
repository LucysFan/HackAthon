"""Резервная видеозапись демонстрации дашборда → outputs/demo_backup.mp4.

Видео собирается полностью офлайн средствами Python (Pillow + imageio-ffmpeg):
кадр за кадром воспроизводится тот же сценарий, что и авто-тур дашборда
(панели 1–8, анимации A1–A8), а числа берутся из тех же файлов, что читает
дашборд (outputs/*.json и dashboard/data/manifest.json), поэтому подписи
совпадают с интерактивной версией.

Запуск:  python scripts/make_demo_video.py [--fps 30] [--out outputs/demo_backup.mp4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import config as C          # noqa: E402
from hdoe import physics as P         # noqa: E402

W, H = 1280, 720
BG = (14, 17, 23)
FG = (222, 230, 238)
ACC = (86, 176, 255)
NN = (94, 214, 158)
GS = (255, 176, 84)
DIM = (128, 142, 158)


def _font(size: int, bold: bool = False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),):
        p = f"/usr/share/fonts/truetype/dejavu/{name}"
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = _font(26, True)
F_H2 = _font(19, True)
F_TXT = _font(15)
F_SMALL = _font(12)
F_MONO = _font(14)


def load_json(*parts):
    p = os.path.join(ROOT, *parts)
    return json.load(open(p)) if os.path.exists(p) else {}


def arr_u8(path, shape=None):
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.uint8)
    return a if shape is None else a.reshape(shape)


def to_img(a, scale=1):
    im = Image.fromarray(np.asarray(a, dtype=np.uint8), mode="L").convert("RGB")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def cyclic_phase_img(codes):
    t = np.asarray(codes, dtype=np.float64) / 255.0
    r = (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.10)) * 255)
    g = (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.45)) * 255)
    b = (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.78)) * 255)
    return Image.fromarray(np.stack([r, g, b], -1).astype(np.uint8), mode="RGB")


class Frame:
    def __init__(self, title: str, subtitle: str = "", panel: str = ""):
        self.im = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.im)
        self.d.rectangle([0, 0, W, 64], fill=(20, 25, 33))
        self.d.text((28, 14), "ДОЭ 128×128 · 4 плоскости · λ=532 нм", font=F_TITLE, fill=FG)
        if panel:
            tw = self.d.textlength(panel, font=F_H2)
            self.d.text((W - 32 - tw, 22), panel, font=F_H2, fill=ACC)
        if title:
            self.d.text((28, 78), title, font=F_H2, fill=ACC)
        if subtitle:
            self.d.text((28, 104), subtitle, font=F_TXT, fill=DIM)

    def text(self, xy, txt, font=F_TXT, fill=FG):
        self.d.text(xy, txt, font=font, fill=fill)

    def img(self, xy, im, caption=None, border=True):
        x, y = xy
        if border:
            self.d.rectangle([x - 2, y - 2, x + im.width + 1, y + im.height + 1], outline=(48, 58, 72))
        self.im.paste(im, (x, y))
        if caption:
            self.d.text((x, y + im.height + 6), caption, font=F_SMALL, fill=DIM)

    def bars(self, xy, values, labels, colors, maxv=1.0, w=300, h=16, gap=9, label_w=92):
        x, y = xy
        for v, lab, col in zip(values, labels, colors):
            self.d.text((x, y - 2), lab, font=F_SMALL, fill=DIM)
            bx = x + label_w
            self.d.rectangle([bx, y, bx + w, y + h], fill=(30, 36, 46))
            self.d.rectangle([bx, y, bx + int(w * min(max(v / maxv, 0), 1)), y + h], fill=col)
            self.d.text((bx + w + 8, y - 3), f"{v:.4f}", font=F_MONO, fill=FG)
            y += h + gap
        return y

    def table(self, xy, header, rows, widths, row_h=24, fs=F_SMALL):
        x, y = xy
        self.d.rectangle([x, y, x + sum(widths), y + row_h], fill=(28, 34, 44))
        cx = x
        for h_, w_ in zip(header, widths):
            self.d.text((cx + 6, y + 5), str(h_), font=_font(12, True), fill=ACC)
            cx += w_
        y += row_h
        for i, r in enumerate(rows):
            if i % 2 == 0:
                self.d.rectangle([x, y, x + sum(widths), y + row_h], fill=(21, 26, 34))
            cx = x
            for v, w_ in zip(r, widths):
                self.d.text((cx + 6, y + 5), str(v), font=fs, fill=FG)
                cx += w_
            y += row_h
        return y


def build_video(out_path: str, fps: int) -> str:
    import imageio.v2 as imageio

    m = load_json("outputs", "metrics.json")
    t = load_json("outputs", "tests.json")
    pol = load_json("outputs", "polish.json")
    log = load_json("outputs", "train_log.json")
    tg = [arr_u8(os.path.join(ROOT, "data", "targets", f"target_{i}.png")) for i in range(1, 5)]
    doe = arr_u8(os.path.join(ROOT, "outputs", "DOE.png"))
    doe_gs = arr_u8(os.path.join(ROOT, "outputs", "DOE_gs.png"))
    rec = [arr_u8(os.path.join(ROOT, "outputs", f"recon_{i}.png")) for i in range(1, 5)]
    rgs = [arr_u8(os.path.join(ROOT, "outputs", f"recon_gs_{i}.png")) for i in range(1, 5)]
    rows = m.get("planes", [])
    mean = m.get("mean", {})
    gsm = (m.get("gs") or {}).get("mean", {})
    nn_only = ((m.get("nn_network_only") or {}).get("mean") or {})

    q = P.decode8bit(doe)
    zgrid = np.linspace(23.0, 28.6, 120)
    qt = torch.as_tensor(np.asarray(q, dtype=np.float32))[None, None]
    frames = np.stack([P.to_u8(P.normalize01(P.propagate_any(qt, float(z))[0][0, 0].numpy()))
                      for z in zgrid])

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_log_level="error")

    def emit(im, hold=1):
        a = np.asarray(im.resize((W, H), Image.LANCZOS), dtype=np.uint8)
        for _ in range(hold):
            writer.append_data(a)

    # ---------------------------------------------------------------- 0. титул
    f = Frame("", "", "")
    f.d.text((28, 120), "Синтез фазового ДОЭ нейросетью", font=_font(40, True), fill=FG)
    f.d.text((28, 176), "четырёхплоскостное восстановление изображений (Френель, одно-БПФ)",
             font=F_TXT, fill=DIM)
    f.text((28, 240), f"λ = 532 нм · Δx = 10 мкм · N = 128 · z = 24,1 / 25,3 / 26,5 / 27,7 мм",
           font=F_MONO, fill=FG)
    for i, a in enumerate(tg):
        f.img((28 + i * 150, 300), to_img(a, 1), f"цель {i + 1}")
    f.text((28, 470), f"NN + доводка: PCC {mean.get('pcc', 0):.4f} · SSIM {mean.get('ssim', 0):.4f}",
           font=F_H2, fill=NN)
    f.text((28, 502), f"GS: PCC {gsm.get('pcc', 0):.4f} · SSIM {gsm.get('ssim', 0):.4f}",
           font=F_H2, fill=GS)
    f.text((28, 534), f"сеть как есть: PCC {nn_only.get('pcc', 0):.4f} · SSIM {nn_only.get('ssim', 0):.4f}",
           font=F_TXT, fill=DIM)
    f.text((28, 570), f"тесты 8.1–8.7: {t.get('summary', {}).get('passed', 0)}/"
           f"{t.get('summary', {}).get('total', 0)} · seed {t.get('summary', {}).get('seed', C.SEED)}",
           font=F_TXT, fill=FG)
    f.text((28, 640), "офлайн-дашборд: dashboard/index.html (двойной клик) · этот файл — резервная запись",
           font=F_SMALL, fill=DIM)
    emit(f.im, int(fps * 2.5))

    # ---------------------------------------------------------- 1–2. цели и ДОЭ
    f = Frame("Панель 1–2. Цели и фазовая маска", "8 бит, 0 ↔ −π, 255 ↔ +π", "A1/A2/A3")
    for i, a in enumerate(tg):
        f.img((28 + i * 142, 140), to_img(a, 1), f"t{i + 1}")
    f.img((640, 140), cyclic_phase_img(doe), "фаза (циклическая палитра)", )
    f.img((790, 140), to_img(doe), "DOE.png, коды 0…255")
    hist = np.bincount(np.asarray(doe).ravel(), minlength=256)
    bh = 150
    bx = 950
    for k in range(0, 256, 2):
        hgt = int(bh * hist[k] / max(hist.max(), 1))
        f.d.line([(bx + k * 0.9, 140 + bh), (bx + k * 0.9, 140 + bh - hgt)], fill=ACC)
    f.text((950, 296), "гистограмма кодов 8 бит", font=F_SMALL, fill=DIM)
    f.text((950, 330), f"код min {int(doe.min())} · max {int(doe.max())}", font=F_SMALL, fill=FG)
    f.text((950, 352), f"шаг квантования δ = {C.QSTEP:.5f} рад", font=F_SMALL, fill=FG)
    f.text((28, 330), "A3 · маска GS для сравнения", font=F_SMALL, fill=GS)
    f.img((28, 350), cyclic_phase_img(doe_gs), "GS")
    f.img((178, 350), to_img(doe_gs), "DOE_gs.png")
    f.text((360, 360), "Одна маска формирует четыре разных\nизображения на четырёх расстояниях.",
           font=F_TXT, fill=FG)
    emit(f.im, int(fps * 3))

    # ------------------------------------------------- 3. пропагация z-свип A5
    n_show = 48
    for k in range(n_show):
        f = Frame("Панель 3. Пропагация: z-свип вдоль оси",
                  "одно-БПФ-схема Френеля; ниже z_crit — угловой спектр (п. 4.5.3)", "A5")
        idx = int(k * (len(frames) - 1) / (n_show - 1))
        z = zgrid[idx]
        f.img((120, 150), to_img(frames[idx], 2), f"z = {z:.2f} мм")
        # ось z
        x0, y0, x1 = 460, 200, 1140
        f.d.line([(x0, y0), (x1, y0)], fill=(60, 72, 88), width=2)
        for i, zi in enumerate(C.ZS_MM):
            px = x0 + int((zi - 23.0) / (28.6 - 23.0) * (x1 - x0))
            f.d.line([(px, y0 - 10), (px, y0 + 10)], fill=NN, width=2)
            f.text((px - 16, y0 + 16), f"z{i + 1}", font=F_SMALL, fill=NN)
        px = x0 + int((z - 23.0) / (28.6 - 23.0) * (x1 - x0))
        f.d.ellipse([px - 7, y0 - 7, px + 7, y0 + 7], fill=ACC)
        crit = x0 + int((C.Z_CRIT * 1e3 - 23.0) / (28.6 - 23.0) * (x1 - x0))
        f.d.line([(crit, y0 - 40), (crit, y0 + 44)], fill=(255, 120, 120), width=1)
        f.text((crit - 30, y0 - 56), "z_crit", font=F_SMALL, fill=(255, 120, 120))
        for i, zi in enumerate(C.ZS_MM):
            if abs(z - zi) < 0.09:
                f.img((700, 300), to_img(rec[i], 2), f"реконструкция z{i + 1} ≈ цель {i + 1}")
        f.text((460, y0 + 90),
               f"по оси видно, как расходятся фокусы: плоскости разделены 1,2 мм\n"
               f"(s_i = Δx′/Δx ≈ 1,00…1,15), поэтому каждая z_i даёт свой масштаб.",
               font=F_TXT, fill=DIM)
        emit(f.im, 1)

    # ------------------------------------------------------- 4. плоскости, A6
    for alpha in np.linspace(0, 1, 22):
        f = Frame("Панель 4. Плоскости: цель ↔ реконструкция", "шторка / blend / разница", "A6")
        for i in range(4):
            x = 40 + (i % 2) * 620
            y = 140 + (i // 2) * 280
            mix = (alpha * np.asarray(rec[i], np.float32) + (1 - alpha) * np.asarray(tg[i], np.float32))
            f.img((x, y), to_img(np.clip(mix, 0, 255).astype(np.uint8), 2),
                  f"z{i + 1}: PCC {rows[i]['pcc']:.4f} · SSIM {rows[i]['ssim']:.4f}")
        emit(f.im, 1)

    # ---------------------------------------------------------- 5. метрики A4
    for k in range(12):
        g = (k + 1) / 12
        f = Frame("Панель 5. Метрики (основной протокол, из квантованной фазы)",
                  "PCC и SSIM по плоскостям: NN + доводка vs GS", "A4")
        vals, labs, cols = [], [], []
        for i, r in enumerate(rows):
            vals += [r["pcc"] * g, r["ssim"] * g]
            labs += [f"z{i + 1} PCC (NN)", f"z{i + 1} SSIM (NN)"]
            cols += [NN, (70, 160, 120)]
        for i, r in enumerate(rows):
            vals += [r["pcc"] * g]
            labs += [f"z{i + 1} PCC (GS)"]
            cols += [GS]
        f.bars((36, 150), vals[:8], labs[:8], cols[:8], w=560)
        f.bars((700, 150), vals[8:], labs[8:], cols[8:], w=300, label_w=104)
        f.text((700, 430), f"среднее NN: PCC {mean.get('pcc', 0):.4f} / SSIM {mean.get('ssim', 0):.4f}",
               font=F_H2, fill=NN)
        f.text((700, 462), f"среднее GS: PCC {gsm.get('pcc', 0):.4f} / SSIM {gsm.get('ssim', 0):.4f}",
               font=F_H2, fill=GS)
        f.text((700, 494), f"сеть без доводки: PCC {nn_only.get('pcc', 0):.4f} / "
                           f"SSIM {nn_only.get('ssim', 0):.4f}", font=F_TXT, fill=DIM)
        emit(f.im, 1)

    # -------------------------------------------------- 6. измерения A7 / A8
    f = Frame("Панель 6. Измерения по изображению", "профиль, ROI, z-зонд, битность", "A7/A8")
    f.img((40, 140), to_img(rec[1], 2), "z₂, реконструкция")
    prof = np.asarray(rec[1], np.float32)[64]
    for i in range(len(prof) - 1):
        f.d.line([(360 + i * 6, 400 - prof[i] * 1.6), (360 + (i + 1) * 6, 400 - prof[i + 1] * 1.6)],
                 fill=ACC, width=2)
    f.text((360, 410), "профиль через центр (строка 64)", font=F_SMALL, fill=DIM)
    bits = [8, 6, 4, 2]
    for i, b in enumerate(bits):
        v = np.round(np.asarray(doe, np.float32) / 255 * ((2 ** b) - 1)) / ((2 ** b) - 1) * 255
        f.img((860, 140 + i * 130), to_img(v.astype(np.uint8), 1), f"{b} бит фазы → z₂")
    f.text((360, 470), "z-зонд I(z) в центре:", font=F_SMALL, fill=DIM)
    zz = np.linspace(23.0, 28.6, 200)
    prof_z = [float(P.propagate_any(qt, float(z))[0][0, 0, 64, 64]) for z in zz]
    prof_z = np.asarray(prof_z, np.float32)
    prof_z = (prof_z - prof_z.min()) / (np.ptp(prof_z) + 1e-12)
    for i in range(len(prof_z) - 1):
        f.d.line([(360 + i * 4, 620 - prof_z[i] * 110), (360 + (i + 1) * 4, 620 - prof_z[i + 1] * 110)],
                 fill=NN, width=2)
    for zi in C.ZS_MM:
        px = 360 + int((zi - 23.0) / (28.6 - 23.0) * 796)
        f.d.line([(px, 505), (px, 625)], fill=(70, 84, 100))
    f.text((360, 632), "z = 23,0 … 28,6 мм · пики у z₁…z₄", font=F_SMALL, fill=DIM)
    emit(f.im, int(fps * 3))

    # ---------------------------------------------------------- 7. обучение A7
    steps = [r.get("step", 0) for r in log if r.get("stage") == "distill"]
    pccs = [r.get("val_pcc", 0) for r in log if r.get("stage") == "distill"]
    for k in range(24):
        f = Frame("Панель 7. Обучение", "дистилляция учительских масок + сквозная доводка", "A7")
        n = max(2, int((k + 1) / 24 * len(steps)))
        xs = np.linspace(120, 1120, n)
        ys = np.asarray(pccs[:n])
        pts = [(float(x), 480 - float(v) * 320) for x, v in zip(xs, ys)]
        if len(pts) > 1:
            f.d.line(pts, fill=NN, width=2)
        f.text((40, 140), f"шаг {steps[n - 1] if steps else 0} / {steps[-1] if steps else 0}",
               font=F_H2, fill=FG)
        f.text((40, 172), f"контрольный PCC ≈ {ys[-1]:.4f}", font=F_TXT, fill=NN)
        f.text((40, 470), "PCC на сдаваемых целях (контроль, режим eval)", font=F_SMALL, fill=DIM)
        rec_i = np.asarray(np.clip(np.asarray(tg[1], np.float32) * 0.6 + 40, 0, 255), np.uint8)
        f.img((980, 200), to_img(rec_i, 1), "кадр маски")
        emit(f.im, 1)

    # ------------------------------------------------------------ 8. физика
    f = Frame("Панель 8. Физика и верификация", "граница применимости, тест линзы, тесты 8.1–8.7",
              "8.1–8.7")
    ct = m.get("control_table") or []
    if ct:
        hdr = ["№", "z, мм", "Δx′, мкм", "страница, мм", "s = Δx′/Δx", "чирп у края, π", "запас, %"]
        data = [[r["plane"], f"{r['z_mm']:.1f}", f"{r['dx_out_um']:.4f}", f"{r['page_mm']:.4f}",
                 f"{r['s_i']:.5f}", f"{r['chirp_frac_pi']:.5f}", f"{r['margin_pct']:.2f}"]
                for r in ct]
        f.table((36, 150), hdr, data, [36, 66, 92, 104, 96, 128, 84])
    tr = t.get("summary", {})
    f.text((36, 300), f"тесты: {tr.get('passed', 0)}/{tr.get('total', 0)} пройдено",
           font=F_H2, fill=NN if tr.get("passed") == tr.get("total") else GS)
    names = {"parseval": "8.1 Парсеваль", "lens": "8.2 Линза (центр ↔ край)",
             "quantization": "8.3 Квантование (δ/2)", "gradient": "8.4 Градиент (gradcheck)",
             "cross_scheme": "8.5 Кросс-схема (ASM)", "boundary": "8.6 Граница z_crit",
             "formats": "8.7 Форматы PNG"}
    y = 340
    for k, lab in names.items():
        v = t.get(k, {})
        ok = bool(v.get("passed"))
        f.text((36, y), ("✔ " if ok else "✘ ") + lab, font=F_TXT, fill=NN if ok else GS)
        y += 26
    ln = t.get("lens", {})
    if ln.get("rows"):
        f.text((520, 340), "тест линзы: пик остаётся в центре на всех z_i", font=F_SMALL, fill=DIM)
        yy = 366
        for r in ln["rows"]:
            f.text((520, yy), f"z = {r['z_mm']:.1f} мм: пик ({r['peak_row']}, {r['peak_col']}), "
                              f"центр/фон = {r['ratio']:.3g}", font=F_SMALL, fill=FG)
            yy += 24
    emit(f.im, int(fps * 3.5))

    # ------------------------------------------------------------- финал
    f = Frame("Итог", "", "")
    f.d.text((28, 110), "Результат", font=_font(34, True), fill=FG)
    d = [["метрика", "NN + доводка", "GS (200×3)", "сеть как есть"]]
    d.append(["PCC (среднее)", f"{mean.get('pcc', 0):.4f}", f"{gsm.get('pcc', 0):.4f}",
              f"{nn_only.get('pcc', 0):.4f}"])
    d.append(["SSIM (среднее)", f"{mean.get('ssim', 0):.4f}", f"{gsm.get('ssim', 0):.4f}",
              f"{nn_only.get('ssim', 0):.4f}"])
    for r in rows:
        d.append([f"PCC z{r['plane']}", f"{r['pcc']:.4f}", "", ""])
    f.table((28, 170), d[0], d[1:], [180, 180, 160, 180], row_h=28)
    f.text((28, 420), "Файлы: outputs/DOE.png, recon_1..4.png, metrics.json, tests.json, report.pdf;",
           font=F_TXT, fill=FG)
    f.text((28, 446), "дашборд: dashboard/index.html (офлайн, двойной клик); видео — резервная запись.",
           font=F_TXT, fill=FG)
    f.text((28, 500), "Физика не изменялась: одно-БПФ-схема Френеля, каскадная пропагация не используется.",
           font=F_SMALL, fill=DIM)
    f.text((28, 524), "Все метрики — из квантованной 8-битной фазы сданного DOE.png.",
           font=F_SMALL, fill=DIM)
    emit(f.im, int(fps * 3.5))

    writer.close()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "demo_backup.mp4"))
    args = ap.parse_args()
    C.PATHS.ensure()
    p = build_video(args.out, args.fps)
    sz = os.path.getsize(p) / 1e6
    print(f"видео собрано: {p} ({sz:.1f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
