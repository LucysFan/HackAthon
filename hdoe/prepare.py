"""Приведение произвольных картинок к формату задачи (128×128, 8 бит, grayscale).

Один модуль на все входы: картинка заказчика, загруженный комплект из четырёх
файлов, «сырой» скриншот любого размера/цвета.

Правила (совпадают с scripts/setup_targets_from_image.py, чтобы дашборд и CLI
давали одинаковый результат):

1. цвет → яркость (ITU-R 601), альфа-канал → композит на белом;
2. авто-кроп по объекту: если фон однотонный, обрезаем поля вокруг содержимого;
3. квадрат по центру, LANCZOS до 128×128;
4. нормировка яркости в [0,1] и отсечение шума < 2 %;
5. инверсия полярности — ДОЭ формирует светящийся объект на тёмном фоне
   (для снимков с белым фоном это обязательно: GS даёт 0.81 против 0.41);
6. выравнивание энергии четырёх плоскостей.

Всё, что делает функцию «умной», собрано в ``prepare``; ``prepare_batch``
дополнительно умеет принимать четыре разных файла одного комплекта.
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image, ImageOps

from . import config as C
from . import targets as T


# --------------------------------------------------------------------- утилиты
def _to_gray(im: Image.Image) -> Image.Image:
    """Цвет/альфа → градации серого на белом фоне."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB") if im.mode not in ("L", "I;16", "F") else im
    g = im.convert("L")
    # EXIF-поворот телефона
    try:
        g = ImageOps.exif_transpose(g)
    except Exception:                                       # noqa: BLE001
        pass
    return g


def auto_crop(g: Image.Image, thr: float = 0.08) -> Image.Image:
    """Обрезка однотонных полей вокруг объекта.

    Работает и для белого фона (объект темнее фона), и для чёрного (объект
    светлее): сравнение идёт с медианой по модулю отклонения.
    """
    a = np.asarray(g, dtype=np.float64) / 255.0
    med = float(np.median(a))
    fg = np.abs(a - med) > thr
    if fg.mean() < 0.002 or fg.mean() > 0.95:   # пусто или «весь кадр объект»
        return g
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(0.04 * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(a.shape[0], y1 + pad), min(a.shape[1], x1 + pad)
    return g.crop((int(x0), int(y0), int(x1), int(y1)))


def preview_steps(src, size: int | None = None) -> list[tuple[str, Image.Image]]:
    """Четыре кадра пути «исходник → цель» для анимации на дашборде.

    Возвращает пары (подпись, картинка) одинакового размера (квадрат), чтобы
    дашборд мог показать их подряд без скачков геометрии.
    """
    n = int(size or C.N)
    im0 = load_image(src)
    g = _to_gray(im0)
    # 1. исходник: как есть, с сохранением пропорций на белом поле
    w, h = g.size
    side0 = max(w, h)
    base = Image.new("L", (side0, side0), 0)
    base.paste(g, ((side0 - w) // 2, (side0 - h) // 2))
    steps = [("исходник", base.resize((n, n), Image.LANCZOS))]
    # 2. только яркость (без кропа) — видно, что цвет ушёл
    steps.append(("яркость", base.resize((n, n), Image.LANCZOS)))
    # 3. авто-кроп объекта
    c = auto_crop(g)
    w1, h1 = c.size
    side1 = max(w1, h1)
    cb = Image.new("L", (side1, side1), int(np.median(np.asarray(c))) if c.size else 0)
    cb.paste(c, ((side1 - w1) // 2, (side1 - h1) // 2))
    steps.append(("кроп", cb.resize((n, n), Image.LANCZOS)))
    # 4. цель: квадрат, 128×128, полярность и 8 бит
    a = to_array(g, invert=True, crop=True)
    steps.append(("цель 128×128 · 8 бит", Image.fromarray(T.to_u8(a), mode="L")))
    return steps


def square_center(g: Image.Image) -> Image.Image:
    side = min(g.size)
    return g.crop(((g.width - side) // 2, (g.height - side) // 2,
                   (g.width + side) // 2, (g.height + side) // 2))


def to_array(g: Image.Image, invert: bool = True, crop: bool = True,
             border: int = 0, denoise: float = 0.02) -> np.ndarray:
    """Картинка → float [0,1] N×N с объектной полярностью."""
    if crop:
        g = auto_crop(g)
    g = square_center(g).resize((C.N, C.N), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64) / 255.0
    a = a - a.min()
    a = a / (a.max() + 1e-12)
    if border > 0:
        a[:border] = a[-border:] = 0.0
        a[:, :border] = a[:, -border:] = 0.0
    if invert:
        a = 1.0 - a
        # после инверсии «объект» должен быть ярче; повторно нормируем
        a = a - a.min()
        a = a / (a.max() + 1e-12)
    a[a < denoise] = 0.0
    return a


def prepare(src, *, invert: bool = True, crop: bool = True,
            border: int = 0) -> np.ndarray:
    """Файл/путь/байты → [4,N,N] в [0,1] с равной энергией плоскостей."""
    g = _to_gray(load_image(src))
    a = to_array(g, invert=invert, crop=crop, border=border)
    arr = np.stack([a] * C.N_PLANES)
    arr = T.equalize_energy(arr)
    return arr / (arr.max() + 1e-12)


def prepare_batch(items, *, invert: bool = True, crop: bool = True,
                  border: int = 0) -> np.ndarray:
    """Комплект: 1 картинка (копируется ×4) либо 2–4 разных.

    Если файлов меньше четырёх, они зацикливаются; если больше — берутся
    первые четыре. Энергия плоскостей выравнивается, как требует п. 5.4.1.
    """
    imgs = [to_array(_to_gray(load_image(it)), invert=invert, crop=crop,
                     border=border) for it in items][:C.N_PLANES]
    if not imgs:
        raise ValueError("пустой комплект")
    while len(imgs) < C.N_PLANES:
        imgs.append(imgs[len(imgs) % len(items)])
    arr = T.equalize_energy(np.stack(imgs[:C.N_PLANES]))
    return arr / (arr.max() + 1e-12)


def load_image(src) -> Image.Image:
    """Путь, байты или file-like → PIL.Image (RGB/L, без EXIF-поворота)."""
    if isinstance(src, Image.Image):
        return ImageOps.exif_transpose(src)
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src)))
    if hasattr(src, "read"):
        return Image.open(src)
    return Image.open(os.fspath(src))


def save_set(arr: np.ndarray, out_dir: str) -> list[str]:
    """Сохранить четыре цели комплекта (PNG 8 бит, режим L)."""
    os.makedirs(out_dir, exist_ok=True)
    return T.save_targets(arr, out_dir)


def preview_grid(src, out_path: str, size: int | None = None) -> str:
    """Сохранить полосу шагов подготовки (для отчёта и проверки глазами)."""
    steps = preview_steps(src, size)
    n = steps[0][1].size[0]
    tile = Image.new("L", (n * len(steps) + 6 * (len(steps) - 1), n), 0)
    for i, (_name, im) in enumerate(steps):
        tile.paste(im if im.size == (n, n) else im.resize((n, n), Image.LANCZOS),
                   (i * (n + 6), 0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tile.save(out_path)
    return out_path


def save_preview(arr: np.ndarray, path: str) -> str:
    """Полоса «источник → 4 цели → края» для анимации преобразования."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = arr.shape[0]
    tile = Image.new("L", (C.N * n + 4 * (n - 1), C.N), 0)
    for i in range(n):
        tile.paste(Image.fromarray(T.to_u8(arr[i]), mode="L"), (i * (C.N + 4), 0))
    tile.save(path)
    return path
