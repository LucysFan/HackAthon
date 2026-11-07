"""Финальный набор из 4 разных «богатых» целей + проверка разрешимости."""
import sys, os, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.ndimage import gaussian_filter, zoom
from hdoe import config as C, polish as POL, gs

torch.set_num_threads(2)
N = C.N
n = np.arange(N) - N // 2
X, Y = np.meshgrid(n, n, indexing="ij")
R = np.hypot(X, Y)
PH = np.arctan2(Y, X)


def grain(seed, sigma=2.0, amp=0.4):
    rng = np.random.default_rng(seed * 7919 + 13)
    g = gaussian_filter(rng.standard_normal((N, N)), sigma)
    return 1.0 + amp * g / (np.abs(g).max() + 1e-9)


def _env(r0=58.0, p=4):
    return np.exp(-(R / r0) ** p)


def target_1(seed=101):
    """Две спиральные ветви (плотный шаг) + зерно."""
    arms = 2
    th = PH + R / 3.6 * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th)) ** 2.0 * _env()
    return np.clip(s * grain(seed) * 1.7, 0, 1)


def target_2(seed=202):
    """Пять закрученных рукавов с кольцевой модуляцией."""
    arms = 5
    th = PH + R / 6.5 * 2 * np.pi / arms
    s = (0.5 + 0.5 * np.cos(arms * th + 1.1 * np.cos(R / 7.0))) ** 2.4 * _env(60.0, 5)
    return np.clip(s * grain(seed, 1.6, 0.35) * 1.8, 0, 1)


def target_3(seed=303):
    """«Роза»: кольца с угловой модуляцией (интерференционная картина)."""
    s = (0.5 + 0.5 * np.cos(0.0042 * R ** 2 + 9.0 * np.sin(3.0 * PH + R / 26.0))) * _env(60.0, 3)
    return np.clip((s ** 1.35) * grain(seed, 2.2, 0.45) * 1.6, 0, 1)


def target_4(seed=404):
    """Спираль Архимеда с пучностями (стоячая волна вдоль рукава) + зерно."""
    th = PH + R / 4.2
    s = (0.5 + 0.5 * np.cos(2.0 * th)) ** 1.8
    s = s * (0.55 + 0.45 * np.cos(0.55 * R)) ** 1.2 * _env(57.0, 4)
    return np.clip(s * grain(seed, 1.8, 0.4) * 1.9, 0, 1)


SET = (target_1, target_2, target_3, target_4)


def make_set():
    arr = np.stack([f() for f in SET])
    arr[arr < 0.02] = 0.0
    return arr / arr.max()


if __name__ == "__main__":
    T = make_set()
    from PIL import Image
    os.makedirs(os.path.join(ROOT, "experiments/_preview"), exist_ok=True)
    for i, t in enumerate(T):
        Image.fromarray((t * 255).astype(np.uint8), mode="L").save(
            os.path.join(ROOT, f"experiments/_preview/t{i+1}.png"))
        print(f"target_{i+1}: max {t.max():.2f} энергия {t.sum():.1f} ненулевых {(t>0.02).mean()*100:.1f}%")
    t0 = time.time(); best = None
    for r in range(3):
        g = gs.gs_run(T, iters=300, seed=r, verbose=False)
        q = POL.quantize(np.asarray(g["phi"][0, 0]))
        m = POL.metrics_of(q, T)
        print(f"GS старт {r}: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
        if best is None or m["pcc"] + m["ssim"] > best[0]:
            best = (m["pcc"] + m["ssim"], q)
    res = POL.polish(T, best[1], steps=700, lr=2.5e-2)
    q = POL.quantize(res["phi"]); m = POL.metrics_of(q, T)
    print(f"ИТОГ +polish700: PCC {m['pcc']:.4f} SSIM {m['ssim']:.4f}", flush=True)
    for r_ in m["rows"]:
        print(f"  z={r_['z_mm']:.1f} PCC {r_['pcc']:.4f} SSIM {r_['ssim']:.4f}")
    np.save(os.path.join(ROOT, "experiments/_setv2_phase.npy"), q)
    print(f"({time.time()-t0:.0f}s)")
