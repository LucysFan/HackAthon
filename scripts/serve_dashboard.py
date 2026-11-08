"""Локальный пульт: отдаёт дашборд и запускает задачи по кнопке.

Запуск (в контейнере — ``docker run -p 8000:8000``):
    python scripts/serve_dashboard.py                 # http://127.0.0.1:8765
    python scripts/serve_dashboard.py --host 0.0.0.0 --port 8000

Возможности:

* статический дашборд из ``dashboard/`` (он же открывается офлайн двойным
  кликом, когда пульт не запущен);
* запуск этапов конвейера по белому списку (``POST /api/run``) с живым логом
  (``GET /api/log``) и остановкой (``POST /api/stop``);
* комплекты изображений: загрузка 1–4 файлов любого размера и цвета
  (``POST /api/sets``), список (``GET /api/sets``), инференс комплекта одним
  прямым проходом сети (``POST /api/infer``);
* данные для вкладок «тест» и «модель»: история обучения (``GET /api/history``),
  статистика весов по слоям (``GET /api/model/graph``), карты активаций для
  анимации распространения сигнала (``GET /api/model/act``);
* отдача артефактов ``GET /api/asset?set=<id>&name=DOE.png`` и файлов из
  ``outputs/`` по белому списку.

Никаких команд из браузера не выполняется: принимаются только идентификаторы
этапов и комплектов. Сервер — единственная точка, которая знает про venv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DASH = os.path.join(ROOT, "dashboard")
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "outputs", "_cache")


# ------------------------------------------------------------------ окружение
def pick_python() -> str:
    """Интерпретатор этапов: venv проекта → текущий → системный."""
    cand = [os.environ.get("HDOE_PYTHON"),
            os.path.join(ROOT, ".venv", "bin", "python"),
            os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
            sys.executable, shutil.which("python3"), shutil.which("python")]
    for c in cand:
        if c and os.path.exists(c):
            return c
    return sys.executable


PY = pick_python()


def have_deps() -> bool:
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:                                      # noqa: BLE001
        return False


def reexec_into_venv() -> None:
    if have_deps() or PY == sys.executable:
        return
    print(f"перезапуск в интерпретаторе проекта: {PY}", flush=True)
    os.execv(PY, [PY, os.path.abspath(__file__)] + sys.argv[1:])


# ------------------------------------------------------------------ этапы
STAGES: dict[str, dict] = {
    "samples": dict(
        title="Пять тестовых комплектов", group="данные",
        cmd=[PY, "scripts/make_sample_sets.py", "--force", "--infer"],
        hint="5 × 4 картинки разных типов + прогон сетью (~5 мин)"),
    "targets": dict(
        title="Цели из картинки", group="данные",
        cmd=[PY, "scripts/setup_targets_from_image.py", "--src", "data/source/object.png"],
        hint="PNG → четыре цели 128×128"),
    "corpus": dict(
        title="Корпус объектов", group="данные",
        cmd=[PY, "-m", "hdoe.corpus"],
        hint="учительские четвёрки для дистилляции"),
    "train": dict(
        title="Обучение сети", group="обучение",
        cmd=[PY, "scripts/train_gpu.py", "--device", "auto", "--distill-steps", "1600",
             "--steps", "0", "--geom-only", "--teachers", "6", "--gs-iters", "250",
             "--gs-restarts", "2", "--val-every", "200", "--snap-every", "200",
             "--phys-steps", "2048", "--phys-gain", "0.35"],
        hint="дистилляция учительских масок GS → outputs/best.pt"),
    "evaluate": dict(
        title="Оценка сдаваемого набора", group="результат",
        cmd=[PY, "scripts/evaluate.py", "--pipeline", "netphys", "--gs-iters", "250",
             "--gs-restarts", "3", "--gs-refine", "100"],
        hint="DOE.png, recon_1..4.png, metrics.json (+GS)"),
    "sweep": dict(
        title="Абляция K", group="результат",
        cmd=[PY, "scripts/sweep_phys.py"],
        hint="вклад физического слоя: K = 0…2048"),
    "tests": dict(
        title="Тесты 8.1–8.9", group="проверка",
        cmd=[PY, "tests/test_all.py"],
        hint="верификация физики и форматов"),
    "bundle": dict(
        title="Сборка дашборда", group="проверка",
        cmd=[PY, "scripts/build_dashboard.py"],
        hint="пересчитать бандл данных"),
    "figures": dict(
        title="Рисунки", group="отчёт",
        cmd=[PY, "scripts/make_figures.py"], hint="figures/*.png"),
    "report": dict(
        title="PDF-отчёт", group="отчёт",
        cmd=[PY, "scripts/make_report.py"], hint="outputs/report.pdf"),
    "video": dict(
        title="Видео-демо", group="отчёт",
        cmd=[PY, "scripts/make_demo_video.py"], hint="outputs/demo_backup.mp4"),
    "probe": dict(
        title="Проверка окружения", group="служебное",
        cmd=[PY, "-c", "import sys, numpy, torch, PIL; "
                       "print('python', sys.version.split()[0]); "
                       "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
        hint="python, torch, CUDA"),
}

PIPELINES: dict[str, dict] = {
    "full": dict(title="Всё с нуля", stages=["targets", "corpus", "train", "evaluate",
                                             "sweep", "tests", "samples", "bundle",
                                             "figures", "report", "video"]),
    "refresh": dict(title="Пересчитать артефакты", stages=["evaluate", "sweep", "tests",
                                                           "samples", "bundle", "figures",
                                                           "report", "video"]),
    "artifacts": dict(title="Только артефакты", stages=["bundle", "figures", "report"]),
    "quickcheck": dict(title="Быстрая проверка", stages=["tests", "bundle"]),
}

# ------------------------------------------------------------------ состояние
LOCK = threading.Lock()
STATE: dict = dict(running=None, title=None, queue=[], command=[], started=None,
                   finished=None, returncode=None, log=[], proc=None, kind=None,
                   set_id=None)
MAX_LOG = 4000


def log_append(text: str) -> None:
    with LOCK:
        STATE["log"].append(text)
        if len(STATE["log"]) > MAX_LOG:
            del STATE["log"][:-MAX_LOG]


def _kill_tree(proc: subprocess.Popen) -> None:
    """Остановить этап вместе с потомками (иначе torch живёт до конца)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:                                      # noqa: BLE001
        try:
            proc.terminate()
        except Exception:                                  # noqa: BLE001
            pass
    for _ in range(40):                                    # до 4 с на мягкий выход
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:                                      # noqa: BLE001
        try:
            proc.kill()
        except Exception:                                  # noqa: BLE001
            pass


def _exec(cmd: list[str], title: str, kind: str = "stage", set_id: str | None = None) -> int:
    with LOCK:
        STATE.update(running=(cmd[1] if len(cmd) > 1 else cmd[0]), title=title,
                     command=cmd, started=time.time(), finished=None,
                     returncode=None, kind=kind, set_id=set_id)
    log_append(f"\n$ {' '.join(cmd)}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env, start_new_session=True)
        with LOCK:
            STATE["proc"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            log_append(line)
        code = proc.wait()
    except FileNotFoundError as exc:
        log_append(f"ОШИБКА: {exc}\n")
        code = 127
    finally:
        with LOCK:
            STATE["proc"] = None
    with LOCK:
        STATE.update(returncode=code, finished=time.time(), running=None, title=None)
    return code


def _worker(stage_ids: list[str]) -> None:
    for sid in stage_ids:
        st = STAGES[sid]
        code = _exec(st["cmd"], st["title"])
        if code != 0:
            log_append(f"\nостановлено на этапе «{st['title']}» (код {code})\n")
            break
    with LOCK:
        STATE.update(running=None, title=None, queue=[])


def start(stage_ids: list[str]) -> tuple[bool, str]:
    with LOCK:
        if STATE["running"]:
            return False, f"уже выполняется: {STATE['title'] or STATE['running']}"
        bad = [s for s in stage_ids if s not in STAGES]
        if bad or not stage_ids:
            return False, f"неизвестные этапы: {bad}"
        STATE.update(log=[], queue=list(stage_ids[1:]))
    threading.Thread(target=_worker, args=(stage_ids,), daemon=True).start()
    return True, "запущено"


def stop_all() -> dict:
    """Остановить текущий этап и очистить очередь. Возвращает новое состояние."""
    with LOCK:
        proc = STATE.get("proc")
        running = STATE["running"]
        STATE["queue"] = []
    killed = False
    if proc is not None:
        _kill_tree(proc)
        killed = True
        log_append("\n■ остановлено пользователем\n")
        with LOCK:
            STATE.update(returncode=-15, finished=time.time(), running=None, title=None,
                         proc=None)
    elif running:
        with LOCK:
            STATE.update(returncode=-15, finished=time.time(), running=None, title=None)
        killed = True
    return dict(ok=True, stopped=bool(killed))


# ------------------------------------------------------------------ файлы/состояние
def _stat(rel: str) -> dict | None:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    return dict(path=rel, mtime=st.st_mtime, size=st.st_size,
                mtime_str=time.strftime("%d.%m %H:%M", time.localtime(st.st_mtime)))


def state_payload() -> dict:
    with LOCK:
        cur = {k: v for k, v in STATE.items() if k != "proc"}
    files = {}
    for key, rel in (("doe", "outputs/DOE.png"), ("metrics", "outputs/metrics.json"),
                     ("tests", "outputs/tests.json"), ("report", "outputs/report.pdf"),
                     ("video", "outputs/demo_backup.mp4"),
                     ("sweep", "outputs/phys_sweep.json"),
                     ("bundle", "dashboard/data_bundle.js"),
                     ("ckpt", "outputs/best.pt")):
        files[key] = _stat(rel)
    metrics = None
    mp = os.path.join(OUT, "metrics.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            metrics = dict(ssim=(m.get("mean") or {}).get("ssim"),
                           pearson=(m.get("mean") or {}).get("pcc"),
                           gs_ssim=((m.get("gs") or {}).get("mean") or {}).get("ssim"),
                           created=m.get("created"),
                           steps=(m.get("phys") or {}).get("steps"),
                           ckpt=(m.get("checkpoint") or {}).get("path"))
        except Exception:                                  # noqa: BLE001
            metrics = None
    try:
        sp = os.path.join(OUT, "sets_summary.json")
        summary = json.load(open(sp)) if os.path.exists(sp) else None
    except Exception:                                      # noqa: BLE001
        summary = None
    return dict(stages=[dict(id=k, title=v["title"], group=v["group"], hint=v["hint"])
                        for k, v in STAGES.items()],
                summary=summary,
                pipelines=[dict(id=k, title=v["title"], stages=v["stages"])
                           for k, v in PIPELINES.items()],
                state=cur, files=files, metrics=metrics, python=PY, root=ROOT,
                sets=list_sets_safe())


def ensure_default_set() -> dict:
    """Сдаваемый набор как комплект ``data/sets/default`` (создаётся один раз).

    Так в списке комплектов сразу есть что прогнать: у него уже лежат DOE.png,
    реконструкции и метрики — те самые, что в отчёте.
    """
    from hdoe import sets as SS
    sid = "default"
    path = os.path.join(SS.SETS_DIR, sid)
    if not SS.has_targets(sid):
        os.makedirs(path, exist_ok=True)
        src = os.path.join(DATA, "targets")
        for i in range(1, 5):
            shutil.copyfile(os.path.join(src, f"target_{i}.png"),
                            os.path.join(path, f"target_{i}.png"))
        for name in ("DOE.png", "recon_1.png", "recon_2.png", "recon_3.png",
                     "recon_4.png", "metrics.json"):
            p = os.path.join(OUT, name)
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(path, name))
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(dict(id=sid, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           sources=["data/targets"], shipped=True), fh,
                      indent=1, ensure_ascii=False)
    return SS.info(sid)


def list_sets_safe() -> list[dict]:
    try:
        from hdoe import sets as SS
        out = [s for s in SS.all_sets() if s["id"] != "default"]
        try:
            out = out + [ensure_default_set()]
        except Exception:                                  # noqa: BLE001
            pass
        return out
    except Exception:                                      # noqa: BLE001
        return []


# ------------------------------------------------------------------ модель/история
def _history() -> dict:
    """Кривые обучения: обучение (дистилляция), валидация, тест."""
    log_path = os.path.join(OUT, "train_log.json")
    rows = json.load(open(log_path)) if os.path.exists(log_path) else []
    train, val = [], []
    for r in rows:
        st = r.get("stage")
        if st == "distill":
            train.append(dict(step=r.get("step"), loss=r.get("loss"),
                              train_ssim=r.get("ssim"), train_pearson=r.get("pcc"),
                              cos=r.get("cos")))
            if r.get("val_ssim") is not None:
                val.append(dict(step=r.get("step"), ssim=r.get("val_ssim"),
                                pearson=r.get("val_pcc")))
        elif st in ("physics_in_net", "e2e"):
            val.append(dict(step=r.get("step"), ssim=r.get("val_ssim"),
                            pearson=r.get("val_pcc")))
    test = None
    mp = os.path.join(OUT, "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        test = dict(ssim=(m.get("mean") or {}).get("ssim"),
                    pearson=(m.get("mean") or {}).get("pcc"),
                    gs_ssim=((m.get("gs") or {}).get("mean") or {}).get("ssim"),
                    gs_pearson=((m.get("gs") or {}).get("mean") or {}).get("pcc"))
    # честный протокол: тренировка / валидация / тест на разных наборах
    # (считает scripts/eval_splits.py)
    splits = None
    sp_path = os.path.join(OUT, "splits.json")
    if os.path.exists(sp_path):
        try:
            sp = json.load(open(sp_path))
            splits = dict(
                train=(sp.get("train") or {}).get("summary"),
                val=(sp.get("val") or {}).get("summary"),
                test=(sp.get("test") or {}).get("summary"),
                sets=(sp.get("test") or {}).get("sets"),
                created=sp.get("created"),
                note=(sp.get("val") or {}).get("note"))
        except Exception:                                  # noqa: BLE001
            splits = None
    return dict(train=train, val=val, test=test, splits=splits,
                note="кривые — дистилляция учительских масок GS; "
                     "валидация в логе — контрольная четвёрка; "
                     "протокол train/val/test — из outputs/splits.json")


def _model_graph() -> dict:
    """Статистика весов по слоям + структура графа (для вкладки «модель»)."""
    cache = os.path.join(CACHE_DIR, "model_graph.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch
    from hdoe.model import build_net
    ck_path = os.path.join(OUT, "best.pt")
    if not os.path.exists(ck_path):
        return dict(layers=[], edges=[], params=0, note="нет outputs/best.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = build_net(ck.get("arch") or dict(width=48, depth=5))
    net.load_state_dict(ck["model"])
    layers, edges = [], []
    idx = 0
    for name, p in net.named_parameters():
        w = p.detach().float()
        nz = float((w.abs() > 1e-6).float().mean())
        layers.append(dict(name=name, shape=list(w.shape), numel=int(w.numel()),
                           mean=float(w.mean()), std=float(w.std()),
                           abs_mean=float(w.abs().mean()), min=float(w.min()),
                           max=float(w.max()), nonzero=nz,
                           hist=_hist(w)))
    # грубая структура: энкодер (blocks/pools) → декодер (ups/decs) → голова
    for i, (name, p) in enumerate(net.named_parameters()):
        kind = ("encoder" if name.startswith(("stem", "blocks", "pools")) else
                "decoder" if name.startswith(("ups", "decs")) else "head")
        layers[i]["group"] = kind
        idx += 1
    for a, b in zip(layers, layers[1:]):
        if a["group"] == b["group"]:
            edges.append([a["name"], b["name"]])
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = dict(layers=layers, edges=edges,
               params=int(sum(l["numel"] for l in layers)),
               arch=ck.get("arch"), pack=ck.get("pack"),
               note="каждый узел — слой с весами; толщина ребра — доля параметров")
    json.dump(out, open(cache, "w"))
    return out


def _hist(w, bins: int = 24) -> list[float]:
    import numpy as np
    a = w.flatten().numpy()
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return [0.0] * bins
    h, _ = np.histogram(a, bins=bins, range=(lo, hi))
    h = h.astype(float) / max(h.max(), 1.0)
    return [round(float(v), 4) for v in h]


def _activations(set_id: str | None, steps: int = 0) -> dict:
    """Карты активаций по стадиям сети для анимации распространения сигнала."""
    import numpy as np
    import torch
    from hdoe import sets as SS
    from hdoe.model import build_net
    ck_path = os.path.join(OUT, "best.pt")
    if not os.path.exists(ck_path):
        return dict(stages=[])
    key = f"act_{set_id or 'default'}_{steps}.json"
    cache = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache):
        return json.load(open(cache))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = build_net(ck.get("arch") or dict(width=48, depth=5))
    net.load_state_dict(ck["model"])
    net.eval()
    tg = SS.load_targets(set_id) if set_id else None
    if tg is None:
        from hdoe import targets as T
        tg = T.load_targets()
    x = torch.as_tensor(tg, dtype=torch.float32)[None]
    captured: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_m, _i, out):
            a = out.detach().float()
            if a.dim() == 4:
                a = a.mean(dim=1, keepdim=True)[0, 0]
            else:
                a = a[0]
            # сжимаем до 24×24 для передачи в браузер
            a = torch.nn.functional.adaptive_avg_pool2d(a[None, None], (24, 24))[0, 0]
            v = a.numpy()
            lo, hi = float(v.min()), float(v.max())
            v = (v - lo) / (hi - lo + 1e-9)
            captured[name] = np.asarray(v * 255, dtype=np.uint8)
        return fn

    handles = []
    # снимаем активации с внутренних модулей (это ResNet-энкодер + U-Net-голова),
    # поэтому обходим дерево, а не только верхний уровень
    wanted = ("backbone", "layer1", "layer2", "layer3", "layer4", "stem", "blocks",
              "pools", "ups", "decs", "head", "neck", "conv")
    for name, m in net.named_modules():
        if name and any(part.split(".")[-1].rstrip("0123456789[]").startswith(w)
                        for part in name.split(".") for w in wanted):
            handles.append(m.register_forward_hook(hook(name)))
    with torch.no_grad():
        try:
            _ = net(x)
        except Exception as exc:                            # noqa: BLE001
            log_append(f"активации: {exc}\n")
    for h in handles:
        h.remove()
    # оставляем «скелет»: верхние модули каждой ветки (stem, блоки, апсемплы, голова)
    def keep(name: str) -> bool:
        leaf = name.split(".")[-1]
        if leaf.isdigit():                       # внутренние слои Sequential
            return False
        return any(k in name for k in ("stem", "blocks", "ups", "decs", "head", "neck"))

    stages = [dict(name=n.replace("net.", ""), size=len(v.flatten()),
                   data=v.flatten().tolist())
              for n, v in captured.items() if keep(n)]
    stages = stages[:24]
    if not stages:                                  # ничего не сняли — не кэшируем
        return dict(stages=[], set=set_id, note="нет активаций")
    out = dict(stages=stages, set=set_id, note="усреднение по каналам, 24×24")
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


# ------------------------------------------------------------------ работа с комплектами
def handle_upload_set(content_type: str, raw: bytes) -> dict:
    from hdoe import sets as SS
    files = parse_multipart(content_type, raw)
    if not files:
        return dict(ok=False, message="не пришло ни одного файла")
    single = len(files) == 1
    sid = SS.new_set_id()
    os.makedirs(SS.set_path(sid), exist_ok=True)
    SS.save_uploads(sid, [f[1] for f in files])
    SS.build_targets(sid, single=single)
    log_append(f"\nпринят комплект {sid}: {len(files)} файл(ов)"
               f"{' (копируется на 4 плоскости)' if single else ''}\n")
    return dict(ok=True, id=sid, info=SS.info(sid))


def parse_multipart(content_type: str, raw: bytes) -> list[tuple[str, bytes]]:
    """Разбор multipart/form-data: [(имя поля, байты файла), ...].

    Разбираем по байтовым границам, а не ``split(b"\\r\\n\\r\\n")``: файл
    может содержать любые байты, включая похожие на разделители.
    """
    if not content_type.startswith("multipart/form-data"):
        return [("raw", raw)] if len(raw) > 128 else []
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        return []
    boundary = b"--" + m.group(1).encode()
    out: list[tuple[str, bytes]] = []
    for chunk in raw.split(boundary):
        if not chunk.strip() or chunk.strip() == b"--":
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        head_end = chunk.find(b"\r\n\r\n")
        if head_end < 0:
            continue
        head, body = chunk[:head_end], chunk[head_end + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        if b"filename=" not in head:
            continue
        name = "file"
        for tag in head.replace(b"\r\n", b";").split(b";"):
            tag = tag.strip()
            if tag.startswith(b"name="):
                name = tag.split(b"=", 1)[1].strip(b' "').decode(errors="ignore")
        out.append((name, body))
    return out


def handle_infer(set_id: str, steps: int) -> dict:
    from hdoe import sets as SS
    if not SS.valid_id(set_id) or not SS.has_targets(set_id):
        return dict(ok=False, message="нет такого комплекта")
    with LOCK:
        if STATE["running"]:
            return dict(ok=False, message=f"уже выполняется: {STATE['title']}")
    sp = SS.set_path(set_id)

    def job():
        cmd = [PY, "scripts/infer_set.py", "--set", sp, "--device", "auto"]
        if steps:
            cmd += ["--steps", str(steps)]
        _exec(cmd, f"инференс комплекта {set_id}", kind="infer", set_id=set_id)

    threading.Thread(target=job, daemon=True).start()
    return dict(ok=True, id=set_id)


# ------------------------------------------------------------------ сервер
ASSET_WHITELIST = ("metrics.json", "tests.json", "report.pdf", "phys_sweep.json",
                   "DOE.png", "DOE_gs.png", "train_log.json", "demo_backup.mp4")


class Handler(BaseHTTPRequestHandler):
    server_version = "hdoe-dashboard/2.0"

    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in (args[0] if args else ""):
            return

    # ---------------------------------------------------------- helpers
    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:                              # noqa: N802
        url = urllib.parse.urlparse(self.path)
        path, q = urllib.parse.unquote(url.path), urllib.parse.parse_qs(url.query)
        if path == "/api/state":
            return self._json(state_payload())
        if path == "/api/log":
            since = int(q.get("since", ["0"])[0])
            with LOCK:
                chunk, total = STATE["log"][since:], len(STATE["log"])
                cur = {k: STATE[k] for k in ("running", "title", "returncode",
                                             "started", "finished", "kind", "set_id",
                                             "queue")}
            return self._json(dict(lines=chunk, total=total, **cur))
        if path == "/api/sets":
            return self._json(dict(sets=list_sets_safe()))
        if path == "/api/history":
            return self._json(_history())
        if path == "/api/model/graph":
            try:
                return self._json(_model_graph())
            except Exception as exc:                        # noqa: BLE001
                return self._json(dict(layers=[], edges=[], error=str(exc)), 200)
        if path == "/api/model/act":
            sid = (q.get("set") or [None])[0]
            try:
                return self._json(_activations(sid))
            except Exception as exc:                        # noqa: BLE001
                return self._json(dict(stages=[], error=str(exc)), 200)
        if path == "/api/asset":
            return self._asset(q)
        if path.startswith("/api/sets/"):
            sid = path.split("/api/sets/")[1].strip("/")
            try:
                from hdoe import sets as SS
                return self._json(SS.info(sid))
            except Exception as exc:                        # noqa: BLE001
                return self._json(dict(ok=False, message=str(exc)), 404)
        return self._static(path)

    def _asset(self, q: dict) -> None:
        """Артефакты комплекта или outputs (белый список имён)."""
        from hdoe import sets as SS
        sid = (q.get("set") or [""])[0]
        name = (q.get("name") or [""])[0]
        if sid:
            if not SS.valid_id(sid):
                return self._json(dict(ok=False), 400)
            base = SS.set_path(sid)
        else:
            base = OUT
        allowed = set(ASSET_WHITELIST) | {
            f"{p}.png" for p in ("recon_1", "recon_2", "recon_3", "recon_4",
                                 "recon_gs_1", "recon_gs_2", "recon_gs_3", "recon_gs_4",
                                 "target_1", "target_2", "target_3", "target_4",
                                 "preview", "ssim")}
        if name not in allowed:
            return self._json(dict(ok=False, message="файл не разрешён"), 403)
        full = os.path.join(base, name)
        if not os.path.isfile(full):
            return self._json(dict(ok=False, message="нет файла"), 404)
        ext = os.path.splitext(name)[1].lower()
        ctype = {".png": "image/png", ".json": "application/json; charset=utf-8",
                 ".pdf": "application/pdf", ".mp4": "video/mp4"}.get(
                     ext, "application/octet-stream")
        return self._bytes(open(full, "rb").read(), ctype)

    def _static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            path = "/index.html"
        rel = os.path.normpath(path.lstrip("/"))
        full = os.path.join(DASH, rel)
        if not os.path.abspath(full).startswith(os.path.abspath(DASH)) or \
                not os.path.isfile(full):
            return self._json(dict(ok=False, message="не найдено"), 404)
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".json": "application/json; charset=utf-8",
                 ".png": "image/png", ".bin": "application/octet-stream"}.get(
                     ext, "application/octet-stream")
        return self._bytes(open(full, "rb").read(), ctype)

    # ----------------------------------------------------------- POST
    def do_POST(self) -> None:                             # noqa: N802
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        raw = self._body()
        if url.path == "/api/run":
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._json(dict(ok=False, message="плохой JSON"), 400)
            if "pipeline" in req:
                pl = PIPELINES.get(str(req["pipeline"]))
                if not pl:
                    return self._json(dict(ok=False, message="нет сценария"), 400)
                ids = list(pl["stages"])
            else:
                ids = [str(s) for s in (req.get("stages") or [])] or [str(req.get("stage"))]
            ok, msg = start(ids)
            return self._json(dict(ok=ok, message=msg, stages=ids), 200 if ok else 409)
        if url.path == "/api/stop":
            return self._json(stop_all())
        if url.path == "/api/sets":
            try:
                return self._json(handle_upload_set(self.headers.get("Content-Type", ""), raw))
            except Exception as exc:                        # noqa: BLE001
                return self._json(dict(ok=False, message=f"не удалось: {exc}"), 400)
        if url.path == "/api/infer":
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._json(dict(ok=False, message="плохой JSON"), 400)
            res = handle_infer(str(req.get("set") or ""), int(req.get("steps") or 0))
            return self._json(res, 200 if res.get("ok") else 409)
        if url.path == "/api/sets/delete":
            from hdoe import sets as SS
            sid = (q.get("id") or [""])[0]
            try:
                return self._json(dict(ok=SS.delete(sid)))
            except Exception as exc:                        # noqa: BLE001
                return self._json(dict(ok=False, message=str(exc)), 400)
        return self._json(dict(ok=False, message="неизвестный метод"), 404)


# ------------------------------------------------------------------ запуск
def main() -> int:
    reexec_into_venv()
    ap = argparse.ArgumentParser(description="Пульт запуска задач и сервер дашборда")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--allow-lan", action="store_true", help="слушать 0.0.0.0")
    args = ap.parse_args()
    host = "0.0.0.0" if args.allow_lan else args.host
    srv = ThreadingHTTPServer((host, args.port), Handler)
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{args.port}/"
    print(f"пульт: {url}\nкаталог: {ROOT}\npython для этапов: {PY}")
    if not os.path.isdir(os.path.join(ROOT, ".venv")) and "docker" not in os.environ.get(
            "HDOE_ENV", ""):
        print("подсказка: .venv не найден — этапы пойдут системным python "
              "(в контейнере это нормально)")
    print("Ctrl+C — остановить")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
