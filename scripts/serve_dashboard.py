"""Локальный пульт: отдаёт дашборд и запускает этапы конвейера по кнопке.

Запуск:
    python scripts/serve_dashboard.py            # затем http://127.0.0.1:8765
    python scripts/serve_dashboard.py --port 9000 --no-browser

Что это даёт. Дашборд остаётся статическим (офлайн, двойной клик по
index.html работает как раньше — это требование ТЗ). Дополнительно этот
локальный сервер:

* отдаёт каталог ``dashboard/`` по HTTP;
* по запросу ``POST /api/run`` запускает **разрешённые** этапы конвейера
  (данные → цели → обучение → оценка → абляция → тесты → дашборд → фигуры →
  отчёт → видео) и стримит их stdout в браузер (``GET /api/log``);
* показывает, что уже посчитано в ``outputs/`` (``GET /api/state``), и умеет
  принимать свою картинку (``POST /api/upload``) — она кладётся в
  ``data/source/object.png`` и из неё собираются четыре цели.

Безопасность: сервер слушает только 127.0.0.1, никаких shell-строк из
запроса не выполняется — принимается лишь имя этапа из белого списка ниже,
аргументы тоже фиксированные. Это локальный пульт для запуска задач, а не
публичный сервис.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
def pick_python() -> str:
    """Интерпретатор для этапов: venv проекта → текущий → системный.

    Пульт надо запускать тем же python, что и остальные скрипты, но если
    его стартовали системным ``python3``, а рядом лежит ``.venv``, этапы
    всё равно пойдут из venv (иначе падает на numpy/torch).
    """
    cand = [os.environ.get("HDOE_PYTHON"), os.path.join(ROOT, ".venv", "bin", "python"),
            os.path.join(ROOT, ".venv", "Scripts", "python.exe"), sys.executable,
            shutil.which("python3"), shutil.which("python")]
    for c in cand:
        if c and os.path.exists(c):
            return c
    return sys.executable


PY = pick_python()

# ------------------------------------------------------------------ этапы
# id → (заголовок, команда, что требуется до запуска, делает ли этап тяжёлый счёт)
STAGES: dict[str, dict] = {
    "targets": dict(
        title="Цели из картинки", group="данные",
        cmd=[PY, "scripts/setup_targets_from_image.py", "--src", "data/source/object.png"],
        hint="PNG → четыре цели 128×128 (кроп, инверсия, выравнивание энергии)"),
    "corpus": dict(
        title="Корпус объектов", group="данные",
        cmd=[PY, "-m", "hdoe.corpus"],
        hint="учительские четвёрки для дистилляции (кэшируется в data/corpus.npy)"),
    "train": dict(
        title="Обучение сети", group="обучение",
        cmd=[PY, "scripts/train_gpu.py", "--device", "auto", "--distill-steps", "1600",
             "--steps", "0", "--geom-only", "--teachers", "6", "--gs-iters", "250",
             "--gs-restarts", "2", "--val-every", "200", "--snap-every", "200",
             "--phys-steps", "2048", "--phys-gain", "0.35"],
        hint="дистилляция учительских масок GS → outputs/best.pt"),
    "evaluate": dict(
        title="Оценка и DOE.png", group="результат",
        cmd=[PY, "scripts/evaluate.py", "--pipeline", "netphys", "--gs-iters", "250",
             "--gs-restarts", "3", "--gs-refine", "100"],
        hint="один прямой проход сети с физикой внутри → metrics.json, recon_1..4.png"),
    "sweep": dict(
        title="Абляция K", group="результат",
        cmd=[PY, "scripts/sweep_phys.py"],
        hint="сколько даёт физический слой внутри сети: K = 0…2048"),
    "tests": dict(
        title="Тесты 8.1–8.9", group="проверка",
        cmd=[PY, "tests/test_all.py"],
        hint="верификация комплекса → tests.json"),
    "bundle": dict(
        title="Сборка дашборда", group="проверка",
        cmd=[PY, "scripts/build_dashboard.py"],
        hint="пересчитать бандл данных — числа на панелях совпадут с metrics.json"),
    "figures": dict(
        title="Рисунки", group="отчёт",
        cmd=[PY, "scripts/make_figures.py"],
        hint="figures/*.png для отчёта"),
    "report": dict(
        title="PDF-отчёт", group="отчёт",
        cmd=[PY, "scripts/make_report.py"],
        hint="outputs/report.pdf"),
    "video": dict(
        title="Видео-демо", group="отчёт",
        cmd=[PY, "scripts/make_demo_video.py"],
        hint="outputs/demo_backup.mp4 (резервная запись защиты)"),
    # служебный этап для самопроверки пульта: печатает окружение и выходит
    "probe": dict(
        title="Проверка окружения", group="служебное",
        cmd=[PY, "-c", "import sys, numpy, torch, PIL; "
                       "print('python', sys.version.split()[0]); "
                       "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
        hint="python, torch, CUDA — диагностика перед длинными этапами"),
}

# «Сценарии» — последовательности этапов для одной кнопки
PIPELINES: dict[str, dict] = {
    "full": dict(title="Всё с нуля", stages=["targets", "corpus", "train", "evaluate",
                                             "sweep", "tests", "bundle", "figures",
                                             "report", "video"]),
    "refresh": dict(title="Пересчитать артефакты", stages=["evaluate", "sweep", "tests",
                                                           "bundle", "figures",
                                                           "report", "video"]),
    "artifacts": dict(title="Только артефакты", stages=["bundle", "figures", "report"]),
    "quickcheck": dict(title="Быстрая проверка", stages=["tests", "bundle"]),
}

# ------------------------------------------------------------------ состояние
LOCK = threading.Lock()
STATE: dict = {
    "running": None,        # id текущего этапа
    "queue": [],            # оставшиеся этапы сценария
    "command": [],
    "started": None,
    "finished": None,
    "returncode": None,
    "log": [],
    "proc": None,
}
MAX_LOG = 4000


def _append(text: str) -> None:
    with LOCK:
        STATE["log"].append(text)
        if len(STATE["log"]) > MAX_LOG:
            del STATE["log"][:-MAX_LOG]


def _run_stage(stage_id: str) -> None:
    st = STAGES[stage_id]
    with LOCK:
        STATE.update(running=stage_id, command=st["cmd"], started=time.time(),
                     finished=None, returncode=None)
    _append(f"\n$ {' '.join(st['cmd'])}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(st["cmd"], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env)
        with LOCK:
            STATE["proc"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            _append(line)
        code = proc.wait()
    except FileNotFoundError as exc:                       # нет venv/скрипта
        _append(f"ОШИБКА: {exc}\n")
        code = 127
    with LOCK:
        STATE.update(returncode=code, finished=time.time(), proc=None)


def _worker(stage_ids: list[str]) -> None:
    for sid in stage_ids:
        _run_stage(sid)
        with LOCK:
            code = STATE["returncode"]
        if code != 0:
            _append(f"\nцепочка остановлена: этап «{STAGES[sid]['title']}» "
                    f"завершился с кодом {code}\n")
            break
    with LOCK:
        STATE.update(running=None, queue=[])


def start(stage_ids: list[str]) -> tuple[bool, str]:
    with LOCK:
        if STATE["running"]:
            return False, f"уже выполняется: {STAGES[STATE['running']]['title']}"
        bad = [s for s in stage_ids if s not in STAGES]
        if bad or not stage_ids:
            return False, f"неизвестные этапы: {bad}"
        STATE.update(log=[], queue=list(stage_ids[1:]))
    threading.Thread(target=_worker, args=(stage_ids,), daemon=True).start()
    return True, "запущено"


def stop() -> bool:
    with LOCK:
        proc = STATE.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        _append("\nостановлено пользователем\n")
        return True
    return False


# ------------------------------------------------------------------ файлы
def _stat(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return dict(path=os.path.relpath(path, ROOT), mtime=st.st_mtime,
                size=st.st_size, age=time.time() - st.st_mtime)


def state_payload() -> dict:
    with LOCK:
        cur = dict(STATE)
        cur.pop("proc", None)
    files = {}
    for key, rel in (("doe", "outputs/DOE.png"), ("metrics", "outputs/metrics.json"),
                     ("tests", "outputs/tests.json"), ("report", "outputs/report.pdf"),
                     ("video", "outputs/demo_backup.mp4"),
                     ("sweep", "outputs/phys_sweep.json"),
                     ("bundle", "dashboard/data_bundle.js"),
                     ("ckpt", "outputs/best.pt")):
        info = _stat(os.path.join(ROOT, rel))
        if info:
            info["mtime_str"] = time.strftime("%d.%m %H:%M", time.localtime(info["mtime"]))
        files[key] = info
    metrics = None
    mp = os.path.join(OUT, "metrics.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            metrics = dict(pcc=(m.get("mean") or {}).get("pcc"),
                           ssim=(m.get("mean") or {}).get("ssim"),
                           gs_pcc=((m.get("gs") or {}).get("mean") or {}).get("pcc"),
                           created=m.get("created"),
                           steps=(m.get("phys") or {}).get("steps"))
        except Exception:                                  # noqa: BLE001
            metrics = None
    return dict(stages=[dict(id=k, title=v["title"], group=v["group"],
                             hint=v["hint"]) for k, v in STAGES.items()],
                pipelines=[dict(id=k, title=v["title"], stages=v["stages"])
                           for k, v in PIPELINES.items()],
                state=cur, files=files, metrics=metrics,
                python=PY, root=ROOT)


# ------------------------------------------------------------------ сервер
class Handler(BaseHTTPRequestHandler):
    server_version = "hdoe-dashboard/1.0"

    def log_message(self, fmt: str, *args) -> None:        # тише в консоли
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

    def _text(self, text: str, code: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:                              # noqa: N802
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        if path == "/api/state":
            return self._json(state_payload())
        if path == "/api/log":
            q = urllib.parse.parse_qs(url.query)
            since = int(q.get("since", ["0"])[0])
            with LOCK:
                chunk = STATE["log"][since:]
                total = len(STATE["log"])
                running = STATE["running"]
                rc = STATE["returncode"]
                started = STATE["started"]
                finished = STATE["finished"]
            return self._json(dict(lines=chunk, total=total, running=running,
                                   returncode=rc, started=started, finished=finished))
        return self._static(path)

    def _static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            path = "/index.html"
        rel = os.path.normpath(path.lstrip("/"))
        full = os.path.join(DASH, rel)
        if not os.path.abspath(full).startswith(os.path.abspath(DASH)) or \
                not os.path.isfile(full):
            return self._text("не найдено", 404)
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".png": "image/png", ".bin": "application/octet-stream"}.get(
                     ext, "application/octet-stream")
        data = open(full, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return self.wfile.write(data)

    # ----------------------------------------------------------- POST
    def do_POST(self) -> None:                             # noqa: N802
        url = urllib.parse.urlparse(self.path)
        raw = self._body()
        if url.path == "/api/run":
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._json(dict(ok=False, message="плохой JSON"), 400)
            if "pipeline" in req:
                pl = PIPELINES.get(str(req["pipeline"]))
                if not pl:
                    return self._json(dict(ok=False, message="нет такого сценария"), 400)
                ids = list(pl["stages"])
            else:
                ids = [str(s) for s in (req.get("stages") or [])] or [str(req.get("stage"))]
            ok, msg = start(ids)
            return self._json(dict(ok=ok, message=msg, stages=ids), 200 if ok else 409)
        if url.path == "/api/stop":
            return self._json(dict(ok=stop()))
        if url.path == "/api/upload":
            return self._upload(raw)
        return self._json(dict(ok=False, message="неизвестный метод"), 404)

    def _upload(self, raw: bytes) -> None:
        """Приём картинки: multipart-поле file либо сырые байты PNG/JPEG."""
        name = "upload.png"
        data = raw
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            boundary = ctype.split("boundary=", 1)[-1].encode()
            parts = raw.split(b"--" + boundary)
            for p in parts:
                if b"filename=" in p and b"\r\n\r\n" in p:
                    head, rest = p.split(b"\r\n\r\n", 1)
                    data = rest.rsplit(b"\r\n", 1)[0]
                    for tag in head.split(b";"):
                        if b"filename=" in tag:
                            name = tag.split(b"filename=", 1)[1].strip(b' "').decode()
                    break
        if len(data) < 128:
            return self._json(dict(ok=False, message="пустой файл"), 400)
        src_dir = os.path.join(ROOT, "data", "source")
        os.makedirs(src_dir, exist_ok=True)
        dst = os.path.join(src_dir, "object.png")
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.save(dst)
            size = im.size
        except Exception as exc:                           # noqa: BLE001
            return self._json(dict(ok=False, message=f"не картинка: {exc}"), 400)
        _append(f"\nпринята картинка {name} {size[0]}×{size[1]} → data/source/object.png\n")
        return self._json(dict(ok=True, file=os.path.relpath(dst, ROOT),
                               width=size[0], height=size[1],
                               next="targets"))


def _have_deps() -> bool:
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:                                      # noqa: BLE001
        return False


def _reexec_into_venv() -> None:
    """Если пульт стартовали системным python без зависимостей, а рядом есть
    .venv — перезапускаем себя интерпретатором venv (иначе не работают ни
    приём картинок, ни сами этапы)."""
    if _have_deps() or PY == sys.executable:
        return
    print(f"перезапуск в интерпретаторе проекта: {PY}", flush=True)
    os.execv(PY, [PY, os.path.abspath(__file__)] + sys.argv[1:])


def main() -> int:
    _reexec_into_venv()
    ap = argparse.ArgumentParser(description="Локальный пульт запуска задач для дашборда")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--allow-lan", action="store_true",
                    help="слушать 0.0.0.0 (по умолчанию только 127.0.0.1)")
    args = ap.parse_args()
    host = "0.0.0.0" if args.allow_lan else args.host
    srv = ThreadingHTTPServer((host, args.port), Handler)
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{args.port}/"
    print(f"пульт: {url}\nкаталог: {ROOT}\npython для этапов: {PY}")
    if not os.path.isdir(os.path.join(ROOT, ".venv")):
        print("ВНИМАНИЕ: .venv не найден — этапы пойдут системным python. "
              "Сначала выполните: bash scripts/setup_env.sh")
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
