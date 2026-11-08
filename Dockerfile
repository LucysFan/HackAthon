# ---- Сборка зависимостей (один слой, без дублирования пакетов) ----
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HDOE_ENV=docker \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg

# ffmpeg — для видео-демо, libgl/libglib — для pillow/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch/torchvision ставим CPU-сборкой: в контейнере всё равно нет GPU, образ втрое меньше
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
    && pip install -r requirements.txt

COPY . .

# Комплекты, которые приносит пользователь, живут в томе — каталог готовим заранее.
# best.pt обязателен: без обученных весов проверять нечего, сборка падает намеренно.
RUN mkdir -p data/sets outputs/_cache \
    && test -f outputs/best.pt \
    && python -c "import torch, numpy, skimage, scipy, PIL, matplotlib, reportlab, imageio; \
print('deps ok', torch.__version__)"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/state', timeout=4).status == 200 else 1)"

# Всё, что нужно заказчику: подняться и показать дашборд
CMD ["python", "scripts/serve_dashboard.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
