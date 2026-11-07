#!/usr/bin/env bash
# Восстановление окружения (Python 3.10+). Используется один раз перед запуском.
#   bash scripts/setup_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --no-cache-dir -r "$ROOT/requirements.txt"
echo "Окружение готово: $VENV/bin/python"
