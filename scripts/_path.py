"""Добавляет корень проекта в sys.path (для запуска `python scripts/<name>.py`)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
