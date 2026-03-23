#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

export PYTHONPATH="$(pwd)"
PORT_TO_USE="${PORT:-8000}"
python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT_TO_USE"
