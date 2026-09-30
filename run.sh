#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
exec .venv/bin/python -m streamlit run app.py --server.headless true
