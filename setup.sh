#!/usr/bin/env bash
# Self-contained setup: create a local virtualenv, install deps, build the motion cache.
# Everything stays inside this folder -- no system or conda changes.
#
#   ./setup.sh            # create .venv and install + build cache
#   source .venv/bin/activate && python run.py
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo ">> Creating virtualenv in .venv (using $PY)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Upgrading pip and installing requirements"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo ">> Building the motion-matching feature cache (data/motion_lib.npz)"
python run.py --build-only

cat <<'EOF'

Setup complete.

  Run the interactive viewer:
      source .venv/bin/activate
      python run.py

  Controls: WASD move, Shift run, Space reset, drag orbit, scroll zoom, Esc quit.

  Headless host? The viewer needs a display. Use a desktop / X-forwarded session
  and ensure MUJOCO_GL=glfw (the default).
EOF
