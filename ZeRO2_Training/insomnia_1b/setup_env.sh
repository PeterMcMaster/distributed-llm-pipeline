#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-insomnia}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] root_dir=$ROOT_DIR"
echo "[setup] venv_dir=$VENV_DIR"
echo "[setup] python_bin=$PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] Could not find $PYTHON_BIN in PATH."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.2.0') else 1)" >/dev/null 2>&1; then
  python -c "import torch; print('[setup] local torch', torch.__version__)"
else
  echo "[setup] Installing pinned local torch wheel for CUDA 12.1..."
  python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.2.0"
  python -c "import torch; print('[setup] local torch', torch.__version__)"
fi

# Install the non-torch dependencies before importing deepspeed. Some of
# DeepSpeed's modules import cpuinfo/numpy immediately at import time.
python -m pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt"

if python -c "from importlib.metadata import version; import sys; sys.exit(0 if version('deepspeed') == '0.16.9' else 1)" >/dev/null 2>&1; then
  python -c "from importlib.metadata import version; print('[setup] local deepspeed', version('deepspeed'))"
else
  DS_BUILD_OPS=0 python -m pip install --no-cache-dir --no-deps "deepspeed==0.16.9"
  python -c "from importlib.metadata import version; print('[setup] local deepspeed', version('deepspeed'))"
fi

python -c "import torch, transformers, datasets; from importlib.metadata import version; print('[setup] torch', torch.__version__); print('[setup] deepspeed', version('deepspeed')); print('[setup] transformers', transformers.__version__); print('[setup] datasets', datasets.__version__)"
