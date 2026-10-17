#!/usr/bin/env bash
# Prepare Python environment for PassiveRadarOffline.
# Usage: scripts/install.sh [--cpu]   (--cpu installs CPU-only PyTorch)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

command -v "$PYTHON" >/dev/null || { echo "Error: $PYTHON not found"; exit 1; }

if [[ ! -d "$VENV" ]]; then
    echo ">> Creating virtualenv in $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

if [[ "${1:-}" == "--cpu" ]]; then
    echo ">> Installing CPU-only PyTorch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ">> Installing requirements"
pip install -r "$ROOT/requirements.txt"

mkdir -p "$ROOT/output_plots"

echo ">> Verifying"
python - <<'EOF'
import numpy, scipy, matplotlib, torch
print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}")
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF

echo ">> Done. Run processing with: scripts/run.sh"
