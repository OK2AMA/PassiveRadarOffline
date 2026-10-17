#!/usr/bin/env bash
# Run passive radar processing.
# Usage:
#   scripts/run.sh                 # default: run_analysis.py --config config.json
#   scripts/run.sh <script.py> [args...]
#   scripts/run.sh --list          # list available processing scripts
# Examples:
#   scripts/run.sh process_dvbt_phased.py
#   scripts/run.sh run_analysis.py --file_a ch1_Fs102MHz_Fc640kHz_FMradio \
#        --file_b ch2_Fs102MHz_Fc640kHz_FMradio --fs 640000 --fc 102e6
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
cd "$ROOT"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Virtualenv not found. Run scripts/install.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ "${1:-}" == "--list" ]]; then
    ls -1 "$ROOT"/{run_,process_,integrate_,scan_,plot_,phased_}*.py 2>/dev/null | xargs -n1 basename
    exit 0
fi

SCRIPT="run_analysis.py"
if [[ "${1:-}" == *.py ]]; then
    SCRIPT="$1"; shift
fi
[[ -f "$SCRIPT" ]] || { echo "Error: $SCRIPT not found (use --list)"; exit 1; }

ARGS=("$@")
if [[ "$SCRIPT" == "run_analysis.py" && ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(--config config.json)
fi

mkdir -p output_plots
echo ">> python $SCRIPT ${ARGS[*]:-}"
exec python "$SCRIPT" "${ARGS[@]}"
