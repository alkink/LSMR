#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_qhead_softpos"
ITER="${1:-30000}"
SUFFIX="${2:-thr03}"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

python analyze_parity_error_decomposition.py \
  "${CFG}" \
  --testiter "${ITER}" \
  --suffix "${SUFFIX}" \
  --split testing \
  --iou-thresh 0.5 \
  --line-width 30 \
  --log-every 50 \
  --out "results/${CFG}/error_decomposition_${ITER}_${SUFFIX}.json" \
  --details-jsonl "results/${CFG}/error_decomposition_${ITER}_${SUFFIX}.jsonl"
