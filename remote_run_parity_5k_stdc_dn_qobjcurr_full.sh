#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_qobjcurr"
ITER="${1:-30000}"
SUFFIX="${2:-thr04}"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "[1/3] train"
bash remote_run_parity_5k_stdc_dn_qobjcurr_train.sh

echo "[2/3] test+eval"
bash remote_run_parity_5k_stdc_dn_qobjcurr_test_eval.sh "${ITER}" "${SUFFIX}"

echo "[3/3] error decomposition"
bash remote_run_parity_5k_stdc_dn_qobjcurr_analyze.sh "${ITER}" "${SUFFIX}"

echo
echo "[summary]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/${CFG}/${ITER}_${SUFFIX}_iou0.5.txt" || true
