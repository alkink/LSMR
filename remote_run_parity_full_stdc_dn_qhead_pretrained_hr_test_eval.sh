#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_full_condlstr_parity_stdc_dn_qhead_pretrained_hr"
ITER="${1:-50000}"
SUFFIX="${2:-thr03}"
TEST_LOG="test_${CFG}_${ITER}_${SUFFIX}.log"
EVAL_LOG="eval_${CFG}_${ITER}_${SUFFIX}.log"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

python -u test.py "${CFG}" --suffix "${SUFFIX}" --modality eval --split testing --testiter "${ITER}" \
  2>&1 | tee "${TEST_LOG}"

bash eval_progressive.sh "${CFG}" "${ITER}" "${SUFFIX}" \
  2>&1 | tee "${EVAL_LOG}"
