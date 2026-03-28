#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_2k_condlstr_parity_stdc_visibility"
LOG_FILE="train_${CFG}.log"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

find cache -maxdepth 1 -name 'culane_*.pkl' -delete

python -m py_compile \
  models/parity_stdc_res34_backbone.py \
  models/condlstr_dense_matcher.py \
  models/condlstr_parity_criterion.py \
  models/condlstr_parity_head.py \
  models/condlstr_parity_postprocess.py \
  models/LSTR_CULANE_condlstr_parity_base.py \
  models/LSTR_CULANE_2k_condlstr_parity_stdc_visibility.py \
  train.py \
  test.py \
  make_seed_alias.py

python -u train.py "${CFG}" --threads 4 2>&1 | tee "${LOG_FILE}"
