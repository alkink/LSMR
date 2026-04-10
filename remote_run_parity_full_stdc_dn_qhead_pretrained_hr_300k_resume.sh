#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
SRC_CFG="LSTR_CULANE_full_condlstr_parity_stdc_dn_qhead_pretrained_hr"
CFG="LSTR_CULANE_full_condlstr_parity_stdc_dn_qhead_pretrained_hr_300k"
START_ITER=150000
LOG_FILE="train_${CFG}_resume_from_${START_ITER}.log"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "cache/nnet/${CFG}"
cp -n "cache/nnet/${SRC_CFG}/${SRC_CFG}_${START_ITER}.pkl" \
      "cache/nnet/${CFG}/${CFG}_${START_ITER}.pkl"

python -m py_compile \
  models/parity_stdc_res34_backbone.py \
  models/condlstr_dn_lane.py \
  models/condlstr_parity_head.py \
  models/condlstr_parity_criterion.py \
  models/condlstr_parity_postprocess.py \
  models/LSTR_CULANE_condlstr_parity_base.py \
  models/LSTR_CULANE_full_condlstr_parity_stdc_dn_qhead_pretrained_hr_300k.py \
  models/py_utils/transformer.py \
  sample/culane.py \
  nnet/py_factory.py \
  train.py \
  test.py

python -u train.py "${CFG}" --iter "${START_ITER}" --threads 4 2>&1 | tee "${LOG_FILE}"
