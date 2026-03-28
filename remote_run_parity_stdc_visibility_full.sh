#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
BASE_CFG="LSTR_CULANE_2k_condlstr_parity_stdc_visibility"
SEED2="${1:-901}"
ITER="${2:-12500}"
SUFFIX="${3:-thr04}"
SEED2_CFG="${BASE_CFG}_seed${SEED2}"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "[1/6] seed317 train"
bash remote_run_parity_stdc_visibility_train.sh

echo "[2/6] seed317 test+eval"
bash remote_run_parity_stdc_visibility_test_eval.sh "${ITER}" "${SUFFIX}"

echo "[3/6] create seed alias: ${SEED2_CFG}"
python make_seed_alias.py "${BASE_CFG}" "${SEED2_CFG}" "${SEED2}"

echo "[4/6] seed${SEED2} train"
find cache -maxdepth 1 -name 'culane_*.pkl' -delete
python -u train.py "${SEED2_CFG}" --threads 4 \
  2>&1 | tee "train_${SEED2_CFG}.log"

echo "[5/6] seed${SEED2} test"
python -u test.py "${SEED2_CFG}" --suffix "${SUFFIX}" --modality eval --split testing --testiter "${ITER}" \
  2>&1 | tee "test_${SEED2_CFG}_${ITER}_${SUFFIX}.log"

echo "[6/6] seed${SEED2} eval"
bash eval_progressive.sh "${SEED2_CFG}" "${ITER}" "${SUFFIX}" \
  2>&1 | tee "eval_${SEED2_CFG}_${ITER}_${SUFFIX}.log"

echo
echo "[summary]"
echo "[${BASE_CFG} seed317]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/${BASE_CFG}/${ITER}_${SUFFIX}_iou0.5.txt" || true

echo
echo "[${SEED2_CFG}]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/${SEED2_CFG}/${ITER}_${SUFFIX}_iou0.5.txt" || true
