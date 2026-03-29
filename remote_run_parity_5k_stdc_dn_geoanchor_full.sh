#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
BASE_CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_geoanchor"
SEED="${1:-901}"
ITER="${2:-30000}"
SUFFIX="${3:-thr04}"

if [[ "${SEED}" == "317" ]]; then
  CFG="${BASE_CFG}"
else
  CFG="${BASE_CFG}_seed${SEED}"
fi

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ "${CFG}" != "${BASE_CFG}" ]]; then
  echo "[1/4] create seed alias: ${CFG}"
  python make_seed_alias.py "${BASE_CFG}" "${CFG}" "${SEED}"
else
  echo "[1/4] using base config: ${CFG}"
fi

echo "[2/4] seed${SEED} train"
find cache -maxdepth 1 -name 'culane_*.pkl' -delete
rm -f "results/${CFG}/match_diag_train.jsonl"
python -u train.py "${CFG}" --threads 4 \
  2>&1 | tee "train_${CFG}.log"

echo "[3/4] seed${SEED} test"
python -u test.py "${CFG}" --suffix "${SUFFIX}" --modality eval --split testing --testiter "${ITER}" \
  2>&1 | tee "test_${CFG}_${ITER}_${SUFFIX}.log"

echo "[4/4] seed${SEED} eval"
bash eval_progressive.sh "${CFG}" "${ITER}" "${SUFFIX}" \
  2>&1 | tee "eval_${CFG}_${ITER}_${SUFFIX}.log"

echo
echo "[summary]"
echo "[${CFG}]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/${CFG}/${ITER}_${SUFFIX}_iou0.5.txt" || true
