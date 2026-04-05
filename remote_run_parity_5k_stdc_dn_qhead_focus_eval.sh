#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_qhead"
ITER="${1:-30000}"
THRESH="0.30"
THRESH_TAG="thr03"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export LSTR_DATA_DIR=/workspace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export LSTR_PARITY_SCORE_THRESH="${THRESH}"

FUSIONS=("linear" "obj_only")

for FUSION in "${FUSIONS[@]}"; do
  OUTPUT_SUFFIX="${THRESH_TAG}_${FUSION}"

  echo
  echo "=========================================="
  echo "fusion=${FUSION} score_thresh=${THRESH} output_suffix=${OUTPUT_SUFFIX}"
  echo "=========================================="

  export LSTR_PARITY_FUSION_MODE="${FUSION}"

  python -u test.py "${CFG}" --suffix "${OUTPUT_SUFFIX}" --modality eval --split testing --testiter "${ITER}" \
    2>&1 | tee "test_${CFG}_${ITER}_${OUTPUT_SUFFIX}.log"

  bash eval_progressive.sh "${CFG}" "${ITER}" "${OUTPUT_SUFFIX}" \
    2>&1 | tee "eval_${CFG}_${ITER}_${OUTPUT_SUFFIX}.log"

  echo "[done] ${OUTPUT_SUFFIX}"
done

echo
echo "[summary]"
for FUSION in "${FUSIONS[@]}"; do
  OUTPUT_SUFFIX="${THRESH_TAG}_${FUSION}"
  RESULT_FILE="results/${CFG}/${ITER}_${OUTPUT_SUFFIX}_iou0.5.txt"
  echo "[${OUTPUT_SUFFIX}]"
  grep -E "precision|recall|Fmeasure|FPS" "${RESULT_FILE}" || true
  echo
done
