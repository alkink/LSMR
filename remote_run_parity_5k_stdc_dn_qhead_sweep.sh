#!/bin/bash
set -euo pipefail

REPO_ROOT="/workspace/LSMR"
CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_qhead"
ITER="${1:-30000}"
shift || true

if [ "$#" -gt 0 ]; then
  SUFFIXES=("$@")
else
  SUFFIXES=("thr02" "thr025" "thr03" "thr04")
fi

cd "${REPO_ROOT}"

for SUFFIX in "${SUFFIXES[@]}"; do
  echo "[sweep] ${CFG} ${ITER} ${SUFFIX}"
  bash remote_run_parity_5k_stdc_dn_qhead_test_eval.sh "${ITER}" "${SUFFIX}"
done

echo
echo "[summary]"
for SUFFIX in "${SUFFIXES[@]}"; do
  RESULT_PATH="results/${CFG}/${ITER}_${SUFFIX}_iou0.5.txt"
  echo "[${SUFFIX}]"
  grep -E "precision|recall|Fmeasure|FPS" "${RESULT_PATH}" || true
  echo
done
