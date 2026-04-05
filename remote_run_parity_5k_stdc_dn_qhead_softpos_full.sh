#!/bin/bash
set -euo pipefail

ITER="${1:-30000}"
SUFFIX="${2:-thr03}"

echo "[1/3] train"
bash remote_run_parity_5k_stdc_dn_qhead_softpos_train.sh

echo "[2/3] test+eval"
bash remote_run_parity_5k_stdc_dn_qhead_softpos_test_eval.sh "${ITER}" "${SUFFIX}"

echo "[3/3] error decomposition"
bash remote_run_parity_5k_stdc_dn_qhead_softpos_analyze.sh "${ITER}" "${SUFFIX}"

echo
echo "[summary]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/LSTR_CULANE_5k_condlstr_parity_stdc_dn_qhead_softpos/${ITER}_${SUFFIX}_iou0.5.txt" || true
