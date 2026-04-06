#!/bin/bash
set -euo pipefail

ITER="${1:-500000}"
SUFFIX="${2:-thr03}"

echo "[1/2] train"
bash remote_run_parity_full_stdc_dn_qhead_train.sh

echo "[2/2] test+eval"
bash remote_run_parity_full_stdc_dn_qhead_test_eval.sh "${ITER}" "${SUFFIX}"

echo
echo "[summary]"
grep -E "precision|recall|Fmeasure|FPS" \
  "results/LSTR_CULANE_full_condlstr_parity_stdc_dn_qhead/${ITER}_${SUFFIX}_iou0.5.txt" || true
