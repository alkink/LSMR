#!/usr/bin/env bash
# Evaluate qhead checkpoint under different fusion modes.
# No retraining — same checkpoint, only postprocess changes.
# Runs: sqrt / linear / obj_only, each at thr025 / thr03 / thr04
#
# Usage:
#   bash remote_run_parity_5k_stdc_dn_qhead_fusion_eval.sh
#
# Requires: qhead checkpoint at iter 30000 already exists.

set -e
cd /workspace/LSMR
conda activate clrernet

CFG="LSTR_CULANE_5k_condlstr_parity_stdc_dn_qhead"
ITER=30000

FUSIONS=("sqrt" "linear" "obj_only")
SUFFIXES=("thr025" "thr03" "thr04")

for FUSION in "${FUSIONS[@]}"; do
    for SUFFIX in "${SUFFIXES[@]}"; do
        EVAL_SUFFIX="${SUFFIX}_${FUSION}"
        echo ""
        echo "=========================================="
        echo "  fusion=${FUSION}  threshold_config=${SUFFIX}  out_suffix=${EVAL_SUFFIX}"
        echo "=========================================="

        LSTR_PARITY_FUSION_MODE="${FUSION}" python test.py \
            "${CFG}" \
            --testiter ${ITER} \
            --suffix "${EVAL_SUFFIX}" \
            --split testing

        python eval_culane.py \
            "${CFG}" \
            --testiter ${ITER} \
            --suffix "${EVAL_SUFFIX}" \
            --split testing \
            | tee "results/${CFG}/${ITER}_${EVAL_SUFFIX}_iou0.5.txt"

        echo ">>> ${EVAL_SUFFIX} done."
    done
done

echo ""
echo "=== ALL FUSION EVALS DONE ==="
echo "Results in: results/${CFG}/"
echo ""
echo "Quick comparison:"
for FUSION in "${FUSIONS[@]}"; do
    echo ""
    echo "--- fusion=${FUSION} ---"
    for SUFFIX in "${SUFFIXES[@]}"; do
        EVAL_SUFFIX="${SUFFIX}_${FUSION}"
        F="${FUSION}"
        RESULT_FILE="results/${CFG}/${ITER}_${EVAL_SUFFIX}_iou0.5.txt"
        if [ -f "${RESULT_FILE}" ]; then
            echo -n "  ${EVAL_SUFFIX}: "
            grep -oE "F1 = [0-9.]+" "${RESULT_FILE}" | head -1 || cat "${RESULT_FILE}"
        fi
    done
done
