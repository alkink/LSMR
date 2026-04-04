#!/bin/bash
set -euo pipefail

ITER="${1:-30000}"
shift || true

echo "[1/2] train"
bash remote_run_parity_5k_stdc_dn_qhead_train.sh

echo "[2/2] threshold sweep"
bash remote_run_parity_5k_stdc_dn_qhead_sweep.sh "${ITER}" "$@"
