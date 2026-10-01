#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
PY=/root/miniconda3/envs/point3r/bin/python
BASE=/root/autodl-tmp/v102_six_bench/results/depth_scannet
OUT="$BASE/scale_shift"
mkdir -p "$OUT"
"$PY" -u -B /root/autodl-tmp/v102_six_bench/depth_scannet.py \
  --model depthcc --scannet_root /root/autodl-tmp/scannetv2 --output_dir "$OUT" \
  --size 512 --align sequence_scale_shift --max_depth 5 --drop_quantile 0.25 --sparse_max_tokens 640 \
  >"$BASE/scale_shift.log" 2>&1
status=$?
echo "SCANNET_SCALE_SHIFT_DONE status=$status $(date -Is)"
exit "$status"
