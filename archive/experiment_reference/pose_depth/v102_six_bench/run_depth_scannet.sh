#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
PY=/root/miniconda3/envs/point3r/bin/python
OUT=/root/autodl-tmp/v102_six_bench/results/depth_scannet
mkdir -p "$OUT/metric"
"$PY" -u -B /root/autodl-tmp/v102_six_bench/depth_scannet.py --model depthcc --scannet_root /root/autodl-tmp/scannetv2 --output_dir "$OUT/metric" --size 512 --align none --max_depth 5 --drop_quantile 0.25 --sparse_max_tokens 640 >"$OUT/infer.log" 2>&1
status=$?
echo "SCANNET_DEPTH_DONE status=$status $(date -Is)"
exit "$status"

