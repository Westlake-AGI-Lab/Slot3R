#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
PY=/root/miniconda3/envs/point3r/bin/python
BASE=/root/autodl-tmp/v102_six_bench/results/depth_bonn
mkdir -p "$BASE/metric" "$BASE/scale_shift"
"$PY" -u -B /root/autodl-tmp/v102_six_bench/depth_bonn.py --model depthcc --bonn_root /root/autodl-tmp/Bonn_unzip --output_dir "$BASE/metric" --size 512 --align none --max_depth 5 --drop_quantile 0.25 --sparse_max_tokens 640 >"$BASE/metric.log" 2>&1
ms=$?
"$PY" -u -B /root/autodl-tmp/v102_six_bench/depth_bonn.py --model depthcc --bonn_root /root/autodl-tmp/Bonn_unzip --output_dir "$BASE/scale_shift" --size 512 --align sequence_scale_shift --max_depth 5 --drop_quantile 0.25 --sparse_max_tokens 640 >"$BASE/scale_shift.log" 2>&1
ss=$?
echo "BONN_DONE metric=$ms scale_shift=$ss $(date -Is)"
exit $((ms != 0 || ss != 0))

