#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/clean_launchers/rayaway_pose_v4/depth_v82k_benchmark
echo '=== JOB ==='
pgrep -af 'run_v82k_depth3|depth_scannet_v82k|depth_bonn_v82k|launch_kitti_v82k' || true
echo '=== PIPELINE ==='
cat "$BASE/pipeline.log" 2>/dev/null || true
echo '=== PROGRESS ==='
for f in "$BASE/run/scannet/metric/summary.tsv" "$BASE/run/scannet/scale_shift/summary.tsv" "$BASE/run/bonn/metric/summary.tsv" "$BASE/run/bonn/scale_shift/summary.tsv"; do
  [[ -f "$f" ]] && echo "$f rows=$(( $(wc -l < "$f") - 1 ))" || true
done
find "$BASE/run/kitti" -type f -name 'frame_*.npy' 2>/dev/null | wc -l | awk '{print "kitti_predictions=" $1}'
echo '=== RECENT ==='
for f in "$BASE/run/scannet/infer.log" "$BASE/run/bonn/metric.log" "$BASE/run/bonn/scale_shift.log" "$BASE/run/kitti/infer.log"; do
  [[ -f "$f" ]] && { echo "--- $f"; tail -5 "$f"; }
done
