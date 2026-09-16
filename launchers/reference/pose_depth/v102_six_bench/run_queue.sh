#!/usr/bin/env bash
set -uo pipefail
BASE=/root/autodl-tmp/v102_six_bench
LOG="$BASE/queue.log"
mkdir -p "$BASE/results"
: >"$LOG"
run_one() {
  name=$1; script=$2
  echo "START $name $(date -Is)" | tee -a "$LOG"
  bash "$script" >>"$LOG" 2>&1
  status=$?
  echo "END $name status=$status $(date -Is)" | tee -a "$LOG"
  return 0
}
run_one pose_tum "$BASE/run_pose_tum.sh"
run_one pose_scannet "$BASE/run_pose_scannet.sh"
run_one depth_bonn "$BASE/run_depth_bonn.sh"
run_one depth_kitti "$BASE/run_depth_kitti.sh"
run_one depth_scannet "$BASE/run_depth_scannet.sh"
echo "ALL_DONE $(date -Is)" | tee -a "$LOG"

