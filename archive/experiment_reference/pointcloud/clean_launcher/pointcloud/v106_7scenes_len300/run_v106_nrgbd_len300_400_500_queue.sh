#!/usr/bin/env bash
set -uo pipefail
RUNNER=/root/autodl-tmp/clean_launcher/pointcloud/v106_7scenes_len300
LOG="$RUNNER/nrgbd_len300_400_500_queue.log"
: >"$LOG"
for run_len in 300 400 500; do
  echo "[QUEUE_START] len=$run_len $(date -Is)" | tee -a "$LOG"
  RUN_LEN="$run_len" bash "$RUNNER/run_v106_nrgbd9_variable_len.sh" >>"$LOG" 2>&1
  status=$?
  echo "[QUEUE_DONE] len=$run_len status=$status $(date -Is)" | tee -a "$LOG"
done
echo "[QUEUE_ALL_DONE] $(date -Is)" | tee -a "$LOG"
