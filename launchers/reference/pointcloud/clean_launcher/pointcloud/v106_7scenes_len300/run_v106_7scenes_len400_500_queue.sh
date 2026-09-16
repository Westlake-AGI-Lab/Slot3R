#!/usr/bin/env bash
set -uo pipefail

RUNNER=/root/autodl-tmp/clean_launcher/pointcloud/v106_7scenes_len300
QUEUE_LOG="$RUNNER/len400_500_queue.log"
: >"$QUEUE_LOG"

for run_len in 400 500; do
  echo "[QUEUE_START] len=$run_len $(date -Is)" | tee -a "$QUEUE_LOG"
  RUN_LEN="$run_len" bash "$RUNNER/run_v106_7scenes18_len300.sh" >>"$QUEUE_LOG" 2>&1
  status=$?
  echo "[QUEUE_DONE] len=$run_len status=$status $(date -Is)" | tee -a "$QUEUE_LOG"
done

echo "[QUEUE_ALL_DONE] $(date -Is)" | tee -a "$QUEUE_LOG"
