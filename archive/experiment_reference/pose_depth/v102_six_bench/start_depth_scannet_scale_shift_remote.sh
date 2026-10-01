#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/v102_six_bench
if pgrep -af 'run_depth_scannet_scale_shift|depth_scannet.py.*sequence_scale_shift' | grep -v pgrep; then
  echo '[NOT_STARTED] ScanNet scale-shift evaluation already exists'
  exit 2
fi
nohup bash "$BASE/run_depth_scannet_scale_shift.sh" >"$BASE/depth_scannet_scale_shift_launcher.log" 2>&1 &
echo "[STARTED] pid=$!"
