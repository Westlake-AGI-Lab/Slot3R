#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/v102_six_bench/results/depth_scannet
pgrep -af 'run_depth_scannet_scale_shift|depth_scannet.py.*sequence_scale_shift' || true
echo -n 'COMPLETED='; grep -c '^\[depth_scene\]' "$BASE/scale_shift.log" 2>/dev/null || true
echo -n 'ERRORS='; grep -Eic 'Traceback|OutOfMemory|CUDA out of memory|status=FAIL' "$BASE/scale_shift.log" 2>/dev/null || true
tail -8 "$BASE/scale_shift.log" 2>/dev/null || true
tail -5 /root/autodl-tmp/v102_six_bench/depth_scannet_scale_shift_launcher.log 2>/dev/null || true
