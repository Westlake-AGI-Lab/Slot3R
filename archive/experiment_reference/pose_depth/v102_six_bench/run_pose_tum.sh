#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
ROOT=/root/autodl-tmp/clean_launchers/rayaway_pose_v4
export POINT3R_EXPERIMENT_TAG=v102_tum_full_clean
POINT3R_SKIP_SINTEL=1 POINT3R_SKIP_TUM=0 bash "$ROOT/run_v54_base_pose.sh"
