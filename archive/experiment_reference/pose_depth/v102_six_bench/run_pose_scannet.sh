#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
export POINT3R_SCANNET_TAG=v102_fullstate_scannet_full
bash /root/autodl-tmp/v102_six_bench/run_scannet_generic.sh

