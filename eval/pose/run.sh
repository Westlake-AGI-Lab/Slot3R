#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLOT3R_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-core}"
DATASET="${DATASET:-sintel}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the pretrained Point3R checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/pose/${MODEL:-core}_${DATASET}}"
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco${PYTHONPATH:+:$PYTHONPATH}"

source "$SCRIPT_DIR/../common_env.sh"
source "$SCRIPT_DIR/../output_guard.sh"

case "$DATASET" in
  sintel)
    DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the Sintel training directory}"
    read -r -a scenes <<< "${SCENES:-alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3}"
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/sintel.py" \
      --weights "$WEIGHTS" --output_dir "$OUTPUT_DIR" \
      --data_root "$DATA_ROOT" --eval_dataset sintel --size 512 --seq_list "${scenes[@]}"
    ;;
  tum|scannet)
    DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the TUM-Dynamic or ScanNet root}"
    if [[ -n "${SCENES:-}" ]]; then
      read -r -a scenes <<< "$SCENES"
      scene_args=(--scenes "${scenes[@]}")
    else
      scene_args=()
    fi
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/scannet_tum.py" \
      --model sparse640_q25 \
      --point3r_repo "$REPO" --point3r_weights "$WEIGHTS" \
      --point3r_bks_backend "$SCRIPT_DIR/backend.py" \
      --dataset "$DATASET" --scannet_root "$DATA_ROOT" \
      --output_dir "$OUTPUT_DIR" --size 512 --pose_eval_stride 1 \
      "${scene_args[@]}"
    ;;
  *)
    echo "DATASET must be sintel, tum, or scannet" >&2
    exit 2
    ;;
esac
