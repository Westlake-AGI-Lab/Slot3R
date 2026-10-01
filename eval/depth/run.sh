#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLOT3R_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-core}"
DATASET="${DATASET:-bonn}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the benchmark dataset root}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the pretrained Point3R checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/depth/${MODEL:-core}_${DATASET}}"
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco${PYTHONPATH:+:$PYTHONPATH}"

source "$SCRIPT_DIR/../common_env.sh"
source "$SCRIPT_DIR/../output_guard.sh"

if [[ -n "${SCENES:-}" ]]; then
  read -r -a scenes <<< "$SCENES"
  scene_args=(--scenes "${scenes[@]}")
else
  scene_args=()
fi

case "$DATASET" in
  bonn)
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/bonn.py" \
      --model depthcc --bonn_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" \
      --point3r_repo "$REPO" --point3r_weights "$WEIGHTS" \
      --align "${ALIGN:-sequence_scale_shift}" --max_depth "${MAX_DEPTH:-5}" \
      --size 512 --kf_every "${KF_EVERY:-1}" --max_frames "${MAX_FRAMES:-0}" \
      --sparse_max_tokens 640 --sparse_global_anchors 128 --drop_quantile 0.25 \
      "${scene_args[@]}"
    ;;
  scannet)
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/scannet.py" \
      --model depthcc --scannet_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" \
      --point3r_repo "$REPO" --point3r_weights "$WEIGHTS" \
      --align "${ALIGN:-sequence_scale_shift}" --max_depth "${MAX_DEPTH:-5}" \
      --size 512 --kf_every "${KF_EVERY:-1}" --max_frames "${MAX_FRAMES:-0}" \
      --sparse_max_tokens 640 --sparse_global_anchors 128 --drop_quantile 0.25 \
      "${scene_args[@]}"
    ;;
  kitti)
    # Prepared root contains image_gathered/ and groundtruth_depth_gathered/.
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/kitti.py" \
      --weights "$WEIGHTS" --output_dir "$OUTPUT_DIR" \
      --data_root "$DATA_ROOT" --align "${ALIGN:-scale_shift}" --eval_dataset kitti --size 512 "${scene_args[@]}"
    ;;
  *)
    echo "DATASET must be bonn, scannet, or kitti" >&2
    exit 2
    ;;
esac
