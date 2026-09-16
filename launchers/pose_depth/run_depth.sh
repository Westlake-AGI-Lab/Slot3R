#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLOT3R_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-bonn}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the benchmark dataset root}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the pretrained Point3R checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/depth/${MODEL:-ours_rayma}_${DATASET}}"
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco${PYTHONPATH:+:$PYTHONPATH}"

source "$SCRIPT_DIR/common_env.sh"
mkdir -p "$OUTPUT_DIR"

if [[ -n "${SCENES:-}" ]]; then
  read -r -a scenes <<< "$SCENES"
  scene_args=(--scenes "${scenes[@]}")
else
  scene_args=()
fi

case "$DATASET" in
  bonn)
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/depth_bonn.py" \
      --model depthcc --bonn_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" \
      --point3r_repo "$REPO" --point3r_weights "$WEIGHTS" \
      --size 512 --kf_every "${KF_EVERY:-1}" --max_frames "${MAX_FRAMES:-0}" \
      --sparse_max_tokens 640 --sparse_global_anchors 128 --drop_quantile 0.25 \
      "${scene_args[@]}"
    ;;
  scannet)
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/depth_scannet.py" \
      --model depthcc --scannet_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" \
      --point3r_repo "$REPO" --point3r_weights "$WEIGHTS" \
      --size 512 --kf_every "${KF_EVERY:-1}" --max_frames "${MAX_FRAMES:-0}" \
      --sparse_max_tokens 640 --sparse_global_anchors 128 --drop_quantile 0.25 \
      "${scene_args[@]}"
    ;;
  kitti)
    # KITTI paths follow eval/video_depth/metadata.py. DATA_ROOT is exported for
    # site-specific metadata files that read it.
    export KITTI_ROOT="$DATA_ROOT"
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/depth_kitti.py" \
      --weights "$WEIGHTS" --output_dir "$OUTPUT_DIR" \
      --eval_dataset kitti --size 512
    ;;
  *)
    echo "DATASET must be bonn, scannet, or kitti" >&2
    exit 2
    ;;
esac
