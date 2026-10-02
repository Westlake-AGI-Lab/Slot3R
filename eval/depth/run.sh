#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLOT3R_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-core}"
DATASET="${DATASET:-bonn}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the prepared dataset root}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the pretrained Point3R checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/depth/${MODEL}_${DATASET}}"
case "$MODEL" in core|vpc_m|vpc_a|point3r|ours|ours_ray|ours_rayma) ;; *) echo "Invalid MODEL: $MODEL" >&2; exit 2 ;; esac
case "$DATASET" in bonn|scannet|kitti) ;; *) echo "Invalid DATASET: $DATASET" >&2; exit 2 ;; esac
extra_args=()
if [[ -n "${SCENES:-}" ]]; then
  read -r -a scenes <<< "$SCENES"
  extra_args+=(--scenes "${scenes[@]}")
fi
if [[ -n "${ALIGN:-}" ]]; then extra_args+=(--align "$ALIGN"); fi
if [[ -n "${MAX_DEPTH:-}" ]]; then extra_args+=(--max_depth "$MAX_DEPTH"); fi
source "$SCRIPT_DIR/../output_guard.sh"
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u -B "$SCRIPT_DIR/launch.py" \
  --model "$MODEL" --dataset "$DATASET" --weights "$WEIGHTS" --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" --size 512 \
  --kf_every "${KF_EVERY:-1}" --max_frames "${MAX_FRAMES:-0}" \
  "${extra_args[@]}" "$@"
