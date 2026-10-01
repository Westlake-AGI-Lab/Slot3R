#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLOT3R_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-core}"
DATASET="${DATASET:-nrgbd}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the 7Scenes or NeuralRGBD root}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the pretrained Point3R checkpoint}"
KF_EVERY="${KF_EVERY:-2}"
MAX_FRAMES="${MAX_FRAMES:-300}"
MAX_POINTS="${MAX_POINTS:-999999}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/pointcloud/${MODEL}_${DATASET}_kf${KF_EVERY}_len${MAX_FRAMES}}"
case "$MODEL" in
  core|vpc_m|vpc_a|point3r|ours|ours_ray|ours_rayma) ;;
  *) echo "MODEL must be core, vpc_m, vpc_a, or point3r (legacy ours aliases accepted)" >&2; exit 2 ;;
esac
case "$DATASET" in
  nrgbd|7scenes) ;;
  *) echo "DATASET must be nrgbd or 7scenes" >&2; exit 2 ;;
esac
scene_args=()
if [[ -n "${SCENES:-}" ]]; then
  read -r -a scenes <<< "$SCENES"
  scene_args=(--scenes "${scenes[@]}")
fi
source "$SCRIPT_DIR/../output_guard.sh"
export PYTHONPATH="$REPO:$REPO/src/croco:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -B "$SCRIPT_DIR/launch.py" \
  --model "$MODEL" --dataset "$DATASET" \
  --point3r_repo "$REPO" --weights "$WEIGHTS" --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" --device cuda --size 512 \
  --kf_every "$KF_EVERY" --max_frames "$MAX_FRAMES" --max_points "$MAX_POINTS" \
  "${scene_args[@]}" "$@"
