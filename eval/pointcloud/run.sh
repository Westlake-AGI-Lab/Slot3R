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

export PYTHONPATH="$SCRIPT_DIR:$REPO:$REPO/src/croco:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export POINT3R_MEMORY_UPDATE_MODE=ordered_kway
export POINT3R_MEMORY_IMPL=tensor
export POINT3R_ORDERED_UPDATE_IMPL=tensor
export POINT3R_KWAY_NUM_SLOTS=8
export POINT3R_ORDERED_THETA_BINS=16
export POINT3R_ORDERED_PHI_BINS=8
export POINT3R_ORDERED_RHO_BINS=32
export POINT3R_SPARSE_READOUT=1
export POINT3R_SPARSE_MAX_TOKENS=640
export POINT3R_SPARSE_GLOBAL_ANCHORS=128
export POINT3R_SPARSE_NEIGHBOR_RANGE=1
export POINT3R_ENCODE_CHUNK_SIZE="${POINT3R_ENCODE_CHUNK_SIZE:-100}"
export POINT3R_CGMC_DROP_QUANTILE="${POINT3R_CGMC_DROP_QUANTILE:-0.25}"
export POINT3R_CGMC_WEIGHTED_MERGE=1
export POINT3R_CONFSELECT_STATS="${POINT3R_CONFSELECT_STATS:-0}"

case "$MODEL" in
  core|ours)
    launcher_prefix=launch_ours
    model_name=slot3r_q25_sparse640
    export POINT3R_RAYAWARE_UPDATE=0
    ;;
  vpc_m|ours_ray)
    launcher_prefix=launch_ours_ray
    model_name=v82e_balanced_predecoder_pose_q25_sparse640
    export POINT3R_RAYAWARE_UPDATE=1
    export POINT3R_RAY_DUAL_BANK=1
    export POINT3R_RAY_BANK_UPDATE_EVERY=4
    export POINT3R_RAY_PAPER_UPDATE=1
    export POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
    export POINT3R_RAY_HYBRID_READOUT=1
    export POINT3R_RAY_POSE_INPUT_ONLY=1
    export POINT3R_RAY_POSE_POST_DECODER_ONLY=0
    export POINT3R_RAY_POSE_ONLY_ENSEMBLE=0
    export POINT3R_RAY_POSE_INPUT_TOKENS=128
    export POINT3R_RAY_POSE_INPUT_MAX_WEIGHT=0.025
    export POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10
    ;;
  vpc_a|ours_rayma)
    launcher_prefix=launch_ours_rayma
    model_name=v106_fresh_bank_pose_q25_sparse640
    export POINT3R_RAYAWARE_UPDATE=1
    export POINT3R_RAY_DUAL_BANK=1
    export POINT3R_RAY_BANK_UPDATE_EVERY=1
    export POINT3R_V106_RAY_BANK_UPDATE_EVERY=1
    export POINT3R_RAY_PAPER_UPDATE=1
    export POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
    export POINT3R_RAY_HYBRID_READOUT=1
    export POINT3R_RAY_POSE_INPUT_ONLY=1
    export POINT3R_RAY_POSE_POST_DECODER_ONLY=0
    export POINT3R_RAY_POSE_ONLY_ENSEMBLE=0
    export POINT3R_RAY_POSE_INPUT_TOKENS=128
    export POINT3R_V106_POSE_INPUT_WEIGHT=0.15
    export POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10
    ;;
  *)
    echo "MODEL must be core, vpc_m, or vpc_a (legacy ours aliases also accepted)" >&2
    exit 2
    ;;
esac

case "$DATASET" in
  nrgbd)
    launcher="$SCRIPT_DIR/${launcher_prefix}_nrgbd.py"
    default_scenes="breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom"
    ;;
  7scenes)
    launcher="$SCRIPT_DIR/${launcher_prefix}_7scenes.py"
    default_scenes="chess/seq-03 chess/seq-05 fire/seq-03 fire/seq-04 heads/seq-01 office/seq-02 office/seq-06 office/seq-07 office/seq-09 pumpkin/seq-01 pumpkin/seq-07 redkitchen/seq-03 redkitchen/seq-04 redkitchen/seq-06 redkitchen/seq-12 redkitchen/seq-14 stairs/seq-01 stairs/seq-04"
    ;;
  *)
    echo "DATASET must be nrgbd or 7scenes" >&2
    exit 2
    ;;
esac

read -r -a scenes <<< "${SCENES:-$default_scenes}"
source "$SCRIPT_DIR/../output_guard.sh"

"$PYTHON_BIN" -B "$launcher" \
  --model "$model_name" \
  --scenes "${scenes[@]}" \
  --point3r_repo "$REPO" \
  --point3r_weights "$WEIGHTS" \
  --nrgbd_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda --size 512 \
  --kf_every "$KF_EVERY" --max_frames "$MAX_FRAMES" \
  --max_points "$MAX_POINTS" \
  --kway_slots 8 --sparse_max_tokens 640 \
  --sparse_global_anchors 128 --sparse_neighbor_range 1
