#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

REPO=/root/autodl-tmp/Point3R_mdf
RUNNER_DIR=/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300
RESULT_BASE=/root/autodl-tmp/clean_launcher/pointcloud/results
MAX_FRAMES=${V82E_NRGBD_MAX_FRAMES:?set V82E_NRGBD_MAX_FRAMES to 400 or 500}
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$RESULT_BASE/v82e_balanced_predecoder_pose_q25_sparse640_nrgbd9_kf2_len${MAX_FRAMES}_843_$STAMP"
MASTER="$ROOT/master.log"
mkdir -p "$ROOT"
: > "$MASTER"

export PYTHONPATH="$RUNNER_DIR:/root/autodl-tmp/clean_launcher/pointcloud:$REPO:$REPO/src/croco:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Slot3R main settings.
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
export POINT3R_ENCODE_CHUNK_SIZE=100
export POINT3R_CGMC_DROP_QUANTILE=0.25
export POINT3R_CGMC_WEIGHTED_MERGE=1
export POINT3R_CONFSELECT_STATS=0

# Geometry-safe motion-gated single pose-readout settings.
export POINT3R_LC_STATE_LOCAL_ROT_MODE=0
export POINT3R_LC_STATE_LOW_RAY_FUSION=1
export POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY=1
export POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION=0
export POINT3R_LC_STATE_RAY_FUSION_PRIOR=1
export POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG=2
export POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG=2
export POINT3R_LC_ODOM_CENTERED_ROTATION=0
export POINT3R_LC_ODOM_ESSENTIAL_ROTATION=0
export POINT3R_RAY_DUAL_BANK=1
export POINT3R_RAY_BANK_UPDATE_EVERY=4
export POINT3R_RAYAWARE_UPDATE=1
export POINT3R_RAY_PAPER_UPDATE=1
export POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
export POINT3R_RAY_HYBRID_READOUT=1
export POINT3R_RAY_POSE_ONLY_ENSEMBLE=0
export POINT3R_RAY_POSE_INPUT_ONLY=1
export POINT3R_RAY_POSE_POST_DECODER_ONLY=0
export POINT3R_RAY_POSE_INPUT_TOKENS=128
export POINT3R_RAY_POSE_INPUT_MAX_WEIGHT=0.025
export POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10
export POINT3R_RAY_POSE_INPUT_JERK_LOW=0.010
export POINT3R_RAY_POSE_INPUT_JERK_HIGH=0.035
export POINT3R_RAY_HYBRID_TOKENS=640
export POINT3R_RAY_HYBRID_STABLE_FRAC=0.60
export POINT3R_RAY_HYBRID_ANCHOR_FRAC=0.10
export POINT3R_RAY_HYBRID_RAY_WEIGHT=0.25
export POINT3R_RAY_POSE_ROT_GATE_DEG=4.0
export POINT3R_RAY_POSE_TRANS_GATE=0.20
export POINT3R_RAY_POSE_MAX_WEIGHT=0.35
export POINT3R_RAY_POINTER_LOOP_GRAPH=0
export POINT3R_RAY_POINTER_LOOP_PGO=0
export POINT3R_LC_ENABLED=0
export POINT3R_EXPERIMENT_TAG=v82e_balanced_predecoder_pose_nrgbd9_len${MAX_FRAMES}

SCENES=(
  breakfast_room complete_kitchen green_room grey_white_room kitchen
  morning_apartment staircase thin_geometry whiteroom
)
if [[ -n "${V79_SCENES:-}" ]]; then
  read -r -a SCENES <<< "$V79_SCENES"
fi
if [[ "${V79_PREFLIGHT:-0}" == "1" ]]; then
  SCENES=(breakfast_room)
  MAX_FRAMES=2
  export POINT3R_EXPERIMENT_TAG=v79_geometry_safe_q25_sparse640_nrgbd_preflight
fi

echo "[ROOT] $ROOT" | tee -a "$MASTER"
echo "[CONFIG] model=v82e predecoder_weight=0.025 ray_bank_every=4 q=0.25 sparse=640 anchors=128 slots=8 NRGBD9 kf=2 len=${MAX_FRAMES}" | tee -a "$MASTER"
sha256sum "$REPO/src/dust3r/point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose.py" | tee "$ROOT/source_sha256.txt" | tee -a "$MASTER"

cd "$REPO"
set +e
/root/miniconda3/envs/point3r/bin/python -B "$RUNNER_DIR/launch_v82e_nrgbd.py" \
  --model v82e_balanced_predecoder_pose_q25_sparse640 \
  --scenes "${SCENES[@]}" \
  --point3r_repo "$REPO" \
  --point3r_weights /root/autodl-tmp/checkpoints/point3r_512.pth \
  --nrgbd_root /root/autodl-tmp/neural_rgbd \
  --output_dir "$ROOT" \
  --device cuda --size 512 --kf_every 2 --max_frames "$MAX_FRAMES" \
  --max_points 999999 --kway_slots 8 --sparse_max_tokens 640 \
  --sparse_global_anchors 128 --sparse_neighbor_range 1 \
  > "$ROOT/terminal.log" 2>&1
STATUS=$?
set -e

grep -E "V79_EFFECTIVE_CONFIG|CGMC_SPARSE512_EFFECTIVE_CONFIG|SINGLE_POSE_READOUT|GEOMETRY_SAFE_POSE_READOUT|pointcloud_scene|Traceback|OutOfMemory|No space|OSError|wrote" \
  "$ROOT/terminal.log" | tail -400 | tee -a "$MASTER" || true
echo "[DONE] status=$STATUS $(date -Is)" | tee -a "$MASTER"
find "$ROOT" -type f \( -name '*.ply' -o -name '*.npy' -o -name '*.npz' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) -delete 2>/dev/null || true
df -h /root/autodl-tmp | tail -1 | tee -a "$MASTER"
exit "$STATUS"
