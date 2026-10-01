#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

REPO=/root/autodl-tmp/Point3R_mdf
RUNNER=/root/autodl-tmp/clean_launcher/pointcloud/v106_7scenes_len300
PROTO=/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300
RESULT_BASE=/root/autodl-tmp/clean_launcher/pointcloud/results
RUN_LEN=${RUN_LEN:?set RUN_LEN to 300, 400, or 500}
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$RESULT_BASE/v106_fresh_bank_pose_q25_sparse640_nrgbd9_kf2_len${RUN_LEN}_843_$STAMP"
mkdir -p "$ROOT"

export PYTHONPATH="$RUNNER:$PROTO:/root/autodl-tmp/clean_launcher/pointcloud:$REPO:$REPO/src/croco:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export POINT3R_MEMORY_UPDATE_MODE=ordered_kway POINT3R_MEMORY_IMPL=tensor POINT3R_ORDERED_UPDATE_IMPL=tensor
export POINT3R_KWAY_NUM_SLOTS=8 POINT3R_ORDERED_THETA_BINS=16 POINT3R_ORDERED_PHI_BINS=8 POINT3R_ORDERED_RHO_BINS=32
export POINT3R_SPARSE_READOUT=1 POINT3R_SPARSE_MAX_TOKENS=640 POINT3R_SPARSE_GLOBAL_ANCHORS=128 POINT3R_SPARSE_NEIGHBOR_RANGE=1
export POINT3R_ENCODE_CHUNK_SIZE=100 POINT3R_CGMC_DROP_QUANTILE=0.25 POINT3R_CGMC_WEIGHTED_MERGE=1 POINT3R_CONFSELECT_STATS=0
export POINT3R_LC_STATE_LOCAL_ROT_MODE=0 POINT3R_LC_STATE_LOW_RAY_FUSION=1 POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY=1
export POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION=0 POINT3R_LC_STATE_RAY_FUSION_PRIOR=1
export POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG=2 POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG=2
export POINT3R_LC_ODOM_CENTERED_ROTATION=0 POINT3R_LC_ODOM_ESSENTIAL_ROTATION=0
export POINT3R_RAY_DUAL_BANK=1 POINT3R_RAY_BANK_UPDATE_EVERY=1 POINT3R_V106_RAY_BANK_UPDATE_EVERY=1
export POINT3R_RAYAWARE_UPDATE=1 POINT3R_RAY_PAPER_UPDATE=1 POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
export POINT3R_RAY_HYBRID_READOUT=1 POINT3R_RAY_POSE_ONLY_ENSEMBLE=0 POINT3R_RAY_POSE_INPUT_ONLY=1 POINT3R_RAY_POSE_POST_DECODER_ONLY=0
export POINT3R_RAY_POSE_INPUT_TOKENS=128 POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10 POINT3R_V106_POSE_INPUT_WEIGHT=0.15
export POINT3R_V101_SAFE_TRANSLATION=0 POINT3R_RAY_HYBRID_TOKENS=640 POINT3R_RAY_HYBRID_STABLE_FRAC=0.60 POINT3R_RAY_HYBRID_ANCHOR_FRAC=0.10
export POINT3R_RAY_HYBRID_RAY_WEIGHT=0.25 POINT3R_RAY_POINTER_LOOP_GRAPH=0 POINT3R_RAY_POINTER_LOOP_PGO=0 POINT3R_LC_ENABLED=0
export POINT3R_EXPERIMENT_TAG="v106_fresh_bank_pose_nrgbd9_len${RUN_LEN}"

SCENES=(breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom)
echo "[ROOT] $ROOT"
echo "[CONFIG] model=v106 q=0.25 sparse=640 anchors=128 slots=8 ray_bank_every=1 pose_weight=0.15 NRGBD9 kf=2 len=$RUN_LEN max_depth=5m"
sha256sum "$REPO/src/dust3r/point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose.py" >"$ROOT/source_sha256.txt"

cd "$REPO"
set +e
/root/miniconda3/envs/point3r/bin/python -B "$RUNNER/launch_v106_nrgbd.py" --model v106_fresh_bank_pose_q25_sparse640 \
  --scenes "${SCENES[@]}" --point3r_repo "$REPO" --point3r_weights /root/autodl-tmp/checkpoints/point3r_512.pth \
  --nrgbd_root /root/autodl-tmp/neural_rgbd --output_dir "$ROOT" --device cuda --size 512 --kf_every 2 --max_frames "$RUN_LEN" \
  --max_points 999999 --kway_slots 8 --sparse_max_tokens 640 --sparse_global_anchors 128 --sparse_neighbor_range 1 \
  >"$ROOT/terminal.log" 2>&1
STATUS=$?
set -e
echo "[DONE] status=$STATUS root=$ROOT $(date -Is)"
find "$ROOT" -type f \( -name '*.ply' -o -name '*.npy' -o -name '*.npz' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) -delete 2>/dev/null || true
exit "$STATUS"
