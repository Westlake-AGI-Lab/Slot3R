#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

REPO=/root/autodl-tmp/Point3R_mdf
RUNNER=/root/autodl-tmp/clean_launcher/pointcloud/v106_7scenes_len300
PROTO=/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300
LAUNCHER="$RUNNER/launch_v106_nrgbd.py"
RESULT_BASE=/root/pointcloud_results/v106_official_nrgbd_kf1_len600_1000_20260910
QUEUE_LOG="$RESULT_BASE/queue.log"
SCENES=(breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom)
LENGTHS=(600 700 800 900 1000)
MODEL=v106_fresh_bank_pose_q25_sparse640

mkdir -p "$RESULT_BASE"
touch "$QUEUE_LOG"

export PYTHONPATH="$RUNNER:$PROTO:/root/autodl-tmp/clean_launcher/pointcloud:$REPO:$REPO/src/croco:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export POINT3R_METRIC_CPU=1

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
export POINT3R_RAY_BANK_UPDATE_EVERY=1
export POINT3R_V106_RAY_BANK_UPDATE_EVERY=1
export POINT3R_RAYAWARE_UPDATE=1
export POINT3R_RAY_PAPER_UPDATE=1
export POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
export POINT3R_RAY_HYBRID_READOUT=1
export POINT3R_RAY_POSE_ONLY_ENSEMBLE=0
export POINT3R_RAY_POSE_INPUT_ONLY=1
export POINT3R_RAY_POSE_POST_DECODER_ONLY=0
export POINT3R_RAY_POSE_INPUT_TOKENS=128
export POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10
export POINT3R_V106_POSE_INPUT_WEIGHT=0.15
export POINT3R_V101_SAFE_TRANSLATION=0
export POINT3R_RAY_HYBRID_TOKENS=640
export POINT3R_RAY_HYBRID_STABLE_FRAC=0.60
export POINT3R_RAY_HYBRID_ANCHOR_FRAC=0.10
export POINT3R_RAY_HYBRID_RAY_WEIGHT=0.25
export POINT3R_RAY_POINTER_LOOP_GRAPH=0
export POINT3R_RAY_POINTER_LOOP_PGO=0
export POINT3R_LC_ENABLED=0

echo "[CONFIG] model=v106 dataset=NRGBD9 kf=1 lengths=600,700,800,900,1000 max_depth=5m q=.25 sparse=640 anchors=128 slots=8 vis_conf_filter=OFF metric_device=CPU" | tee -a "$QUEUE_LOG"
sha256sum "$REPO/src/dust3r/point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose.py" > "$RESULT_BASE/source_sha256.txt"

for length in "${LENGTHS[@]}"; do
  output="$RESULT_BASE/${MODEL}_nrgbd9_kf1_len${length}"
  if [[ -f "$output/summary.tsv" ]] && [[ $(awk 'END{print NR-1}' "$output/summary.tsv") -eq 9 ]]; then
    echo "[SKIP_COMPLETE] len=$length rows=9 output=$output" | tee -a "$QUEUE_LOG"
    continue
  fi
  mkdir -p "$output"
  export POINT3R_EXPERIMENT_TAG="v106_official_nrgbd9_kf1_len${length}"
  echo "[START] len=$length output=$output $(date -Is)" | tee -a "$QUEUE_LOG"

  cd "$REPO"
  set +e
  /root/miniconda3/envs/point3r/bin/python -B "$LAUNCHER" \
    --model "$MODEL" --scenes "${SCENES[@]}" \
    --point3r_repo "$REPO" --point3r_weights /root/autodl-tmp/checkpoints/point3r_512.pth \
    --nrgbd_root /root/autodl-tmp/neural_rgbd --output_dir "$output" \
    --device cuda --size 512 --kf_every 1 --max_frames "$length" --max_points 999999 \
    --kway_slots 8 --sparse_max_tokens 640 --sparse_global_anchors 128 \
    --sparse_neighbor_range 1 > "$output/terminal.log" 2>&1
  status=$?
  set -e

  rows=0
  [[ -f "$output/summary.tsv" ]] && rows=$(awk 'END{print NR-1}' "$output/summary.tsv")
  echo "[DONE] len=$length status=$status rows=$rows output=$output $(date -Is)" | tee -a "$QUEUE_LOG"
  find "$output" -type f \( -name '*.ply' -o -name '*.npy' -o -name '*.npz' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) -delete 2>/dev/null || true
done

echo "[QUEUE_DONE] $(date -Is)" | tee -a "$QUEUE_LOG"
