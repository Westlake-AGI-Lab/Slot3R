#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

REPO=/root/autodl-tmp/Point3R_mdf
PROTO=/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300
V102_WRAP=/root/autodl-tmp/clean_launcher/pointcloud/v102_7scenes_len300/launch_v102_nrgbd.py
RESULT_BASE=/root/pointcloud_results/oursstar_oursstarstar_nrgbd_kf1_len600_1000_20260908
QUEUE_LOG="$RESULT_BASE/queue.log"
SCENES=(breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom)
LENGTHS=(600 700 800 900 1000)
mkdir -p "$RESULT_BASE"
: > "$QUEUE_LOG"

export PYTHONPATH="$PROTO:/root/autodl-tmp/clean_launcher/pointcloud:/root/autodl-tmp/clean_launcher/pointcloud/v102_7scenes_len300:$REPO:$REPO/src/croco:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

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
export POINT3R_ENCODE_CHUNK_SIZE=50
export POINT3R_CGMC_DROP_QUANTILE=0.25
export POINT3R_CGMC_WEIGHTED_MERGE=1
export POINT3R_CONFSELECT_STATS=0
export POINT3R_METRIC_CPU=1

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
export POINT3R_RAY_POSE_INPUT_TEMPERATURE=0.10
export POINT3R_RAY_POSE_INPUT_JERK_LOW=0.010
export POINT3R_RAY_POSE_INPUT_JERK_HIGH=0.035
export POINT3R_RAY_HYBRID_TOKENS=640
export POINT3R_RAY_HYBRID_STABLE_FRAC=0.60
export POINT3R_RAY_HYBRID_ANCHOR_FRAC=0.10
export POINT3R_RAY_HYBRID_RAY_WEIGHT=0.25
export POINT3R_RAY_POINTER_LOOP_GRAPH=0
export POINT3R_RAY_POINTER_LOOP_PGO=0
export POINT3R_LC_ENABLED=0

run_one() {
  local version="$1"
  local length="$2"
  local model launcher output status rows
  if [[ "$version" == "v82k" ]]; then
    model=v82e_balanced_predecoder_pose_q25_sparse640
    launcher="$PROTO/launch_v82e_nrgbd.py"
    export POINT3R_RAY_POSE_INPUT_MAX_WEIGHT=0.025
    export POINT3R_V102_SMOOTH_WEIGHT=0.025
    export POINT3R_V102_JERK_WEIGHT=0.025
    export POINT3R_V101_SAFE_TRANSLATION=0
  else
    model=v102_complementary_jerk_q25_sparse640
    launcher="$V102_WRAP"
    export POINT3R_RAY_POSE_INPUT_MAX_WEIGHT=0.025
    export POINT3R_V102_SMOOTH_WEIGHT=0.15
    export POINT3R_V102_JERK_WEIGHT=0.025
    export POINT3R_V101_SAFE_TRANSLATION=0
  fi

  output="$RESULT_BASE/${version}_${model}_nrgbd9_kf1_len${length}"
  mkdir -p "$output"
  export POINT3R_EXPERIMENT_TAG="${version}_nrgbd9_kf1_len${length}"
  echo "[START] version=$version len=$length output=$output $(date -Is)" | tee -a "$QUEUE_LOG"
  cd "$REPO"
  set +e
  if [[ "$version" == "v82k" ]]; then
    /root/miniconda3/envs/point3r/bin/python -B "$launcher" \
      --model "$model" --scenes "${SCENES[@]}" \
      --point3r_repo "$REPO" --point3r_weights /root/autodl-tmp/checkpoints/point3r_512.pth \
      --nrgbd_root /root/autodl-tmp/neural_rgbd --output_dir "$output" \
      --device cuda --size 512 --kf_every 1 --max_frames "$length" --max_points 999999 \
      --kway_slots 8 --sparse_max_tokens 640 --sparse_global_anchors 128 \
      --sparse_neighbor_range 1 > "$output/terminal.log" 2>&1
    status=$?
  else
    /root/miniconda3/envs/point3r/bin/python -B -c \
      'import runpy; d=runpy.run_path("/root/autodl-tmp/clean_launcher/pointcloud/v102_7scenes_len300/launch_v102_nrgbd.py"); raise SystemExit(d["base"].main())' \
      --model "$model" --scenes "${SCENES[@]}" \
      --point3r_repo "$REPO" --point3r_weights /root/autodl-tmp/checkpoints/point3r_512.pth \
      --nrgbd_root /root/autodl-tmp/neural_rgbd --output_dir "$output" \
      --device cuda --size 512 --kf_every 1 --max_frames "$length" --max_points 999999 \
      --kway_slots 8 --sparse_max_tokens 640 --sparse_global_anchors 128 \
      --sparse_neighbor_range 1 > "$output/terminal.log" 2>&1
    status=$?
  fi
  set -e
  rows=0
  [[ -f "$output/summary.tsv" ]] && rows=$(grep -c "^$model" "$output/summary.tsv" || true)
  echo "[DONE] version=$version len=$length status=$status rows=$rows output=$output $(date -Is)" | tee -a "$QUEUE_LOG"
}

for length in "${LENGTHS[@]}"; do
  run_one v82k "$length"
done
for length in "${LENGTHS[@]}"; do
  run_one v102 "$length"
done
echo "[QUEUE_DONE] $(date -Is)" | tee -a "$QUEUE_LOG"
