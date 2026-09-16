#!/usr/bin/env bash
set -uo pipefail

PY=/root/miniconda3/envs/point3r/bin/python
REPO=/root/autodl-tmp/Point3R_mdf
BASE=/root/autodl-tmp/clean_launchers/rayaway_pose_v4/depth_v82k_benchmark
RUN="$BASE/run"
LAUNCH="$BASE/launchers"
PIPE="$BASE/pipeline.log"
mkdir -p "$RUN" "$LAUNCH"
touch "$PIPE" "$BASE/ENABLE_PAIRED_SCANNET"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1

# Paper Ours front end: ordered K-way + q35 ConfSelect + Sparse640.
export POINT3R_MEMORY_UPDATE_MODE=ordered_kway
export POINT3R_MEMORY_IMPL=tensor
export POINT3R_ORDERED_UPDATE_IMPL=tensor
export POINT3R_KWAY_NUM_SLOTS=8
export POINT3R_KWAY_NUM_SLOTS_FORCE=8
export POINT3R_ORDERED_THETA_BINS=16
export POINT3R_ORDERED_PHI_BINS=8
export POINT3R_ORDERED_RHO_BINS=32
export POINT3R_ORDERED_WAY_POLICY=appearance
export POINT3R_ENCODE_CHUNK_SIZE=100
export POINT3R_CONFSELECT_MERGE_THRESHOLD=0.90
export POINT3R_CONFSELECT_STATS=0
export POINT3R_CGMC=1
export POINT3R_CGMC_DROP_QUANTILE=0.25
export POINT3R_CGMC_MIN_CONF=0.0
export POINT3R_CGMC_WEIGHTED_MERGE=1
export POINT3R_SPARSE_READOUT=1
export POINT3R_SPARSE_MODE=max
export POINT3R_SPARSE_MAX_TOKENS=640
export POINT3R_SPARSE_GLOBAL_ANCHORS=128
export POINT3R_SPARSE_NEIGHBOR_RANGE=1

# Exact v82k/v82e single-forward RayAware state.
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

# Preserve the v82k pose-state configuration; geometry remains decoder-authoritative.
export POINT3R_LC_ENABLED=1 POINT3R_LC_LOOP_DETECTION=0 POINT3R_LC_ODOM_MAX_LAG=3
export POINT3R_LC_FULL_SE3=0 POINT3R_LC_DECOUPLED_SE3=0
export POINT3R_LC_ROTATION_GRAPH=1 POINT3R_LC_DIRECT_RAY_ODOM=0
export POINT3R_LC_STATE_REGULARIZATION=1 POINT3R_LC_STATE_LIE_INCREMENT=1
export POINT3R_LC_STATE_TRANS_STRENGTH=30 POINT3R_LC_STATE_ROT_STRENGTH=30
export POINT3R_LC_STATE_RAY_UNCERTAINTY_POWER=8 POINT3R_LC_STATE_AUTO_ROT_MODE=1
export POINT3R_LC_STATE_HIGH_JERK_RAD=0.035 POINT3R_LC_STATE_LOW_ROT_STRENGTH=0.1
export POINT3R_LC_STATE_LOCAL_ROT_MODE=0
export POINT3R_LC_STATE_LOW_RAY_FUSION=1
export POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY=1
export POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION=0
export POINT3R_LC_STATE_RAY_FUSION_PRIOR=1
export POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG=2
export POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG=2

say() { echo "$*" | tee -a "$PIPE"; }
rows() { local f=$1; [[ -f "$f" ]] && echo $(( $(wc -l < "$f") - 1 )) || echo 0; }
check_module() {
  local log=$1
  grep -q 'point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose' "$log" || {
    say "[fatal] v82e module marker missing: $log"; return 1;
  }
}

say "[pipeline] START v82k depth3 $(date -Is)"

# 1) ScanNet: paired evaluator writes metric and per-sequence summaries.
SCAN="$RUN/scannet"
mkdir -p "$SCAN/metric"
say "[start] ScanNet metric+per_sequence $(date -Is)"
"$PY" -u -B "$LAUNCH/depth_scannet_v82k.py" \
  --model depthcc --scannet_root /root/autodl-tmp/scannetv2 \
  --output_dir "$SCAN/metric" --size 512 --align none --max_depth 10 \
  --drop_quantile 0.25 --sparse_max_tokens 640 > "$SCAN/infer.log" 2>&1
status=$?
check_module "$SCAN/infer.log" || exit 41
scan_metric=$(rows "$SCAN/metric/summary.tsv")
scan_scaled=$(rows "$SCAN/scale_shift/summary.tsv")
say "[done] ScanNet status=$status metric_rows=$scan_metric per_sequence_rows=$scan_scaled $(date -Is)"
[[ "$scan_metric" -eq 100 && "$scan_scaled" -eq 100 ]] || exit 42
touch "$SCAN/DONE.txt"

# 2) Bonn: five official sequences, native metric and sequence-scale alignment.
BONN="$RUN/bonn"
mkdir -p "$BONN/metric" "$BONN/scale_shift"
say "[start] Bonn metric $(date -Is)"
"$PY" -u -B "$LAUNCH/depth_bonn_v82k.py" \
  --model depthcc --bonn_root /root/autodl-tmp/Bonn_unzip \
  --output_dir "$BONN/metric" --size 512 --align none --max_depth 70 \
  --drop_quantile 0.25 --sparse_max_tokens 640 > "$BONN/metric.log" 2>&1
bm=$?
check_module "$BONN/metric.log" || exit 51
say "[start] Bonn per_sequence $(date -Is)"
"$PY" -u -B "$LAUNCH/depth_bonn_v82k.py" \
  --model depthcc --bonn_root /root/autodl-tmp/Bonn_unzip \
  --output_dir "$BONN/scale_shift" --size 512 --align sequence_scale_shift --max_depth 70 \
  --drop_quantile 0.25 --sparse_max_tokens 640 > "$BONN/scale_shift.log" 2>&1
bs=$?
check_module "$BONN/scale_shift.log" || exit 52
bonn_metric=$(rows "$BONN/metric/summary.tsv")
bonn_scaled=$(rows "$BONN/scale_shift/summary.tsv")
say "[done] Bonn metric_status=$bm per_sequence_status=$bs metric_rows=$bonn_metric per_sequence_rows=$bonn_scaled $(date -Is)"
[[ "$bonn_metric" -eq 5 && "$bonn_scaled" -eq 5 ]] || exit 53
touch "$BONN/DONE.txt"

# 3) KITTI: official 1269-frame inference and both official evaluators.
KITTI="$RUN/kitti"
mkdir -p "$KITTI"
ln -sfn /root/autodl-tmp/KITTI "$REPO/data/kitti"
say "[start] KITTI inference $(date -Is)"
"$PY" -u -B "$LAUNCH/launch_kitti_v82k.py" \
  --weights /root/autodl-tmp/checkpoints/point3r_512.pth \
  --output_dir "$KITTI" --eval_dataset kitti --size 512 > "$KITTI/infer.log" 2>&1
ki=$?
check_module "$KITTI/infer.log" || exit 61
preds=$(find "$KITTI" -type f -name 'frame_*.npy' | wc -l)
say "[done] KITTI inference status=$ki predictions=$preds $(date -Is)"
[[ "$preds" -eq 1269 && ( "$ki" -eq 0 || "$ki" -eq 134 ) ]] || exit 62
cd "$REPO" || exit 63
"$PY" -u -B eval/video_depth/eval_depth.py --output_dir "$KITTI" --eval_dataset kitti --align metric > "$KITTI/eval_metric.log" 2>&1
km=$?
"$PY" -u -B eval/video_depth/eval_depth.py --output_dir "$KITTI" --eval_dataset kitti --align 'scale&shift' > "$KITTI/eval_scale_shift.log" 2>&1
ks=$?
say "[done] KITTI metric_status=$km per_sequence_status=$ks $(date -Is)"
[[ "$km" -eq 0 && "$ks" -eq 0 && -f "$KITTI/result_metric.json" && -f "$KITTI/result_scale&shift.json" ]] || exit 64
touch "$KITTI/DONE.txt"

find "$KITTI" -type f \( -name 'frame_*.npy' -o -name 'frame_*.png' -o -name '*.ply' \) -delete 2>/dev/null || true
say "[pipeline] ALL_DONE $(date -Is)"
touch "$BASE/ALL_DONE.txt"
