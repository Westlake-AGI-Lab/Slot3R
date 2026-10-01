#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

PY=/root/miniconda3/envs/point3r/bin/python
REPO=/root/autodl-tmp/Point3R_mdf
ROOT=/root/autodl-tmp/clean_launchers/rayaway_pose_v4
DRIVER=/root/autodl-tmp/clean_launchers/pose/scannet_color90_pose_sparse640_q25.py
BACKEND="$ROOT/point3r_bks_rayaway_pose_v18.py"
TAG=v82e_balanced_predecoder_pose_full
OUT="$ROOT/results/scannet_$TAG"
LOG="$ROOT/logs/scannet_$TAG.log"
rm -rf "$OUT"; mkdir -p "$OUT" "$ROOT/logs"

export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POINT3R_EVAL_SEED=0
export POINT3R_POSE_MODEL_MODULE=dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose

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
export POINT3R_RAY_POINTER_LOOP_GRAPH=0
export POINT3R_RAY_POINTER_LOOP_PGO=0
export POINT3R_EXPERIMENT_TAG="$TAG"

echo "[START] $TAG $(date -Is)" > "$LOG"
cd "$REPO"
set +e
"$PY" -u -B "$DRIVER" --model sparse640_q25 \
  --point3r_bks_backend "$BACKEND" \
  --scannet_root /root/autodl-tmp/scannetv2 \
  --output_dir "$OUT" --size 512 --pose_eval_stride 1 \
  >> "$LOG" 2>&1
status=$?
set -e
echo "[DONE] $TAG status=$status $(date -Is)" >> "$LOG"
grep -E 'pose_scene|SUMMARY|Traceback|ERROR' "$LOG" | tail -200
exit "$status"
