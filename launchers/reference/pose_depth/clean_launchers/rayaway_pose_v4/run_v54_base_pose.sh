#!/usr/bin/env bash
set -uo pipefail
PY=/root/miniconda3/envs/point3r/bin/python
REPO=/root/autodl-tmp/Point3R_mdf
ROOT=/root/autodl-tmp/clean_launchers/rayaway_pose_v4
export PYTHONPATH="$REPO:$REPO/src:$REPO/src/croco" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 POINT3R_EVAL_SEED=0
export POINT3R_POSE_MODEL_MODULE=${POINT3R_POSE_MODEL_MODULE:-dust3r.point3r_kway_frame_sparse_q35_confselect_rayaway_closedloop_v19}

export POINT3R_CGMC=1 POINT3R_CGMC_DROP_QUANTILE=0.25 POINT3R_CGMC_MIN_CONF=0.0 POINT3R_CGMC_WEIGHTED_MERGE=1
export POINT3R_RAYAWARE_UPDATE=${POINT3R_RAYAWARE_UPDATE:-0}
export POINT3R_RAY_PAPER_UPDATE=${POINT3R_RAY_PAPER_UPDATE:-0}
export POINT3R_RAY_KWAY_DIVERSE_UPDATE=${POINT3R_RAY_KWAY_DIVERSE_UPDATE:-0}
export POINT3R_RAY_DUAL_BANK=${POINT3R_RAY_DUAL_BANK:-0}
export POINT3R_RAY_HYBRID_READOUT=${POINT3R_RAY_HYBRID_READOUT:-0}
export POINT3R_POSE_STATIC_REFINE=${POINT3R_POSE_STATIC_REFINE:-0}
export POINT3R_LC_ENABLED=1 POINT3R_LC_LOOP_DETECTION=0 POINT3R_LC_ODOM_MAX_LAG=3
export POINT3R_LC_FULL_SE3=0 POINT3R_LC_DECOUPLED_SE3=0
export POINT3R_LC_ROTATION_GRAPH=1 POINT3R_LC_DIRECT_RAY_ODOM=0
export POINT3R_LC_ODOM_EPS_POS=0.30 POINT3R_LC_ODOM_FEAT_SIM=0.55 POINT3R_LC_ODOM_COST_MARGIN=0.03
export POINT3R_LC_ODOM_MIN_PAIRS=10 POINT3R_LC_ODOM_MIN_RATIO=0.60 POINT3R_LC_ODOM_RANSAC_DEG=2.0 POINT3R_LC_ODOM_RANSAC_ITERS=96
export POINT3R_LC_ODOM_MAX_CORRECTION_DEG=5.0 POINT3R_LC_ODOM_ROT_BLEND=1.0
export POINT3R_LC_ODOM_QUALITY_RMSE_DEG=1.0 POINT3R_LC_ODOM_QUALITY_CORR_DEG=2.0
export POINT3R_LC_ODOM_QUALITY_SUPPORT=50 POINT3R_LC_ODOM_MAX_ROT_WEIGHT=15.0
export POINT3R_LC_ODOM_COVERAGE_ENABLED=0 POINT3R_LC_ODOM_BINARY_QUALITY=0
export POINT3R_LC_ANISOTROPIC_INFORMATION=0 POINT3R_LC_ANISOTROPIC_ROT_GRAPH=0
export POINT3R_LC_ROT_GRAPH_PRIOR_WEIGHT=1.0 POINT3R_LC_ROT_GRAPH_EDGE_WEIGHT=5.0 POINT3R_LC_ROT_GRAPH_ITERS=20
export POINT3R_LC_ROT_GRAPH_IRLS=0 POINT3R_LC_ROT_GRAPH_IRLS_DELTA_DEG=2.0 POINT3R_LC_ROT_GRAPH_IRLS_WARMUP=3
export POINT3R_LC_SYNC_MEMORY=0 POINT3R_LC_SYNC_GEOMETRY=0 POINT3R_LC_ONLINE_CORRECTION=0

# v47: retain the v46 body-frame translation chain, but select the rotation
# prior from predicted trajectory dynamics (never dataset names or GT).
export POINT3R_LC_STATE_REGULARIZATION=1 POINT3R_LC_STATE_LIE_INCREMENT=1
export POINT3R_LC_STATE_TRANS_STRENGTH=${POINT3R_LC_STATE_TRANS_STRENGTH:-30}
export POINT3R_LC_STATE_ROT_STRENGTH=${POINT3R_LC_STATE_ROT_STRENGTH:-30}
export POINT3R_LC_STATE_RAY_UNCERTAINTY_POWER=${POINT3R_LC_STATE_RAY_UNCERTAINTY_POWER:-8}
export POINT3R_LC_STATE_AUTO_ROT_MODE=1
export POINT3R_LC_STATE_HIGH_JERK_RAD=${POINT3R_LC_STATE_HIGH_JERK_RAD:-0.035}
export POINT3R_LC_STATE_LOW_ROT_STRENGTH=${POINT3R_LC_STATE_LOW_ROT_STRENGTH:-0.1}
export POINT3R_LC_STATE_LOCAL_ROT_MODE=${POINT3R_LC_STATE_LOCAL_ROT_MODE:-0}
export POINT3R_CONFSELECT_STATS=0 POINT3R_LC_DEBUG=0

TAG=${POINT3R_EXPERIMENT_TAG:-v47_motion_state_lie}
if [[ "${POINT3R_SKIP_SINTEL:-0}" != "1" ]]; then
  OUT="$ROOT/results/sintel_${TAG}"; LOG="$ROOT/logs/sintel_${TAG}.log"
  export POINT3R_LC_AUDIT_FILE="$ROOT/logs/sintel_${TAG}_events.log"
  rm -rf "$OUT"; mkdir -p "$OUT" "$ROOT/logs"; rm -f "$POINT3R_LC_AUDIT_FILE"
  SCENES=(alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3)
  if [[ -n "${POINT3R_SINTEL_SCENES:-}" ]]; then read -r -a SCENES <<< "$POINT3R_SINTEL_SCENES"; fi
  echo "START $TAG SINTEL $(date -Is)" > "$LOG"; cd "$REPO"
  "$PY" -u -B "$ROOT/pose_sintel_rayaway_v16.py" --weights /root/autodl-tmp/checkpoints/point3r_512.pth --output_dir "$OUT" --eval_dataset sintel --size 512 --seq_list "${SCENES[@]}" >> "$LOG" 2>&1
  echo "END $TAG SINTEL status=$? $(date -Is)" >> "$LOG"
fi

if [[ "${POINT3R_SKIP_TUM:-0}" != "1" ]]; then
  OUT="$ROOT/results/tum_${TAG}"; LOG="$ROOT/logs/tum_${TAG}.log"
  export POINT3R_LC_AUDIT_FILE="$ROOT/logs/tum_${TAG}_events.log"
  rm -rf "$OUT"; mkdir -p "$OUT"; rm -f "$POINT3R_LC_AUDIT_FILE"
  SCENES=(rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz rgbd_dataset_freiburg3_walking_halfsphere rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz)
  if [[ -n "${POINT3R_TUM_SCENES:-}" ]]; then read -r -a SCENES <<< "$POINT3R_TUM_SCENES"; fi
  echo "START $TAG TUM $(date -Is)" > "$LOG"
  "$PY" -u -B "$ROOT/../pose/scannet_color90_pose_sparse640_q25.py" --model sparse640_q25 --point3r_bks_backend "$ROOT/point3r_bks_rayaway_pose_v18.py" --scannet_root /root/autodl-tmp/TUM-Dynamic --scenes "${SCENES[@]}" --output_dir "$OUT" --size 512 --pose_eval_stride 1 >> "$LOG" 2>&1
  echo "END $TAG TUM status=$? $(date -Is)" >> "$LOG"
fi
