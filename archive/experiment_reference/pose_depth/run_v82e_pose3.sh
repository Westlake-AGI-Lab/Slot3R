#!/usr/bin/env bash
set -uo pipefail
ROOT=/root/autodl-tmp/clean_launchers/rayaway_pose_v4

export POINT3R_POSE_MODEL_MODULE=dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose

# Preserve v50's verified trajectory regularization chain.
export POINT3R_LC_STATE_LOCAL_ROT_MODE=0
export POINT3R_LC_STATE_LOW_RAY_FUSION=1
export POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY=1
export POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION=0
export POINT3R_LC_STATE_RAY_FUSION_PRIOR=1
export POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG=2
export POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG=2
export POINT3R_LC_ODOM_CENTERED_ROTATION=0
export POINT3R_LC_ODOM_ESSENTIAL_ROTATION=0

# v56: two independent recurrent pose states.  The stable decoder remains
# authoritative for geometry.  A ray-dominant auxiliary decoder reads the
# retain-or-replace bank and writes its own pose state for the next frame.
export POINT3R_RAY_DUAL_BANK=1
export POINT3R_RAY_BANK_UPDATE_EVERY=4
export POINT3R_RAYAWARE_UPDATE=1
export POINT3R_RAY_PAPER_UPDATE=1
export POINT3R_RAY_KWAY_DIVERSE_UPDATE=0
export POINT3R_RAY_HYBRID_READOUT=1
# Single decoder pass: the main decoder reads one hybrid sparse set composed
# of stable K-way locals, global anchors, and RayAware tokens.  No independent
# ray decoder or auxiliary downstream head is constructed.
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

# Do not mix the rejected v55 pointer-loop graph into this structural test.
export POINT3R_RAY_POINTER_LOOP_GRAPH=0
export POINT3R_RAY_POINTER_LOOP_PGO=0
export POINT3R_EXPERIMENT_TAG=${POINT3R_V82E_TAG:-v82e_balanced_predecoder_pose}

# Explicit requested order: Sintel, TUM, then ScanNet.
POINT3R_SKIP_SINTEL=0 POINT3R_SKIP_TUM=1 \
  bash "$ROOT/run_v54_base_pose.sh"
sintel_status=$?

POINT3R_SKIP_SINTEL=1 POINT3R_SKIP_TUM=0 \
  bash "$ROOT/run_v54_base_pose.sh"
tum_status=$?

PY=/root/miniconda3/envs/point3r/bin/python
REPO=/root/autodl-tmp/Point3R_mdf
DRIVER="$ROOT/../pose/scannet_color90_pose_sparse640_q25.py"
BACKEND="$ROOT/point3r_bks_rayaway_pose_v18.py"
TAG=${POINT3R_EXPERIMENT_TAG}
OUT="$ROOT/results/scannet_${TAG}"
LOG="$ROOT/logs/scannet_${TAG}.log"
export POINT3R_LC_AUDIT_FILE="$ROOT/logs/scannet_${TAG}_events.log"
rm -rf "$OUT"; mkdir -p "$OUT" "$ROOT/logs"; rm -f "$POINT3R_LC_AUDIT_FILE"
echo "START $TAG SCANNET $(date -Is)" > "$LOG"
cd "$REPO"
"$PY" -u -B "$DRIVER" --model sparse640_q25 \
  --point3r_bks_backend "$BACKEND" --scannet_root /root/autodl-tmp/scannetv2 \
  --output_dir "$OUT" --size 512 --pose_eval_stride 1 >> "$LOG" 2>&1
scannet_status=$?
echo "END $TAG SCANNET status=$scannet_status $(date -Is)" >> "$LOG"

echo "END v82e pose3 sintel_status=$sintel_status tum_status=$tum_status scannet_status=$scannet_status $(date -Is)"
exit $(( sintel_status != 0 || tum_status != 0 || scannet_status != 0 ))
