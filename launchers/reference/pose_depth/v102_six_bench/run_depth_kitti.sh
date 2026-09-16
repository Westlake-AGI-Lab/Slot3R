#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/v102_six_bench/v102_common_env.sh
PY=/root/miniconda3/envs/point3r/bin/python
REPO=/root/autodl-tmp/Point3R_mdf
OUT=/root/autodl-tmp/v102_six_bench/results/depth_kitti
mkdir -p "$OUT"; ln -sfn /root/autodl-tmp/KITTI "$REPO/data/kitti"
"$PY" -u -B /root/autodl-tmp/v102_six_bench/launch_kitti.py --weights /root/autodl-tmp/checkpoints/point3r_512.pth --output_dir "$OUT" --eval_dataset kitti --size 512 >"$OUT/infer.log" 2>&1
ki=$?
cd "$REPO"
"$PY" -u -B eval/video_depth/eval_depth.py --output_dir "$OUT" --eval_dataset kitti --align metric >"$OUT/eval_metric.log" 2>&1; km=$?
"$PY" -u -B eval/video_depth/eval_depth.py --output_dir "$OUT" --eval_dataset kitti --align 'scale&shift' >"$OUT/eval_scale_shift.log" 2>&1; ks=$?
echo "KITTI_DONE infer=$ki metric=$km scale_shift=$ks preds=$(find "$OUT" -type f -name 'frame_*.npy' | wc -l) $(date -Is)"
find "$OUT" -type f \( -name 'frame_*.npy' -o -name 'frame_*.png' -o -name '*.ply' \) -delete 2>/dev/null || true
exit $(( (ki != 0 && ki != 134) || km != 0 || ks != 0 ))

