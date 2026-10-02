#!/usr/bin/env python3
"""Unified depth evaluation: --dataset bonn/scannet/kitti --model core/vpc_m/vpc_a."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "src/croco", REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eval.model_config import add_model_arguments, load_task_model
from eval.runtime import run_scenes, mean_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    add_model_arguments(parser)
    parser.add_argument("--dataset", choices=("bonn", "scannet", "kitti"), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--kf_every", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--depth_scale", type=float)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=None, help="Indoor valid-depth cap (default 5m); KITTI uses no cap")
    parser.add_argument("--center_crop", type=int, default=0)
    parser.add_argument("--align", default=None)
    args = parser.parse_args(argv)
    args.align = args.align or ("scale_shift" if args.dataset == "kitti" else "sequence_scale_shift")
    allowed = (("scale_shift", "metric", "scale") if args.dataset == "kitti" else
               ("none", "scale", "median", "scale_shift", "sequence_scale", "sequence_median", "sequence_scale_shift"))
    if args.align not in allowed:
        parser.error(f"--align for {args.dataset} must be one of {allowed}")
    if args.kf_every < 1 or args.max_frames < 0:
        parser.error("--kf_every must be positive and --max_frames nonnegative")
    scales = {"bonn": 5000., "scannet": 1000., "kitti": 256.}
    args.depth_scale = scales[args.dataset] if args.depth_scale is None else args.depth_scale
    if args.dataset != "kitti" and args.max_depth is None:
        args.max_depth = 5.
    if args.depth_scale <= 0 or args.min_depth < 0 or (args.max_depth is not None and args.max_depth <= args.min_depth):
        parser.error("Invalid depth scale or valid-depth bounds")
    if args.dataset == "kitti" and (args.depth_scale != 256. or args.center_crop != 0 or args.max_depth is not None or args.min_depth != 1e-3):
        parser.error("KITTI uses fixed /256 GT decoding and its original helper without a depth cap; indoor depth overrides do not apply")
    return args


def evaluate_scene(args, model, scene):
    from eval.depth.data import load_sequence
    from eval.depth.inference import predict
    from eval.depth.metrics import depth_metrics, sequence_metrics
    batch, depths = load_sequence(args, scene)
    preds, elapsed, peak = predict(args, model, batch)
    if len(preds) != len(batch):
        raise ValueError("Prediction/input frame count mismatch")
    if depths is None:
        metrics = depth_metrics(preds, batch, args)
    else:
        predictions = [p["pts3d_in_self_view"][0, ..., -1].detach().cpu().numpy() for p in preds]
        metrics = sequence_metrics(predictions, depths, args.align, str(args.device).startswith("cuda"))
    return {"frames": len(batch), **metrics, "FPS": len(batch)/max(elapsed,1e-9), "Time": elapsed, "Peak": peak}


def aggregate_metrics(rows, dataset):
    result = mean_metrics(rows)
    if dataset == "kitti":
        total = sum(row["valid_pixels"] for row in rows)
        for key in rows[0]:
            if key not in ("frames", "FPS", "Time", "Peak", "valid_pixels"):
                result[key] = sum(row[key]*row["valid_pixels"] for row in rows)/total
        result["valid_pixels"] = total
    return result


def main(argv=None):
    args = parse_args(argv)
    from eval.depth.data import scene_names
    model = None
    def evaluate(scene):
        nonlocal model
        if model is None:
            model = load_task_model(args)
        return evaluate_scene(args, model, scene)
    return run_scenes(args, scene_names(args), evaluate, lambda rows: aggregate_metrics(rows, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
