#!/usr/bin/env python3
"""Unified pose evaluation: --dataset sintel/scannet/tum --model core/vpc_m/vpc_a."""
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
    parser.add_argument("--dataset", choices=("sintel", "scannet", "tum"), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--pose_eval_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true")
    parser.add_argument("--solve_pose", action="store_true")
    args = parser.parse_args(argv)
    if args.pose_eval_stride < 1 or args.revisit < 1 or args.max_frames < 0:
        parser.error("stride/revisit must be positive; max_frames must be nonnegative")
    return args


def seed_prepared_pose():
    import os
    import random
    import numpy as np
    import torch
    seed = int(os.environ.get("POINT3R_EVAL_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate_scene(args, model, scene):
    import torch
    from dust3r.inference import inference
    from eval.pose.data import DATASETS, load_sequence
    from eval.pose.inference import prepare_input, prepare_output
    from eval.pose.metrics import score_trajectory
    files, gt = load_sequence(args, scene)
    views = prepare_input(files, [True]*len(files), args.size, revisit=args.revisit,
                          update=not args.freeze_state, crop=not args.no_crop)
    with torch.no_grad():
        outputs = inference(views, model, args.device)
    _, poses = prepare_output(outputs, revisit=args.revisit, solve_pose=args.solve_pose)
    if len(poses) != len(files):
        raise ValueError("Prediction/input frame count mismatch")
    path = Path(args.output_dir) / (scene.replace("/", "_")+"_eval_metric.txt")
    metrics = score_trajectory(poses, gt, scene, path, DATASETS[args.dataset]["rpe_stat"])
    return {"frames": len(files), **metrics}


def main(argv=None):
    args = parse_args(argv)
    from eval.pose.data import scene_names
    model = None
    def evaluate(scene):
        nonlocal model
        # Preserve the prepared ScanNet/TUM backend's per-scene seed and model lifetime.
        if args.dataset != "sintel":
            seed_prepared_pose()
            return evaluate_scene(args, load_task_model(args), scene)
        if model is None:
            model = load_task_model(args)
        return evaluate_scene(args, model, scene)
    return run_scenes(args, scene_names(args), evaluate, mean_metrics)


if __name__ == "__main__":
    raise SystemExit(main())
