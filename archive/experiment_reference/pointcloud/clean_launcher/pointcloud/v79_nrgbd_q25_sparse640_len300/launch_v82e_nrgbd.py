#!/usr/bin/env python3
"""Run geometry-safe v79 on the clean official NRGBD protocol."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

import pointcloud_metric_clean as base


MODEL_NAME = "v82e_balanced_predecoder_pose_q25_sparse640"
base.MODELS = tuple(base.MODELS) + (MODEL_NAME,)

_base_load_model = base.load_model
_base_run_model = base.run_model


def build_dataset(args, scene: str):
    """Use only frame IDs having both RGB and depth files."""
    base.add_point3r_paths(args.point3r_repo)
    from eval.mv_recon.data import NRGBD

    resolution = (512, 384) if args.size == 512 else 224
    dataset = NRGBD(
        split="test",
        ROOT=args.nrgbd_root,
        resolution=resolution,
        num_seq=1,
        test_id=scene,
        full_video=True,
        kf_every=args.kf_every,
    )
    image_dir = Path(args.nrgbd_root) / scene / "images"
    depth_dir = Path(args.nrgbd_root) / scene / "depth"
    frame_ids = []
    for image_path in image_dir.glob("img*.png"):
        stem = image_path.stem[3:]
        if stem.isdigit() and (depth_dir / f"depth{stem}.png").is_file():
            frame_ids.append(int(stem))
    frame_ids.sort()
    if not frame_ids:
        raise RuntimeError(f"no paired RGB/depth frames for scene={scene}")
    step = min(args.kf_every, max(len(frame_ids) // 2, 1))
    sampled = frame_ids[::step]
    dataset.tuple_list = [scene + " " + " ".join(map(str, sampled))]
    dataset.scene_list = [scene]
    dataset.num_seq = 1
    print(
        f"[NRGBD_EXISTING_FRAMES] scene={scene} raw={len(frame_ids)} "
        f"kept={len(sampled)} step={step} first={sampled[:3]} last={sampled[-3:]}",
        flush=True,
    )
    return dataset


def load_model(args, model_name: str, device: str):
    if model_name != MODEL_NAME:
        return _base_load_model(args, model_name, device)

    base.configure_point3r_env("sparse512", args)
    base.add_point3r_paths(args.point3r_repo)
    from dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose import Point3R

    model = Point3R.from_pretrained(args.point3r_weights).to(device).eval()
    print(
        "[V79_EFFECTIVE_CONFIG] "
        f"drop_q={os.environ.get('POINT3R_CGMC_DROP_QUANTILE')} "
        f"slots={model.kway_num_slots} "
        f"bins={model.ordered_theta_bins}x{model.ordered_phi_bins}x{model.ordered_rho_bins} "
        f"sparse={int(model.sparse_readout_enabled)} "
        f"sparse_max={model.sparse_readout_max_tokens} "
        f"anchors={model.sparse_readout_global_anchors} "
        f"rayaware={os.environ.get('POINT3R_RAYAWARE_UPDATE')} "
        f"dual_bank={os.environ.get('POINT3R_RAY_DUAL_BANK')} "
        f"pose_input_only={os.environ.get('POINT3R_RAY_POSE_INPUT_ONLY')} "
        f"post_decoder_only={os.environ.get('POINT3R_RAY_POSE_POST_DECODER_ONLY')} "
        f"pose_ensemble={os.environ.get('POINT3R_RAY_POSE_ONLY_ENSEMBLE')}",
        flush=True,
    )
    return model


def run_model(args, model_name: str, model, batch, device: str):
    if model_name != MODEL_NAME:
        return _base_run_model(args, model_name, model, batch, device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    started = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            output = model(batch, point3r_tag=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    return output.ress, output.views, elapsed, peak_gb


base.load_model = load_model
base.run_model = run_model
base.build_dataset = build_dataset


if __name__ == "__main__":
    raise SystemExit(base.main())
