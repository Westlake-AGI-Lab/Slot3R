#!/usr/bin/env python3
"""Load the verified v106 model through the clean point-cloud harness."""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import torch

V97_DIR = Path("/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300")
sys.path.insert(0, str(V97_DIR))
import launch_v97_nrgbd as protocol

base = protocol.base
MODEL_NAME = "v106_fresh_bank_pose_q25_sparse640"
base.MODELS = tuple(base.MODELS) + (MODEL_NAME,)


def load_model(args, model_name: str, device: str):
    if model_name != MODEL_NAME:
        return protocol._base_load_model(args, model_name, device)
    base.configure_point3r_env("sparse512", args)
    base.add_point3r_paths(args.point3r_repo)
    from dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose import Point3R
    model = Point3R.from_pretrained(args.point3r_weights).to(device).eval()
    print(
        "[V106_POINTCLOUD_CONFIG] "
        f"drop_q={os.environ.get('POINT3R_CGMC_DROP_QUANTILE')} "
        f"sparse_max={model.sparse_readout_max_tokens} "
        f"ray_bank_every={os.environ.get('POINT3R_RAY_BANK_UPDATE_EVERY')} "
        f"pose_weight={os.environ.get('POINT3R_V106_POSE_INPUT_WEIGHT', '0.15')}",
        flush=True,
    )
    return model


def run_model(args, model_name: str, model, batch, device: str):
    if model_name != MODEL_NAME:
        return protocol._base_run_model(args, model_name, model, batch, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    started = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            output = model(batch, point3r_tag=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return output.ress, output.views, elapsed, peak_gb


base.load_model = load_model
base.run_model = run_model
base.build_dataset = protocol.build_dataset


if __name__ == "__main__":
    raise SystemExit(base.main())
