#!/usr/bin/env python3
"""Self-contained Point3R ScanNet color_90 BKS-style pose backend.

This file is intentionally only for the Point3R family:
baseline, clean K-way, sparse512, and geoanchor512.  The public launcher
scannet_color90_pose_clean.py owns scene loops and summary files; this backend
runs exactly one scene and writes <scene>_eval_metric.txt.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


POINT3R_ENV_KEYS = (
    "POINT3R_MEMORY_UPDATE_MODE",
    "POINT3R_ORDERED_UPDATE_IMPL",
    "POINT3R_KWAY_NUM_SLOTS",
    "POINT3R_ORDERED_WAY_POLICY",
    "POINT3R_ORDERED_THETA_BINS",
    "POINT3R_ORDERED_PHI_BINS",
    "POINT3R_ORDERED_RHO_BINS",
    "POINT3R_ORDERED_APP_THRESHOLD",
    "POINT3R_ORDERED_BUDGET_EVICT",
    "POINT3R_SPARSE_READOUT",
    "POINT3R_SPARSE_MODE",
    "POINT3R_SPARSE_MAX_TOKENS",
    "POINT3R_SPARSE_GLOBAL_ANCHORS",
    "POINT3R_SPARSE_NEIGHBOR_RANGE",
    "POINT3R_SPARSE_RECENT_TOKENS",
    "POINT3R_SPARSE_RECENT_FRAMES",
    "POINT3R_FIXED_MEMORY_TOKENS",
    "POINT3R_FIXED_MEMORY_MODE",
    "POINT3R_GEOANCHOR",
    "POINT3R_GEOANCHOR_ENABLE",
    "POINT3R_GEOANCHOR_STRIDE",
    "POINT3R_GEOANCHOR_H",
    "POINT3R_GEOANCHOR_FRAMES",
    "POINT3R_GEOANCHOR_MIN_GAP",
    "POINT3R_GEOANCHOR_BUCKETS_PER_KF",
    "POINT3R_GEOANCHOR_SLOTS_PER_KF",
    "POINT3R_GEOANCHOR_MAX_KFS",
    "POINT3R_PROFILE",
)


def add_path(path: str | Path) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def add_repo_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    for p in (repo, repo / "src" / "croco", repo / "src"):
        add_path(p)


def configure_env(method: str) -> None:
    for key in POINT3R_ENV_KEYS:
        os.environ.pop(key, None)

    if method == "baseline":
        return

    os.environ.update(
        {
            "POINT3R_MEMORY_UPDATE_MODE": "ordered_kway",
            "POINT3R_ORDERED_UPDATE_IMPL": "tensor",
            "POINT3R_KWAY_NUM_SLOTS": "8",
            "POINT3R_ORDERED_WAY_POLICY": "appearance",
            "POINT3R_ORDERED_THETA_BINS": "16",
            "POINT3R_ORDERED_PHI_BINS": "8",
            "POINT3R_ORDERED_RHO_BINS": "32",
        }
    )

    if method in ("sparse512", "geoanchor512", "sparse640_q25"):
        os.environ.update(
            {
                "POINT3R_SPARSE_READOUT": "1",
                "POINT3R_SPARSE_MODE": "max",
                "POINT3R_SPARSE_MAX_TOKENS": "640",
                "POINT3R_SPARSE_GLOBAL_ANCHORS": "128",
                "POINT3R_SPARSE_NEIGHBOR_RANGE": "1",
            }
        )

    if method == "geoanchor512":
        os.environ.update(
            {
                "POINT3R_GEOANCHOR": "1",
                "POINT3R_GEOANCHOR_ENABLE": "1",
                "POINT3R_GEOANCHOR_STRIDE": "8",
                "POINT3R_GEOANCHOR_H": "8",
                "POINT3R_GEOANCHOR_FRAMES": "4",
                "POINT3R_GEOANCHOR_MIN_GAP": "16",
                "POINT3R_GEOANCHOR_BUCKETS_PER_KF": "32",
                "POINT3R_GEOANCHOR_SLOTS_PER_KF": "32",
                "POINT3R_GEOANCHOR_MAX_KFS": "2048",
            }
        )


def load_model_class(method: str):
    if method != "sparse640_q25":
        raise ValueError(f"unsupported method={method}")
    model_module = os.environ.get(
        "POINT3R_POSE_MODEL_MODULE",
        "dust3r.point3r_kway_frame_sparse_q35_confselect",
    )
    Point3R = importlib.import_module(model_module).Point3R
    print(
        f"[RAYAWAY_POSE_V17_CONFIG] module={model_module} "
        f"drop_q={os.environ.get('POINT3R_CGMC_DROP_QUANTILE','0.25')} "
        f"min_conf={os.environ.get('POINT3R_CGMC_MIN_CONF','0.0')} "
        f"weighted_merge={os.environ.get('POINT3R_CGMC_WEIGHTED_MERGE','1')} "
        f"sparse_max={os.environ.get('POINT3R_SPARSE_MAX_TOKENS','640')}",
        flush=True,
    )
    return Point3R


def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import weighted_procrustes

    bsz, height, width, _ = pts3ds_self.shape
    pp = (
        torch.tensor([width // 2, height // 2], device=pts3ds_self.device)
        .float()
        .repeat(bsz, 1)
        .reshape(bsz, 1, 2)
    )
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    pts3ds_self = pts3ds_self.reshape(bsz, -1, 3)
    pts3ds_other = pts3ds_other.reshape(bsz, -1, 3)
    conf_self = conf_self.reshape(bsz, -1)
    conf_other = conf_other.reshape(bsz, -1)
    c2w = weighted_procrustes(
        pts3ds_self,
        pts3ds_other,
        torch.log(conf_self) * torch.log(conf_other),
        use_weights=True,
        return_T=True,
    )
    return c2w, focal, pp.reshape(bsz, 2)


def prepare_input(img_paths, img_mask, size, revisit=1, update=True, crop=True):
    from src.dust3r.utils.image import load_images_for_eval as load_images

    images = load_images(img_paths, size=size, crop=crop)
    views = []
    for i, image in enumerate(images):
        img = image["img"]
        views.append(
            {
                "img": img,
                "ray_map": torch.full(
                    (img.shape[0], 6, img.shape[-2], img.shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(bool(img_mask[i])).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(bool(update and img_mask[i])).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )

    if revisit <= 1:
        return views

    repeated = []
    for r in range(revisit):
        for i, view in enumerate(views):
            new_view = deepcopy(view)
            new_view["idx"] = r * len(views) + i
            new_view["instance"] = str(r * len(views) + i)
            if r > 0 and not update:
                new_view["update"] = torch.tensor(False).unsqueeze(0)
            repeated.append(new_view)
    return repeated


def prepare_output(outputs, revisit=1, solve_pose=False):
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]

    if solve_pose:
        pr_poses, focal, pp = recover_cam_params(
            torch.cat(pts3ds_self, 0),
            torch.cat(pts3ds_other, 0),
            torch.cat(conf_self, 0),
            torch.cat(conf_other, 0),
        )
    else:
        pts3ds_self_cat = torch.cat(pts3ds_self, 0)
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)
        bsz, height, width, _ = pts3ds_self_cat.shape
        pp = (
            torch.tensor([width // 2, height // 2], device=pts3ds_self_cat.device)
            .float()
            .repeat(bsz, 1)
            .reshape(bsz, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self_cat, pp, focal_mode="weiszfeld")

    cam_dict = {"focal": focal.cpu().numpy(), "pp": pp.cpu().numpy()}
    return cam_dict, pr_poses


def run_one_scene(args) -> tuple[float, float, float]:
    from eval.relpose.metadata import dataset_metadata
    from eval.relpose.utils import eval_metrics, get_tum_poses, load_traj
    from src.dust3r.inference import inference

    metadata = dataset_metadata.get(args.dataset)
    if metadata is None:
        raise RuntimeError("dataset_metadata has no scannet entry")

    img_path = args.scannet_root
    scene = args.scene
    dir_path = metadata["dir_path_func"](img_path, scene)
    filelist = sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    filelist = filelist[:: args.pose_eval_stride]
    if not filelist:
        raise RuntimeError(f"no images for scene={scene} dir={dir_path}")

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
    print(f"[check] deterministic_seed={seed}", flush=True)

    configure_env(args.method)
    Point3R = load_model_class(args.method)
    print(f"[check] method={args.method}", flush=True)
    print(f"[check] model={Point3R.__module__}", flush=True)
    print(
        "[check] sparse_max={} anchors={} neighbor={} geoanchor={}".format(
            os.environ.get("POINT3R_SPARSE_MAX_TOKENS", ""),
            os.environ.get("POINT3R_SPARSE_GLOBAL_ANCHORS", ""),
            os.environ.get("POINT3R_SPARSE_NEIGHBOR_RANGE", ""),
            os.environ.get("POINT3R_GEOANCHOR", os.environ.get("POINT3R_GEOANCHOR_ENABLE", "")),
        ),
        flush=True,
    )
    model = Point3R.from_pretrained(args.weights).to(args.device).eval()

    views = prepare_input(
        filelist,
        [True for _ in filelist],
        size=args.size,
        crop=not args.no_crop,
        revisit=args.revisit,
        update=not args.freeze_state,
    )

    with torch.no_grad():
        outputs = inference(views, model, args.device)

    cam_dict, pr_poses = prepare_output(
        outputs,
        revisit=args.revisit,
        solve_pose=args.solve_pose,
    )
    pred_traj = get_tum_poses(pr_poses)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anno_path = metadata.get("anno_path", None)
    gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, scene)
    traj_format = metadata.get("traj_format", None)
    if traj_format is None:
        raise RuntimeError(f"scannet metadata has no traj_format; gt={gt_traj_file}")
    gt_traj = load_traj(
        gt_traj_file=gt_traj_file,
        traj_format=traj_format,
        stride=args.pose_eval_stride,
    )

    ate, rpe_trans, rpe_rot = eval_metrics(
        pred_traj,
        gt_traj,
        seq=scene,
        filename=str(output_dir / f"{scene}_eval_metric.txt"),
    )
    return float(ate), float(rpe_trans), float(rpe_rot)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("sparse640_q25",), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", choices=("scannet", "tum"), default="scannet")
    parser.add_argument("--scannet_root", default="")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pose_eval_stride", type=int, default=1)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true")
    parser.add_argument("--solve_pose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_repo_paths(args.repo)
    from eval.add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(args.weights)
    ate, rpe_t, rpe_rot = run_one_scene(args)
    print(
        f"[pose_scene] method={args.method} scene={args.scene} "
        f"ATE_RMSE={ate:.6f} RPE_t={rpe_t:.6f} RPE_rot={rpe_rot:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
