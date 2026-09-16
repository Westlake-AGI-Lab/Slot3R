#!/usr/bin/env python3
"""Clean Bonn depth evaluator aligned to the MonST3R prepared 110-frame split.

Dataset protocol:
  <bonn_root>/rgbd_bonn_<scene>/rgb_110/*.png
  <bonn_root>/rgbd_bonn_<scene>/depth_110/*.png
  <bonn_root>/rgbd_bonn_<scene>/groundtruth_110.txt

Depth protocol:
  - depth png scale: /5000.0 meters
  - MonST3R no-crop geometry: resize long edge to --size, then make H/W multiples of 16
  - valid GT: min_depth < depth < max_depth
  - default alignment: scale_shift least squares on valid pixels

Supported models:
  point3r, kway, sparse512, geoanchor512, cut3r, ghost, ttt3r
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


MODELS = ("point3r", "kway", "depthcc", "sparse512", "geoanchor512", "cut3r", "ghost", "ttt3r")
DEFAULT_SCENES = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")
BONN_INTRINSICS = np.array(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def add_path(path: str | Path) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def add_point3r_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    for p in (repo, repo / "src" / "croco", repo / "src"):
        add_path(p)


def add_src_repo_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    for p in (repo, repo / "src", repo / "src" / "croco"):
        add_path(p)


def clear_point3r_env() -> None:
    for key in (
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
    ):
        os.environ.pop(key, None)


def configure_point3r_env(model_name: str, args) -> None:
    clear_point3r_env()
    if model_name == "point3r":
        return
    os.environ.update(
        {
            "POINT3R_MEMORY_UPDATE_MODE": "ordered_kway",
            "POINT3R_ORDERED_UPDATE_IMPL": "tensor",
            "POINT3R_KWAY_NUM_SLOTS": str(args.kway_slots),
            "POINT3R_ORDERED_WAY_POLICY": "appearance",
            "POINT3R_ORDERED_THETA_BINS": str(args.theta_bins),
            "POINT3R_ORDERED_PHI_BINS": str(args.phi_bins),
            "POINT3R_ORDERED_RHO_BINS": str(args.rho_bins),
        }
    )
    if model_name in ("depthcc", "sparse512", "geoanchor512"):
        os.environ.update(
            {
                "POINT3R_SPARSE_READOUT": "1",
                "POINT3R_SPARSE_MODE": "max",
                "POINT3R_SPARSE_MAX_TOKENS": str(args.sparse_max_tokens),
                "POINT3R_SPARSE_GLOBAL_ANCHORS": str(args.sparse_global_anchors),
                "POINT3R_SPARSE_NEIGHBOR_RANGE": str(args.sparse_neighbor_range),
                "POINT3R_CGMC_DROP_QUANTILE": str(args.drop_quantile),
                "POINT3R_CGMC_MIN_CONF": "0.0",
            }
        )
    if model_name == "geoanchor512":
        os.environ.update({"POINT3R_GEOANCHOR": "1", "POINT3R_GEOANCHOR_ENABLE": "1"})


def read_tum_poses_ordered(path: str | Path) -> list[np.ndarray]:
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("scipy is required for groundtruth_110 pose loading") from exc

    poses = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        tx, ty, tz = (float(x) for x in parts[1:4])
        qx, qy, qz, qw = (float(x) for x in parts[4:8])
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        poses.append(pose)
    return poses


def resize_like_monst3r_no_crop(image, depth, intrinsics, long_edge_size: int):
    h, w = image.shape[:2]
    scale = float(long_edge_size) / float(max(w, h))
    w1 = int(round(w * scale))
    h1 = int(round(h * scale))
    cx, cy = w1 // 2, h1 // 2
    out_w = ((2 * cx) // 16) * 16
    out_h = ((2 * cy) // 16) * 16
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"bad resize from {(w, h)} to {(out_w, out_h)}")
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    image = cv2.resize(image, (out_w, out_h), interpolation=interp)
    depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    intrinsics = intrinsics.copy()
    intrinsics[0, :] *= float(out_w) / float(w)
    intrinsics[1, :] *= float(out_h) / float(h)
    return image, depth, intrinsics


def load_bonn_scene(args, scene: str):
    # Accept both the short protocol name (``balloon2``) and the on-disk
    # directory name (``rgbd_bonn_balloon2``).  This makes SCENES usable with
    # names copied directly from the dataset directory without duplicating the
    # prefix.
    scene_dir = scene if scene.startswith("rgbd_bonn_") else f"rgbd_bonn_{scene}"
    root = Path(args.bonn_root) / scene_dir
    rgb_dir = root / "rgb_110"
    depth_dir = root / "depth_110"
    pose_path = root / "groundtruth_110.txt"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not depth_dir.is_dir():
        raise FileNotFoundError(depth_dir)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    depth_files = sorted(p for p in depth_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    poses = read_tum_poses_ordered(pose_path)
    n = min(len(rgb_files), len(depth_files), len(poses))
    frames = []
    for idx in range(0, n, max(1, args.kf_every)):
        rgb = cv2.imread(str(rgb_files[idx]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"could not read rgb {rgb_files[idx]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_raw = cv2.imread(str(depth_files[idx]), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise IOError(f"could not read depth {depth_files[idx]}")
        depth = np.nan_to_num(depth_raw.astype(np.float32), 0.0) / float(args.depth_scale)
        depth[depth < 1e-3] = 0.0
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        rgb, depth, intrinsics = resize_like_monst3r_no_crop(rgb, depth, BONN_INTRINSICS, args.size)
        frames.append(
            {
                "rgb": rgb,
                "depth": depth.astype(np.float32),
                "pose": poses[idx].astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "rgb_path": str(rgb_files[idx]),
                "depth_path": str(depth_files[idx]),
            }
        )
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError(f"no frames for scene={scene}")
    return frames


def frames_to_batch(frames):
    batch = []
    for i, frame in enumerate(frames):
        img = torch.from_numpy(frame["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1)
        img = img * 2.0 - 1.0
        depth = torch.from_numpy(frame["depth"]).unsqueeze(0)
        valid = (depth > 0).bool()
        batch.append(
            {
                "img": img.unsqueeze(0),
                "ray_map": torch.full((1, 6, img.shape[-2], img.shape[-1]), torch.nan),
                "true_shape": torch.tensor([[img.shape[-2], img.shape[-1]]], dtype=torch.int32),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(frame["pose"]).unsqueeze(0),
                "camera_intrinsics": torch.from_numpy(frame["intrinsics"]).unsqueeze(0),
                "depthmap": depth.unsqueeze(0),
                "valid_mask": valid.unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return batch


def move_batch_to_device(batch, device: str):
    keep_cpu = {"depthmap", "valid_mask", "idx", "instance", "true_shape"}
    for view in batch:
        for key, value in list(view.items()):
            if key in keep_cpu:
                continue
            if torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return batch


def move_tree_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_tree_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_tree_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_tree_to_device(v, device) for v in obj)
    return obj


def ceil_to_multiple(x: int, multiple: int) -> int:
    return int(((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple))


def resize_map_to_hw(x, hw):
    if not torch.is_tensor(x):
        return x
    h, w = hw
    if x.ndim == 4:
        if x.shape[-1] <= 8:
            y = x.permute(0, 3, 1, 2)
            y = torch.nn.functional.interpolate(y.float(), size=(h, w), mode="bilinear", align_corners=False)
            return y.permute(0, 2, 3, 1).to(dtype=x.dtype)
        return torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False).to(dtype=x.dtype)
    if x.ndim == 3:
        y = torch.nn.functional.interpolate(x[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)
        return y[:, 0].to(dtype=x.dtype)
    return x


def build_ghost_views_from_batch(batch, multiple: int = 14):
    views, original_shapes = [], []
    for view in batch:
        img = view["img"]
        h, w = img.shape[-2:]
        hh, ww = ceil_to_multiple(h, multiple), ceil_to_multiple(w, multiple)
        new_view = dict(view)
        if (hh, ww) != (h, w):
            new_view["img"] = torch.nn.functional.interpolate(img, size=(hh, ww), mode="bilinear", align_corners=False)
        views.append(new_view)
        original_shapes.append((h, w))
    return views, original_shapes


def resize_preds_to_batch(preds, original_shapes):
    out = []
    for pred, hw in zip(preds, original_shapes):
        pred = dict(pred)
        for key in ("pts3d_in_other_view", "pts3d", "pts3d_in_self_view", "depth", "conf", "depth_conf", "valid_mask"):
            if key in pred:
                pred[key] = resize_map_to_hw(pred[key], hw)
        out.append(pred)
    return out


def build_ttt3r_views_from_batch(batch, device):
    views = []
    for i, view in enumerate(batch):
        img = view["img"].detach().to(device)
        views.append(
            {
                "img": img,
                "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan, device=device),
                "true_shape": torch.tensor([[img.shape[-2], img.shape[-1]]], dtype=torch.int32, device=device),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0),
                "img_mask": torch.tensor(True, device=device).unsqueeze(0),
                "ray_mask": torch.tensor(False, device=device).unsqueeze(0),
                "update": torch.tensor(True, device=device).unsqueeze(0),
                "reset": torch.tensor(False, device=device).unsqueeze(0),
            }
        )
    return views


def load_model(args, model_name: str):
    device = args.device
    if model_name in ("point3r", "kway", "depthcc", "sparse512", "geoanchor512"):
        # depthcc uses the frame-sparse ConfSelect implementation. Configure it
        # as sparse readout so the requested token budget is not cleared.
        configure_point3r_env(model_name, args)
        add_point3r_paths(args.point3r_repo)
        if model_name == "point3r":
            from dust3r.point3r import Point3R
        elif model_name in ("kway", "depthcc"):
            print(f"[check] before kway import env POINT3R_KWAY_NUM_SLOTS={os.environ.get('POINT3R_KWAY_NUM_SLOTS')}")
            os.environ["POINT3R_KWAY_NUM_SLOTS"] = os.environ.get("POINT3R_KWAY_NUM_SLOTS_FORCE", os.environ.get("POINT3R_KWAY_NUM_SLOTS", "16"))
            print(f"[check] before kway import forced POINT3R_KWAY_NUM_SLOTS={os.environ.get('POINT3R_KWAY_NUM_SLOTS')}")
            if model_name == "depthcc":
                module_name = os.environ.get(
                    "POINT3R_DEPTH_MODEL_MODULE",
                    "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose",
                )
                Point3R = importlib.import_module(module_name).Point3R
            else:
                from dust3r.point3r_ordered_kway_clean import Point3R
        elif model_name == "sparse512":
            from dust3r.point3r_kway_frame_sparse import Point3R
        else:
            from dust3r.point3r_ordered_kway_clean_geoanchor import Point3R
        print(f"[check] model={model_name} module={Point3R.__module__}", flush=True)
        if model_name == "depthcc":
            print(
                "[Q_ABLATION_CONFIG] "
                f"drop_quantile={os.environ.get('POINT3R_CGMC_DROP_QUANTILE')} "
                f"sparse_max_tokens={os.environ.get('POINT3R_SPARSE_MAX_TOKENS')}",
                flush=True,
            )
        return Point3R.from_pretrained(args.point3r_weights).to(device).eval()

    if model_name == "cut3r":
        add_src_repo_paths(args.cut3r_repo)
        from dust3r.model import ARCroco3DStereo
        from dust3r.inference import inference

        model = ARCroco3DStereo.from_pretrained(args.cut3r_weights).to(device).eval()
        model._depth_clean_inference = inference
        return model

    if model_name == "ghost":
        add_src_repo_paths(args.ghost_repo)
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT(total_budget=args.ghost_total_budget).to(device).eval()
        ckpt = torch.load(args.ghost_weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        return model

    if model_name == "ttt3r":
        add_src_repo_paths(args.ttt3r_repo)
        from src.dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(args.ttt3r_weights).to(device).eval()
        model.config.model_update_type = "ttt3r"
        return model

    raise ValueError(model_name)


def run_model(args, model_name: str, model, batch):
    device = args.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    t0 = time.time()
    with torch.no_grad():
        if model_name in ("point3r", "kway", "depthcc", "sparse512", "geoanchor512"):
            batch_dev = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=False):
                output = model(batch_dev, point3r_tag=True)
            preds = output.ress
        elif model_name == "cut3r":
            batch_dev = move_batch_to_device(batch, device)
            output = model._depth_clean_inference(batch_dev, model, device)
            if isinstance(output, tuple):
                output = output[0]
            preds = output["pred"] if isinstance(output, dict) else output.pred
        elif model_name == "ghost":
            batch_dev = move_batch_to_device(batch, device)
            reset = getattr(getattr(model, "aggregator", None), "reset_kv_repository", None)
            if reset is not None:
                reset()
            ghost_batch, original_shapes = build_ghost_views_from_batch(batch_dev, args.ghost_patch_multiple)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                output = model.inference(ghost_batch)
            preds = resize_preds_to_batch(output.ress, original_shapes)
            if reset is not None:
                reset()
        elif model_name == "ttt3r":
            from src.dust3r.inference import inference_recurrent_lighter

            output, _ = inference_recurrent_lighter(build_ttt3r_views_from_batch(batch, device), model, device)
            preds = output["pred"]
        else:
            raise ValueError(model_name)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return preds, elapsed, peak


def pred_to_depth(pred, hw, model_name=None):
    if model_name == "ghost":
        source = os.environ.get("GHOST_DEPTH_SOURCE", "depth")

        if source == "depth":
            if "depth" not in pred:
                raise KeyError(f"ghost prediction missing depth keys={list(pred.keys())}")
            depth = pred["depth"]
            if depth.ndim == 4:
                depth = depth[:, 0]
            return resize_map_to_hw(depth, hw)[0].detach().float().cpu().numpy()

        if source in ("pts3d_in_self_view", "pts3d", "pts3d_in_other_view"):
            if source not in pred:
                raise KeyError(f"ghost prediction missing {source} keys={list(pred.keys())}")
            pts = pred[source]
            if pts.ndim == 4 and pts.shape[-1] == 3:
                depth = pts[..., 2]
            elif pts.ndim == 4 and pts.shape[1] == 3:
                depth = pts[:, 2]
            else:
                raise ValueError(f"unexpected ghost {source} shape={tuple(pts.shape)}")
            return resize_map_to_hw(depth, hw)[0].detach().float().cpu().numpy()

        raise ValueError(f"unknown GHOST_DEPTH_SOURCE={source}")

    if "depth" in pred:
        depth = pred["depth"]
        if depth.ndim == 4:
            depth = depth[:, 0]
        return resize_map_to_hw(depth, hw)[0].detach().float().cpu().numpy()

    for key in ("pts3d_in_self_view", "pts3d", "pts3d_in_other_view"):
        if key in pred:
            pts = pred[key]
            if pts.ndim == 4 and pts.shape[-1] == 3:
                depth = pts[..., 2]
            elif pts.ndim == 4 and pts.shape[1] == 3:
                depth = pts[:, 2]
            else:
                continue
            return resize_map_to_hw(depth, hw)[0].detach().float().cpu().numpy()

    raise KeyError(f"cannot find depth-like prediction keys={list(pred.keys())}")

def align_depth(pred, gt, mask, mode: str):
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    if len(p) < 16:
        return pred
    if mode == "none":
        return pred
    if mode == "scale":
        denom = float(np.dot(p, p))
        scale = float(np.dot(p, g) / denom) if denom > 1e-12 else 1.0
        return pred * scale
    if mode == "median":
        med_p = float(np.median(p))
        med_g = float(np.median(g))
        return pred * (med_g / med_p) if abs(med_p) > 1e-12 else pred
    if mode == "scale_shift":
        a = np.stack([p, np.ones_like(p)], axis=1)
        scale, shift = np.linalg.lstsq(a, g, rcond=None)[0]
        return pred * float(scale) + float(shift)
    raise ValueError(mode)


def depth_metrics(preds, batch, args):
    records = []
    for pred, view in zip(preds, batch):
        gt = view["depthmap"][0, 0].detach().cpu().numpy().astype(np.float32)
        hw = gt.shape
        pd = pred_to_depth(pred, hw, args.model).astype(np.float32)
        mask = np.isfinite(gt) & np.isfinite(pd) & (gt > args.min_depth) & (gt < args.max_depth) & (pd > 0)
        if args.center_crop > 0:
            h, w = gt.shape
            crop = min(args.center_crop, h, w)
            y0 = (h - crop) // 2
            x0 = (w - crop) // 2
            sl = np.s_[y0 : y0 + crop, x0 : x0 + crop]
            gt, pd, mask = gt[sl], pd[sl], mask[sl]
        if mask.sum() >= 16:
            records.append({"gt": gt, "pd": pd, "mask": mask})

    if not records:
        return {k: float("nan") for k in ("AbsRel", "SqRel", "RMSE", "RMSElog", "Delta1", "Delta2", "Delta3", "Valid")}

    if args.align.startswith("sequence_"):
        p_all = np.concatenate([r["pd"][r["mask"]].astype(np.float64) for r in records])
        g_all = np.concatenate([r["gt"][r["mask"]].astype(np.float64) for r in records])
        seq_mode = args.align[len("sequence_") :]
        if seq_mode == "scale":
            denom = float(np.dot(p_all, p_all))
            scale = float(np.dot(p_all, g_all) / denom) if denom > 1e-12 else 1.0
            shift = 0.0
        elif seq_mode == "median":
            med_p = float(np.median(p_all))
            med_g = float(np.median(g_all))
            scale = float(med_g / med_p) if abs(med_p) > 1e-12 else 1.0
            shift = 0.0
        elif seq_mode == "scale_shift":
            a = np.stack([p_all, np.ones_like(p_all)], axis=1)
            scale, shift = np.linalg.lstsq(a, g_all, rcond=None)[0]
            scale, shift = float(scale), float(shift)
        else:
            raise ValueError(args.align)
        for r in records:
            r["pd"] = r["pd"] * scale + shift
        metric_align = "none"
    else:
        metric_align = args.align

    vals = []
    for r in records:
        gt, pd, mask = r["gt"], r["pd"], r["mask"]
        pd = align_depth(pd, gt, mask, metric_align)
        mask = mask & np.isfinite(pd) & (pd > args.min_depth) & (pd < args.max_depth)
        if mask.sum() < 16:
            continue
        p = pd[mask].astype(np.float64)
        g = gt[mask].astype(np.float64)
        thresh = np.maximum(g / p, p / g)
        vals.append(
            {
                "AbsRel": float(np.mean(np.abs(g - p) / g)),
                "SqRel": float(np.mean(((g - p) ** 2) / g)),
                "RMSE": float(np.sqrt(np.mean((g - p) ** 2))),
                "RMSElog": float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
                "Delta1": float(np.mean(thresh < 1.25)),
                "Delta2": float(np.mean(thresh < 1.25**2)),
                "Delta3": float(np.mean(thresh < 1.25**3)),
                "Valid": float(mask.sum()),
            }
        )
    if not vals:
        return {k: float("nan") for k in ("AbsRel", "SqRel", "RMSE", "RMSElog", "Delta1", "Delta2", "Delta3", "Valid")}
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def eval_one_scene(args, model_name: str, model, scene: str):
    frames = load_bonn_scene(args, scene)
    batch = frames_to_batch(frames)
    preds, elapsed, peak = run_model(args, model_name, model, batch)
    n = min(len(batch), len(preds))
    batch, preds = batch[:n], preds[:n]
    metrics = depth_metrics(preds, batch, args)
    fps = n / max(elapsed, 1e-9)
    return {
        "model": model_name,
        "scene": scene,
        "frames": n,
        **metrics,
        "FPS": fps,
        "Time": elapsed,
        "Peak": peak,
    }


def existing_scenes(root: str | Path, scenes: list[str] | None):
    if scenes:
        return scenes
    root = Path(root)
    out = []
    for scene in DEFAULT_SCENES:
        if (root / f"rgbd_bonn_{scene}" / "rgb_110").is_dir():
            out.append(scene)
    return out


def fmt(x, nd=6):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_stats(path: Path, rows: list[dict], failures: list[str], header: list[str]):
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    lines = ["========== DEPTH_SCENES ==========", "\t".join(header)]
    for row in rows:
        lines.append("\t".join(fmt(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header))
    if failures:
        lines += ["", "========== FAILURES ==========", *failures]
    lines += ["", "========== SUMMARY =========="]
    lines.append("| Method | n | frames | AbsRel | SqRel | RMSE | RMSElog | Delta1 | Delta2 | Delta3 | FPS | Peak |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, rs in sorted(by_model.items()):
        def mean(key):
            arr = np.array([float(r[key]) for r in rs], dtype=np.float64)
            return float(np.nanmean(arr)) if len(arr) else float("nan")
        lines.append(
            f"| {model} | {len(rs)} | {mean('frames'):.1f} | {mean('AbsRel'):.6f} | {mean('SqRel'):.6f} | "
            f"{mean('RMSE'):.6f} | {mean('RMSElog'):.6f} | {mean('Delta1'):.6f} | {mean('Delta2'):.6f} | "
            f"{mean('Delta3'):.6f} | {mean('FPS'):.4f} | {mean('Peak'):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonn_root", default="/root/autodl-tmp/Bonn_unzip")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/clean_launchers/depth/results/bonn_depth")
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--kf_every", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--depth_scale", type=float, default=5000.0)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=70.0)
    parser.add_argument(
        "--align",
        choices=("none", "scale", "median", "scale_shift", "sequence_scale", "sequence_median", "sequence_scale_shift"),
        default="scale_shift",
    )
    parser.add_argument("--center_crop", type=int, default=0)

    parser.add_argument("--point3r_repo", default="/root/autodl-tmp/Point3R_mdf")
    parser.add_argument("--point3r_weights", default="/root/autodl-tmp/checkpoints/point3r_512.pth")
    parser.add_argument("--cut3r_repo", default="/root/autodl-tmp/CUT3R")
    parser.add_argument("--cut3r_weights", default="/root/autodl-tmp/checkpoints/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--ghost_repo", default="/root/autodl-tmp/GHOST")
    parser.add_argument("--ghost_weights", default="/root/checkpoints/checkpoints.pth")
    parser.add_argument("--ghost_total_budget", type=int, default=1200000)
    parser.add_argument("--ghost_patch_multiple", type=int, default=14)
    parser.add_argument("--ttt3r_repo", default="/root/autodl-tmp/TTT3R")
    parser.add_argument("--ttt3r_weights", default="/root/autodl-tmp/checkpoints/cut3r_512_dpt_4_64.pth")

    parser.add_argument("--kway_slots", type=int, default=8)
    parser.add_argument("--theta_bins", type=int, default=16)
    parser.add_argument("--phi_bins", type=int, default=8)
    parser.add_argument("--rho_bins", type=int, default=32)
    parser.add_argument("--sparse_max_tokens", type=int, default=512)
    parser.add_argument("--sparse_global_anchors", type=int, default=128)
    parser.add_argument("--sparse_neighbor_range", type=int, default=1)
    parser.add_argument("--drop_quantile", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models if args.models is not None else ([args.model] if args.model else None)
    if not models:
        raise SystemExit("pass --model NAME or --models NAME ...")
    scenes = existing_scenes(args.bonn_root, args.scenes)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.tsv"
    log = out / "run.log"
    stats = out / "stats_only.log"
    header = ["model", "scene", "frames", "AbsRel", "SqRel", "RMSE", "RMSElog", "Delta1", "Delta2", "Delta3", "Valid", "FPS", "Time", "Peak"]
    summary.write_text("\t".join(header) + "\n")
    log.write_text("")
    rows, failures = [], []
    with log.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(
            f"[check] bonn_root={args.bonn_root} scenes={scenes} size={args.size} "
            f"align={args.align} max_depth={args.max_depth} "
            f"drop_quantile={args.drop_quantile} sparse_max_tokens={args.sparse_max_tokens}\n"
        )
        for model_name in models:
            print(f"[check] loading model={model_name}", flush=True)
            lf.write(f"[check] loading model={model_name}\n")
            model = load_model(args, model_name)
            for scene in scenes:
                try:
                    row = eval_one_scene(args, model_name, model, scene)
                    rows.append(row)
                    line = "\t".join(fmt(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header)
                    print(f"[depth_scene] {line}", flush=True)
                    lf.write(f"[depth_scene] {line}\n")
                    with summary.open("a", encoding="utf-8") as sf:
                        sf.write(line + "\n")
                except Exception as exc:
                    msg = f"[depth_scene] model={model_name} scene={scene} status=FAIL error={type(exc).__name__}: {exc}"
                    print(msg, flush=True)
                    lf.write(msg + "\n")
                    lf.write(traceback.format_exc() + "\n")
                    failures.append(msg)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_stats(stats, rows, failures, header)
    print(f"wrote {summary}")
    print(f"wrote {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
