#!/usr/bin/env python3
"""Clean NRGBD point-cloud metric launcher.

One file, one NRGBD protocol, one metric protocol. It follows the two older
Point3R point-cloud launchers: NRGBD test split, optional max-frame truncation,
Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True), 224 center crop,
Open3D point-to-point ICP, then Acc/Comp/NC.

Supported model names:
  point3r, kway, sparse512, geoanchor, ghost, cut3r, ttt3r

The output directory keeps only text stats:
  run.log, summary.tsv, stats_only.log
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


MODELS = ("point3r", "kway", "sparse512", "geoanchor", "ghost", "cut3r", "ttt3r")
SCENES = (
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
)


def add_path(path: str | Path, *, front: bool = True) -> None:
    p = str(Path(path).resolve())
    if p in sys.path:
        return
    if front:
        sys.path.insert(0, p)
    else:
        sys.path.append(p)


def add_point3r_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    add_path(repo)
    add_path(repo / "src" / "croco")
    add_path(repo / "src")


def promote_paths(*paths: str | Path) -> None:
    for path in reversed([str(Path(p).resolve()) for p in paths]):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def force_point3r_metric_imports(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    promote_paths(repo, repo / "src" / "croco", repo / "src")
    for name in ("eval.mv_recon.criterion", "eval.mv_recon.utils"):
        sys.modules.pop(name, None)


def add_src_repo_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    add_path(repo)
    add_path(repo / "src")
    add_path(repo / "src" / "croco")


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

    if model_name in ("sparse512", "geoanchor"):
        os.environ.update(
            {
                "POINT3R_SPARSE_READOUT": "1",
                "POINT3R_SPARSE_MODE": "max",
                "POINT3R_SPARSE_MAX_TOKENS": str(args.sparse_max_tokens),
                "POINT3R_SPARSE_GLOBAL_ANCHORS": str(args.sparse_global_anchors),
                "POINT3R_SPARSE_NEIGHBOR_RANGE": str(args.sparse_neighbor_range),
            }
        )

    if model_name == "geoanchor":
        os.environ.update(
            {
                "POINT3R_GEOANCHOR": "1",
                "POINT3R_GEOANCHOR_STRIDE": str(args.geoanchor_stride),
                "POINT3R_GEOANCHOR_H": str(args.geoanchor_h),
                "POINT3R_GEOANCHOR_FRAMES": str(args.geoanchor_frames),
                "POINT3R_GEOANCHOR_MIN_GAP": str(args.geoanchor_min_gap),
                "POINT3R_GEOANCHOR_BUCKETS_PER_KF": str(args.geoanchor_buckets_per_kf),
                "POINT3R_GEOANCHOR_SLOTS_PER_KF": str(args.geoanchor_slots_per_kf),
                "POINT3R_GEOANCHOR_MAX_KFS": str(args.geoanchor_max_kfs),
            }
        )


def move_batch_to_device(batch, device: str):
    ignore_keys = {
        "depthmap",
        "dataset",
        "label",
        "instance",
        "idx",
        "true_shape",
        "rng",
    }
    for view in batch:
        for name, value in list(view.items()):
            if name in ignore_keys:
                continue
            if isinstance(value, (tuple, list)):
                view[name] = [
                    x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                    for x in value
                ]
            elif torch.is_tensor(value):
                view[name] = value.to(device, non_blocking=True)
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


def build_ttt3r_views_from_batch(batch):
    # Keep this aligned with the already-working pose clean launcher:
    # CPU views + minimal TTT3R fields; inference_recurrent_lighter handles device.
    views = []
    for i, view in enumerate(batch):
        img = view["img"].detach().cpu()
        ts = view.get("true_shape")
        if torch.is_tensor(ts):
            ts = ts.detach().cpu()
        else:
            ts = torch.tensor([[img.shape[-2], img.shape[-1]]], dtype=torch.int32)

        views.append({
            "img": img,
            "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan),
            "true_shape": ts,
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


def ceil_to_multiple(x: int, multiple: int) -> int:
    return int(((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple))


def build_ghost_views_from_batch(batch, multiple: int = 14):
    views = []
    original_shapes = []
    for view in batch:
        img = view["img"]
        h, w = img.shape[-2:]
        hh = ceil_to_multiple(h, multiple)
        ww = ceil_to_multiple(w, multiple)
        new_view = dict(view)
        if (hh, ww) != (h, w):
            new_view["img"] = torch.nn.functional.interpolate(
                img, size=(hh, ww), mode="bilinear", align_corners=False
            )
            if torch.is_tensor(new_view.get("true_shape")):
                new_view["true_shape"] = torch.tensor(
                    [[hh, ww]],
                    dtype=new_view["true_shape"].dtype,
                    device=new_view["img"].device,
                )
        views.append(new_view)
        original_shapes.append((h, w))
    return views, original_shapes


def resize_map_to_hw(x, hw):
    if not torch.is_tensor(x):
        return x
    h, w = hw
    orig_dtype = x.dtype
    if x.ndim == 4:
        if x.shape[-1] <= 8:
            y = x.float().permute(0, 3, 1, 2)
            y = torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            y = y.permute(0, 2, 3, 1)
        else:
            y = torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False)
        return (y > 0.5) if orig_dtype is torch.bool else y.to(dtype=orig_dtype)
    if x.ndim == 3:
        y = torch.nn.functional.interpolate(x[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        return (y > 0.5) if orig_dtype is torch.bool else y.to(dtype=orig_dtype)
    return x


def resize_ghost_preds_to_batch(preds, original_shapes):
    resized = []
    for pred, hw in zip(preds, original_shapes):
        pred = dict(pred)
        for key in ("pts3d_in_other_view", "pts3d", "pts3d_in_self_view"):
            if key in pred:
                pred[key] = resize_map_to_hw(pred[key], hw)
        for key in ("conf", "depth", "depth_conf", "valid_mask"):
            if key in pred:
                pred[key] = resize_map_to_hw(pred[key], hw)
        resized.append(pred)
    return resized

def center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return arr
    h, w = arr.shape[:2]
    if h < crop or w < crop:
        return arr
    cy, cx = h // 2, w // 2
    r = crop // 2
    return arr[cy - r : cy + r, cx - r : cx + r]


def maybe_subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), max_points, replace=False)]


def compute_open3d_metrics(pts_all, pts_gt_all, masks_all, images_all, args) -> dict[str, float]:
    force_point3r_metric_imports(args.point3r_repo)
    import open3d as o3d
    from eval.mv_recon.utils import accuracy, completion

    pred_chunks = []
    gt_chunks = []
    color_chunks = []
    for i in range(len(pts_all)):
        pts = center_crop(pts_all[i], args.center_crop)
        gt = center_crop(pts_gt_all[i], args.center_crop)
        mask = center_crop(masks_all[i], args.center_crop) > 0
        image = center_crop(images_all[i], args.center_crop)
        pred_chunks.append(pts[mask])
        gt_chunks.append(gt[mask])
        color_chunks.append(image[mask])

    pred = np.concatenate(pred_chunks, axis=0).reshape(-1, 3)
    gt = np.concatenate(gt_chunks, axis=0).reshape(-1, 3)
    color = np.concatenate(color_chunks, axis=0).reshape(-1, 3)

    pred_mask = np.isfinite(pred).all(axis=-1)
    gt_mask = np.isfinite(gt).all(axis=-1)
    keep = pred_mask & gt_mask
    pred = pred[keep]
    gt = gt[keep]
    color = color[keep]

    if len(pred) < 100 or len(gt) < 100:
        return {
            "Acc": float("nan"),
            "Comp": float("nan"),
            "NC1": float("nan"),
            "NC2": float("nan"),
            "Acc_med": float("nan"),
            "Comp_med": float("nan"),
            "NC1_med": float("nan"),
            "NC2_med": float("nan"),
        }

    pred = maybe_subsample(pred, args.max_points, args.seed)
    gt = maybe_subsample(gt, args.max_points, args.seed + 1)
    if len(color) != len(pred):
        color = np.ones_like(pred)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(color, 0.0, 1.0))

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        args.icp_thresh,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd = pcd.transform(reg_p2p.transformation)
    pcd.estimate_normals()
    pcd_gt.estimate_normals()

    gt_normal = np.asarray(pcd_gt.normals)
    pred_normal = np.asarray(pcd.normals)

    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    return {
        "Acc": float(acc),
        "Comp": float(comp),
        "NC1": float(nc1),
        "NC2": float(nc2),
        "Acc_med": float(acc_med),
        "Comp_med": float(comp_med),
        "NC1_med": float(nc1_med),
        "NC2_med": float(nc2_med),
    }


def collect_metric_arrays(batch, preds, args):
    force_point3r_metric_imports(args.point3r_repo)
    from eval.mv_recon.criterion import L21, Regr3D_t_ScaleShiftInv
    from dust3r.utils.geometry import geotrf

    metric_device = batch[0]["img"].device if torch.is_tensor(batch[0].get("img")) else torch.device("cpu")
    preds = move_tree_to_device(preds, metric_device)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    gt_pts, pred_pts, _gt_factor, _pr_factor, masks, monitoring = criterion.get_all_pts3d_t(batch, preds)
    gt_shift_z = monitoring.get("gt_shift_z", torch.zeros((), device=pred_pts[0].device))

    pts_all = []
    pts_gt_all = []
    images_all = []
    masks_all = []
    base_camera = batch[0]["camera_pose"][0].detach().cpu()
    for j, view in enumerate(batch):
        image = view["img"].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        image = (image + 1.0) / 2.0
        mask = view["valid_mask"].detach().cpu().numpy()[0]

        pts = pred_pts[j].detach().cpu().numpy()[0]
        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

        shift = gt_shift_z.detach().cpu().numpy()
        if np.ndim(shift) > 0:
            shift = float(np.ravel(shift)[0])
        else:
            shift = float(shift)

        # Matches the legacy Point3R NRGBD point-cloud launchers.
        pts[..., -1] += shift
        pts = geotrf(base_camera, pts)
        pts_gt[..., -1] += shift
        pts_gt = geotrf(base_camera, pts_gt)

        pts_all.append(pts[None])
        pts_gt_all.append(pts_gt[None])
        images_all.append(image[None])
        masks_all.append(mask[None])

    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(pts_gt_all, axis=0),
        np.concatenate(masks_all, axis=0),
        np.concatenate(images_all, axis=0),
    )


def save_scene_pointclouds(
    pts_all, pts_gt_all, masks_all, images_all, args, model_name: str, scene: str
) -> None:
    """Save a reproducibly sampled prediction/GT PLY pair for one scene."""
    if not args.save_ply:
        return

    force_point3r_metric_imports(args.point3r_repo)
    import open3d as o3d

    pred_chunks = []
    gt_chunks = []
    color_chunks = []
    for pts, gt, mask, image in zip(pts_all, pts_gt_all, masks_all, images_all):
        keep = mask > 0
        pred_chunks.append(pts[keep])
        gt_chunks.append(gt[keep])
        color_chunks.append(image[keep])

    pred = np.concatenate(pred_chunks, axis=0).reshape(-1, 3)
    gt = np.concatenate(gt_chunks, axis=0).reshape(-1, 3)
    color = np.concatenate(color_chunks, axis=0).reshape(-1, 3)
    keep = (
        np.isfinite(pred).all(axis=-1)
        & np.isfinite(gt).all(axis=-1)
        & np.isfinite(color).all(axis=-1)
    )
    pred, gt, color = pred[keep], gt[keep], color[keep]

    if args.ply_max_points > 0 and len(pred) > args.ply_max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pred), args.ply_max_points, replace=False)
        pred, gt, color = pred[idx], gt[idx], color[idx]

    ply_root = Path(args.ply_output_dir or args.output_dir)
    scene_dir = ply_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred)
    pred_pcd.colors = o3d.utility.Vector3dVector(np.clip(color, 0.0, 1.0))
    pred_path = scene_dir / f"{model_name}.ply"
    o3d.io.write_point_cloud(
        str(pred_path), pred_pcd, write_ascii=False, compressed=False
    )

    gt_path = scene_dir / "gt.ply"
    if not gt_path.exists():
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt)
        gt_pcd.colors = o3d.utility.Vector3dVector(np.clip(color, 0.0, 1.0))
        o3d.io.write_point_cloud(
            str(gt_path), gt_pcd, write_ascii=False, compressed=False
        )

    print(
        f"[PLY_SAVED] scene={scene} model={model_name} "
        f"pred={pred_path} gt={gt_path} points={len(pred)}",
        flush=True,
    )


def build_dataset(args, scene: str):
    add_point3r_paths(args.point3r_repo)
    from eval.mv_recon.data import NRGBD

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError(f"unsupported size={args.size}")

    return NRGBD(
        split="test",
        ROOT=args.nrgbd_root,
        resolution=resolution,
        num_seq=1,
        test_id=scene,
        full_video=True,
        kf_every=args.kf_every,
    )


def load_model(args, model_name: str, device: str):
    if model_name in ("point3r", "kway", "sparse512", "geoanchor"):
        configure_point3r_env(model_name, args)
        add_point3r_paths(args.point3r_repo)
        if model_name == "point3r":
            from dust3r.point3r import Point3R
        elif model_name == "kway":
            from dust3r.point3r_ordered_kway_clean import Point3R
        elif model_name == "sparse512":
            from dust3r.point3r_kway_frame_sparse import Point3R
        else:
            from dust3r.point3r_ordered_kway_clean_geoanchor import Point3R
        return Point3R.from_pretrained(args.point3r_weights).to(device).eval()

    if model_name == "ghost":
        add_src_repo_paths(args.ghost_repo)
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT(total_budget=args.ghost_total_budget).to(device).eval()
        ckpt = torch.load(args.ghost_weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        return model

    if model_name in ("cut3r", "ttt3r"):
        add_src_repo_paths(args.ttt3r_repo)
        from src.dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(args.ttt3r_weights).to(device).eval()
        model.config.model_update_type = model_name
        return model

    raise ValueError(model_name)


def run_model(args, model_name: str, model, batch, device: str):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.time()
    with torch.no_grad():
        if model_name in ("point3r", "kway", "sparse512", "geoanchor"):
            with torch.cuda.amp.autocast(enabled=False):
                output = model(batch, point3r_tag=True)
            preds, out_views = output.ress, output.views
        elif model_name == "ghost":
            reset = getattr(getattr(model, "aggregator", None), "reset_kv_repository", None)
            if reset is not None:
                reset()
            ghost_batch, original_shapes = build_ghost_views_from_batch(batch, args.ghost_patch_multiple)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                try:
                    output = model.inference(ghost_batch, frame_writer=None, cache_results=True)
                except TypeError:
                    output = model.inference(ghost_batch)
            if output is None or getattr(output, "ress", None) is None:
                raise RuntimeError("GHOST inference returned no ress")
            preds = resize_ghost_preds_to_batch(output.ress, original_shapes)
            out_views = batch
            if reset is not None:
                reset()
        elif model_name in ("cut3r", "ttt3r"):
            from src.dust3r.inference import inference_recurrent_lighter

            output, _ = inference_recurrent_lighter(build_ttt3r_views_from_batch(batch), model, device)
            preds, out_views = output["pred"], batch
        else:
            raise ValueError(model_name)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return preds, out_views, elapsed, peak_gb


def eval_one_scene(args, model_name: str, model, scene: str) -> dict:
    from torch.utils.data._utils.collate import default_collate

    dataset = build_dataset(args, scene)
    if len(dataset) < 1:
        raise RuntimeError(f"no sequence for scene={scene}")

    batch = default_collate([dataset[0]])
    orig_len = len(batch)
    if args.max_frames > 0 and len(batch) > args.max_frames:
        batch = batch[: args.max_frames]

    device = args.device
    if model_name == "ttt3r":
        preds, _out_views, elapsed, peak_gb = run_model(args, model_name, model, batch, device)
        batch = move_batch_to_device(batch, device)
    else:
        batch = move_batch_to_device(batch, device)
        preds, batch, elapsed, peak_gb = run_model(args, model_name, model, batch, device)

    valid_len = len(preds)
    if len(batch) != valid_len:
        n = min(len(batch), valid_len)
        batch, preds = batch[:n], preds[:n]

    # Long sequences can fit model inference but OOM while the scale/shift
    # criterion concatenates every dense point map on CUDA.  The criterion has
    # no model dependency, so optionally release those output tensors from the
    # GPU and evaluate the exact same arrays in host memory.
    if os.environ.get("POINT3R_METRIC_CPU", "0") == "1":
        cpu = torch.device("cpu")
        batch = move_batch_to_device(batch, cpu)
        preds = move_tree_to_device(preds, cpu)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pts_all, pts_gt_all, masks_all, images_all = collect_metric_arrays(batch, preds, args)
    save_scene_pointclouds(
        pts_all, pts_gt_all, masks_all, images_all, args, model_name, scene
    )
    metrics = compute_open3d_metrics(pts_all, pts_gt_all, masks_all, images_all, args)
    fps = len(batch) / max(elapsed, 1e-9)

    row = {
        "model": model_name,
        "scene": scene,
        "orig_len": orig_len,
        "frames": len(batch),
        "Acc": metrics["Acc"],
        "Comp": metrics["Comp"],
        "NC1": metrics["NC1"],
        "NC2": metrics["NC2"],
        "NC": (metrics["NC1"] + metrics["NC2"]) / 2.0,
        "Acc_med": metrics["Acc_med"],
        "Comp_med": metrics["Comp_med"],
        "NC1_med": metrics["NC1_med"],
        "NC2_med": metrics["NC2_med"],
        "NC_med": (metrics["NC1_med"] + metrics["NC2_med"]) / 2.0,
        "FPS": fps,
        "Time": elapsed,
        "Peak": peak_gb,
    }
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def fnum(x: float, nd: int = 6) -> str:
    try:
        if np.isnan(float(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_stats(out_dir: Path, rows: list[dict], failures: list[str]) -> None:
    stats = out_dir / "stats_only.log"
    header = [
        "model",
        "scene",
        "orig_len",
        "frames",
        "Acc",
        "Comp",
        "NC1",
        "NC2",
        "NC",
        "FPS",
        "Time",
        "Peak",
        "Acc_med",
        "Comp_med",
        "NC_med",
    ]
    lines = []
    lines.append("========== POINTCLOUD_SCENES ==========")
    lines.append("\t".join(header))
    for row in rows:
        lines.append("\t".join(fnum(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header))

    if failures:
        lines.append("")
        lines.append("========== FAILURES ==========")
        lines.extend(failures)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    lines.append("")
    lines.append("========== SUMMARY ==========")
    lines.append("| Method | n | frames_avg | Acc | Comp | NC1 | NC2 | NC | FPS | Time | Peak | Acc_med | Comp_med | NC_med |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_name, model_rows in sorted(by_model.items()):
        def mean(key: str) -> float:
            vals = np.array([float(r[key]) for r in model_rows], dtype=np.float64)
            return float(np.nanmean(vals)) if len(vals) else float("nan")

        lines.append(
            f"| {model_name} | {len(model_rows)} | {mean('frames'):.1f} | "
            f"{mean('Acc'):.6f} | {mean('Comp'):.6f} | {mean('NC1'):.6f} | "
            f"{mean('NC2'):.6f} | {mean('NC'):.6f} | {mean('FPS'):.4f} | "
            f"{mean('Time'):.4f} | {mean('Peak'):.2f} | {mean('Acc_med'):.6f} | "
            f"{mean('Comp_med'):.6f} | {mean('NC_med'):.6f} |"
        )
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser("Clean NRGBD point-cloud metric")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--output_dir", default="/root/autodl-tmp/clean_launchers/pointcloud/results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--nrgbd_root", default="/root/autodl-tmp/neural_rgbd")
    parser.add_argument("--kf_every", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--center_crop", type=int, default=224)
    parser.add_argument("--icp_thresh", type=float, default=0.1)
    parser.add_argument("--max_points", type=int, default=999999, help="Maximum sampled points for Open3D ICP/NC metric; 0 keeps all points")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--ply_max_points", type=int, default=999999)
    parser.add_argument("--ply_output_dir", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--point3r_repo", default="/root/autodl-tmp/Point3R_mdf")
    parser.add_argument("--point3r_weights", default="/root/autodl-tmp/checkpoints/point3r_512.pth")
    parser.add_argument("--ghost_repo", default="/root/autodl-tmp/GHOST")
    parser.add_argument("--ghost_weights", default="/root/autodl-tmp/checkpoints/checkpoints.pth")
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
    parser.add_argument("--geoanchor_stride", type=int, default=8)
    parser.add_argument("--geoanchor_h", type=int, default=8)
    parser.add_argument("--geoanchor_frames", type=int, default=4)
    parser.add_argument("--geoanchor_min_gap", type=int, default=16)
    parser.add_argument("--geoanchor_buckets_per_kf", type=int, default=32)
    parser.add_argument("--geoanchor_slots_per_kf", type=int, default=32)
    parser.add_argument("--geoanchor_max_kfs", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.tsv"
    log = out_dir / "run.log"
    header = [
        "model",
        "scene",
        "orig_len",
        "frames",
        "Acc",
        "Comp",
        "NC1",
        "NC2",
        "NC",
        "FPS",
        "Time",
        "Peak",
        "Acc_med",
        "Comp_med",
        "NC_med",
    ]
    summary.write_text("\t".join(header) + "\n", encoding="utf-8")
    log.write_text("", encoding="utf-8")

    rows: list[dict] = []
    failures: list[str] = []

    with log.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(
            f"[check] model={args.model} scenes={args.scenes} root={args.nrgbd_root} "
            f"kf_every={args.kf_every} max_frames={args.max_frames} size={args.size}\n"
        )
        print(f"[check] loading model={args.model}", flush=True)
        model = load_model(args, args.model, args.device)
        for scene in args.scenes:
            try:
                row = eval_one_scene(args, args.model, model, scene)
                rows.append(row)
                line = "\t".join(fnum(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header)
                print(f"[pointcloud_scene] {line}", flush=True)
                lf.write(f"[pointcloud_scene] {line}\n")
                with summary.open("a", encoding="utf-8") as sf:
                    sf.write(line + "\n")
            except Exception as exc:
                msg = f"[pointcloud_scene] model={args.model} scene={scene} status=FAIL error={type(exc).__name__}: {exc}"
                print(msg, flush=True)
                traceback.print_exc()
                lf.write(msg + "\n")
                lf.write(traceback.format_exc() + "\n")
                lf.write(traceback.format_exc() + "\n")
                failures.append(msg)
    write_stats(out_dir, rows, failures)
    print(f"wrote {summary}")
    print(f"wrote {out_dir / 'stats_only.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
