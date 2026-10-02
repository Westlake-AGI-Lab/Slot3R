"""Depth alignment and scoring; preserve indoor and KITTI protocols."""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from eval.video_depth.tools import depth_evaluation

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

def sequence_metrics(predictions, depth_paths, align, use_gpu):
    gt = []
    for path in depth_paths:
        raw = np.array(Image.open(path), dtype=int)
        if raw.max() <= 255:
            raise ValueError(f"Expected 16-bit KITTI depth: {path}")
        depth = raw.astype(float) / 256.0
        depth[raw == 0] = -1.0
        gt.append(depth)
    gt = np.stack(gt)
    if len(predictions) != len(gt):
        raise ValueError("Prediction/ground-truth frame count mismatch")
    pred = np.stack([cv2.resize(value, (gt.shape[2], gt.shape[1]),
                               interpolation=cv2.INTER_CUBIC) for value in predictions])
    flags = {"scale_shift": {"align_with_lad2": True},
             "scale": {"align_with_scale": True}, "metric": {"metric_scale": True}}
    # Same helper, resize, alignment and valid-pixel weighting as the original evaluator.
    result, _, _, _ = depth_evaluation(pred, gt, max_depth=None, use_gpu=use_gpu, **flags[align])
    if result["valid_pixels"] <= 0:
        raise ValueError("No valid ground-truth depth pixels")
    return result
