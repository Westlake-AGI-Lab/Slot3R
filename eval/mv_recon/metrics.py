"""Shared point-cloud alignment, ICP, accuracy, completion and normal metrics."""
from __future__ import annotations

import numpy as np
import torch
from .model_registry import force_point3r_metric_imports, move_tree_to_device


def center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return arr
    h, w = arr.shape[:2]
    if h < crop or w < crop:
        return arr
    cy, cx = h // 2, w // 2
    r = crop // 2
    return arr[cy - r : cy + r, cx - r : cx + r]


def subsample_correspondences(pred, gt, color, max_points: int, seed: int):
    """Keep the same RGB/depth correspondences as the original table protocol."""
    if not len(pred) == len(gt) == len(color):
        raise ValueError("point-cloud sampling requires paired prediction/GT/color arrays")
    if max_points <= 0 or len(pred) <= max_points:
        return pred, gt, color
    indices = np.random.default_rng(seed).choice(len(pred), max_points, replace=False)
    return pred[indices], gt[indices], color[indices]


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

    pred, gt, color = subsample_correspondences(
        pred, gt, color, args.max_points, args.seed
    )

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
