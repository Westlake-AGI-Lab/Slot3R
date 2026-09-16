#!/usr/bin/env python3
"""FFEval-style ScanNet color_90 camera-pose evaluator.

No nested launchers. One dataset protocol, one metric protocol, thin model adapters.

Protocol:
  input  : <scannet_root>/<scene>/color_90/*.jpg and pose_90.txt
  metric : color_90 evo protocol
  output : run.log, summary.tsv

Adapters return camera-to-world matrices [N,4,4]. Metrics match the
ScanNet color_90 relpose table: APE translation RMSE, plus RPE translation
mean and RPE rotation-angle mean.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MODELS = (
    "baseline",
    "kway",
    "sparse512",
    "geoanchor512",
    "sparse640_q25",
    "cut3r",
    "streamvggt",
    "ghost",
    "ttt3r",
)


def add_path(path: str | Path) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def add_repo_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    for p in (repo, repo / "src", repo / "src" / "croco"):
        add_path(p)


def import_symbol(spec: str):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def default_scenes() -> list[str]:
    return [f"scene{i:04d}_00" for i in range(707, 807)]


def scannet_files(scannet_root: str, scene: str, stride: int) -> tuple[list[str], Path]:
    root = Path(scannet_root) / scene
    color = root / "color_90"
    pose = root / "pose_90.txt"
    if not color.is_dir():
        raise FileNotFoundError(color)
    if not pose.is_file():
        raise FileNotFoundError(pose)
    files = sorted(
        str(p)
        for p in color.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[::stride]
    if not files:
        raise RuntimeError(f"no image files in {color}")
    return files, pose


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(x*x + y*y + z*z + w*w)
    n[n == 0] = 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R

def load_replica_c2w(path: str | Path, stride: int = 1, n: int | None = None) -> np.ndarray:
    raw = Path(path).read_text(errors="ignore")
    if re_search_invalid_pose(raw):
        raise ValueError(f"invalid_gt_pose: {path}")
    arr = np.loadtxt(path).astype(np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[1] == 16:
        poses = arr.reshape(-1, 4, 4)
    elif arr.shape[1] == 12:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], arr.shape[0], axis=0)
        poses[:, :3, :4] = arr.reshape(-1, 3, 4)
    elif arr.shape[1] == 8:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], arr.shape[0], axis=0)
        poses[:, :3, 3] = arr[:, 1:4]
        poses[:, :3, :3] = _quat_xyzw_to_matrix(arr[:, 4:8])
    else:
        raise ValueError(f"bad pose_90 shape {arr.shape} in {path}")
    poses = poses[::stride]
    if n is not None:
        poses = poses[:n]
    return poses


def re_search_invalid_pose(text: str) -> bool:
    import re

    return re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", text, re.I) is not None


def c2w_to_tum(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    c2w = np.asarray(c2w, np.float64)
    xyz = c2w[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    ts = np.arange(len(c2w), dtype=float)
    return np.concatenate([xyz, quat_wxyz], axis=1), ts


def color90_evo_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    """Match the ScanNet color_90 relpose table parser.

    The table used ATE rmse, but RPE translation/rotation mean from the evo text block.
    """
    from copy import deepcopy

    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    pred_tum, pred_ts = c2w_to_tum(pred_c2w)
    gt_tum, gt_ts = c2w_to_tum(gt_c2w)
    traj_est = PoseTrajectory3D(
        positions_xyz=pred_tum[:, :3],
        orientations_quat_wxyz=pred_tum[:, 3:],
        timestamps=pred_ts,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_tum[:, :3],
        orientations_quat_wxyz=gt_tum[:, 3:],
        timestamps=gt_ts,
    )
    if len(traj_est.timestamps) == len(traj_ref.timestamps):
        traj_est.timestamps = traj_ref.timestamps
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    rpe_rot_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_t_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    return {
        "ATE_RMSE": float(ate_result.stats["rmse"]),
        "ATE_mean": float(ate_result.stats["mean"]),
        "RPE_t": float(rpe_t_result.stats["mean"]),
        "RPE_t_RMSE": float(rpe_t_result.stats["rmse"]),
        "RPE_rot": float(rpe_rot_result.stats["mean"]),
        "RPE_rot_RMSE": float(rpe_rot_result.stats["rmse"]),
    }


def parse_color90_metric_file(metric_path: str | Path) -> dict[str, float]:
    txt = Path(metric_path).read_text(errors="ignore")

    def block(title: str) -> str:
        m = re.search(title + r".*?(?=\n[A-Z][A-Za-z ]+ w\.r\.t\.|\Z)", txt, re.S)
        return m.group(0) if m else ""

    def val(text: str, key: str) -> float:
        m = re.search(rf"^\s*{key}\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else float("nan")

    ape = block(r"APE w\.r\.t\. translation part")
    rper = block(r"RPE w\.r\.t\. rotation angle")
    rpet = block(r"RPE w\.r\.t\. translation part")
    return {
        "ATE_RMSE": val(ape, "rmse"),
        "ATE_mean": val(ape, "mean"),
        "RPE_t": val(rpet, "mean"),
        "RPE_t_RMSE": val(rpet, "rmse"),
        "RPE_rot": val(rper, "mean"),
        "RPE_rot_RMSE": val(rper, "rmse"),
    }


def build_dust3r_views(files: list[str], size: int, crop: bool, device: str):
    import torch
    from dust3r.utils.image import load_images_for_eval

    with contextlib.redirect_stdout(io.StringIO()):
        images = load_images_for_eval(files, size=size, crop=crop)
    views = []
    for i, im in enumerate(images):
        img = im["img"].to(device)
        views.append(
            {
                "img": img,
                "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan, device=device),
                "true_shape": torch.from_numpy(im["true_shape"]).to(device),
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


def point3r_child_env(method: str, repo: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    keys = (
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
    )
    for key in keys:
        env.pop(key, None)

    repo = Path(repo)
    env["PYTHONPATH"] = f"{repo}:{repo / 'src' / 'croco'}:{repo / 'src'}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env["OPENBLAS_NUM_THREADS"] = "8"
    env["NUMEXPR_NUM_THREADS"] = "8"

    if method == "baseline":
        return env

    env.update(
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
    if method in ("sparse512", "geoanchor512"):
        env.update(
            {
                "POINT3R_SPARSE_READOUT": "1",
                "POINT3R_SPARSE_MODE": "max",
                "POINT3R_SPARSE_MAX_TOKENS": "512",
                "POINT3R_SPARSE_GLOBAL_ANCHORS": "128",
                "POINT3R_SPARSE_NEIGHBOR_RANGE": "1",
            }
        )
    if method == "geoanchor512":
        env.update(
            {
                "POINT3R_GEOANCHOR": "1",
                "POINT3R_GEOANCHOR_STRIDE": "8",
                "POINT3R_GEOANCHOR_H": "8",
                "POINT3R_GEOANCHOR_FRAMES": "4",
                "POINT3R_GEOANCHOR_MIN_GAP": "16",
                "POINT3R_GEOANCHOR_BUCKETS_PER_KF": "32",
                "POINT3R_GEOANCHOR_SLOTS_PER_KF": "32",
                "POINT3R_GEOANCHOR_MAX_KFS": "2048",
            }
        )
    return env


class Point3RBKSAdapter:
    """Run the self-contained Point3R BKS-style backend per scene.

    This keeps the clean evaluator as the single user-facing entrypoint while
    keeping the Point3R family on the same BKS-style protocol.
    """

    def __init__(self, args, method: str):
        self.args = args
        self.method = method
        self.repo = Path(args.point3r_repo)
        self.weights = args.point3r_weights
        self.device = args.device
        self.size = args.size
        self.output_dir = Path(args.output_dir)
        self.tmp_root = self.output_dir / "_point3r_bks_tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.backend = Path(args.point3r_bks_backend)
        if not self.backend.is_file():
            raise FileNotFoundError(f"Point3R BKS backend not found: {self.backend}")

    def _prepare_one_scene_symlink(self, scene: str) -> None:
        src = Path(self.args.scannet_root) / scene
        one_eval = Path("/root/autodl-tmp/scannetv2_one_eval")
        disk_eval_parent = Path("/mnt/disk5/data/eval")
        disk_eval = disk_eval_parent / "scannet_eval"

        if one_eval.exists() or one_eval.is_symlink():
            shutil.rmtree(one_eval, ignore_errors=True)
        one_eval.mkdir(parents=True, exist_ok=True)
        link = one_eval / scene
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)

        disk_eval_parent.mkdir(parents=True, exist_ok=True)
        if disk_eval.exists() or disk_eval.is_symlink():
            if disk_eval.is_symlink() or disk_eval.is_file():
                disk_eval.unlink()
            else:
                shutil.rmtree(disk_eval)
        disk_eval.symlink_to(one_eval)

    def eval_scene(self, scene: str) -> dict:
        files, pose_file = scannet_files(self.args.scannet_root, scene, self.args.pose_eval_stride)
        _ = load_replica_c2w(pose_file, stride=self.args.pose_eval_stride, n=len(files))
        scene_out = self.tmp_root / f"{scene}_{self.method}"
        shutil.rmtree(scene_out, ignore_errors=True)
        scene_out.mkdir(parents=True, exist_ok=True)
        raw_log = scene_out / "raw.log"
        cmd = [
            sys.executable,
            "-B",
            str(self.backend),
            "--method",
            self.method,
            "--scene",
            scene,
            "--scannet_root",
            str(self.args.scannet_root),
            "--repo",
            str(self.repo),
            "--weights",
            self.weights,
            "--device",
            self.device,
            "--output_dir",
            str(scene_out),
            "--size",
            str(self.size),
            "--pose_eval_stride",
            str(self.args.pose_eval_stride),
        ]
        if self.args.no_crop:
            cmd.append("--no_crop")
        env = point3r_child_env(self.method, self.repo)
        with raw_log.open("wb") as f:
            status = subprocess.run(cmd, cwd=str(self.repo), env=env, stdout=f, stderr=subprocess.STDOUT).returncode

        metric = scene_out / f"{scene}_eval_metric.txt"
        if not metric.is_file():
            tail = raw_log.read_text(errors="ignore").splitlines()[-40:]
            shutil.rmtree(scene_out, ignore_errors=True)
            raise RuntimeError(
                f"NO_METRIC_FILE exit={status} backend={self.backend} "
                f"tail={' | '.join(tail)}"
            )

        metrics = parse_color90_metric_file(metric)
        row = {
            "model": self.method,
            "scene": scene,
            "n": len(files),
            "ATE_RMSE": float(metrics["ATE_RMSE"]),
            "ATE_mean": float(metrics["ATE_mean"]),
            "RPE_t": float(metrics["RPE_t"]),
            "RPE_rot": float(metrics["RPE_rot"]),
            "FPS": float("nan"),
        }
        shutil.rmtree(scene_out, ignore_errors=True)
        return row


class Cut3RAdapter:
    def __init__(self, repo: str, weights: str, device: str, size: int, crop: bool):
        add_repo_paths(repo)
        import torch
        from dust3r.model import ARCroco3DStereo

        self.torch = torch
        self.repo = repo
        self.device = device
        self.size = size
        self.crop = crop
        self.model = ARCroco3DStereo.from_pretrained(weights).to(device).eval()

    def predict_c2w(self, files: list[str]) -> np.ndarray:
        from dust3r.inference import inference
        from dust3r.utils.camera import pose_encoding_to_camera

        views = build_dust3r_views(files, self.size, self.crop, self.device)
        with self.torch.no_grad():
            out = inference(views, self.model, self.device)
        if isinstance(out, tuple):
            out = out[0]
        preds = out["pred"] if isinstance(out, dict) else out.pred
        c2w = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).detach().cpu().numpy()[0]
            for pred in preds
        ]
        return np.stack(c2w, axis=0)


def build_ttt3r_views(files: list[str], size: int, crop: bool):
    import torch
    from src.dust3r.utils import image as image_utils

    with contextlib.redirect_stdout(io.StringIO()):
        if hasattr(image_utils, "load_images_for_eval"):
            images = image_utils.load_images_for_eval(files, size=size, crop=crop)
        else:
            images = image_utils.load_images(files, size=size)
    views = []
    for i, im in enumerate(images):
        img = im["img"]
        views.append(
            {
                "img": img,
                "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan),
                "true_shape": torch.from_numpy(im["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views


class TTT3RAdapter:
    def __init__(self, repo: str, weights: str, device: str, size: int, crop: bool):
        repo = Path(repo).resolve()
        for p in (repo, repo / "src", repo / "src" / "croco", Path(weights).resolve().parent):
            add_path(p)
        import torch
        from src.dust3r.model import ARCroco3DStereo

        self.torch = torch
        self.device = device
        self.size = size
        self.crop = crop
        self.model = ARCroco3DStereo.from_pretrained(weights).to(device).eval()
        self.model.config.model_update_type = "ttt3r"

    def predict_c2w(self, files: list[str]) -> np.ndarray:
        from src.dust3r.inference import inference_recurrent_lighter
        from src.dust3r.utils.camera import pose_encoding_to_camera

        views = build_ttt3r_views(files, self.size, self.crop)
        with self.torch.no_grad():
            outputs, _ = inference_recurrent_lighter(views, self.model, self.device)
        preds = outputs["pred"]
        c2w = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).detach().cpu().numpy()[0]
            for pred in preds
        ]
        return np.stack(c2w, axis=0)


class VGGTStyleAdapter:
    """Adapter for StreamVGGT-like APIs: load images -> model.inference(frames)."""

    def __init__(
        self,
        repo: str,
        weights: str,
        device: str,
        model_class: str,
        image_loader: str,
        pose_decoder: str,
        model_kwargs: str = "",
    ):
        add_repo_paths(repo)
        import torch

        self.torch = torch
        self.device = device
        self.load_images = import_symbol(image_loader)
        self.pose_decode = import_symbol(pose_decoder)
        cls = import_symbol(model_class)
        kwargs = {}
        if model_kwargs:
            for item in model_kwargs.split(","):
                k, v = item.split("=", 1)
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                kwargs[k] = v
        self.model = cls(**kwargs).to(device).eval()
        ckpt = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)

    def predict_c2w(self, files: list[str]) -> np.ndarray:
        images = self.load_images(files).to(self.device)
        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
        reset = getattr(getattr(self.model, "aggregator", None), "reset_kv_repository", None)
        if reset is not None:
            reset()
        with self.torch.no_grad():
            dtype = self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32
            with self.torch.cuda.amp.autocast(enabled=self.torch.cuda.is_available(), dtype=dtype):
                try:
                    out = self.model.inference(frames, frame_writer=None, cache_results=True)
                except TypeError:
                    out = self.model.inference(frames)
        if getattr(out, "ress", None) is None:
            raise RuntimeError("model.inference returned no ress; set cache_results=True or use a result writer")
        pose_enc = self.torch.stack([r["camera_pose"].squeeze(0) for r in out.ress], dim=0)
        extri, _ = self.pose_decode(
            pose_enc.unsqueeze(0) if pose_enc.ndim == 2 else pose_enc,
            images.shape[-2:],
        )
        extri = extri.squeeze(0)
        row = self.torch.tensor([0, 0, 0, 1], device=extri.device, dtype=extri.dtype).expand(extri.shape[0], 1, 4)
        c2w = self.torch.cat([extri, row], dim=1)
        if reset is not None:
            reset()
        return c2w.detach().cpu().numpy()


def make_adapter(args, model_name: str):
    if model_name in ("baseline", "kway", "sparse512", "geoanchor512", "sparse640_q25"):
        adapter = Point3RBKSAdapter(args, model_name)
        adapter.display_name = model_name
        return adapter

    if model_name == "cut3r":
        return Cut3RAdapter(args.cut3r_repo, args.cut3r_weights, args.device, args.size, not args.no_crop)

    if model_name == "streamvggt":
        return VGGTStyleAdapter(
            args.streamvggt_repo,
            args.streamvggt_weights,
            args.device,
            args.streamvggt_class,
            args.streamvggt_image_loader,
            args.streamvggt_pose_decoder,
            args.streamvggt_model_kwargs,
        )

    if model_name == "ghost":
        return VGGTStyleAdapter(
            args.ghost_repo,
            args.ghost_weights,
            args.device,
            args.ghost_class,
            args.ghost_image_loader,
            args.ghost_pose_decoder,
            args.ghost_model_kwargs,
        )

    if model_name == "ttt3r":
        return TTT3RAdapter(args.ttt3r_repo, args.ttt3r_weights, args.device, args.size, not args.no_crop)

    raise ValueError(model_name)


def eval_one_scene(args, adapter, model_name: str, scene: str) -> dict:
    if hasattr(adapter, "eval_scene"):
        return adapter.eval_scene(scene)

    files, pose_file = scannet_files(args.scannet_root, scene, args.pose_eval_stride)
    gt_c2w = load_replica_c2w(pose_file, stride=args.pose_eval_stride, n=len(files))
    t0 = time.time()
    pred_c2w = adapter.predict_c2w(files)
    if len(pred_c2w) != len(gt_c2w):
        n = min(len(pred_c2w), len(gt_c2w))
        pred_c2w, gt_c2w, files = pred_c2w[:n], gt_c2w[:n], files[:n]
    m = color90_evo_pose_metrics(pred_c2w, gt_c2w)
    fps = len(files) / max(time.time() - t0, 1e-9)
    return {
        "model": model_name,
        "scene": scene,
        "n": len(files),
        "ATE_RMSE": float(m["ATE_RMSE"]),
        "ATE_mean": float(m["ATE_mean"]),
        "RPE_t": float(m["RPE_t"]),
        "RPE_rot": float(m["RPE_rot"]),
        "FPS": fps,
    }


def fmt(x, nd=6):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_stats(stats_path: Path, rows: list[dict], failures: list[str], header: list[str]) -> None:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    lines = []
    lines.append("========== POSE_SCENES ==========")
    lines.append("\t".join(header))
    for row in rows:
        lines.append("\t".join(fmt(row[k], 4 if k == "FPS" else 6) for k in header))
    if failures:
        lines.append("")
        lines.append("========== FAILURES ==========")
        lines.extend(failures)
    lines.append("")
    lines.append("========== SUMMARY ==========")
    lines.append("| Method | n | ATE_RMSE | ATE_mean | RPE_t mean | RPE_rot mean | FPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, model_rows in sorted(by_model.items()):
        def mean(key: str) -> float:
            return float(sum(float(r[key]) for r in model_rows) / len(model_rows))

        lines.append(
            f"| {model} | {len(model_rows)} | {mean('ATE_RMSE'):.6f} | {mean('ATE_mean'):.6f} | "
            f"{mean('RPE_t'):.6f} | {mean('RPE_rot'):.6f} | {mean('FPS'):.4f} |"
        )
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ffeval_root", default="/root/autodl-tmp/FeedForward_Eval")
    p.add_argument("--scannet_root", default="/root/autodl-tmp/scannetv2")
    p.add_argument("--output_dir", default="/root/autodl-tmp/results/ffeval_pose_scannet_color90")
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--models", nargs="+", choices=MODELS, default=None)
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no_crop", action="store_true")
    p.add_argument("--pose_eval_stride", type=int, default=1)

    p.add_argument("--point3r_repo", default="/root/autodl-tmp/Point3R_mdf")
    p.add_argument("--point3r_weights", default="/root/autodl-tmp/checkpoints/point3r_512.pth")
    p.add_argument(
        "--point3r_bks_backend",
        default="/root/autodl-tmp/clean_launchers/pose/point3r_bks_sparse640_q25.py",
    )

    p.add_argument("--cut3r_repo", default="/root/autodl-tmp/CUT3R")
    p.add_argument("--cut3r_weights", default="/root/autodl-tmp/checkpoints/cut3r_512_dpt_4_64.pth")

    p.add_argument("--streamvggt_repo", default="/root/autodl-tmp/StreamVGGT")
    p.add_argument("--streamvggt_weights", default="/root/autodl-tmp/checkpoints/checkpoints.pth")
    p.add_argument("--streamvggt_class", default="streamvggt.models.streamvggt:StreamVGGT")
    p.add_argument("--streamvggt_image_loader", default="streamvggt.utils.load_fn:load_and_preprocess_images")
    p.add_argument("--streamvggt_pose_decoder", default="streamvggt.utils.pose_enc:pose_encoding_to_extri_intri")
    p.add_argument("--streamvggt_model_kwargs", default="")

    p.add_argument("--ghost_repo", default="/root/autodl-tmp/GHOST")
    p.add_argument("--ghost_weights", default="/root/autodl-tmp/checkpoints/checkpoints.pth")
    p.add_argument("--ghost_class", default="streamvggt.models.streamvggt:StreamVGGT")
    p.add_argument("--ghost_image_loader", default="streamvggt.utils.load_fn:load_and_preprocess_images")
    p.add_argument("--ghost_pose_decoder", default="streamvggt.utils.pose_enc:pose_encoding_to_extri_intri")
    p.add_argument("--ghost_model_kwargs", default="total_budget=1200000")

    p.add_argument("--ttt3r_repo", default="/root/autodl-tmp/TTT3R")
    p.add_argument("--ttt3r_weights", default="/root/autodl-tmp/checkpoints/cut3r_512_dpt_4_64.pth")

    args = p.parse_args()
    add_path(args.ffeval_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.tsv"
    log = out / "run.log"
    stats = out / "stats_only.log"
    scenes = args.scenes or default_scenes()
    models = args.models if args.models is not None else ([args.model] if args.model else None)
    if not models:
        p.error("pass --model NAME for one model, or --models NAME ... for an explicit batch")

    header = ["model", "scene", "n", "ATE_RMSE", "ATE_mean", "RPE_t", "RPE_rot", "FPS"]
    summary.write_text("\t".join(header) + "\n")
    log.write_text("")
    stats.write_text("")
    rows: list[dict] = []
    failures: list[str] = []

    with log.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(
            f"[check] ffeval_root={args.ffeval_root} scannet_root={args.scannet_root} "
            f"size={args.size} crop={not args.no_crop} stride={args.pose_eval_stride}\n"
        )
        for model_name in models:
            lf.write(f"[check] loading model={model_name}\n")
            print(f"[check] loading model={model_name}", flush=True)
            adapter = make_adapter(args, model_name)
            for scene in scenes:
                try:
                    row = eval_one_scene(args, adapter, model_name, scene)
                    line = "\t".join(fmt(row[k], 4 if k == "FPS" else 6) for k in header)
                    print(f"[pose_scene] {line}", flush=True)
                    lf.write(f"[pose_scene] {line}\n")
                    rows.append(row)
                    with summary.open("a", encoding="utf-8") as sf:
                        sf.write(line + "\n")
                except Exception as e:
                    msg = f"[pose_scene] model={model_name} scene={scene} status=FAIL error={type(e).__name__}: {e}"
                    print(msg, flush=True)
                    lf.write(msg + "\n")
                    failures.append(msg)
    write_stats(stats, rows, failures, header)
    print(f"wrote {summary}")
    print(f"wrote {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
