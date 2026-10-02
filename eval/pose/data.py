"""Pose dataset registry, scene selection, and ground-truth loading."""
from pathlib import Path
import numpy as np
from eval.relpose.metadata import SINTEL_SCENES

def default_scenes() -> list[str]:
    return [f"scene{i:04d}_00" for i in range(707, 807)]

def scannet_files(scannet_root: str, scene: str, stride: int, dataset: str = "scannet") -> tuple[list[str], Path]:
    root = Path(scannet_root) / scene
    color = root / ("rgb_90" if dataset == "tum" else "color_90")
    pose = root / ("groundtruth_90.txt" if dataset == "tum" else "pose_90.txt")
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


DATASETS = {
    "sintel": {"images": "final", "trajectory": "sintel", "rpe_stat": "rmse"},
    "scannet": {"images": "color_90", "trajectory": "replica", "rpe_stat": "mean"},
    "tum": {"images": "rgb_90", "trajectory": "tum", "rpe_stat": "mean"},
}

def scene_names(args):
    if args.scenes:
        return args.scenes
    if args.dataset == "sintel":
        return list(SINTEL_SCENES)
    if args.dataset == "scannet":
        return default_scenes()
    return sorted(p.name for p in Path(args.data_root).iterdir()
                  if p.is_dir() and (p / "groundtruth_90.txt").is_file())


def load_sequence(args, scene):
    from eval.relpose.evo_utils import load_traj
    if args.dataset == "sintel":
        folder = Path(args.data_root) / "final" / scene
        files = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[::args.pose_eval_stride]
        pose = Path(args.data_root) / "camdata_left" / scene
    else:
        files, pose = scannet_files(args.data_root, scene, args.pose_eval_stride, args.dataset)
        load_replica_c2w(pose, stride=args.pose_eval_stride, n=len(files))  # Preserve GT validation.
    if args.max_frames:
        files = files[:args.max_frames]
    gt = load_traj(str(pose), traj_format=DATASETS[args.dataset]["trajectory"],
                   stride=args.pose_eval_stride)
    if args.max_frames:
        gt = (gt[0][:args.max_frames], gt[1][:args.max_frames])
    if len(files) < 3 or len(gt[0]) != len(files):
        raise ValueError(f"Expected at least 3 paired images/poses for {scene}: {len(files)}/{len(gt[0])}")
    return files, gt
