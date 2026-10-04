"""Prepared depth datasets and their frame-loading protocols."""
from pathlib import Path
import re
import cv2
import numpy as np
import torch

BONN_SCENES = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")
BONN_INTRINSICS = np.array(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

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

def load_bonn_scene(args, scene: str):
    # Accept both the short protocol name (``balloon2``) and the on-disk
    # directory name (``rgbd_bonn_balloon2``).  This makes SCENES usable with
    # names copied directly from the dataset directory without duplicating the
    # prefix.
    scene_dir = scene if scene.startswith("rgbd_bonn_") else f"rgbd_bonn_{scene}"
    root = Path(args.data_root) / scene_dir
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

def load_scannet_scene(args, scene: str):
    root = Path(args.data_root) / scene
    rgb_dir = root / "color_90"
    depth_dir = root / "depth_90"
    pose_path = root / "pose_90.txt"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not depth_dir.is_dir():
        raise FileNotFoundError(depth_dir)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    rgb_files = sorted((p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")), key=lambda p: p.stem)
    depth_by_stem = {p.stem: p for p in depth_dir.iterdir() if p.suffix.lower() == ".png"}
    rgb_files = [p for p in rgb_files if p.stem in depth_by_stem]
    depth_files = [depth_by_stem[p.stem] for p in rgb_files]
    pose_raw = np.loadtxt(pose_path, dtype=np.float32)
    if pose_raw.ndim == 1:
        pose_raw = pose_raw[None]
    if pose_raw.shape[1] != 16:
        raise ValueError(f"bad pose_90 shape {pose_raw.shape} in {pose_path}")
    poses = pose_raw.reshape(-1, 4, 4)
    intrinsics = np.loadtxt(root / "intrinsic" / "intrinsic_color.txt", dtype=np.float32)[:3, :3]
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
        rgb, depth, intrinsics_resized = resize_like_monst3r_no_crop(rgb, depth, intrinsics, args.size)
        frames.append(
            {
                "rgb": rgb,
                "depth": depth.astype(np.float32),
                "pose": poses[idx].astype(np.float32),
                "intrinsics": intrinsics_resized.astype(np.float32),
                "rgb_path": str(rgb_files[idx]),
                "depth_path": str(depth_files[idx]),
            }
        )
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError(f"no frames for scene={scene}")
    return frames

def scene_pairs(root, scene):
    """Pair the prepared KITTI RGB and GT files by name, never by list position."""
    root = Path(root)
    def frame_key(path):
        return re.sub(r"_(?:image|groundtruth_depth)_(?=\d+_image_\d+\.png$)", "_", path.name)
    def indexed(folder):
        paths = list(folder.glob("*.png"))
        entries = {frame_key(p): p for p in paths}
        if len(entries) != len(paths):
            raise ValueError(f"Duplicate KITTI frame IDs: {folder}")
        return entries
    images = indexed(root / "image_gathered" / scene)
    depths = indexed(root / "groundtruth_depth_gathered" / scene)
    if not images or images.keys() != depths.keys():
        raise ValueError(f"Missing or unmatched KITTI RGB/GT frames for {scene}")
    names = sorted(images)
    return [str(images[n]) for n in names], [depths[n] for n in names]


DATASETS = {
    "bonn": {"loader": load_bonn_scene, "depth_scale": 5000., "align": "sequence_scale_shift"},
    "scannet": {"loader": load_scannet_scene, "depth_scale": 1000., "align": "sequence_scale_shift"},
    "kitti": {"depth_scale": 256., "align": "scale_shift"},
}

def scene_names(args):
    if args.scenes:
        return args.scenes
    root = Path(args.data_root)
    if args.dataset == "bonn":
        return [s for s in BONN_SCENES if (root / f"rgbd_bonn_{s}" / "rgb_110").is_dir()]
    if args.dataset == "scannet":
        return [p.name for p in sorted(root.glob("scene*"))
                if (p / "color_90").is_dir() and (p / "depth_90").is_dir()]
    return sorted(p.name for p in (root / "image_gathered").iterdir() if p.is_dir())


def load_sequence(args, scene):
    if args.dataset == "kitti":
        from eval.pose.inference import prepare_input
        files, depths = scene_pairs(args.data_root, scene)
        files, depths = files[::args.kf_every], depths[::args.kf_every]
        if args.max_frames:
            files, depths = files[:args.max_frames], depths[:args.max_frames]
        return prepare_input(files, [True]*len(files), size=args.size, crop=False), depths
    return frames_to_batch(DATASETS[args.dataset]["loader"](args, scene)), None
