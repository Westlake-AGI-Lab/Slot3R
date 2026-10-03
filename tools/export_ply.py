#!/usr/bin/env python3
"""Reconstruct an ordered RGB folder without ground truth and export colored PLY."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "src/croco"):
    sys.path.insert(0, str(path))

from eval.model_config import MODEL_MODULES, add_model_arguments, configure_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--kf_every", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=200, help="After subsampling; 0 means all")
    parser.add_argument("--conf_quantile", type=float, default=0.20,
                        help="Discard this bottom fraction of finite output confidence values")
    parser.add_argument("--max_save_points", type=int, default=3000000, help="0 means no cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--frustum_count", type=int, default=12)
    parser.add_argument("--frustum_scale", type=float, default=0.08,
                        help="Frustum depth in the model's coordinate units")
    args = parser.parse_args(argv)
    if args.kf_every < 1 or args.max_frames < 0 or args.max_save_points < 0:
        parser.error("kf_every must be positive; max_frames and max_save_points must be nonnegative")
    if not 0 <= args.conf_quantile < 1 or not 0 <= args.drop_quantile < 1:
        parser.error("confidence quantiles must be in [0, 1)")
    if args.frustum_count < 1 or not 0 < args.frustum_scale < float("inf"):
        parser.error("frustum_count and frustum_scale must be positive and finite")
    return args


def select_frames(directory, stride, limit):
    def key(path):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]
    files = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}), key=key)
    files = files[::stride]
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError(f"No RGB images selected from {directory}")
    return files


def prepare_output(directory):
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}; choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)


def collect_cloud(views, predictions):
    import numpy as np
    if not predictions or len(views) != len(predictions):
        raise ValueError("Expected one prediction per input view")
    points, colors, confidence = [], [], []
    for view, pred in zip(views, predictions):
        xyz = pred["pts3d_in_other_view"][0].detach().float().cpu().numpy()
        conf = pred["conf"][0].detach().float().cpu().numpy()
        rgb = view["img"][0].detach().float().cpu().permute(1, 2, 0).numpy()
        if xyz.shape != rgb.shape or conf.shape != xyz.shape[:2]:
            raise ValueError(f"Point/RGB/confidence shape mismatch: {xyz.shape}, {rgb.shape}, {conf.shape}")
        points.append(xyz.reshape(-1, 3))
        colors.append(np.clip((rgb.reshape(-1, 3) + 1) * 0.5, 0, 1))
        confidence.append(conf.reshape(-1))
    return np.concatenate(points), np.concatenate(colors), np.concatenate(confidence)


def filter_cloud(points, colors, confidence, quantile, cap, seed):
    import numpy as np
    finite = np.isfinite(points).all(1) & np.isfinite(colors).all(1) & np.isfinite(confidence)
    points, colors, confidence = points[finite], colors[finite], confidence[finite]
    if not len(points):
        raise ValueError("Model produced no finite colored points")
    threshold = float(np.quantile(confidence, quantile))
    keep = confidence >= threshold
    points, colors = points[keep], colors[keep]
    if cap and len(points) > cap:
        indices = np.sort(np.random.default_rng(seed).choice(len(points), cap, replace=False))
        points, colors = points[indices], colors[indices]
    return points, colors, threshold


def write_cloud(path, points, colors):
    """Binary little-endian XYZ + RGB, readable in MeshLab/Open3D/CloudCompare."""
    import numpy as np
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for i, field in enumerate(("x", "y", "z")):
        data[field] = points[:, i]
    for i, field in enumerate(("red", "green", "blue")):
        data[field] = rgb[:, i]
    with path.open("wb") as out:
        out.write(("ply\nformat binary_little_endian 1.0\n"
                   f"element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\n"
                   "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        data.tofile(out)


def write_edges(path, vertices, colors, edges):
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element edge {len(edges)}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
        for xyz, rgb in zip(vertices, colors):
            out.write(" ".join(f"{v:.9g}" for v in xyz) + " " + " ".join(str(int(v)) for v in rgb) + "\n")
        for a, b in edges:
            out.write(f"{a} {b}\n")


def export_cameras(directory, poses, count, scale):
    import numpy as np
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Expected finite camera-to-world matrices with shape (N, 4, 4)")
    np.save(directory / "c2w.npy", poses)
    centers = poses[:, :3, 3]
    anchors = np.array([[33, 150, 243], [0, 188, 190], [255, 193, 7], [235, 63, 52]])
    t = np.linspace(0, 3, len(poses))
    colors = np.rint(np.column_stack([np.interp(t, np.arange(4), anchors[:, c]) for c in range(3)])).astype(int)
    write_edges(directory / "trajectory.ply", centers, colors, [(i, i+1) for i in range(len(poses)-1)])
    np.savetxt(directory / "camera_centers.csv", np.column_stack([np.arange(len(poses)), centers]),
               delimiter=",", header="frame_index,x,y,z", comments="", fmt=["%d", "%.9g", "%.9g", "%.9g"])
    vertices, vertex_colors, edges = [], [], []
    corners = np.array([[-.75, -.5, 1], [.75, -.5, 1], [.75, .5, 1], [-.75, .5, 1]]) * scale
    for i in np.linspace(0, len(poses)-1, min(count, len(poses)), dtype=int):
        base = len(vertices)
        vertices.extend([centers[i], *(corners @ poses[i, :3, :3].T + centers[i])])
        vertex_colors.extend([colors[i]] * 5)
        edges.extend((base, base+j) for j in range(1, 5))
        edges.extend((base+j, base+(j % 4)+1) for j in range(1, 5))
    write_edges(directory / "camera_frustums.ply", vertices, vertex_colors, edges)
    project = ET.Element("MeshLabProject")
    group = ET.SubElement(project, "MeshGroup")
    for name in ("cloud.ply", "trajectory.ply", "camera_frustums.ply"):
        layer = ET.SubElement(group, "MLMesh", label=name, filename=name)
        ET.SubElement(layer, "MLMatrix44").text = "\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    ET.indent(project)
    (directory / "scene.mlp").write_text('<!DOCTYPE MeshLabDocument>\n' + ET.tostring(project, encoding="unicode"), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    files = select_frames(args.image_dir, args.kf_every, args.max_frames)
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    import numpy as np
    import torch
    from dust3r.utils.image import load_images

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA environment or explicitly select --device cpu")
    prepare_output(args.output_dir)
    configure_model(args)
    torch.manual_seed(args.seed)
    module = importlib.import_module(MODEL_MODULES[args.model])
    model = module.Point3R.from_pretrained(args.weights).to(device).eval()
    views = load_images([str(p) for p in files], size=args.size, verbose=False)
    if len(views) != len(files):
        raise ValueError("Image loader skipped selected input frames")
    for view in views:
        _, _, h, w = view["img"].shape
        view["true_shape"] = torch.as_tensor(view["true_shape"])
        view.update(img_mask=torch.ones(1, dtype=torch.bool), ray_mask=torch.zeros(1, dtype=torch.bool),
                    ray_map=torch.full((1, 6, h, w), torch.nan), update=torch.ones(1, dtype=torch.bool),
                    reset=torch.zeros(1, dtype=torch.bool))
        for key, value in list(view.items()):
            if isinstance(value, torch.Tensor) and key != "true_shape":
                view[key] = value.to(device)
    print(f"[input] model={args.model} frames={len(files)} first={files[0].name} last={files[-1].name}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(views, point3r_tag=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if len(output.ress) != len(files):
        raise ValueError("Inference did not return one prediction per selected frame")
    points, colors, confidence = collect_cloud(output.views, output.ress)
    points, colors, threshold = filter_cloud(points, colors, confidence, args.conf_quantile,
                                             args.max_save_points, args.seed)
    poses = None
    if args.save_trajectory:
        from dust3r.utils.camera import pose_encoding_to_camera
        poses = np.concatenate([pose_encoding_to_camera(p["camera_pose"].detach().clone()).float().cpu().numpy()
                                for p in output.ress], axis=0)
        export_cameras(args.output_dir, poses, args.frustum_count, args.frustum_scale)
    write_cloud(args.output_dir / "cloud.ply", points, colors)
    (args.output_dir / "frames.txt").write_text("".join(str(p.resolve()) + "\n" for p in files), encoding="utf-8")
    stats = dict(model=args.model, frames=len(files), kf_every=args.kf_every,
                 saved_points=len(points), conf_quantile=args.conf_quantile, conf_threshold=threshold,
                 elapsed_sec=elapsed, fps=len(files)/elapsed,
                 peak_cuda_gb=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0,
                 trajectory_frames=len(poses) if poses is not None else 0,
                 model_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 model_environment={k: v for k, v in os.environ.items() if k.startswith("POINT3R_")})
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {args.output_dir / 'cloud.ply'} ({len(points):,} points)", flush=True)


if __name__ == "__main__":
    main()
