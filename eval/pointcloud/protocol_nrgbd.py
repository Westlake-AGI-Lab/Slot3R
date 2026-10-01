"""Shared NeuralRGBD frame-pairing protocol; no model selection side effects."""
from pathlib import Path
import pointcloud_metric_clean as base

_base_load_model = base.load_model
_base_run_model = base.run_model

def build_dataset(args, scene: str):
    """Use only frame IDs having both RGB and depth files."""
    base.add_point3r_paths(args.point3r_repo)
    from eval.mv_recon.data import NRGBD

    resolution = (512, 384) if args.size == 512 else 224
    dataset = NRGBD(
        split="test",
        ROOT=args.nrgbd_root,
        resolution=resolution,
        num_seq=1,
        test_id=scene,
        full_video=True,
        kf_every=args.kf_every,
    )
    image_dir = Path(args.nrgbd_root) / scene / "images"
    depth_dir = Path(args.nrgbd_root) / scene / "depth"
    frame_ids = []
    for image_path in image_dir.glob("img*.png"):
        stem = image_path.stem[3:]
        if stem.isdigit() and (depth_dir / f"depth{stem}.png").is_file():
            frame_ids.append(int(stem))
    frame_ids.sort()
    if not frame_ids:
        raise RuntimeError(f"no paired RGB/depth frames for scene={scene}")
    step = min(args.kf_every, max(len(frame_ids) // 2, 1))
    sampled = frame_ids[::step]
    dataset.tuple_list = [scene + " " + " ".join(map(str, sampled))]
    dataset.scene_list = [scene]
    dataset.num_seq = 1
    print(
        f"[NRGBD_EXISTING_FRAMES] scene={scene} raw={len(frame_ids)} "
        f"kept={len(sampled)} step={step} first={sampled[:3]} last={sampled[-3:]}",
        flush=True,
    )
    return dataset

base.build_dataset = build_dataset
