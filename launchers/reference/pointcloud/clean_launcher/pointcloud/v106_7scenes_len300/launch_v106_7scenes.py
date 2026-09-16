#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_v106_nrgbd as v106


def build_dataset(args, scene_spec: str):
    v106.base.add_point3r_paths(args.point3r_repo)
    from eval.mv_recon.data import SevenScenes
    scene, seq_id = scene_spec.split("/", 1)
    resolution = (512, 384) if args.size == 512 else 224
    dataset = SevenScenes(split="test", ROOT=args.nrgbd_root, resolution=resolution,
                          num_seq=1, test_id=scene, seq_id=seq_id,
                          full_video=True, kf_every=args.kf_every)
    print(f"[7SCENES_SEQUENCE] scene={scene} seq={seq_id} dataset_items={len(dataset)} kf={args.kf_every}", flush=True)
    return dataset


v106.base.build_dataset = build_dataset
if __name__ == "__main__":
    raise SystemExit(v106.base.main())

