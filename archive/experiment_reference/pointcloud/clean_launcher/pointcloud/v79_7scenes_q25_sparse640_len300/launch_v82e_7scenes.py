#!/usr/bin/env python3
"""Run v81 geometry with the v76 pre-decoder pose path on 7Scenes."""

from __future__ import annotations

import sys
from pathlib import Path


NRGBD_RUNNER = Path(
    "/root/autodl-tmp/clean_launcher/pointcloud/v79_nrgbd_q25_sparse640_len300"
)
sys.path.insert(0, str(NRGBD_RUNNER))
import launch_v82e_nrgbd as v82e


def build_dataset(args, scene_spec: str):
    v82e.base.add_point3r_paths(args.point3r_repo)
    from eval.mv_recon.data import SevenScenes

    if "/" not in scene_spec:
        raise ValueError(f"expected scene/seq-NN, got {scene_spec!r}")
    scene, seq_id = scene_spec.split("/", 1)
    resolution = (512, 384) if args.size == 512 else 224
    dataset = SevenScenes(
        split="test",
        ROOT=args.nrgbd_root,
        resolution=resolution,
        num_seq=1,
        test_id=scene,
        seq_id=seq_id,
        full_video=True,
        kf_every=args.kf_every,
    )
    print(
        f"[7SCENES_SEQUENCE] scene={scene} seq={seq_id} "
        f"dataset_items={len(dataset)} kf={args.kf_every}",
        flush=True,
    )
    return dataset


v82e.base.build_dataset = build_dataset


if __name__ == "__main__":
    raise SystemExit(v82e.base.main())
