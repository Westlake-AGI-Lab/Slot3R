"""Prepared pose datasets used by the paper; roots are supplied by launchers."""
import os

SINTEL_SCENES = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4",
                 "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2",
                 "temple_2", "temple_3"]
dataset_metadata = {
    "scannet": {
        "img_path": os.environ.get("SCANNET_ROOT", "data/scannet"), "mask_path": None,
        "dir_path_func": lambda root, seq: os.path.join(root, seq, "color_90"),
        "gt_traj_func": lambda root, anno, seq: os.path.join(root, seq, "pose_90.txt"),
        "traj_format": "replica", "seq_list": None, "full_seq": True,
    },
    "tum": {
        "img_path": os.environ.get("TUM_ROOT", "data/tum"), "mask_path": None,
        "dir_path_func": lambda root, seq: os.path.join(root, seq, "rgb_90"),
        "gt_traj_func": lambda root, anno, seq: os.path.join(root, seq, "groundtruth_90.txt"),
        "traj_format": "tum", "seq_list": None, "full_seq": True,
    },
    "sintel": {
        "img_path": "data/sintel/training/final", "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda root, seq: os.path.join(root, seq),
        "gt_traj_func": lambda root, anno, seq: os.path.join(anno, seq),
        "traj_format": None, "seq_list": SINTEL_SCENES, "full_seq": False,
    },
}
