from copy import deepcopy
import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
from scipy.spatial.transform import Rotation
from eval.relpose.evo_utils import *

def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation

    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def get_tum_poses(poses):
    """
    poses: list of 4x4 arrays
    """
    tt = np.arange(len(poses)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in poses]
    tum_poses = np.stack(tum_poses, 0)
    return [tum_poses, tt]


def save_tum_poses(poses, path):
    traj = get_tum_poses(poses)
    save_trajectory_tum_format(traj, path)
    return traj[0]  # return the poses


def save_focals(cam_dict, path):
    # convert focal to txt
    focals = cam_dict["focal"]
    np.savetxt(path, focals, fmt="%.6f")
    return focals


def save_intrinsics(cam_dict, path):
    K_raw = np.eye(3)[None].repeat(len(cam_dict["focal"]), axis=0)
    K_raw[:, 0, 0] = cam_dict["focal"]
    K_raw[:, 1, 1] = cam_dict["focal"]
    K_raw[:, :2, 2] = cam_dict["pp"]
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K, fmt="%.6f")
    return K_raw












# tensor
