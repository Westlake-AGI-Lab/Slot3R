"""Shared pose view construction and camera recovery."""
from copy import deepcopy
import numpy as np
import torch

def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import weighted_procrustes

    bsz, height, width, _ = pts3ds_self.shape
    pp = (
        torch.tensor([width // 2, height // 2], device=pts3ds_self.device)
        .float()
        .repeat(bsz, 1)
        .reshape(bsz, 1, 2)
    )
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    pts3ds_self = pts3ds_self.reshape(bsz, -1, 3)
    pts3ds_other = pts3ds_other.reshape(bsz, -1, 3)
    conf_self = conf_self.reshape(bsz, -1)
    conf_other = conf_other.reshape(bsz, -1)
    c2w = weighted_procrustes(
        pts3ds_self,
        pts3ds_other,
        torch.log(conf_self) * torch.log(conf_other),
        use_weights=True,
        return_T=True,
    )
    return c2w, focal, pp.reshape(bsz, 2)

def prepare_input(img_paths, img_mask, size, revisit=1, update=True, crop=True):
    from src.dust3r.utils.image import load_images_for_eval as load_images

    images = load_images(img_paths, size=size, crop=crop)
    views = []
    for i, image in enumerate(images):
        img = image["img"]
        views.append(
            {
                "img": img,
                "ray_map": torch.full(
                    (img.shape[0], 6, img.shape[-2], img.shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(bool(img_mask[i])).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(bool(update and img_mask[i])).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )

    if revisit <= 1:
        return views

    repeated = []
    for r in range(revisit):
        for i, view in enumerate(views):
            new_view = deepcopy(view)
            new_view["idx"] = r * len(views) + i
            new_view["instance"] = str(r * len(views) + i)
            if r > 0 and not update:
                new_view["update"] = torch.tensor(False).unsqueeze(0)
            repeated.append(new_view)
    return repeated

def prepare_output(outputs, revisit=1, solve_pose=False):
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]

    if solve_pose:
        pr_poses, focal, pp = recover_cam_params(
            torch.cat(pts3ds_self, 0),
            torch.cat(pts3ds_other, 0),
            torch.cat(conf_self, 0),
            torch.cat(conf_other, 0),
        )
    else:
        pts3ds_self_cat = torch.cat(pts3ds_self, 0)
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)
        bsz, height, width, _ = pts3ds_self_cat.shape
        pp = (
            torch.tensor([width // 2, height // 2], device=pts3ds_self_cat.device)
            .float()
            .repeat(bsz, 1)
            .reshape(bsz, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self_cat, pp, focal_mode="weiszfeld")

    cam_dict = {"focal": focal.cpu().numpy(), "pp": pp.cpu().numpy()}
    return cam_dict, pr_poses
