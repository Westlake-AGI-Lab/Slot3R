"""KITTI video-depth metrics, evaluated in memory without prediction/media exports."""
import os
import sys
import importlib
import argparse
import json
import re
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.video_depth.tools import depth_evaluation
from eval.add_ckpt_path import add_path_to_dust3r


def get_args_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_root", required=True,
                        help="Root containing image_gathered/ and groundtruth_depth_gathered/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_dataset", choices=("kitti",), default="kitti")
    parser.add_argument("--align", choices=("scale_shift", "metric", "scale"), default="scale_shift")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    return parser


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


def sequence_metrics(predictions, depth_paths, align, use_gpu):
    gt = []
    for path in depth_paths:
        raw = np.array(Image.open(path), dtype=int)
        if raw.max() <= 255:
            raise ValueError(f"Expected 16-bit KITTI depth: {path}")
        depth = raw.astype(float) / 256.0
        depth[raw == 0] = -1.0
        gt.append(depth)
    gt = np.stack(gt)
    if len(predictions) != len(gt):
        raise ValueError("Prediction/ground-truth frame count mismatch")
    pred = np.stack([cv2.resize(value, (gt.shape[2], gt.shape[1]),
                               interpolation=cv2.INTER_CUBIC) for value in predictions])
    flags = {"scale_shift": {"align_with_lad2": True},
             "scale": {"align_with_scale": True}, "metric": {"metric_scale": True}}
    # Same helper, resize, alignment and valid-pixel weighting as the original evaluator.
    result, _, _, _ = depth_evaluation(pred, gt, max_depth=None, use_gpu=use_gpu, **flags[align])
    if result["valid_pixels"] <= 0:
        raise ValueError("No valid ground-truth depth pixels")
    return result


def evaluate(args, model):
    from dust3r.inference import inference
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    scenes = args.scenes or sorted(p.name for p in (root / "image_gathered").iterdir() if p.is_dir())
    if not scenes:
        raise ValueError("No KITTI scenes found")
    model = model.to(args.device).eval()
    results, failures = {}, {}
    for scene in tqdm(scenes):
        try:
            files, depths = scene_pairs(root, scene)
            views = prepare_input(files, [True] * len(files), size=args.size, crop=False)
            with torch.no_grad():
                outputs = inference(views, model, args.device)
            predictions = [p["pts3d_in_self_view"][0, ..., -1].detach().cpu().numpy()
                           for p in outputs["pred"]]
            results[scene] = sequence_metrics(predictions, depths, args.align,
                                              use_gpu=str(args.device).startswith("cuda"))
            del outputs, predictions, views
        except Exception as exc:
            failures[scene] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out / "scenes.json").write_text(json.dumps({"metrics": results, "failures": failures}, indent=2), encoding="utf-8")
    if failures or not results:
        for name in ("result_scale&shift.json", "result_scale.json", "result_metric.json"):
            (out / name).unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete KITTI evaluation: {failures}")
    values = list(results.values())
    average = {key: float(np.average([m[key] for m in values], weights=[m["valid_pixels"] for m in values]))
               for key in values[0] if key != "valid_pixels"}
    suffix = "scale&shift" if args.align == "scale_shift" else args.align
    (out / f"result_{suffix}.json").write_text(json.dumps(average, indent=2), encoding="utf-8")
    print(average)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    module_name = os.environ.get(
        "POINT3R_DEPTH_MODEL_MODULE",
        "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose",
    )
    Point3R = importlib.import_module(module_name).Point3R
    from dust3r.utils.camera import pose_encoding_to_camera

    print(
        "[Q_ABLATION_CONFIG] "
        f"module={Point3R.__module__} "
        f"drop_quantile={os.environ.get('POINT3R_CGMC_DROP_QUANTILE')} "
        f"sparse_max_tokens={os.environ.get('POINT3R_SPARSE_MAX_TOKENS')}",
        flush=True,
    )

    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        # colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            None,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = Point3R.from_pretrained(args.weights).eval()
    evaluate(args, model)
