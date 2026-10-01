"""Model registry, variant configuration and inference adapters."""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import torch

MODEL_MODULES = {
    "core": "dust3r.point3r_kway_frame_sparse_q35_confselect",
    "vpc_m": "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose",
    "vpc_a": "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose",
    "point3r": "dust3r.point3r",
}
MODEL_ALIASES = {"ours": "core", "ours_ray": "vpc_m", "ours_rayma": "vpc_a"}
MODELS = tuple(MODEL_MODULES) + ("ghost", "cut3r", "ttt3r")
RAY_SETTINGS = {
    "POINT3R_RAYAWARE_UPDATE": "1",
    "POINT3R_RAY_DUAL_BANK": "1",
    "POINT3R_RAY_PAPER_UPDATE": "1",
    "POINT3R_RAY_KWAY_DIVERSE_UPDATE": "0",
    "POINT3R_RAY_HYBRID_READOUT": "1",
    "POINT3R_RAY_POSE_INPUT_ONLY": "1",
    "POINT3R_RAY_POSE_POST_DECODER_ONLY": "0",
    "POINT3R_RAY_POSE_ONLY_ENSEMBLE": "0",
    "POINT3R_RAY_POSE_INPUT_TOKENS": "128",
    "POINT3R_RAY_POSE_INPUT_TEMPERATURE": "0.10",
}
VARIANT_SETTINGS = {
    "core": {"POINT3R_RAYAWARE_UPDATE": "0"},
    "vpc_m": {**RAY_SETTINGS, "POINT3R_RAY_BANK_UPDATE_EVERY": "4",
              "POINT3R_RAY_POSE_INPUT_MAX_WEIGHT": "0.025"},
    "vpc_a": {**RAY_SETTINGS, "POINT3R_RAY_BANK_UPDATE_EVERY": "1",
              "POINT3R_V106_RAY_BANK_UPDATE_EVERY": "1",
              "POINT3R_V106_POSE_INPUT_WEIGHT": "0.15"},
}


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def configure_model(args):
    """One configuration path for both direct Python and shell entrypoints."""
    clear_point3r_env()
    for settings in VARIANT_SETTINGS.values():
        for key in settings:
            os.environ.pop(key, None)
    if args.model not in VARIANT_SETTINGS:
        return
    os.environ.update({
        "POINT3R_MEMORY_UPDATE_MODE": "ordered_kway",
        "POINT3R_MEMORY_IMPL": "tensor",
        "POINT3R_ORDERED_UPDATE_IMPL": "tensor",
        "POINT3R_KWAY_NUM_SLOTS": str(args.kway_slots),
        "POINT3R_ORDERED_WAY_POLICY": "appearance",
        "POINT3R_ORDERED_THETA_BINS": str(args.theta_bins),
        "POINT3R_ORDERED_PHI_BINS": str(args.phi_bins),
        "POINT3R_ORDERED_RHO_BINS": str(args.rho_bins),
        "POINT3R_SPARSE_READOUT": "1",
        "POINT3R_SPARSE_MODE": "max",
        "POINT3R_SPARSE_MAX_TOKENS": str(args.sparse_max_tokens),
        "POINT3R_SPARSE_GLOBAL_ANCHORS": str(args.sparse_global_anchors),
        "POINT3R_SPARSE_NEIGHBOR_RANGE": str(args.sparse_neighbor_range),
        "POINT3R_ENCODE_CHUNK_SIZE": str(args.encode_chunk_size),
        "POINT3R_CGMC_DROP_QUANTILE": str(args.drop_quantile),
        "POINT3R_CGMC_WEIGHTED_MERGE": "1",
    })
    os.environ.setdefault("POINT3R_CONFSELECT_STATS", "0")
    os.environ.update(VARIANT_SETTINGS[args.model])


def add_path(path: str | Path, *, front: bool = True) -> None:
    p = str(Path(path).resolve())
    if p in sys.path:
        return
    if front:
        sys.path.insert(0, p)
    else:
        sys.path.append(p)


def add_point3r_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    add_path(repo)
    add_path(repo / "src" / "croco")
    add_path(repo / "src")


def promote_paths(*paths: str | Path) -> None:
    for path in reversed([str(Path(p).resolve()) for p in paths]):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def force_point3r_metric_imports(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    promote_paths(repo, repo / "src" / "croco", repo / "src")
    for name in ("eval.mv_recon.criterion", "eval.mv_recon.utils"):
        sys.modules.pop(name, None)


def add_src_repo_paths(repo: str | Path) -> None:
    repo = Path(repo).resolve()
    add_path(repo)
    add_path(repo / "src")
    add_path(repo / "src" / "croco")


def clear_point3r_env() -> None:
    for key in (
        "POINT3R_MEMORY_UPDATE_MODE",
        "POINT3R_ORDERED_UPDATE_IMPL",
        "POINT3R_KWAY_NUM_SLOTS",
        "POINT3R_ORDERED_WAY_POLICY",
        "POINT3R_ORDERED_THETA_BINS",
        "POINT3R_ORDERED_PHI_BINS",
        "POINT3R_ORDERED_RHO_BINS",
        "POINT3R_ORDERED_APP_THRESHOLD",
        "POINT3R_ORDERED_BUDGET_EVICT",
        "POINT3R_SPARSE_READOUT",
        "POINT3R_SPARSE_MODE",
        "POINT3R_SPARSE_MAX_TOKENS",
        "POINT3R_SPARSE_GLOBAL_ANCHORS",
        "POINT3R_SPARSE_NEIGHBOR_RANGE",
        "POINT3R_SPARSE_RECENT_TOKENS",
        "POINT3R_SPARSE_RECENT_FRAMES",
        "POINT3R_FIXED_MEMORY_TOKENS",
        "POINT3R_FIXED_MEMORY_MODE",
        "POINT3R_GEOANCHOR",
        "POINT3R_GEOANCHOR_STRIDE",
        "POINT3R_GEOANCHOR_H",
        "POINT3R_GEOANCHOR_FRAMES",
        "POINT3R_GEOANCHOR_MIN_GAP",
        "POINT3R_GEOANCHOR_BUCKETS_PER_KF",
        "POINT3R_GEOANCHOR_SLOTS_PER_KF",
        "POINT3R_GEOANCHOR_MAX_KFS",
        "POINT3R_PROFILE",
    ):
        os.environ.pop(key, None)


def move_batch_to_device(batch, device: str):
    ignore_keys = {
        "depthmap",
        "dataset",
        "label",
        "instance",
        "idx",
        "true_shape",
        "rng",
    }
    for view in batch:
        for name, value in list(view.items()):
            if name in ignore_keys:
                continue
            if isinstance(value, (tuple, list)):
                view[name] = [
                    x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                    for x in value
                ]
            elif torch.is_tensor(value):
                view[name] = value.to(device, non_blocking=True)
    return batch


def move_tree_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_tree_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_tree_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_tree_to_device(v, device) for v in obj)
    return obj


def build_ttt3r_views_from_batch(batch):
    # Keep this aligned with the already-working pose clean launcher:
    # CPU views + minimal TTT3R fields; inference_recurrent_lighter handles device.
    views = []
    for i, view in enumerate(batch):
        img = view["img"].detach().cpu()
        ts = view.get("true_shape")
        if torch.is_tensor(ts):
            ts = ts.detach().cpu()
        else:
            ts = torch.tensor([[img.shape[-2], img.shape[-1]]], dtype=torch.int32)

        views.append({
            "img": img,
            "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan),
            "true_shape": ts,
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


def ceil_to_multiple(x: int, multiple: int) -> int:
    return int(((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple))


def build_ghost_views_from_batch(batch, multiple: int = 14):
    views = []
    original_shapes = []
    for view in batch:
        img = view["img"]
        h, w = img.shape[-2:]
        hh = ceil_to_multiple(h, multiple)
        ww = ceil_to_multiple(w, multiple)
        new_view = dict(view)
        if (hh, ww) != (h, w):
            new_view["img"] = torch.nn.functional.interpolate(
                img, size=(hh, ww), mode="bilinear", align_corners=False
            )
            if torch.is_tensor(new_view.get("true_shape")):
                new_view["true_shape"] = torch.tensor(
                    [[hh, ww]],
                    dtype=new_view["true_shape"].dtype,
                    device=new_view["img"].device,
                )
        views.append(new_view)
        original_shapes.append((h, w))
    return views, original_shapes


def resize_map_to_hw(x, hw):
    if not torch.is_tensor(x):
        return x
    h, w = hw
    orig_dtype = x.dtype
    if x.ndim == 4:
        if x.shape[-1] <= 8:
            y = x.float().permute(0, 3, 1, 2)
            y = torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            y = y.permute(0, 2, 3, 1)
        else:
            y = torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False)
        return (y > 0.5) if orig_dtype is torch.bool else y.to(dtype=orig_dtype)
    if x.ndim == 3:
        y = torch.nn.functional.interpolate(x[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        return (y > 0.5) if orig_dtype is torch.bool else y.to(dtype=orig_dtype)
    return x


def resize_ghost_preds_to_batch(preds, original_shapes):
    resized = []
    for pred, hw in zip(preds, original_shapes):
        pred = dict(pred)
        for key in ("pts3d_in_other_view", "pts3d", "pts3d_in_self_view"):
            if key in pred:
                pred[key] = resize_map_to_hw(pred[key], hw)
        for key in ("conf", "depth", "depth_conf", "valid_mask"):
            if key in pred:
                pred[key] = resize_map_to_hw(pred[key], hw)
        resized.append(pred)
    return resized


def load_model(args, model_name: str, device: str):
    configure_model(args)
    if model_name in MODEL_MODULES:
        add_point3r_paths(args.point3r_repo)
        model_class = importlib.import_module(MODEL_MODULES[model_name]).Point3R
        model = model_class.from_pretrained(args.point3r_weights).to(device).eval()
        print(f"[MODEL] name={model_name} module={MODEL_MODULES[model_name]}", flush=True)
        if model_name in VARIANT_SETTINGS:
            print(
                f"[SLOT3R_CONFIG] drop_q={os.environ['POINT3R_CGMC_DROP_QUANTILE']} "
                f"slots={model.kway_num_slots} sparse_max={model.sparse_readout_max_tokens} "
                f"anchors={model.sparse_readout_global_anchors} "
                f"variant={VARIANT_SETTINGS[model_name]}", flush=True,
            )
        return model

    if model_name == "ghost":
        add_src_repo_paths(args.ghost_repo)
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT(total_budget=args.ghost_total_budget).to(device).eval()
        ckpt = torch.load(args.ghost_weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        return model

    if model_name in ("cut3r", "ttt3r"):
        add_src_repo_paths(args.ttt3r_repo)
        from src.dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(args.ttt3r_weights).to(device).eval()
        model.config.model_update_type = model_name
        return model

    raise ValueError(model_name)


def run_model(args, model_name: str, model, batch, device: str):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.time()
    with torch.no_grad():
        if model_name in MODEL_MODULES:
            with torch.cuda.amp.autocast(enabled=False):
                output = model(batch, point3r_tag=True)
            preds, out_views = output.ress, output.views
        elif model_name == "ghost":
            reset = getattr(getattr(model, "aggregator", None), "reset_kv_repository", None)
            if reset is not None:
                reset()
            ghost_batch, original_shapes = build_ghost_views_from_batch(batch, args.ghost_patch_multiple)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                try:
                    output = model.inference(ghost_batch, frame_writer=None, cache_results=True)
                except TypeError:
                    output = model.inference(ghost_batch)
            if output is None or getattr(output, "ress", None) is None:
                raise RuntimeError("GHOST inference returned no ress")
            preds = resize_ghost_preds_to_batch(output.ress, original_shapes)
            out_views = batch
            if reset is not None:
                reset()
        elif model_name in ("cut3r", "ttt3r"):
            from src.dust3r.inference import inference_recurrent_lighter

            output, _ = inference_recurrent_lighter(build_ttt3r_views_from_batch(batch), model, device)
            preds, out_views = output["pred"], batch
        else:
            raise ValueError(model_name)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return preds, out_views, elapsed, peak_gb
