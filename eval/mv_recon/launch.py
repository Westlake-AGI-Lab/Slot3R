#!/usr/bin/env python3
"""Evaluate a selected model and dataset through one point-cloud pipeline."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

# Support both python -m eval.mv_recon.launch and python eval/mv_recon/launch.py.
REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "src/croco", REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from eval.mv_recon.data import DEFAULT_SCENES, build_dataset
from eval.mv_recon.model_registry import (
    MODELS, MODEL_MODULES, canonical_model, load_model, run_model,
    move_batch_to_device, move_tree_to_device,
)
from eval.mv_recon.metrics import collect_metric_arrays, compute_open3d_metrics


def eval_one_scene(args, model_name: str, model, scene: str) -> dict:
    from torch.utils.data._utils.collate import default_collate

    dataset = build_dataset(args, scene)
    if len(dataset) < 1:
        raise RuntimeError(f"no sequence for scene={scene}")

    batch = default_collate([dataset[0]])
    orig_len = len(batch)
    if args.max_frames > 0 and len(batch) > args.max_frames:
        batch = batch[: args.max_frames]

    device = args.device
    if model_name == "ttt3r":
        preds, _out_views, elapsed, peak_gb = run_model(args, model_name, model, batch, device)
        batch = move_batch_to_device(batch, device)
    else:
        batch = move_batch_to_device(batch, device)
        preds, batch, elapsed, peak_gb = run_model(args, model_name, model, batch, device)

    valid_len = len(preds)
    if len(batch) != valid_len:
        n = min(len(batch), valid_len)
        batch, preds = batch[:n], preds[:n]

    # Long sequences can fit model inference but OOM while the scale/shift
    # criterion concatenates every dense point map on CUDA.  The criterion has
    # no model dependency, so optionally release those output tensors from the
    # GPU and evaluate the exact same arrays in host memory.
    if os.environ.get("POINT3R_METRIC_CPU", "0") == "1":
        cpu = torch.device("cpu")
        batch = move_batch_to_device(batch, cpu)
        preds = move_tree_to_device(preds, cpu)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pts_all, pts_gt_all, masks_all, images_all = collect_metric_arrays(batch, preds, args)
    metrics = compute_open3d_metrics(pts_all, pts_gt_all, masks_all, images_all, args)
    fps = len(batch) / max(elapsed, 1e-9)

    row = {
        "model": model_name,
        "scene": scene,
        "orig_len": orig_len,
        "frames": len(batch),
        "Acc": metrics["Acc"],
        "Comp": metrics["Comp"],
        "NC1": metrics["NC1"],
        "NC2": metrics["NC2"],
        "NC": (metrics["NC1"] + metrics["NC2"]) / 2.0,
        "Acc_med": metrics["Acc_med"],
        "Comp_med": metrics["Comp_med"],
        "NC1_med": metrics["NC1_med"],
        "NC2_med": metrics["NC2_med"],
        "NC_med": (metrics["NC1_med"] + metrics["NC2_med"]) / 2.0,
        "FPS": fps,
        "Time": elapsed,
        "Peak": peak_gb,
    }
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def fnum(x: float, nd: int = 6) -> str:
    try:
        if np.isnan(float(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_stats(out_dir: Path, rows: list[dict], failures: list[str]) -> None:
    stats = out_dir / "stats_only.log"
    header = [
        "model",
        "scene",
        "orig_len",
        "frames",
        "Acc",
        "Comp",
        "NC1",
        "NC2",
        "NC",
        "FPS",
        "Time",
        "Peak",
        "Acc_med",
        "Comp_med",
        "NC_med",
    ]
    lines = []
    lines.append("========== POINTCLOUD_SCENES ==========")
    lines.append("\t".join(header))
    for row in rows:
        lines.append("\t".join(fnum(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header))

    if failures:
        lines.append("")
        lines.append("========== FAILURES ==========")
        lines.extend(failures)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    lines.append("")
    lines.append("========== SUMMARY ==========")
    lines.append("| Method | n | frames_avg | Acc | Comp | NC1 | NC2 | NC | FPS | Time | Peak | Acc_med | Comp_med | NC_med |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_name, model_rows in sorted(by_model.items()):
        def mean(key: str) -> float:
            vals = np.array([float(r[key]) for r in model_rows], dtype=np.float64)
            return float(np.nanmean(vals)) if len(vals) else float("nan")

        lines.append(
            f"| {model_name} | {len(model_rows)} | {mean('frames'):.1f} | "
            f"{mean('Acc'):.6f} | {mean('Comp'):.6f} | {mean('NC1'):.6f} | "
            f"{mean('NC2'):.6f} | {mean('NC'):.6f} | {mean('FPS'):.4f} | "
            f"{mean('Time'):.4f} | {mean('Peak'):.2f} | {mean('Acc_med'):.6f} | "
            f"{mean('Comp_med'):.6f} | {mean('NC_med'):.6f} |"
        )
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Slot3R point-cloud evaluation")
    parser.add_argument("--model", type=canonical_model, choices=MODELS, default="core")
    parser.add_argument("--dataset", choices=tuple(DEFAULT_SCENES), default="nrgbd")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--kf_every", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--center_crop", type=int, default=224)
    parser.add_argument("--icp_thresh", type=float, default=0.1)
    parser.add_argument("--max_points", type=int, default=999999, help="Maximum sampled points for Open3D ICP/NC metric; 0 keeps all points")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--point3r_repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--weights", "--point3r_weights", dest="point3r_weights", default="")
    parser.add_argument("--ghost_repo", default="")
    parser.add_argument("--ghost_weights", default="")
    parser.add_argument("--ghost_total_budget", type=int, default=1200000)
    parser.add_argument("--ghost_patch_multiple", type=int, default=14)
    parser.add_argument("--ttt3r_repo", default="")
    parser.add_argument("--ttt3r_weights", default="")

    parser.add_argument("--kway_slots", type=int, default=8)
    parser.add_argument("--theta_bins", type=int, default=16)
    parser.add_argument("--phi_bins", type=int, default=8)
    parser.add_argument("--rho_bins", type=int, default=32)
    parser.add_argument("--sparse_max_tokens", type=int, default=640)
    parser.add_argument("--sparse_global_anchors", type=int, default=128)
    parser.add_argument("--sparse_neighbor_range", type=int, default=1)
    parser.add_argument("--encode_chunk_size", type=int, default=int(os.environ.get("POINT3R_ENCODE_CHUNK_SIZE", "100")))
    parser.add_argument("--drop_quantile", type=float, default=float(os.environ.get("POINT3R_CGMC_DROP_QUANTILE", "0.25")))
    args = parser.parse_args(argv)
    if args.kf_every < 1:
        parser.error("--kf_every must be positive")
    if args.model in MODEL_MODULES and not args.point3r_weights:
        parser.error("--weights is required for Slot3R/Point3R")
    if args.model == "ghost" and not (args.ghost_repo and args.ghost_weights):
        parser.error("GHOST requires --ghost_repo and --ghost_weights")
    if args.model in ("cut3r", "ttt3r") and not (args.ttt3r_repo and args.ttt3r_weights):
        parser.error("CUT3R/TTT3R require --ttt3r_repo and --ttt3r_weights")
    args.scenes = args.scenes or list(DEFAULT_SCENES[args.dataset])
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.tsv"
    log = out_dir / "run.log"
    header = [
        "model",
        "scene",
        "orig_len",
        "frames",
        "Acc",
        "Comp",
        "NC1",
        "NC2",
        "NC",
        "FPS",
        "Time",
        "Peak",
        "Acc_med",
        "Comp_med",
        "NC_med",
    ]
    summary.write_text("\t".join(header) + "\n", encoding="utf-8")
    log.write_text("", encoding="utf-8")

    rows: list[dict] = []
    failures: list[str] = []

    with log.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(
            f"[check] model={args.model} scenes={args.scenes} dataset={args.dataset} root={args.data_root} "
            f"kf_every={args.kf_every} max_frames={args.max_frames} size={args.size}\n"
        )
        print(f"[check] loading model={args.model}", flush=True)
        model = load_model(args, args.model, args.device)
        for scene in args.scenes:
            try:
                row = eval_one_scene(args, args.model, model, scene)
                rows.append(row)
                line = "\t".join(fnum(row[k], 4 if k in ("FPS", "Time", "Peak") else 6) for k in header)
                print(f"[pointcloud_scene] {line}", flush=True)
                lf.write(f"[pointcloud_scene] {line}\n")
                lf.flush()
                with summary.open("a", encoding="utf-8") as sf:
                    sf.write(line + "\n")
            except Exception as exc:
                msg = f"[pointcloud_scene] model={args.model} scene={scene} status=FAIL error={type(exc).__name__}: {exc}"
                print(msg, flush=True)
                traceback.print_exc()
                lf.write(msg + "\n")
                lf.write(traceback.format_exc() + "\n")
                failures.append(msg)
    write_stats(out_dir, rows, failures)
    print(f"wrote {summary}")
    print(f"wrote {out_dir / 'stats_only.log'}")
    return 1 if failures or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
