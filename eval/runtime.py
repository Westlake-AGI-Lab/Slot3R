"""Shared scene execution and metric-only result writing for pose/depth."""
import csv
import gc
import json
import math
import traceback
from pathlib import Path


def run_scenes(args, scenes, evaluate, aggregate):
    if not scenes:
        raise ValueError("No evaluation scenes selected")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    results, failures = {}, {}
    with (out / "run.log").open("w", encoding="utf-8") as log:
        for scene in scenes:
            try:
                row = evaluate(scene)
                if not row or any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
                    raise ValueError("Empty or non-finite metrics")
                results[scene] = row
                with (out / "summary.tsv").open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["model", "scene", *row], delimiter="\t")
                    if len(results) == 1:
                        writer.writeheader()
                    writer.writerow({"model": args.model, "scene": scene, **row})
                message = f"[scene] {scene}: {row}"
            except Exception as exc:
                failures[scene] = f"{type(exc).__name__}: {exc}"
                message = f"[FAIL] {scene}: {failures[scene]}\n{traceback.format_exc()}"
            print(message, flush=True)
            log.write(message+"\n")
            log.flush()
            (out / "scenes.json").write_text(json.dumps({"metrics": results, "failures": failures}, indent=2)+"\n")
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures or not results:
        (out / "stats_only.log").write_text("INCOMPLETE: see scenes.json; no aggregate published.\n")
        return 1
    summary = aggregate(list(results.values()))
    if not all(math.isfinite(v) for v in summary.values()):
        raise ValueError("Non-finite aggregate metrics")
    (out / "result.json").write_text(json.dumps(summary, indent=2)+"\n")
    (out / "stats_only.log").write_text(json.dumps(summary, indent=2)+"\n")
    return 0


def mean_metrics(rows):
    return {key: sum(float(row[key]) for row in rows)/len(rows) for key in rows[0]}
