#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np


def parse_metric(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"(?=APE w\.r\.t\.|RPE w\.r\.t\.)", text)
    ate = rpe_rotation = rpe_translation = None
    for section in sections:
        if section.startswith("APE w.r.t. translation"):
            match = re.search(r"^\s*rmse\s+([0-9.eE+-]+)", section, re.MULTILINE)
            ate = float(match.group(1)) if match else None
        elif section.startswith("RPE w.r.t. rotation"):
            match = re.search(r"^\s*rmse\s+([0-9.eE+-]+)", section, re.MULTILINE)
            rpe_rotation = float(match.group(1)) if match else None
        elif section.startswith("RPE w.r.t. translation"):
            match = re.search(r"^\s*rmse\s+([0-9.eE+-]+)", section, re.MULTILINE)
            rpe_translation = float(match.group(1)) if match else None
    if None in (ate, rpe_translation, rpe_rotation):
        raise ValueError(f"incomplete metric file: {path}")
    return ate, rpe_translation, rpe_rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.result_dir).glob("*_eval_metric.txt")):
        scene = path.name[: -len("_eval_metric.txt")]
        metric = parse_metric(path)
        rows.append(metric)
        print(f"{scene}\t{metric[0]:.6f}\t{metric[1]:.6f}\t{metric[2]:.6f}")
    if not rows:
        raise RuntimeError(f"no metric files under {args.result_dir}")
    mean = np.mean(np.asarray(rows), axis=0)
    print(
        f"MEAN\tn={len(rows)}\tATE={mean[0]:.6f}\t"
        f"TRANS={mean[1]:.6f}\tROT={mean[2]:.6f}"
    )


if __name__ == "__main__":
    main()
