#!/usr/bin/env python3
"""Summarize evo-style per-scene pose metric text files."""

import pathlib
import sys


def parse_metric(path):
    section = None
    values = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("APE w.r.t. translation"):
            section = "ate"
        elif line.startswith("RPE w.r.t. rotation"):
            section = "rot"
        elif line.startswith("RPE w.r.t. translation"):
            section = "trans"
        elif section and line.startswith("mean"):
            values[f"{section}_mean"] = float(line.split()[-1])
        elif section and line.startswith("rmse"):
            values[f"{section}_rmse"] = float(line.split()[-1])
    return values["ate_rmse"], values["trans_mean"], values["rot_mean"]


def load_dir(root):
    result = {}
    for path in sorted(pathlib.Path(root).glob("*_eval_metric.txt")):
        scene = path.name.removesuffix("_eval_metric.txt")
        result[scene] = parse_metric(path)
    return result


def main():
    named = []
    for item in sys.argv[1:]:
        name, root = item.split("=", 1)
        named.append((name, load_dir(root)))
    scenes = sorted(set.intersection(*(set(data) for _, data in named)))
    print("scene\t" + "\t".join(
        f"{name}_ATE\t{name}_trans\t{name}_rot" for name, _ in named
    ))
    for scene in scenes:
        print(scene + "\t" + "\t".join(
            "\t".join(f"{value:.6f}" for value in data[scene])
            for _, data in named
        ))
    print("MEAN\t" + "\t".join(
        "\t".join(
            f"{sum(data[scene][metric] for scene in scenes) / len(scenes):.6f}"
            for metric in range(3)
        )
        for _, data in named
    ))


if __name__ == "__main__":
    main()
