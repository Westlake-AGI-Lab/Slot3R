# Slot3R: Set-Associative Spatial Memory for Streaming 3D Reconstruction

Private release candidate for the training-free Slot3R retrofit of Point3R.
The pretrained backbone stays frozen. This checkout focuses on the numerical
experiments in the paper; evaluation does not export point-cloud visualizations,
PLY files, prediction NPY files, depth images, or trajectory plots.

## Models

| `MODEL` | Paper label | Historical alias | Implementation |
| --- | --- | --- | --- |
| `core` | Slot3R (Core) | `ours` | K-way / ConfSelect / sparse readout |
| `vpc_m` | Slot3R-VPC-M | `ours_ray` | v82e motion-gated pose conditioning |
| `vpc_a` | Slot3R-VPC-A | `ours_rayma` | v106 agreement-only conditioning, fresh bank |

All variants reuse the Point3R checkpoint. See [model provenance](docs/MODEL_VARIANTS.md)
and `MODEL_CHECKSUMS.sha256`; the evaluated model files are unchanged.

## Installation

Use Linux, Python 3.11 and a CUDA-enabled PyTorch environment. Follow
[installation](docs/installation.md) for dependencies and checkpoint setup.
Datasets and checkpoints are not included.

## Evaluate

All maintained entrypoints are under `eval/`:

```text
eval/
  mv_recon/launch.py  # 7Scenes / NeuralRGBD: Acc, Comp, NC, FPS
  pose/launch.py      # ScanNet / Sintel / TUM-Dynamic: ATE, RPE
  depth/launch.py     # Bonn / ScanNet / KITTI: depth metrics
  model_config.py    # shared model names and task configurations
  runtime.py         # shared pose/depth scene loop and result writer
```

Each task has one `launch.py` and an optional `run.sh` wrapper. Select datasets
and models with `--dataset` and `--model`; dataset loading and numerical metrics
live in `data.py` and `metrics.py`, rather than separate dataset executables.

Example point-cloud run (paths are placeholders):

```bash
MODEL=core DATASET=nrgbd DATA_ROOT=/path/to/neural_rgbd \
WEIGHTS=/path/to/point3r_512.pth KF_EVERY=2 MAX_FRAMES=300 \
bash eval/mv_recon/run.sh
```

See [evaluation instructions](docs/eval.md) for every table, required dataset
layouts, metric-scale versus sequence-aligned depth, output files, and AutoDL
smoke tests. Existing non-empty output directories are rejected to avoid mixing
results. Failed scenes produce a nonzero exit code; partial results are not full
table reproductions.

## Validation status

Launcher, path, export-policy, model-hash and optional CUDA kernel checks:

```bash
python -m unittest discover -s tests -v
sha256sum -c MODEL_CHECKSUMS.sha256
```

Core NeuralRGBD evaluation (`kf=2`, up to 200 frames, nine scenes) has passed a
real RTX 4090 run. Aggregate metrics are close to the historical reference.
The unified point-cloud entrypoint also passes six 20-frame GPU smoke runs
(Core/VPC-M/VPC-A on 7Scenes/NeuralRGBD). All 22 automated checks pass on AutoDL,
including pose/depth CLI routing, alignment and aggregation checks.
The 18 pose/depth GPU format-fixture checks also pass; these are not target
benchmark results. Full-table reproduction and pose/depth benchmark-data
validation remain pending.
See the [cleanup audit](docs/REPOSITORY_AUDIT.md). The legacy training scripts are inherited
from Point3R and are not a supported Slot3R training workflow in this release.

## Provenance and acknowledgements

Based on [Point3R](https://github.com/YkiWu/Point3R), with components from DUSt3R,
CroCo, CUT3R and the evaluation ecosystems credited in the source files.
Original licenses and notices are retained. The [archive](archive/README.md)
contains experiment-machine scripts and retired Point3R evaluation entrypoints
for provenance; they are not portable commands or supported public entrypoints.
