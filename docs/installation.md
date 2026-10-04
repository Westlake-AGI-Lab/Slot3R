# Installation

Slot3R uses the pretrained Point3R backbone for inference and evaluation. Use Linux, Python 3.11 and an NVIDIA GPU with a compatible CUDA-enabled PyTorch installation.

## Environment

```bash
git clone https://github.com/xyzhang-ashley/Slot3R.git
cd Slot3R
conda create -n slot3r python=3.11 -y
conda activate slot3r

# Install torch and torchvision for your CUDA environment first.
pip install -r requirements.txt

# Build the 2D and fused 3D rotary-position-encoding kernels.
(cd src/croco/models/curope && python setup.py build_ext --inplace)
```

Rebuild the extension in the active environment after changing PyTorch or CUDA.

## Checkpoint

Download `point3r_512.pth` through the [Point3R repository](https://github.com/YkiWu/Point3R). All three Slot3R variants reuse this checkpoint; no additional Slot3R training or weights are needed. Checkpoints and datasets are downloaded separately.

Pass the local checkpoint path as `--weights /path/to/point3r_512.pth` for RGB-folder reconstruction, or `WEIGHTS=/path/to/point3r_512.pth` for an evaluation shell command.

## Run

- [RGB-folder reconstruction and point-cloud export](export.md).
- [Benchmark evaluation and dataset layouts](eval.md).
- [Model variants](../README.md#model-variants).

Run commands from the repository root.
