# Installation

The maintained workflow is inference/evaluation on Linux with an NVIDIA GPU.
Dependency versions inherited from Point3R are not yet a locked release
environment; task-specific validation is recorded in `REPOSITORY_AUDIT.md`.

```bash
git clone git@github.com:xyzhang-ashley/Slot3r.git
cd Slot3r
conda create -n slot3r python=3.11 -y
conda activate slot3r
# Install a CUDA-enabled torch/torchvision pair compatible with the machine first.
pip install -r requirements.txt
# Build the 2D and fused 3D RoPE kernels with the active PyTorch environment.
(cd src/croco/models/curope && python setup.py build_ext --inplace)
```

Download the pretrained Point3R checkpoint using the link in the
[upstream repository](https://github.com/YkiWu/Point3R), and set `WEIGHTS` to its
local path. Slot3R does not require an additional trained checkpoint.

The shell entrypoints establish the repository's Python import paths automatically.
They do not require an installed FeedForward_Eval checkout for Slot3R pose evaluation.
Prepare datasets according to [eval.md](eval.md); dataset roots are explicit
arguments, never hardcoded experiment-machine locations.

Rebuild the extension after changing PyTorch/CUDA; do not copy a compiled `.so`
from another installation. The recovered fused RoPE3D implementation is used by
the original table experiments. `POINT3R_DISABLE_FUSED_ROPE3D=1` selects the slower
PyTorch reference for diagnostics; its FPS is not comparable to the fused runs.
The maintained workflow is inference only.

Run the checks before an evaluation (the RoPE parity test runs when CUDA is available):

```bash
python -m unittest discover -s tests -v
sha256sum -c MODEL_CHECKSUMS.sha256
```
