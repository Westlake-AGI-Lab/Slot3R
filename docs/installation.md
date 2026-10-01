# Installation

The maintained workflow is inference/evaluation on Linux with an NVIDIA GPU.
Full AutoDL verification is pending; dependency versions inherited from Point3R
are not yet a locked, validated release environment.

```bash
git clone git@github.com:xyzhang-ashley/Slot3r.git
cd Slot3r
conda create -n slot3r python=3.11 -y
conda activate slot3r
# Install a CUDA-enabled torch/torchvision pair compatible with the machine first.
pip install -r requirements.txt
```

Download the pretrained Point3R checkpoint using the link in the
[upstream repository](https://github.com/YkiWu/Point3R), and set `WEIGHTS` to its
local path. Slot3R does not require an additional trained checkpoint.

The shell entrypoints establish the repository's Python import paths automatically.
They do not require an installed FeedForward_Eval checkout for Slot3R pose evaluation.
Prepare datasets according to [eval.md](eval.md); dataset roots are explicit
arguments, never hardcoded experiment-machine locations.

Run the CPU-only checks before running a GPU smoke test:

```bash
python -m unittest discover -s tests -v
sha256sum -c MODEL_CHECKSUMS.sha256
```
