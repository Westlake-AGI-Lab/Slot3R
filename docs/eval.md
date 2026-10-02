# Table evaluation

Use `MODEL=core`, `vpc_m`, or `vpc_a`. Legacy aliases `ours`, `ours_ray`, and
`ours_rayma` are also accepted. All commands run inference and metrics; none write
PLY, point-cloud NPY, depth PNG/GIF, comparison images, or trajectory plots.
Numerical arrays stay in memory. Outputs are metric TSV/JSON/TXT and logs.

Every task uses one `launch.py`, selected with `--dataset` and `--model`.
`run.sh` is only an environment-variable convenience wrapper. Pose and depth
support `core`, `vpc_m`, `vpc_a`, and `point3r`; their configuration is applied
in Python through `eval/model_config.py`, so direct CLI runs use the same model
settings as shell runs. Each task's `data.py` handles input formats and
`metrics.py` handles scoring. There are no dataset-specific executables.

## Required dataset layouts

| Task | `DATASET` | `DATA_ROOT` layout |
| --- | --- | --- |
| Point cloud | `7scenes` | `<scene>/seq-NN/` with the upstream 7Scenes RGB/depth/pose format |
| Point cloud | `nrgbd` | `<scene>/images/imgN.png`, `depth/depthN.png`, original trajectory files |
| Pose | `scannet` | `<scene>/color_90/*.{jpg,png}`, `pose_90.txt` |
| Pose | `tum` | `<scene>/rgb_90/*.{jpg,png}`, `groundtruth_90.txt` |
| Pose | `sintel` | `final/<scene>/*.png`, `camdata_left/<scene>/*.cam` (the training directory) |
| Depth | `bonn` | prepared `rgbd_bonn_<scene>/rgb_110/`, `depth_110/`, `groundtruth_110.txt` |
| Depth | `scannet` | prepared `<scene>/color_90/`, `depth_90/`, `pose_90.txt`, intrinsic files |
| Depth | `kitti` | `image_gathered/<scene>/*.png`, `groundtruth_depth_gathered/<scene>/*.png` |

These are prepared evaluation datasets, not arbitrary raw downloads. Follow the
existing Point3R/MonST3R preprocessing convention. KITTI supports identical RGB/GT
basenames or official `..._image_<frame>_image_<camera>` versus
`..._groundtruth_depth_<frame>_image_<camera>` names; missing pairs fail explicitly.

## Point cloud: main and long-sequence tables

All model/dataset combinations use `eval/mv_recon/launch.py`. Dataset loading is
selected in `data.py`, model variants in `model_registry.py`, and the shared numerical
pipeline in `metrics.py`. The old `eval/pointcloud/launch_ours_*` files are removed.
The model configuration is set in Python, so the module entrypoint and shell
wrapper use the same settings. For example:

```bash
python -m eval.mv_recon.launch --model core --dataset nrgbd \
  --data_root /path/to/neural_rgbd --weights /path/to/point3r_512.pth \
  --kf_every 2 --max_frames 200 --output_dir outputs/core_nrgbd_200
```

Use `--model vpc_m` / `vpc_a` or `--dataset 7scenes` with its root. Default scene
lists live in `data.py`; `--scenes` overrides them. The shell wrapper below accepts
the same model/dataset names via environment variables. `point3r` is also
available as a baseline. Existing external GHOST/CUT3R/TTT3R adapters remain
available through the Python CLI with their explicit repository/weight arguments.

```bash
MODEL=core DATASET=nrgbd DATA_ROOT=/path/to/neural_rgbd \
WEIGHTS=/path/to/point3r_512.pth KF_EVERY=2 MAX_FRAMES=300 \
bash eval/mv_recon/run.sh
```

Repeat `MAX_FRAMES=300,400,500` separately for the main table. Use `DATASET=7scenes`
with the corresponding root for 7Scenes. For the long-sequence table, use
`KF_EVERY=1` and `MAX_FRAMES=600,700,800,900,1000` separately. Default scene lists
are the existing 18 7Scenes and 9 NeuralRGBD sequences. `SCENES="name ..."` selects
a subset; subset metrics are smoke tests, not table reproductions.

Outputs: `summary.tsv`, `stats_only.log`, `run.log` under
`outputs/pointcloud/<model>_<dataset>_kf<k>_len<n>/` by default.
FPS excludes data loading and metric computation. Model inference and metric
formulas are preserved from the recovered experiment code.
The point cap uses shared sampling indices for predictions and GT, matching the
original table launcher. Sampling uses a fixed seed (0 by default); historical
runs used an unseeded sample, so their last digits can vary.

## Camera pose

```bash
python -m eval.pose.launch --model core --dataset sintel \
  --data_root /path/to/sintel/training --weights /path/to/point3r_512.pth \
  --output_dir outputs/pose/core_sintel
```

```bash
MODEL=core DATASET=sintel DATA_ROOT=/path/to/sintel/training \
WEIGHTS=/path/to/point3r_512.pth OUTPUT_DIR=outputs/pose/core_sintel \
bash eval/pose/run.sh
```

Change `DATASET` and `DATA_ROOT` for `scannet` or `tum`. Sintel defaults to the
14 paper sequences; TUM discovers prepared scene directories. ScanNet retains
the original candidate list `scene0707_00`–`scene0806_00`. The exact 94 valid
ScanNet scene manifest and the TUM benchmark subset must be confirmed on the
experiment machine and supplied through `SCENES` for a full table run. Invalid
or missing GT is a failure, not a zero-valued or silently excluded result.

Pose uses the original Sim(3)-aligned evo implementation and writes per-scene
metric text. The inherited reporting conventions remain explicit: Sintel uses
RPE RMSE; ScanNet/TUM use RPE mean. Both statistics are retained in results.
`--pose_eval_stride` controls frame subsampling; `--max_frames` limits a smoke
test. Full evaluation defaults to stride 1 without a frame cap. No external
FeedForward_Eval repo or per-scene backend subprocess is needed.

## Video depth: both table settings

```bash
python -m eval.depth.launch --model core --dataset bonn \
  --data_root /path/to/bonn --weights /path/to/point3r_512.pth \
  --align sequence_scale_shift --output_dir outputs/depth/core_bonn
```

Bonn and ScanNet, per-sequence scale-and-shift alignment:

```bash
MODEL=core DATASET=bonn DATA_ROOT=/path/to/bonn \
WEIGHTS=/path/to/point3r_512.pth ALIGN=sequence_scale_shift MAX_DEPTH=5 \
OUTPUT_DIR=outputs/depth/core_bonn_sequence bash eval/depth/run.sh
```

For metric-scale Bonn/ScanNet, use `ALIGN=none` and a fresh output directory.
The 5 m maximum depth and sequence alignment follow the recovered experiment
launchers; this corrects the previous portable wrapper's inconsistent defaults.

KITTI:

```bash
MODEL=core DATASET=kitti DATA_ROOT=/path/to/val_selection_cropped \
WEIGHTS=/path/to/point3r_512.pth ALIGN=scale_shift \
OUTPUT_DIR=outputs/depth/core_kitti_sequence bash eval/depth/run.sh
```

For metric-scale KITTI use `ALIGN=metric` and a fresh output directory. The KITTI
runner computes metrics directly in memory using the original depth helper,
including cubic resize and valid-pixel-weighted aggregation. Bonn/ScanNet retain
frame-mean metrics followed by scene means. KITTI uses /256 GT decoding and the
original positive-GT mask without an upper depth cap; indoor `--max_depth` and
crop overrides are rejected for KITTI. Bonn/ScanNet default to a 5 m cap.

Pose/depth share the same output structure: `config.json`, `summary.tsv`,
`scenes.json`, `run.log`, `stats_only.log`, and `result.json`. `result.json` is
written only when all selected scenes succeed with finite metrics. Failed runs
retain diagnostic per-scene results but publish no aggregate. Older KITTI
`result_scale&shift.json` / `result_metric.json` consumers should now read
`result.json` and its accompanying `config.json` alignment setting.

## AutoDL smoke test

Start with one real prepared scene for each task, for example:

```bash
MODEL=core DATASET=nrgbd SCENES=breakfast_room MAX_FRAMES=10 KF_EVERY=2 \
DATA_ROOT=/path/to/neural_rgbd WEIGHTS=/path/to/point3r_512.pth \
OUTPUT_DIR=outputs/smoke/core_cloud bash eval/mv_recon/run.sh

MODEL=vpc_m DATASET=sintel SCENES=alley_2 \
DATA_ROOT=/path/to/sintel/training WEIGHTS=/path/to/point3r_512.pth \
OUTPUT_DIR=outputs/smoke/vpc_m_pose bash eval/pose/run.sh

MODEL=vpc_a DATASET=bonn SCENES=balloon2 MAX_FRAMES=10 \
DATA_ROOT=/path/to/bonn WEIGHTS=/path/to/point3r_512.pth \
OUTPUT_DIR=outputs/smoke/vpc_a_depth bash eval/depth/run.sh
```

Check nonzero exit status, actual evaluated scene/frame counts, finite metrics,
absence of prediction/media artifacts, and model selection printed in the logs.
Then test TUM and KITTI explicitly, followed by all three variants. Small smoke
runs validate wiring only; compare full runs to the paper separately.
