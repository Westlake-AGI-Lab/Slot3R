# Evaluation

Select `core`, `vpc_m`, `vpc_a`, or the `point3r` baseline with `--model`
in a Python command, or `MODEL` in a shell command. Select the dataset with
`--dataset` or `DATASET`. Evaluation writes metrics and logs.
For point clouds and trajectory visualization, see [export](export.md).

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

For point-cloud reconstruction:

```bash
python -m eval.mv_recon.launch --model core --dataset nrgbd \
  --data_root /path/to/neural_rgbd --weights /path/to/point3r_512.pth \
  --kf_every 2 --max_frames 200 --output_dir outputs/core_nrgbd_200
```

Use `--model vpc_m` / `vpc_a` or `--dataset 7scenes` with its root.
Use `--scenes` to select specific sequences.

```bash
MODEL=core DATASET=nrgbd DATA_ROOT=/path/to/neural_rgbd \
WEIGHTS=/path/to/point3r_512.pth KF_EVERY=2 MAX_FRAMES=300 \
bash eval/mv_recon/run.sh
```

Repeat `MAX_FRAMES=300,400,500` separately for the main table. Use `DATASET=7scenes`
with the corresponding root for 7Scenes. For the long-sequence table, use
`KF_EVERY=1` and `MAX_FRAMES=600,700,800,900,1000` separately. Default scene lists
are 18 7Scenes and 9 NeuralRGBD sequences. `SCENES="name ..."` selects a subset.

Outputs: `summary.tsv`, `stats_only.log`, `run.log` under
`outputs/pointcloud/<model>_<dataset>_kf<k>_len<n>/` by default.
Sampling uses a fixed seed (0 by default).

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
14 paper sequences; TUM discovers prepared scene directories. ScanNet uses
candidates `scene0707_00`–`scene0806_00`. For the paper tables, select the
corresponding 94 ScanNet sequences or 8 TUM-Dynamic sequences with `SCENES`.
Evaluation requires valid ground truth for every selected sequence.

Pose evaluation uses Sim(3) alignment. Sintel reports RPE RMSE; ScanNet/TUM
report RPE mean. `--pose_eval_stride` controls frame subsampling;
`--max_frames` limits the sequence length. Full evaluation defaults to stride 1
without a frame cap.

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
Indoor depth evaluation uses a 5 m maximum depth.

KITTI:

```bash
MODEL=core DATASET=kitti DATA_ROOT=/path/to/val_selection_cropped \
WEIGHTS=/path/to/point3r_512.pth ALIGN=scale_shift \
OUTPUT_DIR=outputs/depth/core_kitti_sequence bash eval/depth/run.sh
```

For metric-scale KITTI use `ALIGN=metric` and a fresh output directory. The KITTI
runner computes metrics directly in memory,
including cubic resize and valid-pixel-weighted aggregation. Bonn/ScanNet retain
frame-mean metrics followed by scene means. KITTI uses /256 GT decoding and the
positive-GT mask without an upper depth cap; indoor `--max_depth` and
crop overrides are rejected for KITTI. Bonn/ScanNet default to a 5 m cap.

Pose/depth share the same output structure: `config.json`, `summary.tsv`,
`scenes.json`, `run.log`, `stats_only.log`, and `result.json`. `result.json` is
written only when all selected scenes succeed with finite metrics. Failed runs
retain diagnostic per-scene results but publish no aggregate.
