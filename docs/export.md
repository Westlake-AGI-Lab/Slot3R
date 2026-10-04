# Export point clouds from an RGB folder

Export a colored point cloud directly from an RGB image folder.
Install the [environment and Point3R checkpoint](installation.md) first.
The example uses Core on the teaser's NeuralRGBD `morning_apartment` scene.
Set the image and checkpoint paths to your local copies and run from the repository root:

```bash
python tools/export_ply.py \
  --image_dir /path/to/neural_rgbd/morning_apartment/images \
  --weights /path/to/point3r_512.pth \
  --output_dir outputs/core_morning_apartment \
  --model core --kf_every 2 --max_frames 200 \
  --conf_quantile 0.25 --max_save_points 3000000
```

This exports the point cloud only, using the teaser's 200-frame, stride-2 input
selection. Add `--save_trajectory` when camera poses and trajectory layers are needed.

Select `core`, `vpc_m`, `vpc_a`, or the `point3r` baseline with `--model`.

## Input ordering and length

Place RGB JPG/JPEG/PNG/BMP files directly in `--image_dir`; files are naturally
sorted (`frame2.png` precedes `frame10.png`). Other files and subdirectories are
ignored. Use an RGB-only directory: mixed RGB/depth PNG folders are unsuitable.
`--kf_every 2` selects every second image, **then** `--max_frames 200` keeps the
first 200 selected images. `--max_frames 0` uses all selected images. Folders
with fewer images use their actual available count, recorded in `stats.json`.

The default device is CUDA and image size is 512. CPU can be selected explicitly
with `--device cpu` but is slow. The current model API retains the sequence's
views and predictions; long sequences can exhaust GPU or host RAM. Start with
20 frames for a smoke run, then increase the length. The point cap only limits
the exported file; it does not cap inference memory.

## Outputs

| File | Contents |
| --- | --- |
| `cloud.ply` | Colored XYZ point cloud in the model's shared coordinate frame |
| `frames.txt` | Exact ordered input paths after frame selection |
| `stats.json` | Frame count, point count, filters, timing and configuration |
| `c2w.npy`* | Predicted camera-to-world matrices, `(N, 4, 4)` |
| `trajectory.ply`* | Time-colored camera centers and connecting edges |
| `camera_frustums.ply`* | Sparse camera frustums as colored vertices/edges |
| `camera_centers.csv`* | Frame index and XYZ camera center; indexes correspond to `frames.txt` |
| `scene.mlp`* | MeshLab project with cloud, trajectory and frustum layers |

\* Written with `--save_trajectory`. These are **predicted** poses, not GT.
Frustums illustrate camera orientation; their rectangle is schematic and is not
calibrated to estimated intrinsics. Set `--frustum_count` (default 12) and
`--frustum_scale` (default 0.08 model units) for visibility. Some point-cloud
viewers ignore PLY edges; use MeshLab for the trajectory layers.

No alignment to ground truth, scaling to meters, or surface meshing is performed.
`--conf_quantile 0.20` discards the lowest-confidence 20% of finite colored
points globally (ties at the threshold remain). This export filter differs from
the model memory's `--drop_quantile 0.25`. Set the former to 0 to disable filtering.
`--max_save_points 3000000` caps the output using seeded random sampling;
0 preserves all surviving points. RGB values come from the model's resized/cropped
input images. Non-finite geometry, colors and confidence are removed together.

Use a new or empty output directory for each run.
For benchmark metrics and ground-truth alignment, see [evaluation](eval.md).
