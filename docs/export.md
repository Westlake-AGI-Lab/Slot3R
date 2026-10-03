# Export point clouds from an RGB folder

`tools/export_ply.py` is a visualization/inference entrypoint. It does not need
depth, camera intrinsics, ground-truth poses, or a benchmark dataset layout.
Install the normal [runtime and Point3R checkpoint](installation.md) first.
Run from the repository root:

```bash
python tools/export_ply.py \
  --image_dir /path/to/scene/images \
  --weights /path/to/point3r_512.pth \
  --output_dir outputs/my_scene_core \
  --model core --kf_every 2 --max_frames 200 --save_trajectory
```

Select `core`, `vpc_m`, `vpc_a`, or the `point3r` baseline. Historical aliases
`ours`, `ours_ray`, and `ours_rayma` are accepted. The Slot3R variants reuse
`eval/model_config.py`'s reconstruction configuration; model sources are not
patched or copied. Evaluation under `eval/` still produces numerical results only.
The export-only Point3R loader accepts both dictionary and Namespace checkpoint
metadata, including the retained `point3r_512.pth` checkpoint.

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
| `stats.json` | Actual frames, point count, filter, timing, configuration and model source hash |
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

Use a new or empty output directory for each run. Existing outputs are never
silently overwritten. Do not compare these unaligned visualization files with
benchmark metrics; use the [evaluation entrypoints](eval.md) for those.

## Provenance and validation

This entrypoint adapts the retained AutoDL RGB-folder teaser inference workflow
and its camera-pose export into a portable command. It preserves natural frame
ordering, streaming view fields, shared-frame point maps, normalized-image RGB
recovery, global confidence filtering, and seeded output sampling. It uses the
current shared model configuration rather than historical machine-specific paths.
Pure NumPy export tests exercise point/color correspondence, binary PLY payloads,
frame selection and camera/frustum transforms. A real CUDA/checkpoint inference
run is additionally required to validate a particular deployment.

AutoDL validation (2026-10-03): all 28 tests pass without skips on an RTX 4090,
Python 3.11.15, PyTorch 2.5.1 / CUDA 12.1, with the repository's RoPE extension
built in that environment. Six export tests include dictionary/Namespace
checkpoint-loading coverage. Each of `core`, `vpc_m`, `vpc_a`, and `point3r`
successfully reconstructed 20 green_room RGB frames with the retained
`point3r_512.pth` checkpoint, image size 512, stride 2, a 100,000-point cap, and
`--save_trajectory`; all checkpoint keys matched.

Open3D 0.19 independently read each colored cloud. Additional checks verified
finite coordinates/colors, 20 valid rigid camera-to-world matrices, trajectory
centers matching those matrices, 19 connecting edges, frustum edge indexes,
CSV/frame counts, MeshLab layer references, and model-source hashes. These are
short-sequence functional checks, not full-scene quality or FPS benchmarks.
