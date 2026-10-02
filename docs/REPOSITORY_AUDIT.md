# Evaluation cleanup audit

Source revision: `b94b525b3627ca72165417c0406db7d95fea6fc8`.

## Implemented

- Canonical table entrypoints now live under `eval/mv_recon`, `eval/pose` and
  `eval/depth`. Old Point3R launchers and experiment-machine scripts are archived
  separately with their provenance. The five checksummed Slot3R variant files
  are unchanged; the AutoDL follow-up restores their missing fused RoPE dependency.
- Removed the active point-cloud PLY export function, CLI flags and call site.
  Removed active pose trajectory plotting and depth/RGB/confidence image exporters.
  KITTI now scores predictions in memory rather than emitting NPY/PNG and stopping
  before evaluation. Required metric calculations and dataset NPY readers remain.
- Fixed KITTI root handling, paired frame validation, metrics aggregation, and
  explicit alignment selection. RGB and GT mismatch is an error.
- Fixed TUM routing to `rgb_90` / `groundtruth_90.txt`, added an explicit Sintel
  training root, and removed the unnecessary FeedForward_Eval requirement.
- Removed an unused function that rewrote shared experiment-machine dataset
  symlinks and deleted shared directories. Portable entrypoints do not do this.
- Fixed Bonn/ScanNet wrapper alignment and 5 m depth defaults to follow the
  recovered table-run scripts. Both per-sequence and metric-scale modes are
  documented; these are distinct experiments and must use separate outputs.
- Failed/empty point-cloud, pose and depth runs return failure. Malformed pose
  metric files cannot become zero scores. Non-empty output directories are
  rejected to prevent stale-run contamination.
- Removed an unused v97 model-selection side effect from NeuralRGBD protocol
  setup. Restricted CLI choices that referenced absent historical models.
- Added `core`, `vpc_m`, `vpc_a` names while retaining old `ours` aliases.
- Replaced the inherited Point3R landing README with Slot3R setup/evaluation
  instructions. Historical material and original licenses remain available.

## Checks performed before point-cloud entrypoint unification

- Eleven unittest cases (all passed on AutoDL), including 24 model/dataset shell dispatch
  combinations, dataset layouts, KITTI frame pairing, failure propagation,
  stale-output protection, shell/Python syntax, no active media exporters,
  paired point sampling, and CUDA RoPE parity with/without a pose token.
- All five hashes in `MODEL_CHECKSUMS.sha256` match the source revision.
- Initial AST comparison confirmed unchanged point-cloud array collection/metrics,
  Bonn/ScanNet depth alignment/metrics, evo pose metrics, and KITTI depth helper.
  The AutoDL follow-up corrects point subsampling to use shared prediction/GT
  indices, as in the original table launcher, instead of independent samples.
- The local Windows machine has no CUDA/checkpoint/datasets; nine tests pass
  there and two dependency/GPU tests skip. All eleven pass on the AutoDL machine.

## AutoDL Core NeuralRGBD validation (2026-10-01)

Nine scenes completed with exit code 0, `kf=2`, `max_frames=200`, size 512,
999,999 paired metric points, seed 0, K=8, Sparse640, 128 anchors and q=0.25.
`thin_geometry` has only 198 sampled frames in both the historical and current
runs; the total is 1,798 frames. No prediction NPY, PLY, or images were produced.
No existing data files were deleted; about 1.1 GB remained on the data disk.

| Metric | Historical K=8 run | Current source build | Difference |
| --- | ---: | ---: | ---: |
| Acc | 0.038787 | 0.038736 | -0.000051 (-0.13%) |
| Comp | 0.017417 | 0.017595 | +0.000178 (+1.02%) |
| NC | 0.679724 | 0.679711 | -0.000012 |
| FPS | 17.958 | 19.944 | +1.986 |

These are close aggregate values, not an exact per-scene reproduction. In
particular, green_room NC is 0.632212 versus 0.647219 historically. Historical
point sampling was unseeded, and the CUDA extension was rebuilt for this GPU;
their separate contributions to the remaining differences have not been isolated.
This 200-frame check does not validate the paper's 300/400/500-frame tables.

The first complete run without fused RoPE3D scored Acc 0.039387, Comp 0.017855,
NC 0.678012 and 5.157 FPS. The original experiment directory included fused
RoPE3D sources missing from the release. Those sources were restored, with the
3D wrapper reusing the 2D wrapper's loaded extension rather than importing the
same binary under another name. A diagnostic using the old binary aborted at
shutdown; a fresh source build with the single import completed normally.
FP32 parity checks (head dimensions 64/72, with/without pose tokens) found a
maximum absolute error of 1.08e-6 against the PyTorch reference. The five
checksummed Slot3R variant source files remain byte-identical.

Environment: RTX 4090, Python 3.11.15, PyTorch 2.5.1 / CUDA 12.1, Open3D 0.19.0,
NumPy 2.4.6; the extension was compiled for sm_89. This records a working test
environment, not a validated install lock for every task. Full per-scene values,
source hashes and settings are in
[`validation/nrgbd_kf2_len200_20261001.json`](validation/nrgbd_kf2_len200_20261001.json).

## Unified point-cloud entrypoint (2026-10-01)

The six model-by-dataset launchers have been replaced by one
`eval/mv_recon/launch.py`, selected with `--model` and `--dataset`.
`data.py` owns the dataset factory and default scene lists;
`model_registry.py` owns model imports and variant settings; `metrics.py` owns
the shared point-cloud scoring pipeline. `run.sh` only translates environment
variables to CLI arguments. The old `eval/pointcloud` entrypoints are removed.
The generic `--data_root` replaces the dataset-specific `--nrgbd_root` argument.
Legacy model aliases remain accepted, while result rows use canonical model names.

The Python and shell entrypoints now share the same Sparse640 and q=0.25 defaults.
Variant settings are cleared before selecting another model. The five recorded
model source hashes remain unchanged. AST comparison against the preceding main
revision confirms that the four shared crop/sampling/collection/scoring functions
and both dataset classes are unchanged. Pose and depth pipelines are untouched.

Fifteen tests pass on AutoDL, including real module/direct-script imports of all
four shipped model classes and CUDA RoPE parity. Twelve pass locally, with three
dependency-dependent checks skipped. The real-import test covers a collision
found during the first smoke attempt: naming the registry `models.py` shadowed
CroCo's `models` package when launching a script directly. The final registry
name is `model_registry.py`.

All six real GPU smoke runs passed: Core, VPC-M and VPC-A on NeuralRGBD
`thin_geometry` and 7Scenes `heads/seq-01`, each with `kf=2`, 20 frames and the
settings above. Every run returned exit code 0 with finite Acc/Comp/NC/FPS and
produced only metrics/logs, with no visualization exports. The tested CUDA
extension was reused from the preceding successful source build after checking
source equality with line endings normalized. Source hashes, settings and
per-run metrics are recorded in
[`validation/unified_pointcloud_smoke_20261001.json`](validation/unified_pointcloud_smoke_20261001.json).
These short runs verify the new routing and execution; they do not establish
full-table numerical reproduction or meaningful benchmark FPS.

## Unified pose/depth entrypoints (2026-10-01)

Pose and depth now each have one `launch.py` and a thin `run.sh` wrapper.
Dataset selection is a CLI argument; there are no executable `sintel.py`,
`scannet_tum.py`, `bonn.py`, `scannet.py` or `kitti.py` launchers. Each task groups
input protocols, inference and scoring in `data.py`, `inference.py`, and
`metrics.py`. The per-scene pose subprocess backend is also removed.
Model names/settings are centralized in `eval/model_config.py` and the shared
pose/depth scene loop/result writer lives in `eval/runtime.py`.

The maintained pose/depth model choices are Core, VPC-M, VPC-A and Point3R.
Unused external-model and historical experiment switches from the old standalone
scripts are retired; point-cloud external-model adapters remain available.
Pose/depth results now consistently use `config.json`, `scenes.json`,
`summary.tsv`, `result.json`, `run.log` and `stats_only.log`. Failed scenes or
non-finite scores make the process fail and prevent publishing an aggregate.
Pose still emits the underlying per-scene evo metric text for inspection.

Protocol preservation checks:

- The 23 Core, 55 VPC-M and 56 VPC-A environment settings exactly match the former
  shell configuration, excluding the two obsolete module-selection variables.
- AST comparisons verify 12 unchanged numerical/inference/configuration helpers:
  indoor depth resizing, depth extraction, alignment and scoring; KITTI scoring;
  pose input construction, camera recovery and output decoding; evo result parsing;
  and the point-cloud model configuration helpers moved into the common module.
- Sintel retains RPE RMSE; ScanNet/TUM retain RPE mean. Both values remain in the
  output. ScanNet/TUM also retain their per-scene deterministic seed/model lifetime.
- Bonn/ScanNet keep their /5000 and /1000 depth decoding, no-crop resizing, 5 m
  valid-depth cap and frame-mean/scene-mean aggregation. KITTI keeps /256 decoding,
  cubic resize, original alignment helper and valid-pixel-weighted aggregation.
- All five recorded model source hashes are unchanged. All 22 tests pass in the
  AutoDL environment; 17 pass locally with five dependency-dependent checks skipped.

No prepared Sintel, ScanNet, TUM, Bonn or KITTI benchmark data was found on 13795.
Format fixtures used for GPU execution checks are six 7Scenes frames repackaged
into the five expected input layouts. These are explicitly **not benchmark data**;
their scores must not be used as paper results or evidence of reproduction.
All 18 fixture-based GPU executions passed (three Slot3R variants times six
task/dataset routes), six frames each, with finite results and no visualization
exports. Status, environment and source hashes are recorded in
[`validation/unified_pose_depth_20261001.json`](validation/unified_pose_depth_20261001.json);
fixture scores are deliberately excluded from that record.

## AutoDL verification still required

1. Install and record exact working dependency/CUDA versions; the inherited
   requirements file is not an environment lock. Run one scene per task/model,
   including the corrected TUM and KITTI branches. Commands are in `eval.md`.
2. Recover the exact 94-scene ScanNet and TUM-Dynamic evaluation manifests from
   the experiment machine. The inherited ScanNet candidate range has 100 names;
   do not silently drop failed scenes or claim a full table match from a subset.
3. Preserve and verify the inherited pose reporting convention: ScanNet/TUM's
   wrapper reads RPE **mean** from evo metric files, while Sintel summarizes RPE
   **RMSE**. This cleanup does not silently change those definitions; compare
   with the recorded table runs before normalizing them.
4. Confirm prepared RGB/depth/pose frame correspondence on Bonn/ScanNet. Those
   legacy loaders still rely on the original preprocessing/alignment convention.
5. Run full table settings only after smoke tests; compare finite metrics,
   scene/frame counts, OOM handling, FPS protocol and output artifacts against
   experiment records. The previous Core NeuralRGBD 200-frame run and the six
   unified-entrypoint point-cloud smoke runs above have completed, as have the
   18 pose/depth format-fixture executions. Pose/depth benchmark-data GPU validation,
   full-sequence VPC-M/VPC-A results, external baseline adapters and full paper
   table settings remain pending.

Legacy training/fine-tuning documentation is explicitly marked unsupported for
this Slot3R release. Old experiment scripts are archival evidence, not portable
alternatives to the maintained `eval/*/run.sh` entrypoints.
