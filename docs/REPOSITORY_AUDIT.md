# Evaluation cleanup audit

Source revision: `b94b525b3627ca72165417c0406db7d95fea6fc8`.

## Implemented

- Canonical table entrypoints now live under `eval/pointcloud`, `eval/pose` and
  `eval/depth`. Old Point3R launchers and experiment-machine scripts are archived
  separately with their provenance. No model source file was changed.
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

## Checks performed locally

- Nine CPU-only unittest cases, including 24 model/dataset shell dispatch
  combinations, dataset layouts, KITTI frame pairing, failure propagation,
  stale-output protection, shell/Python syntax, and no active media exporters.
- All five hashes in `MODEL_CHECKSUMS.sha256` match the source revision.
- AST comparison confirms unchanged point-cloud array collection/metrics,
  Bonn/ScanNet depth alignment/metrics, evo pose metrics, and KITTI depth helper.
- No checkpoint, datasets or CUDA environment are available locally. These checks
  do not establish GPU runtime correctness or reproduce the paper's numbers.

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
   experiment records. No GPU numerical result has been claimed yet.

Legacy training/fine-tuning documentation is explicitly marked unsupported for
this Slot3R release. Old experiment scripts are archival evidence, not portable
alternatives to the maintained `eval/*/run.sh` entrypoints.
