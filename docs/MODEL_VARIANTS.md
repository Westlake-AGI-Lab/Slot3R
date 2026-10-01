# Model variants and provenance

This file is the canonical mapping between paper labels, historical experiment
names, source files, and verified SHA-256 hashes.

| Paper label | Historical name | Source file under `src/dust3r/` | SHA-256 |
| --- | --- | --- | --- |
| Slot3R (Core), `core` (formerly Ours) | base K-way / ConfSelect | `point3r_kway_frame_sparse_q35_confselect.py` | `87373a6e7705b6478a9598c0e8be1b3b232168d96fee3c1cbd06964e6221d145` |
| Slot3R-VPC-M, `vpc_m` (formerly Ours-Ray) | v82e / v82k | `point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose.py` | `6614815354b4b65500e4cf254f1d8e87d3156a322419e4fd5c87dd8ac4b7fd18` |
| Slot3R-VPC-A, `vpc_a` (formerly Ours-RayMA) | v106 | `point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose.py` | `0d38c0cdca24d778fb0f2b856d0fea1a5cab5c17c65dc01b912fb9f397c55209` |

## Important v82e disambiguation

Two files with the same historical v82e filename existed on the point-cloud
machine. The paper runs and corrected teaser exports record the `661481...`
source hash above. The other `a7cba2...` file is an obsolete working copy and
must not be used as Ours-Ray.

## Ours-RayMA dependency chain

The v106 file is a small reproducibility wrapper around v102. Its import chain
is:

```text
v106_fresh_bank_pose
  -> v102_complementary_jerk
     -> v101_safe_translation
        -> v82e_balanced_predecoder_pose
```

All four files are therefore retained. Historical pose scripts sometimes
loaded v102 directly and set `POINT3R_RAY_BANK_UPDATE_EVERY=1` plus equal
`POINT3R_V102_SMOOTH_WEIGHT` and `POINT3R_V102_JERK_WEIGHT` values of `0.15`.
That configuration is behaviorally represented by the v106 wrapper, which is
the canonical Ours-RayMA entry point in this repository.

## Checkpoint

These variants reuse the pretrained Point3R checkpoint. Checkpoint binaries
are intentionally excluded from Git and must be supplied separately.
