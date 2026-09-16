# Evaluation launchers

The repository separates maintained launchers from immutable experiment
references:

- `pointcloud/`: portable 7Scenes and NeuralRGBD point-cloud entry points.
- `pose_depth/`: portable ScanNet, Sintel, TUM-Dynamic, Bonn, and KITTI entry
  points.
- `reference/`: exact scripts recovered from the two evaluation machines.
  They are retained for provenance and may contain machine-specific paths.

Generated results, point clouds, NumPy arrays, trajectories, frustums,
checkpoints, datasets, and logs are excluded by the repository `.gitignore`.

The portable wrappers are still under reproducibility review. Until that audit
is complete, use the reference scripts only to compare configuration values,
not as copy-paste commands on a new machine.
