# Historical source archive

These files preserve provenance from commit
`b94b525b3627ca72165417c0406db7d95fea6fc8` before evaluation cleanup.

- `experiment_reference/`: exact recovered experiment-machine scripts.
- `point3r_eval/`: retired upstream launchers and unused metric/export helpers.
- `POINT3R_README.md`, `LAUNCHERS_ORIGINAL.md`: previous documentation.

Do not run these as release entrypoints. They contain machine-specific paths,
legacy visualization/export code and assumptions that do not apply to a clean
checkout. Maintained table evaluation is documented in `../docs/eval.md` and
lives under `../eval/`. Metric algorithms were preserved rather than discarded
while relocating the old entrypoints.
