#!/usr/bin/env bash
# Source only after OUTPUT_DIR is set. Refuse to overwrite prior results.
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "OUTPUT_DIR is not empty; choose a new directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
