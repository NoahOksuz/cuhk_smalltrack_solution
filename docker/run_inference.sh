#!/usr/bin/env bash
# CUHK-X Small Model Track — ONE-COMMAND inference.
#
#   ./run_inference.sh /path/to/testing_dir
#
# Runs the prebuilt image on your GPU, finds the SM_test_XXXX/ clips anywhere
# under the given path, and writes submission.csv into your CURRENT folder.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMG="${CUHK_INFER_IMAGE:-cuhk-small-infer:latest}"

if [ "$#" -lt 1 ]; then
  cat <<EOF
Usage: $0 /path/to/testing_dir

  /path/to/testing_dir may be the small_model_track_test dir or any parent
  of it (e.g. .../Testing); the SM_test_XXXX/ clips are located automatically.

Output: $(pwd)/submission.csv  (your current folder)

Build the image first (one time, from the repository root):
      docker build -f "$HERE/Dockerfile" -t cuhk-small-infer:latest "$ROOT"
EOF
  exit 1
fi

docker run --rm \
  --gpus all \
  -v "${1%/}":/data:ro \
  -v "$(pwd)":/out \
  "$IMG"

echo "Done. Wrote: $(pwd)/submission.csv"
