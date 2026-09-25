# CUHK-X Small Model Track — Submission Inference (Docker)

Self-contained inference image for the CUHK-X Challenge 2026, **Small Model
Track**. It runs the top-15 verification release
(`release_joint_interact_w69_guard`) on a directory of raw test-clip folders
and writes a Kaggle `submission.csv`.

## What the image contains

- **All model weights in a single file**, `unified_model.pt`: the 4 R(2+1)D-34
  members (1 body+hands joint with YOLO pose hand-crops, 1 depth+IR 160px,
  1 thermal, 1 IR-only), group-quantized (int6/int5) and losslessly
  lzma-delta-coded, **plus** the **YOLO11n** person detector and
  **YOLO11n-pose** (hand crops). Total model assets **96,058,917 bytes**
  (≤ the 100 MB track cap). The entrypoint loads everything from this one
  file and verifies its sha256 against `report.json`.
- The **train-only transition prior** (`sequence_prior.json`).
- **No training data, no downloads at inference.** `sota_model.Model()` is
  built with `pretrained=False`, so the IG65M hub checkpoint is never loaded.

## Build

Build from the repository root, not from `docker/`, because the image installs
the same `pyproject.toml` + `code/` package used by native inference:

```bash
docker build -f docker/Dockerfile -t cuhk-small-infer:latest .
```

The base is `python:3.13-slim-bookworm` with
**torch 2.14.0 + CUDA 13.0** (pinned; matches the RTX 5090 sm_120 host that
produced the recorded submission) plus `torchvision 0.29.0`, `numpy 2.5.2`,
`scipy 1.18.1`, `pillow 12.3.0`, `ultralytics 8.4.142`. OpenCV is pulled
transitively by ultralytics; the base image ships the X/GL shared libraries
cv2 dlopen's.

## Run

```bash
docker run --rm --gpus all -v /path/to/testing_dir:/data:ro -v "$(pwd)":/out cuhk-small-infer:latest
```

That's all. The image locates the `SM_test_XXXX/` clip folders automatically
(anywhere under `/path/to/testing_dir`, up to 5 levels deep — the clips root
or any parent of it, e.g. `.../Testing`) and writes `submission.csv` into
`$(pwd)`, your current folder.

`run_inference.sh /path/to/testing_dir` is a one-line wrapper that runs
exactly the `docker run` above.

## Verification (reproduced)

On the full 405-clip public test set, fresh per-clip detection, no cached
boxes:

| metric | value |
| --- | --- |
| rows | 405 |
| hands view | ok 399 / capped 1 / fallback 1 / no IR 4 |
| decoder guard | 1 of 151 takes fell back |
| wall time | ~105 s |
| peak GPU | 1.72 GB |
| `output_sha256` | `f77d1d28d9797ce8fd49916bc5d18201371765ec682d0b3b5e23c3878ebc3893` |

The container's output is **byte-identical** (`cmp`) to the recorded
`final_submission.csv`, i.e. the exact submission that scored
**0.89552 (180/201, 11th)**. Docker's `CMD` invokes the same installed Python
CLI as native inference: `samhar /data /out/submission.csv`.

## Layout

The Docker directory contains only Docker-specific files:

```
docker/
  Dockerfile          installs the repository-root Python package
  requirements.txt    pinned non-torch runtime dependencies
  run_inference.sh    optional host-side docker-run wrapper
```

The canonical implementation and only checkpoint live outside this directory:

```
pyproject.toml
code/
  cuhk_infer/cli.py
  sota_*.py
  checkpoints/unified_model.pt
  report.json
  sequence_prior.json
  sample_submission.csv
```

The Docker build copies `code/` once into the image; the archive does not keep
a second Docker-specific checkpoint or source copy.

## Notes for the technical report

- Backbone: R(2+1)D-34 fine-tuned from the IG65M→Kinetics clip-32 init.
  The temporal strides of the six (2,1,1) convolutions in layers 2–4 are set
  to 1 at inference (`strideless`) so all 32 input frames keep a feature
  position — the largest single generalization win in this work.
- Fusion: 4 members combined with fixed weights (0.69 / 0.12 / 0.07 / 0.12),
  missing-modality masking, then a recording-structure decoder with a
  per-take fallback guard.
- The decoder exploits the thermal frame-counter / timestamp repeat structure
  of the scripted recordings. It is the part most likely to degrade on the
  organizer's unseen 8-participant data; the guard (and the strong raw base)
  bound that risk.
