# CUHK-X Challenge 2026 | Small Model Track

**Noah Deniz Oksuz's solution** for the [CUHK-X Small Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track): four R(2+1)D-34 video members, person and wrist detection, and a recording-structure decoder. The full method, validation and limitations are in [the technical report](cuhk_PAPER.pdf).

| Result | Value |
| --- | --- |
| Public leaderboard | 0.89552 (180/201; 11th) |
| Private leaderboard | 0.90196 (184/204; 8th) |
| Held-out-subject CV | 0.7770 raw / 0.8521 decoded (3,036 clips) |
| Shipped model assets | 96,081,702 bytes (checkpoint + transition prior; under the 100 MB cap) |
| Recorded predictions | [`final_submission.csv`](final_submission.csv), SHA-256 `f77d1d28d9797ce8fd49916bc5d18201371765ec682d0b3b5e23c3878ebc3893` |

## Quick start (Docker)

Requires Docker, NVIDIA Container Toolkit and an NVIDIA GPU. Build from the repository root:

```bash
docker build -f docker/Dockerfile -t cuhk-small-infer:latest .
docker run --rm --gpus all \
  -v /path/to/testing_dir:/data:ro \
  -v "$(pwd)":/out \
  cuhk-small-infer:latest
```

The container discovers `SM_test_XXXX/` folders beneath `/data` and writes `submission.csv` in the current directory. Use a separate output directory to avoid overwriting the recorded CSV. The supplied 405-row template is for the released test set; for another evaluation set, use the native CLI with `--template` and a matching clip directory. See [Docker details](docker/README.md) and [native installation](#1-native-inference-python-package).

---

## Layout

```
pyproject.toml          install metadata + `samhar` / `cuhk-infer` commands
inference.sh            compatibility wrapper that calls `samhar "$@"`
code/                   single canonical inference package
  cuhk_infer/           Python CLI
  sota_*.py             model, decoders, data, ensemble, joint
  checkpoints/
    unified_model.pt    all 4 video members + both YOLO detectors (one file)
  report.json           member spec, decoder config, checkpoint sha256
  sequence_prior.json   train-only transition prior
  sample_submission.csv bundled Kaggle template
final_submission.csv    recorded test-set predictions (405 rows)
submission.csv          identical copy supplied with the original submission
docker/                 Dockerfile + optional host wrapper; installs `code/`
cuhk_PAPER.pdf          technical report (method, CV, efficiency and compliance)
```

There is one inference implementation and one checkpoint in the archive.
Native inference installs `code/` through `pyproject.toml`; Docker copies and
installs that same tree.

## 1. Native inference (Python package)

### Environment
Linux x86_64, NVIDIA GPU, and Python 3.13 with:

```
torch==2.14.0            (CUDA 13.0 build, from https://download.pytorch.org/whl/cu130)
torchvision==0.29.0      (cu130)
numpy==2.5.2
scipy==1.18.1
pillow==12.3.0
ultralytics==8.4.142
pi-heif==1.4.0
```

Install from the repository root:

```
pip install torch==2.14.0 torchvision==0.29.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

Run in the simple `INPUT [OUTPUT]` form:

```
samhar /path/to/testing_dir
samhar /path/to/testing_dir output.csv

# Equivalent compatibility wrapper:
./inference.sh /path/to/testing_dir [output.csv]
```

`cuhk-infer` is an equivalent alias. `testing_dir` may be the directory that
directly contains `SM_test_XXXX/`, or any parent up to five levels above it
(for example `.../Testing`). If omitted, `OUTPUT` defaults to
`./submission.csv` in the current directory.

The command performs per-clip YOLO detection on the raw IR frames, decodes the
four modality caches, ensembles, and runs the structure decoder. No training
data and no network downloads are used at inference.

## 2. Docker inference (turnkey, recommended for verification)

Build once from the repository root (the root build context is required so
Docker can copy the same `code/` tree used by native inference):

```
docker build -f docker/Dockerfile -t cuhk-small-infer:latest .
```

Run (NVIDIA Container Toolkit; `--gpus all`). The image invokes
`samhar /data /out/submission.csv`, locates the clip root automatically, and
writes `submission.csv` into `$(pwd)`, your current folder:

```
docker run --rm --gpus all \
  -v /path/to/testing_dir:/data:ro \
  -v "$(pwd)":/out \
  cuhk-small-infer:latest
```

Or use the host wrapper:

```
bash docker/run_inference.sh /path/to/testing_dir
```

The image is `python:3.13-slim-bookworm` + torch 2.14.0 cu130 (pinned to the
exact build used to produce the recorded submission) + the pinned deps above,
with the X/GL shared libraries OpenCV dlopen's.

## Reproduction result (recorded verification)

The supplied verification package reports the following result on the full
405-clip released test set with fresh raw-folder input, per-clip detection and
no cached boxes. The CSV and artifact checksums were independently checked
when preparing this repository; the full GPU inference was **not rerun** as
part of the GitHub packaging.

| metric | value |
| --- | --- |
| rows | 405 |
| hands view | ok 399 / capped 1 / fallback 1 / no IR 4 |
| decoder guard | 1 of 151 takes fell back |
| wall time | ~83 s (native) / ~87–105 s (Docker) |
| peak GPU memory | 1.71 GB |
| `output_sha256` | `f77d1d28…78ebc3893` |

Both the native and Docker paths produce a `submission.csv` that is
**byte-identical** (`cmp`) to `final_submission.csv`. The pipeline is
deterministic on a fixed GPU/driver, so a rerun on the organizer's fresh
sample data will land well within 10% of the private score (the public set
itself reproduces exactly).

## Model efficiency

- **4 members**, each R(2+1)D-34 (63.5 M params) fine-tuned from the IG65M→
  Kinetics clip-32 init, then **group-quantized** (int6 for the joint body+
  hands member, int5 for the other three) and **losslessly lzma-delta-coded**
  against the first member. Total 96.08 MB.
- Inference runs at **fp16** with horizontal-flip TTA on a single consumer
  GPU (RTX 5090); peak ~1.7 GB.
- The IG65M hub checkpoint is **not** loaded at inference — `Model()` is built
  `pretrained=False`, so there is no download and no large backbone shipped.

## Data usage declaration (required by the rules)

- **Small Model Track Training set**: all 3,036 clips used for the all-data
  fits. The `sequence_prior.json` transition prior is fit on **Training labels
  only** (never test labels).
- **Large Model Track data**: the organizer permitted the public Large-Model-
  Track data for pre-training/co-training in the Small Model Track. We
  audited it for leakage (it contains only the 18 Small-track users; the test
  users appear nowhere) and **did not use it in any shipped model** — it added
  no new subjects and almost no new labeled action footage. Declared here for
  completeness; reproducible without it.
- **No test labels** were used in any form.

## IG65M backbone note

The R(2+1)D-34 is initialized from the IG65M→Kinetics-clip-32 pretrained
weights (architecture corrections per the MIT-licensed
`moabitcoin/ig65m-pytorch`; license in `code/IG65M_LICENSE.txt`). This is a
63.5 M-parameter video backbone; the "no large pretrained backbones" clause of
the Small Model Track is left to the organizers' interpretation, and we declare
it here. The 100 MB model-size cap is the binding constraint and is met.

## Licensing and compliance

This private repository preserves the submission artifacts and report; it is
not a claim that third-party components are proprietary. The IG65M-derived
architecture is attributed in [`code/IG65M_LICENSE.txt`](code/IG65M_LICENSE.txt).
The bundled checkpoint includes stock Ultralytics YOLO11n and YOLO11n-pose
weights; Ultralytics is distributed under AGPL-3.0. Consult the [technical
report](cuhk_PAPER.pdf) for the complete data-use, pretraining and detector
declarations before redistributing or changing the visibility of this repo.
No signed honor statement was present in the supplied folder; add one only if
the organizers require it and it has actually been signed.
