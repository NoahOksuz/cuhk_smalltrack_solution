# CUHK-X Challenge 2026 | Small Model Track

**Noah Deniz Oksuz's solution** for the [CUHK-X Small Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track): four R(2+1)D-34 video members, person and wrist detection, and a recording-structure decoder. The full method, validation and limitations are in [the technical report](cuhk_PAPER.pdf).

| Result | Value |
| --- | --- |
| Public leaderboard | 0.89552 (180/201; 11th) |
| Private leaderboard | 0.90196 (184/204; 8th) |
| Held-out-subject CV | 0.7770 raw / 0.8521 decoded (3,036 clips) |
| Shipped model assets | 98,451,640 bytes (video members + two YOLO11 programs + transition prior; under the 100 MB cap) |
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
  sota_torch_yolo.py    runs the YOLO11 models as torch.export programs
  sota_onnx_yolo.py     YOLO pre-/post-processing (letterbox, NMS, rescaling)
  checkpoints/
    video_members.pt    the 4 video members (packed, lzma-delta-coded)
    yolo11n.pt2.xz      YOLO11n person detector (torch.export program, AGPL-3.0)
    yolo11n-pose.pt2.xz YOLO11n-pose (torch.export program, AGPL-3.0)
    unified_model.pt    original submission checkpoint; export-tool input only, not in the Docker image
  report.json           member spec, decoder config, asset sha256
  sequence_prior.json   train-only transition prior
  sample_submission.csv bundled Kaggle template
tools/
  export_yolo_torch.py  offline: unified_model.pt -> the three files above (AGPL-3.0)
final_submission.csv    recorded test-set predictions (405 rows)
submission.csv          identical copy supplied with the original submission
docker/                 Dockerfile + optional host wrapper; installs `code/`
cuhk_PAPER.pdf          technical report (method, CV, efficiency and compliance)
LICENSE                 Apache License 2.0 (original work)
NOTICE                  attribution notices
THIRD_PARTY_LICENSES    third-party scope by path + full MIT / AGPL-3.0 texts
```

There is one inference implementation. Native inference installs `code/`
through `pyproject.toml`; Docker copies and installs that same tree. The
Ultralytics package is not needed to run it: the detectors load as
`torch.export` programs with plain PyTorch.

## 1. Native inference (Python package)

### Environment
Linux x86_64, NVIDIA GPU, and Python 3.13 with:

```
torch==2.14.0            (CUDA 13.0 build, from https://download.pytorch.org/whl/cu130)
torchvision==0.29.0      (cu130)
numpy==2.5.2
scipy==1.18.1
pillow==12.3.0
opencv-python-headless==5.0.0.93
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

After the detectors were moved from Ultralytics to `torch.export` programs,
the full 405-clip inference was rerun natively on an RTX 5090 with the
`ultralytics` and `onnxruntime` modules blocked from importing. The CSV is
byte-identical to `final_submission.csv`, and every intermediate (person
crops, pose hand crops, feature caches, member probabilities) matches the
original Ultralytics run exactly. Wall time was 85.7 s. The Docker image has
not been rebuilt since that change.

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
weights released by `facebookresearch/VMZ` (Apache-2.0), as converted to
PyTorch by the MIT-licensed `moabitcoin/ig65m-pytorch`, whose architecture
corrections `code/sota_model.py` follows (license in `code/IG65M_LICENSE.txt`). This is a
63.5 M-parameter video backbone; the "no large pretrained backbones" clause of
the Small Model Track is left to the organizers' interpretation, and we declare
it here. The 100 MB model-size cap is the binding constraint and is met.

## License

Original work in this repository is licensed under the
[Apache License 2.0](LICENSE). Third-party components keep their own licenses
and are **not** covered by that grant:

| Path | License | What it is |
| --- | --- | --- |
| everything not listed below | Apache-2.0 | original code, configs, docs, recorded outputs |
| `code/checkpoints/video_members.pt` | Apache-2.0 | original fine-tuned weights; initialised from IG65M→Kinetics weights (VMZ, Apache-2.0; PyTorch conversion by ig65m-pytorch, MIT) |
| `code/sota_model.py` | Apache-2.0, with MIT notice | architecture corrections adapted from `moabitcoin/ig65m-pytorch` |
| `code/IG65M_LICENSE.txt` | MIT | original ig65m-pytorch license notice |
| `code/checkpoints/yolo11n.pt2.xz` | AGPL-3.0 | Ultralytics YOLO11n pretrained model, exported with `torch.export` |
| `code/checkpoints/yolo11n-pose.pt2.xz` | AGPL-3.0 | Ultralytics YOLO11n-pose pretrained model, exported with `torch.export` |
| `code/checkpoints/unified_model.pt` | AGPL-3.0 and Apache-2.0 | original submission checkpoint (YOLO11 weights + video members); export-tool input only |
| `tools/export_yolo_torch.py` | AGPL-3.0 | offline export tool; the only file that imports Ultralytics |

The Ultralytics package is not a runtime dependency. Inference uses only the
pretrained YOLO11 models, loaded with `torch.export.load`; letterboxing,
non-maximum suppression and coordinate rescaling are implemented in
`code/sota_onnx_yolo.py` under Apache-2.0. Full license texts and
attributions are in [`NOTICE`](NOTICE) and
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES). See the
[technical report](cuhk_PAPER.pdf) for the data-use, pretraining and detector
declarations.
