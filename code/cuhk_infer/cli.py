"""cuhk-infer CLI: CUHK-X Small Model Track inference.

SAM-style usage:
    cuhk-infer /path/to/testing_dir            # writes ./submission.csv
    cuhk-infer /path/to/testing_dir out.csv    # writes to out.csv

The input path may be the small_model_track_test dir or any parent of it
(e.g. .../Testing); the SM_test_XXXX/ clips are located automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make the project root (which holds the flat sota_*.py inference modules and
# sota_release_infer2.py) importable, no matter how we were launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sota_release_infer2 import main as release_main  # noqa: E402

TEMPLATE = _PROJECT_ROOT / "sample_submission.csv"


def find_clips_root(start: Path, max_depth: int = 5) -> Path:
    """Locate the dir that directly holds SM_test_0001, searching up to max_depth
    levels under `start`. Returns that dir (the clips root)."""
    start = Path(start)
    if not start.is_dir():
        raise SystemExit(f"error: {start} is not a directory")
    for d in start.rglob("SM_test_0001"):
        if d.is_dir() and len(d.relative_to(start).parts) <= max_depth:
            return d.parent
    raise SystemExit(f"error: no SM_test_0001 folder found under: {start}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CUHK-X Small Model Track inference: raw test clips -> submission.csv",
    )
    ap.add_argument("input", type=Path,
                    help="test dir (or any parent of the small_model_track_test dir)")
    ap.add_argument("output", type=Path, nargs="?", default=Path("submission.csv"),
                    help="output CSV path (default: ./submission.csv)")
    ap.add_argument("--template", type=Path, default=TEMPLATE,
                    help="Kaggle template csv (default: the bundled one)")
    ap.add_argument("--package", type=Path, default=None,
                    help="package dir holding checkpoints/unified_model.pt + report.json "
                         "(default: the package root)")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch dir for decoded caches (default: a temp dir)")
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA device index (default: 0)")
    return ap


def run(args: argparse.Namespace) -> None:
    clips = find_clips_root(args.input)
    output = args.output.resolve()
    work = args.work or Path(tempfile.mkdtemp(prefix="cuhk_infer_"))
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    package = (args.package or _PROJECT_ROOT).resolve()

    print(f"clips:    {clips}", flush=True)
    print(f"template: {args.template}", flush=True)
    print(f"output:   {output}", flush=True)

    # Reuse the release entrypoint verbatim so the native and Docker paths stay
    # identical.
    old_argv = sys.argv
    sys.argv = [
        "sota_release_infer2.py",
        "--package", str(package),
        "--input", str(clips),
        "--template", str(args.template),
        "--work", str(work),
        "--output", str(output),
    ]
    try:
        release_main()
    finally:
        sys.argv = old_argv


def main(argv=None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
