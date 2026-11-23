"""Offline build step: export the bundled YOLO11 detectors to ONNX.

This is the only script in the repository that imports Ultralytics, and it is
not part of inference. It reads the stock YOLO11n / YOLO11n-pose checkpoints
stored in the original unified checkpoint and writes plain ONNX graphs that
`code/sota_onnx_yolo.py` runs with ONNX Runtime. It also writes
`video_members.pt`, the four video members alone, which loads with
`torch.load(..., weights_only=True)` and needs no Ultralytics classes.

Precisions mirror the original pipeline: the person detector ran in fp32 and
the pose model in fp16 (`quantize=16`).

    python tools/export_yolo_onnx.py code/checkpoints/unified_model.pt code/checkpoints

Requires `ultralytics==8.4.142`, `onnx` and a CUDA GPU (fp16 export).
"""
import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import torch

EXPORTS = [
    # (checkpoint key, output file, fp16)
    ('yolo11n', 'yolo11n.onnx', False),
    ('yolo11n-pose', 'yolo11n-pose.onnx', True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('unified', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--extra-fp32-pose', action='store_true', help='also write yolo11n-pose-fp32.onnx')
    ap.add_argument('--opset', type=int, default=None)
    args = ap.parse_args()
    from ultralytics import YOLO

    ck = torch.load(args.unified, map_location='cpu', weights_only=False)
    jobs = list(EXPORTS)
    if args.extra_fp32_pose:
        jobs.append(('yolo11n-pose', 'yolo11n-pose-fp32.onnx', False))
    args.out.mkdir(parents=True, exist_ok=True)
    video = {k: v for k, v in ck.items() if k.startswith('video_')}
    torch.save(video, args.out / 'video_members.pt')
    torch.load(args.out / 'video_members.pt', map_location='cpu', weights_only=True)
    written = ['video_members.pt']
    for key, name, half in jobs:
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / f'{key}.pt'
            torch.save(ck[key], pt)
            onnx_path = YOLO(str(pt)).export(format='onnx', imgsz=640, dynamic=True, half=half,
                                             simplify=True, device=0, verbose=False,
                                             **({'opset': args.opset} if args.opset else {}))
            shutil.copyfile(onnx_path, args.out / name)
        written.append(name)
    for name in written:
        data = (args.out / name).read_bytes()
        print(f'{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}')


if __name__ == '__main__':
    main()
