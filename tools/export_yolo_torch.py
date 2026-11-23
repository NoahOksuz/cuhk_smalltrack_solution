# SPDX-License-Identifier: AGPL-3.0-only
# This tool imports Ultralytics (AGPL-3.0) and is distributed under AGPL-3.0;
# see THIRD_PARTY_LICENSES. It is not part of the Apache-2.0 inference package.
"""Offline build step: save the bundled YOLO11 models as `torch.export` programs.

This is the only place Ultralytics is imported, and it is not part of
inference. It builds each model exactly as `YOLO.predict` does (BatchNorm
folded in the stored precision, then cast to the run precision: fp32 detector,
fp16 pose) and exports *that* module with `torch.export` (dynamic batch).
Ultralytics' own export paths cast to fp32 before folding, which gives
slightly different weights, so they are not used.

Outputs:
  yolo11n.pt2.xz, yolo11n-pose.pt2.xz   (xz-compressed programs)
  video_members.pt                      (video members only, weights_only-loadable)

    python tools/export_yolo_torch.py code/checkpoints/unified_model.pt code/checkpoints

Requires `ultralytics==8.4.142`, torch 2.14 and a CUDA GPU. Programs are
exported at the IR frame letterbox size, 480x640.
"""
import argparse
import hashlib
import json
import lzma
import tempfile
from pathlib import Path

import numpy as np
import torch

SHAPE = (480, 640)
EXPORTS = [('yolo11n', False), ('yolo11n-pose', True)]   # (checkpoint key, fp16)


class _Prediction(torch.nn.Module):
    """Return only the prediction tensor (drop the per-level feature maps)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        y = self.m(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('unified', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    from ultralytics import YOLO

    ck = torch.load(args.unified, map_location='cpu', weights_only=False)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in ck.items() if k.startswith('video_')}, args.out / 'video_members.pt')
    written = ['video_members.pt']
    dummy = np.zeros((*SHAPE, 3), np.uint8)
    for key, half in EXPORTS:
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / f'{key}.pt'
            torch.save(ck[key], pt)
            yolo = YOLO(str(pt))
            yolo.predict([dummy], imgsz=640, device=0, verbose=False, **({'quantize': 16} if half else {}))
            model = yolo.predictor.model.model.eval()      # the exact module predict() runs
        head = model.model[-1]
        head.dynamic = True        # anchors from (h, w) each call (same values), so batch can stay symbolic
        x = torch.zeros(2, 3, *SHAPE, device='cuda:0', dtype=torch.float16 if half else torch.float32)
        with torch.inference_mode():
            program = torch.export.export(_Prediction(model), (x,), strict=False,
                                          dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=64)}})
        meta = {
            'description': f'Ultralytics {key.upper()} (stock COCO weights), exported from the YOLO.predict module',
            'author': 'Ultralytics', 'license': 'AGPL-3.0 (https://ultralytics.com/license)',
            'ultralytics_version': ck[key].get('version'), 'stride': int(max(model.stride)),
            'task': 'pose' if hasattr(head, 'kpt_shape') else 'detect', 'imgsz': list(SHAPE),
            'precision': 'fp16' if half else 'fp32',
        }
        if hasattr(head, 'kpt_shape'):
            meta['kpt_shape'] = list(head.kpt_shape)
        name = f'{key}.pt2.xz'
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'model.pt2'
            torch.export.save(program, str(raw), extra_files={'config.txt': json.dumps(meta)})
            (args.out / name).write_bytes(lzma.compress(raw.read_bytes(), preset=9 | lzma.PRESET_EXTREME))
        written.append(name)
    for name in written:
        data = (args.out / name).read_bytes()
        print(f'{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}')


if __name__ == '__main__':
    main()
