"""Run the bundled YOLO11 models as `torch.export` programs on plain PyTorch.

The `.pt2` programs (tools/export_yolo_torch.py) hold the exported ATen graph
and the stock Ultralytics weights (AGPL-3.0, recorded in each program's
config.txt). Loading them needs only `torch.export.load`; no Ultralytics code
is imported. The graph runs op by op on PyTorch's own kernels in the original
precision (fp32 detector, fp16 pose), so it reproduces the original
pipeline's numbers bit for bit.

The pre- and post-processing are shared with sota_onnx_yolo. Programs are
exported for any batch size but one input size (480x640 for these IR frames);
frames whose minimum-rectangle letterbox differs from it are letterboxed to
the exported size. `.xz` files are decompressed in memory.
"""
import io
import json
import lzma
from pathlib import Path

import cv2
import torch

from sota_onnx_yolo import OnnxYolo, PAD_VALUE


class TorchYolo(OnnxYolo):
    def __init__(self, path, device=0, conf=.25, iou=.7, classes=None, max_det=300):
        use_cuda = device is not None and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{int(device)}' if use_cuda else 'cpu')
        data = Path(path).read_bytes()
        if str(path).endswith('.xz'):
            data = lzma.decompress(data)
        extra = {'config.txt': ''}
        program = torch.export.load(io.BytesIO(data), extra_files=extra)
        self.net = program.module().to(self.device)
        meta = json.loads(extra['config.txt'])
        self.meta = meta
        self.fp16 = meta['precision'] == 'fp16'
        self.stride = int(meta['stride'])
        self.kpt_shape = meta.get('kpt_shape')
        self.shape = tuple(meta['imgsz'])                  # exported (h, w)
        self.imgsz = max(self.shape)
        self.conf, self.iou, self.classes, self.max_det = conf, iou, classes, max_det

    def _letterbox(self, img, auto):
        out = super()._letterbox(img, auto)
        if out.shape[:2] == self.shape:
            return out
        h, w = img.shape[:2]                               # fixed-size fallback to the exported shape
        r = min(self.shape[0] / h, self.shape[1] / w)
        nw, nh = round(w * r), round(h * r)
        dw, dh = (self.shape[1] - nw) / 2, (self.shape[0] - nh) / 2
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(img, round(dh - .1), round(dh + .1), round(dw - .1), round(dw + .1),
                                  cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)

    @torch.inference_mode()
    def predict(self, frames, batch=1):
        results = []
        for s in range(0, len(frames), batch):
            chunk = frames[s:s + batch]
            auto = len({f.shape for f in chunk}) == 1
            x = self._tensor([self._letterbox(f, auto) for f in chunk])
            raw = self.net(x)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            for k, f in enumerate(chunk):
                results.append(self._result(self._nms(raw[k]), x.shape[2:], f.shape[:2]))
        return results
