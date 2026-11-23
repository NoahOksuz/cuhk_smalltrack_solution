"""ONNX Runtime inference for the bundled person detector and pose model.

Runs the exported YOLO11n / YOLO11n-pose graphs (see tools/export_yolo_onnx.py)
without importing Ultralytics. The steps the pipeline depends on are written
here from their definitions:

1. aspect-preserving resize to a 640 long side, padded with grey (114) up to
   the next multiple of the model stride (32), centred;
2. one forward pass;
3. best-class confidence filter, optional class filter, per-class NMS
   (IoU 0.7, at most 300 detections);
4. boxes and keypoints mapped back to the source frame and clipped to it.

Frames are passed as HWC uint8 arrays and, like the original pipeline, their
channel order is reversed before inference. Post-processing runs in torch on
the inference device and in the graph's precision so the arithmetic follows
the original fp32 detector / fp16 pose runs.
"""
import ast
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

PAD_VALUE = 114
MAX_WH = 7680          # class offset for per-class NMS
MAX_NMS = 30000        # candidates kept before NMS


@dataclass
class Detections:
    """Detections for one frame; arrays are numpy in the graph's precision."""
    orig_shape: tuple        # (h, w)
    xyxy: np.ndarray         # [n, 4] pixels
    xyxyn: np.ndarray        # [n, 4] normalised by (w, h)
    conf: np.ndarray         # [n]
    cls: np.ndarray          # [n]
    kpt_xyn: np.ndarray | None = None   # [n, k, 2]
    kpt_conf: np.ndarray | None = None  # [n, k]

    def __len__(self):
        return len(self.conf)


class OnnxYolo:
    def __init__(self, path, device=0, imgsz=640, conf=.25, iou=.7, classes=None, max_det=300, tf32=True):
        if device is None or not torch.cuda.is_available():
            providers, self.device = ['CPUExecutionProvider'], torch.device('cpu')
        else:
            providers = [('CUDAExecutionProvider', {'device_id': int(device), 'use_tf32': int(bool(tf32)),
                                                    'cudnn_conv_algo_search': 'DEFAULT'}),
                         'CPUExecutionProvider']
            self.device = torch.device(f'cuda:{int(device)}')
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fp16 = inp.type == 'tensor(float16)'
        meta = self.session.get_modelmeta().custom_metadata_map
        stride = ast.literal_eval(meta.get('stride', '32'))
        self.stride = int(max(stride) if isinstance(stride, (list, tuple)) else stride)
        self.kpt_shape = ast.literal_eval(meta['kpt_shape']) if 'kpt_shape' in meta else None
        self.imgsz, self.conf, self.iou, self.classes, self.max_det = imgsz, conf, iou, classes, max_det

    # -- pre-processing -------------------------------------------------
    def _letterbox(self, img, auto):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        dw, dh = self.imgsz - nw, self.imgsz - nh
        if auto:
            dw, dh = dw % self.stride, dh % self.stride
        dw, dh = dw / 2, dh / 2
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - .1), round(dh + .1)
        left, right = round(dw - .1), round(dw + .1)
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)

    def _tensor(self, padded):
        x = torch.from_numpy(np.stack(padded)).to(self.device)
        x = x.permute(0, 3, 1, 2).flip(1).contiguous()
        return (x.half() if self.fp16 else x.float()).div_(255)

    # -- post-processing ------------------------------------------------
    @staticmethod
    def _xywh2xyxy(x):
        y = torch.empty_like(x)
        xy, wh = x[..., :2], x[..., 2:] / 2
        y[..., :2] = xy - wh
        y[..., 2:] = xy + wh
        return y

    def _nms(self, pred):
        """pred [C, N] -> [n, 6 + extra] rows (x1, y1, x2, y2, conf, cls, extra...)."""
        nk = int(np.prod(self.kpt_shape)) if self.kpt_shape else 0
        nc = pred.shape[0] - 4 - nk
        x = pred.transpose(0, 1).clone()
        x[:, :4] = self._xywh2xyxy(x[:, :4])
        box, cls, extra = x.split((4, nc, nk), 1)
        conf, j = cls.max(1, keepdim=True)
        keep = conf.view(-1) > self.conf
        x = torch.cat((box, conf, j.to(x.dtype), extra), 1)[keep]
        if self.classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(self.classes, device=x.device)).any(1)]
        if not len(x):
            return x
        if len(x) > MAX_NMS:
            x = x[x[:, 4].argsort(descending=True)[:MAX_NMS]]
        boxes = x[:, :4] + x[:, 5:6] * MAX_WH
        i = torchvision.ops.nms(boxes, x[:, 4], self.iou)[:self.max_det]
        return x[i]

    def _scale(self, img1_shape, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - .1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - .1)
        return gain, pad_x, pad_y

    def _result(self, det, img1_shape, img0_shape):
        h, w = img0_shape
        gain, pad_x, pad_y = self._scale(img1_shape, img0_shape)
        b = det[:, :4]
        b[..., 0] -= pad_x
        b[..., 1] -= pad_y
        b[..., 2] -= pad_x
        b[..., 3] -= pad_y
        b[..., 0] /= gain
        b[..., 1] /= gain
        b[..., 2] /= gain
        b[..., 3] /= gain
        b[..., 0].clamp_(0, w)
        b[..., 1].clamp_(0, h)
        b[..., 2].clamp_(0, w)
        b[..., 3].clamp_(0, h)
        bn = b.clone()
        bn[..., [0, 2]] /= w
        bn[..., [1, 3]] /= h
        out = Detections((h, w), b.cpu().numpy(), bn.cpu().numpy(), det[:, 4].cpu().numpy(), det[:, 5].cpu().numpy())
        if self.kpt_shape:
            k = det[:, 6:].view(len(det), *self.kpt_shape)
            k[..., 0] -= pad_x
            k[..., 1] -= pad_y
            k[..., 0] /= gain
            k[..., 1] /= gain
            k[..., 0].clamp_(0, w)
            k[..., 1].clamp_(0, h)
            xy = k[..., :2].clone()
            xy[..., 0] /= w
            xy[..., 1] /= h
            out.kpt_xyn, out.kpt_conf = xy.cpu().numpy(), k[..., 2].cpu().numpy()
        return out

    # -- public ---------------------------------------------------------
    @torch.inference_mode()
    def predict(self, frames, batch=1):
        """frames: list of HWC uint8 arrays -> list of Detections (same order)."""
        results = []
        for s in range(0, len(frames), batch):
            chunk = frames[s:s + batch]
            auto = len({f.shape for f in chunk}) == 1
            x = self._tensor([self._letterbox(f, auto) for f in chunk])
            feed = x.cpu().numpy()
            raw = torch.from_numpy(self.session.run(None, {self.input_name: feed})[0]).to(self.device)
            for k, f in enumerate(chunk):
                results.append(self._result(self._nms(raw[k]), x.shape[2:], f.shape[:2]))
        return results
