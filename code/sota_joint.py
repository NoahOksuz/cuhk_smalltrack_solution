"""Body + hands joint IR member for releases: model, pose-probe hand crops, hand-view decode, paired scoring.

Mirrors gen_joint_train.JointModel (same parameter names: encoder.*, head.*, hand_head.*, fusion.*) on top of the
shipped sota_model.Model with temporal striding removed. The hands view is a fixed per-clip square from yolo11n-pose
wrists on the eight IR probe frames (gen_hand_cache.hand_crop geometry), decoded like the ir128 cache. Only the fused
head is used for predictions. Independent per-clip computation; no cross-clip operations.

Two hands-view kinds: 'irhand128' (IR only, 1 channel, 128 px; body = IR channel of union32, 32 frames, strideless) and
'depthirhand160' (Depth_Color RGB + IR, 4 channels, 160 px via sota_data_res.decode; body = union32_160 'combined',
16 frames with both temporal offsets, strided). Defaults reproduce the IR joint member.

interaction=True adds gen_joint_train.Interaction (cross-view spatio-temporal transformer whose logits are added to the
fused head); members without it keep the original encoder(x) path, so earlier packages reproduce byte-for-byte.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from sota_data import files, read
from sota_data_res import decode as visual_decode
from sota_model import Model, strideless

CROP = dict(kp_conf=.5, hand_extension=.5, pad=.08, min_side=96)
POSE = dict(probes=8, imgsz=640, conf=.25, quantize=16)
WRISTS, ELBOWS = (9, 10), (7, 8)
HANDS = {'irhand128': dict(channels=1, size=128), 'depthirhand160': dict(channels=4, size=160)}
MEAN = torch.tensor([.43216, .394666, .37645, (.43216 + .394666 + .37645) / 3]).view(4, 1, 1, 1)
STD = torch.tensor([.22803, .22145, .216989, (.22803 + .22145 + .216989) / 3]).view(4, 1, 1, 1)


class Interaction(nn.Module):
    """Identical to gen_joint_train.Interaction (same parameter names and computation)."""

    def __init__(self, frames=32, width=256, layers=2, heads=4):
        super().__init__()
        self.project = nn.Linear(512, width)
        self.slot = nn.Parameter(torch.zeros(1, 1, 8, width))
        self.time = nn.Parameter(torch.zeros(1, frames, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(.3)
        self.out = nn.Linear(width, 40)

    def forward(self, body_map, hand_map):
        tokens = [F.adaptive_avg_pool3d(m, (m.shape[2], 2, 2)).flatten(3).permute(0, 2, 3, 1) for m in (body_map, hand_map)]
        x = self.project(torch.cat(tokens, 2)) + self.slot + self.time[:, :tokens[0].shape[1]]
        b, t = x.shape[:2]
        x = self.blocks(x.reshape(b, t * 8, -1))
        return self.out(self.drop(self.norm(x).mean(1)))


class JointModel(nn.Module):
    def __init__(self, strided=False, interaction=False, frames=32):
        super().__init__()
        base = Model()
        self.encoder, self.head = base.encoder, base.head
        self.removed = [] if strided else strideless(self.encoder)
        self.hand_head = nn.Sequential(nn.Dropout(.3), nn.Linear(512, 40))
        self.fusion = nn.Sequential(nn.Dropout(.3), nn.Linear(1024, 40))
        self.interaction = Interaction(frames) if interaction else None

    def forward(self, body, hands):
        n = len(body)
        if self.interaction is None:
            z = self.encoder(torch.cat([body, hands]))
            return self.fusion(torch.cat([z[:n], z[n:]], 1))
        e = self.encoder
        maps = e.layer4(e.layer3(e.layer2(e.layer1(e.stem(torch.cat([body, hands]))))))
        z = maps.mean((-1, -2, -3))
        return self.fusion(torch.cat([z[:n], z[n:]], 1)) + self.interaction(maps[:n], maps[n:])


def pose_probes(detector, folder, body):
    """Keypoints [8, 17, 3] (normalized x, y, confidence; NaN when no person) and native (w, h) for one clip."""
    paths = files(folder)
    keypoints = np.full((POSE['probes'], 17, 3), np.nan, dtype=np.float32)
    size = (0, 0)
    if not paths:
        return keypoints, size
    frames, slots = [], []
    for t, j in enumerate(np.linspace(0, len(paths) - 1, POSE['probes']).round().astype(int)):
        im = read(paths[j], 'RGB')
        a = None if im is None else np.asarray(im)
        if a is None or not a.any():
            continue
        size = (a.shape[1], a.shape[0])
        frames.append(a)
        slots.append(t)
    if not frames:
        return keypoints, size
    results = detector.predict(frames, imgsz=POSE['imgsz'], conf=POSE['conf'], quantize=POSE['quantize'], verbose=False, batch=len(frames))
    for t, r in zip(slots, results):
        if r.keypoints is None or not len(r.boxes):
            continue
        xyxy, conf = r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()
        if body is not None:
            c = np.asarray(body)
            inter = (np.clip(np.minimum(xyxy[:, 2], c[2]) - np.maximum(xyxy[:, 0], c[0]), 0, None)
                     * np.clip(np.minimum(xyxy[:, 3], c[3]) - np.maximum(xyxy[:, 1], c[1]), 0, None))
            area = np.maximum((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]), 1e-9)
            k = int(np.lexsort((-conf, -(inter / area).round(2)))[0])
        else:
            k = int(conf.argmax())
        keypoints[t, :, :2] = r.keypoints.xyn[k].cpu().numpy()
        keypoints[t, :, 2] = r.keypoints.conf[k].cpu().numpy()
    return keypoints, size


def hand_crop(kp, body, size):
    """Normalized square hands crop and status (identical geometry to gen_hand_cache.hand_crop)."""
    if not size[0]:
        return body, 'no_ir'
    w, h = int(size[0]), int(size[1])
    scale = np.array([w, h], dtype=np.float64)
    box = np.asarray(body if body is not None else [0, 0, 1, 1], dtype=np.float64)
    body_side = float(max((box[2] - box[0]) * w, (box[3] - box[1]) * h))
    points = []
    for f in kp:
        if np.isnan(f[0, 2]):
            continue
        ok = f[:, 2] >= CROP['kp_conf']
        for wr, el in zip(WRISTS, ELBOWS):
            if ok[wr]:
                points.append(f[wr, :2] * scale)
                if ok[el]:
                    points.append((f[wr, :2] + CROP['hand_extension'] * (f[wr, :2] - f[el, :2])) * scale)
    if not points:
        return body, 'fallback'
    pts = np.clip(np.asarray(points), 0, scale)
    lo, hi = pts.min(0), pts.max(0)
    side = float(max((hi - lo).max() + 2 * CROP['pad'] * body_side, CROP['min_side']))
    status = 'capped' if side >= body_side else 'ok'
    side = min(side, body_side, float(h))
    centre = (lo + hi) / 2
    x0 = float(np.clip(centre[0] - side / 2, 0, w - side))
    y0 = float(np.clip(centre[1] - side / 2, 0, h - side))
    return [x0 / w, y0 / h, (x0 + side) / w, (y0 + side) / h], status


def decode_ir(job):
    """(i, folder, crop) -> (i, uint8 [32, 1, 128, 128]): frames linspace(0, n-1, 32).round(), integer crop, bilinear."""
    i, folder, crop = job
    a = np.zeros((32, 1, 128, 128), dtype=np.uint8)
    paths = files(Path(folder)) if folder else []
    for t, j in enumerate(np.linspace(0, max(len(paths) - 1, 0), 32).round().astype(int)):
        im = read(paths[j], 'L') if paths else None
        if im is None:
            continue
        if crop is not None:
            w, h = im.size
            im = im.crop(tuple(round(v * s) for v, s in zip(crop, [w, h, w, h])))
        a[t, 0] = np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR))
    return i, a


def decode_hands(job, kind):
    """(i, row, base, crop) -> (i, uint8 hands clip) for a HANDS kind; rows hold IR / Depth_Color folders relative to base."""
    i, row, base, crop = job
    if kind == 'irhand128':
        return decode_ir((i, str(Path(base) / row['IR']) if row.get('IR') else None, crop))
    return visual_decode((i, row, base, crop, HANDS[kind]['size']))


class Paired(torch.utils.data.Dataset):
    """Body row of a union cache (any leading row offset) and the matching hands row.

    view 'ir': body = IR channel of union32, hands = 1-channel IR, both replicated to 4 channels.
    view 'combined': body and hands are 4-channel Depth_Color RGB + IR as stored. frames 16 reads offset::2."""

    def __init__(self, body_path, body_rows, hand_path, frames=32, offset=0, view='ir'):
        self.body = np.load(body_path, mmap_mode='r')
        self.rows = np.asarray(body_rows)
        self.hands = np.load(hand_path, mmap_mode='r')
        assert len(self.hands) == len(self.rows) and view in ('ir', 'combined') and frames in (16, 32)
        self.frames, self.offset, self.view = frames, offset, view

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        body, hands = self.body[self.rows[i]], self.hands[i]
        if self.frames == 16:
            body, hands = body[self.offset::2], hands[self.offset::2]
        body = torch.from_numpy(body.copy()).permute(1, 0, 2, 3).float() / 255
        hands = torch.from_numpy(hands.copy()).permute(1, 0, 2, 3).float() / 255
        if self.view == 'ir':
            body, hands = body[3:4].expand(4, -1, -1, -1), hands.expand(4, -1, -1, -1)
        return (body - MEAN) / STD, (hands - MEAN) / STD


@torch.inference_mode()
def predict_joint(model, body_path, body_rows, hand_path, batch=4, workers=0, frames=32, view='ir'):
    """Flip-averaged fused logits -> softmax in row order; 16-frame members average both temporal offsets."""
    model.eval()
    passes = []
    for offset in ([0, 1] if frames == 16 else [0]):
        out = []
        data = Paired(body_path, body_rows, hand_path, frames=frames, offset=offset, view=view)
        for body, hands in torch.utils.data.DataLoader(data, batch_size=batch, num_workers=workers):
            body, hands = body.cuda(), hands.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                z = (model(body, hands).float() + model(body.flip(-1), hands.flip(-1)).float()) / 2
            out.append(z.float().softmax(1).cpu().numpy())
        passes.append(np.concatenate(out))
    return passes[0] if len(passes) == 1 else np.mean(passes, axis=0)
