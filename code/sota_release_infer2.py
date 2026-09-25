"""Standalone raw-folder inference for releases built by sota_release_build2.py (adds body+hands joint members)."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from sota_data import files, read, window
from sota_data_res import decode as visual_decode
from sota_data_zoom import zoom_crop
from sota_thermal32 import decode as thermal_decode
from sota_blocks import clip_meta, scenes
from sota_group_relax import assign_relaxed, assign_adjacent
from sota_decoder import ensemble_decode
from sota_delta_pack import decode as decode_weights
from sota_model import Model, strideless
from sota_views import input_channels
from sota_ensemble import availability, combine
from sota_joint import JointModel, pose_probes, hand_crop, decode_hands, predict_joint, HANDS

MEAN = torch.tensor([.43216, .394666, .37645, (.43216 + .394666 + .37645) / 3]).view(4, 1, 1, 1)
STD = torch.tensor([.22803, .22145, .216989, (.22803 + .22145 + .216989) / 3]).view(4, 1, 1, 1)


def yolo_from_unified(ckpt):
    """Build a YOLO object from a unified-checkpoint dict entry (a full ultralytics
    checkpoint with a DetectionModel/PoseModel under 'model'). YOLO() only accepts
    paths or modules, so round-trip the entry through a temp file; the resulting
    state_dict is byte-identical to loading the original .pt directly."""
    from ultralytics import YOLO
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tf:
        torch.save(ckpt, tf.name)
        path = tf.name
    try:
        return YOLO(path)
    finally:
        os.unlink(path)


def channels(x, view):
    return input_channels(x, view)


class Data(torch.utils.data.Dataset):
    def __init__(self, path, offset, view, frames=16):
        self.mm = np.load(path, mmap_mode='r')
        self.offset = offset
        self.view = view
        self.frames = frames

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, i):
        window = self.mm[i] if self.frames == 32 else self.mm[i, self.offset::2]
        x = torch.from_numpy(window.copy()).permute(1, 0, 2, 3).float() / 255
        return (channels(x, self.view) - MEAN) / STD


@torch.inference_mode()
def predict(model, path, view, frames=16):
    model.eval()
    result = []
    for offset in [0, 1]:
        outputs = []
        for x in torch.utils.data.DataLoader(Data(path, offset, view, frames), batch_size=8, num_workers=0):
            x = x.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                z = (model(x).float() + model(x.flip(-1)).float()) / 2
            outputs.append(z.float().softmax(1).cpu().numpy())
        result.append(np.concatenate(outputs))
    return np.mean(result, axis=0)


def cache_kind(cache):
    if cache.startswith('thermal32'):
        return 'thermal', 128
    if cache.startswith('zoom32_'):
        return 'zoom', int(cache.split('_')[1])
    assert cache.startswith('union32'), cache
    return 'union', int(cache.split('_')[1]) if '_' in cache else 128


def validate_raw_input(root, ids):
    """Reject wrong roots before detector/cache work; individual sensors may be absent."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'--input is not a directory: {root}')
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Template must contain nonempty, unique clip IDs')
    for clip in ids:
        if not clip or clip in {'.', '..'} or Path(clip).name != clip:
            raise ValueError(f'Invalid template clip ID: {clip!r}')
        folder = root / clip
        if not folder.is_dir():
            raise ValueError(f'--input must directly contain every template clip folder; missing {folder}')
        paths = [p for mod in ('IR', 'Depth_Color', 'Thermal') for p in files(folder / mod)]
        if not paths or not any(read(p, 'L') is not None for p in paths):
            raise ValueError(f'No readable sensor images for clip {clip} under --input {root}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True, help='Directory containing SM_test_* folders')
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    start = time.time()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    report = json.loads((args.package / 'report.json').read_text())
    # Load all six model members from the one canonical checkpoint. Accept a
    # package-root file for compatibility, but the source layout keeps it under
    # checkpoints/.
    ckpt = args.package / 'checkpoints' / 'unified_model.pt'
    if not ckpt.exists():
        ckpt = args.package / 'unified_model.pt'
    for name, expected in report['sha256'].items():
        path = ckpt if name == 'unified_model.pt' else args.package / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    model_weights = torch.load(ckpt, map_location='cpu', weights_only=False)
    with open(args.template) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        template = list(reader)
    ids = [r[fields[0]].rstrip('/').split('/')[-1] for r in template]
    validate_raw_input(args.input, ids)
    args.work.mkdir(parents=True, exist_ok=True)
    rows = [dict(id=i, **{m: f'{i}/{m}' for m in ['Depth_Color', 'IR', 'Thermal', 'IMU']}) for i in ids]
    detector = yolo_from_unified(model_weights['yolo11n'])
    boxes_per_clip = []
    for i, row in enumerate(rows):
        paths = files(args.input / row['IR'])
        batch = []
        for j in sorted(set(np.linspace(0, max(len(paths) - 1, 0), 8).round().astype(int))):
            im = read(paths[j], 'RGB') if paths else None
            if im is not None and np.asarray(im).any():
                batch.append(np.asarray(im))
        boxes = []
        if batch:
            for res in detector.predict(batch, classes=[0], conf=.25, imgsz=640, device=0, verbose=False):
                if len(res.boxes):
                    h, w = res.orig_shape
                    boxes.append((res.boxes.xyxy[res.boxes.conf.argmax()].cpu().numpy() / [w, h, w, h]).tolist())
        boxes_per_clip.append(boxes)
        if i % 100 == 0:
            print('detected', i, flush=True)
    del detector
    torch.cuda.empty_cache()
    fallback = report.get('fallback_detector')
    if fallback:
        # Second detector only where the primary found nobody; every other crop is unchanged.
        from sota_fallback_det import load_detector, detect
        detector = load_detector(args.package / fallback['asset'], args.work)
        rescued = []
        for i, row in enumerate(rows):
            if not boxes_per_clip[i]:
                boxes_per_clip[i] = detect(detector, args.input / row['IR'])
                if boxes_per_clip[i]:
                    rescued.append(row['id'])
        print('fallback rescued', rescued, flush=True)
        del detector
        torch.cuda.empty_cache()
    crops = dict(union=[window(b, 'union') for b in boxes_per_clip], zoom=[zoom_crop(b) for b in boxes_per_clip])
    hand_caches = {}
    if any(item.get('kind') == 'joint' for item in report['models']):
        # Hands view: yolo11n-pose wrists on the eight IR probe frames, relative to the clip's person crop.
        pose = yolo_from_unified(model_weights['yolo11n-pose'])
        crops['hands'], status = [], []
        for i, row in enumerate(rows):
            kp, size = pose_probes(pose, args.input / row['IR'], crops['union'][i])
            crop, state = hand_crop(kp, crops['union'][i], size)
            crops['hands'].append(crop)
            status.append(state)
        del pose
        torch.cuda.empty_cache()
        for kind in sorted({item.get('hand_cache') or 'irhand128' for item in report['models'] if item.get('kind') == 'joint'}):
            c, size = HANDS[kind]['channels'], HANDS[kind]['size']
            hand_caches[kind] = args.work / f'{kind}.npy'
            mm = np.lib.format.open_memmap(hand_caches[kind], mode='w+', dtype=np.uint8, shape=(len(rows), 32, c, size, size))
            jobs = [(i, row, str(args.input), crops['hands'][i]) for i, row in enumerate(rows)]
            with ThreadPoolExecutor(8) as pool:
                for i, a in pool.map(lambda job, kind=kind: decode_hands(job, kind), jobs):
                    mm[i] = a
            mm.flush()
            del mm
        print('hands view', {s: status.count(s) for s in set(status)}, flush=True)
    (args.work / 'crops.json').write_text(json.dumps(crops))
    caches = {}
    for cache in dict.fromkeys(item['cache'] for item in report['models']):
        kind, size = cache_kind(cache)
        path = args.work / f'{cache}.npy'
        mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(len(rows), 32, 4, size, size))
        if kind == 'thermal':
            jobs = [(i, str(args.input / row['Thermal'])) for i, row in enumerate(rows)]
            fn = thermal_decode
        else:
            jobs = [(i, row, str(args.input), crops[kind][i], size) for i, row in enumerate(rows)]
            fn = visual_decode
        with ThreadPoolExecutor(8) as pool:
            for i, a in pool.map(fn, jobs):
                mm[i] = a
        mm.flush()
        del mm
        caches[cache] = path
        print('decoded', cache, flush=True)
    with ThreadPoolExecutor(8) as pool:
        meta = list(pool.map(lambda r: clip_meta(args.input, r), rows))
    blocks = (assign_adjacent if 'loose_gap' in report['grouping'] else assign_relaxed)(meta, **report['grouping'])
    scene, members = scenes(meta, blocks)
    (args.work / 'metadata.json').write_text(json.dumps(dict(meta=meta, blocks=blocks)))
    torch.set_float32_matmul_precision('high')
    probs, reference = [], None
    for k, item in enumerate(report['models']):
        ck = model_weights[f"video_{item['tag']}"]
        assert ck['base_tag'] == (None if k == 0 else report['models'][0]['tag'])
        state, ref = decode_weights(ck['model'], reference)
        if reference is None:
            reference = ref
        if item.get('kind') == 'joint':
            model = JointModel(strided=not item.get('strideless', True), interaction=item.get('fusion') == 'interact',
                               frames=int(item.get('frames', 32))).cuda().eval()
            model.load_state_dict(state, strict=True)
            del state, ref, ck
            probs.append(predict_joint(model, caches[item['cache']], np.arange(len(rows)), hand_caches[item.get('hand_cache') or 'irhand128'],
                                       frames=int(item.get('frames', 32)), view=item['view']))
        else:
            model = Model().cuda().eval()
            # Shape-identical to a strided checkpoint; the package records which.
            if item.get('strideless') or ck.get('strideless'):
                print('strideless', item['tag'], len(strideless(model.encoder)), flush=True)
            model.load_state_dict(state)
            del state, ref, ck
            probs.append(predict(model, caches[item['cache']], item['view'], int(item.get('frames', 16))))
        del model
        torch.cuda.empty_cache()
        print('scored', item['tag'], flush=True)
    weights = [item['weight'] for item in report['models']]
    assert np.isclose(sum(weights), 1)
    visual = next(c for c in caches if cache_kind(c)[0] != 'thermal')
    thermal = caches.get('thermal32')
    avail = availability(np.load(caches[visual], mmap_mode='r'),
                         np.load(thermal, mmap_mode='r') if thermal else np.ones((len(rows), 32, 1, 1, 1), np.uint8))
    if report.get('missing_modality_masking'):
        combined = combine(probs, weights, [item['need'] for item in report['models']], avail)
    else:
        combined = sum(w * p for w, p in zip(weights, probs))
    assert combined.shape == (len(rows), 40) and np.isfinite(combined).all()
    assert np.allclose(combined.sum(1), 1, atol=1e-5)
    prior = {k: np.array(v) for k, v in json.loads((args.package / 'sequence_prior.json').read_text()).items()}
    pred = ensemble_decode(combined, members, scene, meta, prior, **report['decoder'])
    assert all(len(set(pred[r])) == len(r) for r in members.values())
    if report.get('decoder_guard'):
        from sota_decoder_guard import guard
        pred, fallen = guard(combined, pred, members, **report['decoder_guard'])
        print('decoder guard: takes fell back', fallen, 'of', len(members), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, label in zip(template, pred, strict=True):
            row[fields[1]] = int(label)
            writer.writerow(row)
    np.save(args.work / 'probabilities.npy', combined)
    verification = dict(rows=len(rows), seconds=time.time() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                        input_mode='fresh raw folders, per-clip detection; no training files or cached boxes used')
    (args.work / 'verification.json').write_text(json.dumps(verification, indent=2))
    print('RAW_INFERENCE', verification, flush=True)


if __name__ == '__main__':
    main()
