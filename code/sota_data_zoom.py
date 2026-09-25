"""32-frame Depth+IR cache zoomed onto the upper body.

The shipped crop keeps a 1.4 margin and a 224 px minimum side, so a seated or
distant subject fills a third of it and hands and small objects span a few
pixels at 128 px. The confusions left after decoding are hand-object pairs
(Take_medicine/Drink_water, Tableware/Stir, Turn_pages/Read, Play_games/Phone).
This view crops the upper part of the same YOLO union box, so no new asset is
needed at inference.
"""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sota_data import ROOT, OUT, manifests, fingerprint
from sota_data_res import decode, FRAMES

ZOOM = dict(top=-.05, bottom=.62, widen=1.15, min_side=.22)


def zoom_crop(boxes, top=ZOOM['top'], bottom=ZOOM['bottom'], widen=ZOOM['widen'], min_side=ZOOM['min_side']):
    """Square normalized crop over the upper body of the union person box, or None."""
    if not boxes:
        return None
    b = np.array(boxes)
    x0, y0 = b[:, :2].min(0)
    x1, y1 = b[:, 2:].max(0)
    w, h = (x1 - x0) * 640, (y1 - y0) * 480
    cx, cy = (x0 + x1) / 2 * 640, y0 * 480 + h * (top + bottom) / 2
    half = max(w * widen, h * (bottom - top) * widen, min_side * 640) / 2
    return [max(cx - half, 0) / 640, max(cy - half, 0) / 480,
            min(cx + half, 640) / 640, min(cy + half, 480) / 480]


def prepare(size, workers):
    train, test = manifests()
    signature = fingerprint(train, test)
    tag = f'zoom32_{size}'
    complete = OUT / f'{tag}_complete.json'
    target = OUT / f'{tag}.npy'
    if complete.exists() and json.loads(complete.read_text())['fingerprint'] == signature:
        print('already built', tag)
        return
    boxes = json.loads((OUT / 'boxes.json').read_text())
    assert boxes['fingerprint'] == signature
    crops = [zoom_crop(b) for b in boxes['boxes']]
    records = ([(r, str(ROOT / 'Training/data/HAR/data')) for r in train]
               + [(r, str(ROOT / 'Testing/data/small_model_track_test')) for r in test])
    mm = np.lib.format.open_memmap(target, mode='w+', dtype='uint8',
                                   shape=(len(records), FRAMES, 4, size, size))
    jobs = [(i, r, base, crops[i], size) for i, (r, base) in enumerate(records)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, a in pool.map(decode, jobs, chunksize=4):
            mm[i] = a
            if i % 500 == 0:
                print('decode', tag, i, len(records), flush=True)
    mm.flush()
    complete.write_text(json.dumps(dict(fingerprint=signature, rows=len(records), train_rows=len(train),
                                        config=dict(ZOOM, frames=FRAMES, size=size, fallback=sum(c is None for c in crops)))))
    print('cache complete', tag, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    prepare(a.size, a.workers)
