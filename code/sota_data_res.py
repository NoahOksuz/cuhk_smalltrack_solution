"""32-frame caches at a configurable spatial resolution.

At 128 px a whole-body person crop leaves the hands about ten pixels across,
which is the likely cause of the confusions between close-range hand actions
(Turn_pages vs Read_documents, Stir vs Pour, Write vs Tap_keyboard). The crop
geometry is unchanged; only the sampled resolution moves.
"""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from sota_data import ROOT, OUT, CONFIG, manifests, fingerprint, files, read, window

FRAMES = 32


def decode(job):
    row_id, row, base, crop, size = job
    a = np.zeros((FRAMES, 4, size, size), dtype=np.uint8)
    for mod, mode, start, ch in [('Depth_Color', 'RGB', 0, 3), ('IR', 'L', 3, 1)]:
        paths = files(Path(base) / row[mod]) if row.get(mod) else []
        idx = np.linspace(0, max(len(paths) - 1, 0), FRAMES).round().astype(int)
        for t, j in enumerate(idx):
            im = read(paths[j], mode) if paths else None
            if im is None:
                continue
            if crop is not None:
                w, h = im.size
                im = im.crop(tuple(round(v * s) for v, s in zip(crop, [w, h, w, h])))
            x = np.asarray(im.resize((size, size), Image.Resampling.BILINEAR))
            a[t, start:start + ch] = x.transpose(2, 0, 1) if ch == 3 else x[None]
    return row_id, a


def prepare(mode, size, margin, workers):
    train, test = manifests()
    signature = fingerprint(train, test)
    tag = f'{mode}32_{size}' + ('' if margin == CONFIG['margin'] else f'_m{margin:g}')
    complete = OUT / f'{tag}_complete.json'
    target = OUT / f'{tag}.npy'
    if complete.exists() and json.loads(complete.read_text())['fingerprint'] == signature:
        print('already built', tag)
        return
    boxes = json.loads((OUT / 'boxes.json').read_text())
    assert boxes['fingerprint'] == signature
    saved = CONFIG['margin']
    CONFIG['margin'] = margin
    try:
        crops = [window(b, mode) for b in boxes['boxes']]
    finally:
        CONFIG['margin'] = saved
    records = ([(r, str(ROOT / 'Training/data/HAR/data')) for r in train]
               + [(r, str(ROOT / 'Testing/data/small_model_track_test')) for r in test])
    mm = np.lib.format.open_memmap(target, mode='w+', dtype='uint8',
                                   shape=(len(records), FRAMES, 4, size, size))
    jobs = [(i, r, base, crops[i], size) for i, (r, base) in enumerate(records)]
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        for i, a in pool.map(decode, jobs, chunksize=4):
            mm[i] = a
            if i % 500 == 0:
                print('decode', tag, i, len(records), flush=True)
    mm.flush()
    complete.write_text(json.dumps(dict(fingerprint=signature, rows=len(records),
                                        train_rows=len(train),
                                        config=dict(CONFIG, frames=FRAMES, size=size, margin=margin))))
    print('cache complete', tag, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['union', 'median'], default='union')
    p.add_argument('--size', type=int, default=160)
    p.add_argument('--margin', type=float, default=1.4)
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    prepare(a.mode, a.size, a.margin, a.workers)
