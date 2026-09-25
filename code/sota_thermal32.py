"""32-frame thermal cache, so the thermal stream gets the same temporal
augmentation and multi-window inference as the depth+IR stream."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
from PIL import Image

from sota_data import ROOT, OUT, CONFIG, manifests, fingerprint, files, read

FRAMES = 32


def decode(job):
    row_id, path = job
    a = np.zeros((FRAMES, 4, 128, 128), dtype=np.uint8)
    paths = files(Path(path))
    idx = np.linspace(0, max(len(paths) - 1, 0), FRAMES).round().astype(int)
    for t, j in enumerate(idx):
        im = read(paths[j], 'RGB') if paths else None
        if im is None:
            continue
        x = np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR))
        a[t, :3] = x.transpose(2, 0, 1)
        a[t, 3] = np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BILINEAR))
    return row_id, a


if __name__ == '__main__':
    train, test = manifests()
    sig = fingerprint(train, test)
    complete = OUT / 'thermal32_complete.json'
    target = OUT / 'thermal32.npy'
    if complete.exists() and json.loads(complete.read_text())['fingerprint'] == sig:
        print('already built')
    else:
        records = ([str(ROOT / 'Training/data/HAR/data' / r['Thermal']) for r in train]
                   + [str(ROOT / 'Testing/data/small_model_track_test' / r['Thermal']) for r in test])
        mm = np.lib.format.open_memmap(target, mode='w+', dtype='uint8',
                                       shape=(len(records), FRAMES, 4, 128, 128))
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(10, mp_context=ctx) as pool:
            for i, a in pool.map(decode, list(enumerate(records)), chunksize=4):
                mm[i] = a
                if i % 500 == 0:
                    print('thermal32', i, len(records), flush=True)
        mm.flush()
        complete.write_text(json.dumps(dict(fingerprint=sig, config=dict(CONFIG, frames=FRAMES),
                                            rows=len(records), train_rows=len(train))))
        print('thermal32 complete', flush=True)
