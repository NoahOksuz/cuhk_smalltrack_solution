"""Manifest-aligned, fixed-person-crop Depth RGB + IR preprocessing."""
import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/nyx/comps/cuhk_data/Small-Model-Track')
OUT = Path(__file__).resolve().parent / 'sota_runs'
CONFIG = dict(version=1, frames=16, size=128, probes=8, confidence=.25,
              margin=1.4, min_side=.35, detector_size=640)


def manifests(root=ROOT):
    with open(root/'manifests/train_manifest.csv') as f:
        train = list(csv.DictReader(f))
    with open(root/'manifests/test_manifest.csv') as f:
        test = list(csv.DictReader(f))
    return train, test


def fingerprint(train, test):
    return hashlib.sha256(json.dumps([CONFIG, train, test], sort_keys=True).encode()).hexdigest()


def files(path):
    return sorted((p for p in path.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}),
                  key=lambda p: [(0,int(s)) if s.isdigit() else (1,s) for s in re.split(r'(\d+)', p.name)]) if path.is_dir() else []


def read(path, mode):
    try:
        with Image.open(path) as im:
            return im.convert(mode)
    except (OSError, ValueError):
        return None


def indices(n):
    return np.linspace(0, max(n-1, 0), CONFIG['frames']).round().astype(int)


def window(boxes, mode='union'):
    if not boxes:
        return None
    b = np.array(boxes)
    lo, hi = b[:,:2].min(0), b[:,2:].max(0)
    center = (lo+hi)/2
    extent = hi-lo
    if mode == 'median':
        center = np.median((b[:,:2]+b[:,2:])/2, axis=0)
        extent = (b[:,2:]-b[:,:2]).max(0)
    # Match the contender's 640x480 reference geometry and clipped bounds.
    side = max(np.max(extent*np.array([640,480]))*CONFIG['margin'], CONFIG['min_side']*640)
    half = side/np.array([640,480])/2
    return np.r_[np.maximum(center-half,0), np.minimum(center+half,1)].tolist()


def decode(job):
    row_id, row, base, crop = job
    a = np.zeros((16,4,128,128), dtype=np.uint8)
    bad = [0,0]
    for mi,(mod,mode,start,ch) in enumerate([('Depth_Color','RGB',0,3),('IR','L',3,1)]):
        paths = files(Path(base)/row[mod]) if row.get(mod) else []
        for t,j in enumerate(indices(len(paths))):
            im = read(paths[j], mode) if paths else None
            if im is None:
                bad[mi] += 1
                continue
            if crop is not None:
                w,h = im.size
                im = im.crop(tuple(round(v*s) for v,s in zip(crop,[w,h,w,h])))
            x = np.asarray(im.resize((128,128),Image.Resampling.BILINEAR))
            a[t,start:start+ch] = x.transpose(2,0,1) if ch==3 else x[None]
            if not x.any(): bad[mi] += 1
    return row_id,a,bad


def prepare(root=ROOT, out=OUT, mode='union', detector=None, workers=8, clip_detection=False):
    out.mkdir(parents=True,exist_ok=True)
    train,test = manifests(root)
    signature = fingerprint(train,test)
    policy = 'per_clip' if clip_detection else 'global_32'
    complete = out/f'{mode}_complete.json'
    target = out/f'{mode}.npy'
    if complete.exists():
        meta=json.loads(complete.read_text())
        assert meta['fingerprint']==signature, 'Cache manifest/config changed'
        assert meta.get('detector_batch_policy','global_32')==policy
        assert list(np.load(target,mmap_mode='r').shape)==[len(train)+len(test),16,4,128,128]
        return
    records = [(r, str(root/'Training/data/HAR/data')) for r in train] + [(r,str(root/'Testing/data/small_model_track_test')) for r in test]
    boxes_file = out/'boxes.json'
    if boxes_file.exists():
        obj=json.loads(boxes_file.read_text())
        assert obj['fingerprint']==signature
        assert obj.get('detector_batch_policy','global_32')==policy
        boxes=obj['boxes']
    else:
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(6)
        torch.set_float32_matmul_precision('highest')
        det=YOLO(str(detector or Path('/home/nyx/comps/cuhk_small/yolo11n.pt')))
        boxes=[[] for _ in records]
        batch,owners=[],[]
        def flush():
            if not batch: return
            results=det.predict(batch,classes=[0],conf=.25,imgsz=640,device=0,verbose=False)
            for owner,res in zip(owners,results,strict=True):
                if len(res.boxes):
                    h,w=res.orig_shape
                    boxes[owner].append((res.boxes.xyxy[res.boxes.conf.argmax()].cpu().numpy()/[w,h,w,h]).tolist())
            batch.clear();owners.clear()
        for i,(r,base) in enumerate(records):
            paths=files(Path(base)/r['IR']) if r.get('IR') else []
            for j in sorted(set(np.linspace(0,max(len(paths)-1,0),8).round().astype(int))):
                im=read(paths[j],'RGB') if paths else None
                if im is not None and np.asarray(im).any():
                    batch.append(np.asarray(im));owners.append(i)
                if len(batch)>=32: flush()
            if clip_detection: flush()
            if i%200==0: print('detect',i,len(records),flush=True)
        flush()
        boxes_file.write_text(json.dumps(dict(fingerprint=signature,boxes=boxes,detector_batch_policy=policy)))
        del det
        torch.cuda.empty_cache()
    crops=[window(b,mode) for b in boxes]
    mm=np.lib.format.open_memmap(target,mode='w+',dtype='uint8',shape=(len(records),16,4,128,128))
    jobs=[(i,r,base,crops[i]) for i,(r,base) in enumerate(records)]
    bad=[]
    # Spawn avoids inheriting a live CUDA runtime in decoder workers.
    import multiprocessing
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,a,missing in pool.map(decode,jobs,chunksize=4):
            mm[i]=a;bad.append(missing)
            if i%200==0: print('decode',mode,i,len(records),flush=True)
    mm.flush()
    complete.write_text(json.dumps(dict(fingerprint=signature,config=CONFIG,detector_batch_policy=policy,rows=len(records),train_rows=len(train),
                                       missing_frames=np.sum(bad,0).tolist(),fallback_crops=sum(c is None for c in crops))))
    print('cache complete',complete.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['union','median'],default='union')
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--detector',type=Path);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--clip-detection',action='store_true')
    a=p.parse_args();prepare(a.root,a.out,a.mode,a.detector,a.workers,a.clip_detection)
