"""Recover recording blocks from clip metadata (timestamps + thermal frame counter).

A block is one continuous recording of one user performing one scripted action
sequence.  Within a block every action is distinct (verified 780/780 on train),
and blocks are recoverable from metadata alone (verified 890/890 pure).
"""
import collections
import json
import os
import re
from pathlib import Path

TS = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.(\d+)")
TH = re.compile(r"frame_(\d+)")
THERMAL_FPS = 24.0


def clip_meta(base, row):
    """Timestamp, thermal counter and IMU device set for one clip."""
    out = {}
    ir = Path(base) / row["IR"]
    if ir.is_dir():
        names = sorted(os.listdir(ir))
        if names:
            a, b = TS.search(names[0]), TS.search(names[-1])
            if a:
                out["date"] = a.group(1)
                out["t0"] = int(a.group(2)) * 3600 + int(a.group(3)) * 60 + int(a.group(4)) + int(a.group(5)) / 1000
            if b:
                out["t1"] = int(b.group(2)) * 3600 + int(b.group(3)) * 60 + int(b.group(4)) + int(b.group(5)) / 1000
    th = Path(base) / row["Thermal"]
    if th.is_dir():
        names = sorted(os.listdir(th))
        if names:
            a = TH.search(names[0])
            if a: out["th0"] = int(a.group(1))
    imu = Path(base) / row["IMU"]
    devices = set()
    if imu.is_dir():
        for name in sorted(os.listdir(imu)):
            try:
                with open(imu / name, encoding="utf-8", errors="replace") as f:
                    for line in list(f)[1:400]:
                        parts = line.split(",")
                        if len(parts) > 2: devices.add(parts[1])
            except OSError:
                pass
    out["devices"] = sorted(devices)
    return out


def assign(meta):
    """Block id per clip index. Clips without usable metadata become singletons."""
    groups = collections.defaultdict(list)
    loose = []
    for i, m in enumerate(meta):
        if m.get("t0") is None or m.get("th0") is None: loose.append(i)
        else: groups[(m["date"], tuple(m["devices"]))].append(i)
    block = {}
    n = 0
    for key, idx in groups.items():
        idx.sort(key=lambda i: meta[i]["t0"])
        prev = None
        for i in idx:
            m = meta[i]
            new = prev is None
            if prev is not None:
                dt = m["t0"] - meta[prev]["t0"]
                dth = m["th0"] - meta[prev]["th0"]
                # Same recording iff the thermal counter advanced in step with the clock.
                new = not (dth > 0 and dt >= 0 and abs(dth - THERMAL_FPS * dt) < max(30, .35 * THERMAL_FPS * max(dt, 1)))
            if new: n += 1
            block[i] = n
            prev = i
    for i in loose:
        n += 1
        block[i] = n
    return [block[i] for i in range(len(meta))]


def scenes(meta, blocks, gap=250):
    """Group blocks that are repetitions of the same scripted scene."""
    members = collections.defaultdict(list)
    for i, b in enumerate(blocks): members[b].append(i)
    for b in members: members[b].sort(key=lambda i: meta[i].get("t0", 0))
    keyed = collections.defaultdict(list)
    for b, idx in members.items():
        m = meta[idx[0]]
        if m.get("t0") is None: continue
        keyed[(m["date"], tuple(m["devices"]))].append((m["t0"], b))
    scene = {}
    n = 0
    for key, lst in keyed.items():
        lst.sort()
        prev = None
        for t0, b in lst:
            same = (prev is not None and len(members[b]) == len(members[prev])
                    and t0 - meta[members[prev][-1]]["t0"] < gap)
            if not same: n += 1
            scene[b] = n
            prev = b
    for b in members:
        if b not in scene:
            n += 1
            scene[b] = n
    return scene, members


if __name__ == "__main__":
    import csv, sys
    ROOT = Path("/home/nyx/comps/cuhk_data/Small-Model-Track")
    out = Path("/home/nyx/comps/cuhk_recovery/sota_runs")
    with open(ROOT / "manifests/train_manifest.csv") as f: train = list(csv.DictReader(f))
    with open(ROOT / "manifests/test_manifest.csv") as f: test = list(csv.DictReader(f))
    tb, teb = str(ROOT / "Training/data/HAR/data"), str(ROOT / "Testing/data/small_model_track_test")
    trm = [clip_meta(tb, r) for r in train]
    tem = [clip_meta(teb, r) for r in test]
    json.dump({"train": trm, "test": tem,
               "train_blocks": assign(trm), "test_blocks": assign(tem)},
              open(out / "blocks.json", "w"))
    print("train blocks", len(set(assign(trm))), "test blocks", len(set(assign(tem))))


def regions(meta, blocks, gap=600):
    """Contiguous stretches of one recording session, as cross-validation groups.

    The evaluation clips were carved out as whole stretches of session time: no
    test block sits within 286 s of a training block, while sibling repetitions
    of a scene are only ~70 s apart.  Grouping on blocks alone therefore leaves
    a near-identical retake of every held-out clip in the training set.  These
    groups keep a whole stretch together so the split matches the real one.
    """
    import collections
    members = collections.defaultdict(list)
    for i, b in enumerate(blocks): members[b].append(i)
    for b in members: members[b].sort(key=lambda i: meta[i].get("t0", 0))
    keyed = collections.defaultdict(list)
    loose = []
    for b, idx in members.items():
        m = meta[idx[0]]
        if m.get("t0") is None: loose.append(b)
        else: keyed[(m["date"], tuple(m["devices"]))].append((m["t0"], b))
    group = {}
    n = 0
    for key, lst in keyed.items():
        lst.sort()
        prev_end = None
        for t0, b in lst:
            if prev_end is None or t0 - prev_end > gap: n += 1
            group[b] = n
            prev_end = meta[members[b][-1]]["t0"]
    for b in loose:
        n += 1
        group[b] = n
    return [group[b] for b in blocks], members
