"""Weighted member pooling that ignores members whose input clip is blank.

Shared by validation, release building and raw-folder inference. A clip with no
thermal frames gets no vote from the thermal member; a clip with no IR frames
gets none from IR-only members. Remaining weights are renormalised per clip.
Masking the three-member ensemble lifted decoded OOF from 0.84453 to 0.84684.
"""
import numpy as np

VIEW_NEEDS = {'combined': 'visual', 'irdepth': 'visual', 'ir': 'ir', 'irnorm': 'ir', 'depth': 'depth', 'depthz': 'depth'}


def need_for(cache, view):
    return 'thermal' if cache.startswith('thermal') else VIEW_NEEDS[view]


def availability(visual, thermal, stride=8):
    """visual: (N,32,4,H,W) Depth RGB + IR cache; thermal: (N,32,4,h,w) cache. Boolean per clip."""
    ir = np.array([bool(visual[i, ::stride, 3].any()) for i in range(len(visual))])
    depth = np.array([bool(visual[i, ::stride, :3].any()) for i in range(len(visual))])
    th = np.array([bool(thermal[i, ::stride].any()) for i in range(len(thermal))])
    return dict(ir=ir, depth=depth, visual=ir | depth, thermal=th)


def combine(probs, weights, needs, avail):
    w = np.stack([wi * avail[need] for wi, need in zip(weights, needs)], 1).astype(float)
    blind = w.sum(1) == 0
    w[blind] = weights
    w /= w.sum(1, keepdims=True)
    return sum(w[:, [m]] * p for m, p in enumerate(probs))
