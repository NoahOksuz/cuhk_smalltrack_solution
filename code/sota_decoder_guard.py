"""Opt-in take guard for the sequence decoder (validated by gen_decoder_guard_fixed.py, 2026-09-14).

After decoding, a take falls back to the per-clip argmax of the combined probabilities when it is longer than
`max_take` clips (the longest take in the training metadata has 10) or when more than `max_dup_pairs` pairs of its clips
share the same argmax with confidence >= `conf` — both signs that the take is not one scripted recording of distinct
actions. Held-out-subject stress test on 3,036 clips: correct takes lose 7 clips (85.54% -> 85.31%); merged takes recover
from 58.63% to 72.13%; missing, partial or reordered structure is unaffected. Label-free, per clip group only.
"""
import numpy as np

GUARD = dict(max_take=10, max_dup_pairs=2, conf=.5)


def guard(prob, pred, members, max_take=10, max_dup_pairs=2, conf=.5):
    """prob [N, C] combined probabilities, pred [N] decoded labels, members {take: row positions}."""
    raw, top = prob.argmax(1), prob.max(1)
    result = np.array(pred, copy=True)
    fallen = 0
    for rows in members.values():
        rows = np.asarray(rows)
        confident = rows[top[rows] >= conf]
        counts = np.bincount(raw[confident], minlength=prob.shape[1])
        if len(rows) > max_take or int((counts * (counts - 1) // 2).sum()) > max_dup_pairs:
            result[rows] = raw[rows]
            fallen += 1
    return result, fallen
