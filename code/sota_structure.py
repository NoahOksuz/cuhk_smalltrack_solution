"""Recording-structure post-processing for clip probabilities.

Two properties of how the data was recorded constrain the labels:

  distinctness  Every clip inside one recording block has a different action.
                Verified on all 780 labelled training blocks.
  repetition    A scene is recorded two or three times in a row, and each
                repetition performs the same action sequence in the same order,
                so matched positions across repetitions share a label.

Both are recovered from clip metadata alone (timestamps, thermal frame counter,
IMU device set), so they apply unchanged to the unlabelled test clips.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def average_repetitions(prob, members, scene, weight=.5):
    """Blend each clip toward the mean of its matched positions in sibling blocks."""
    out = prob.copy()
    groups = {}
    for block, scene_id in scene.items():
        groups.setdefault(scene_id, []).append(block)
    for blocks in groups.values():
        if len(blocks) < 2:
            continue
        blocks = sorted(blocks, key=lambda b: members[b][0])
        if len({len(members[b]) for b in blocks}) != 1:
            continue
        for position in range(len(members[blocks[0]])):
            rows = [members[b][position] for b in blocks]
            mean = prob[rows].mean(0)
            for r in rows:
                out[r] = (1 - weight) * prob[r] + weight * mean
    return out


def assign_distinct(prob, members):
    """Argmax subject to one action per clip within each block."""
    pred = prob.argmax(1)
    for rows in members.values():
        if len(rows) < 2:
            continue
        cost = -np.log(prob[rows] + 1e-12)
        r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            pred[rows[a]] = b
    return pred


def apply(prob, members, scene, weight=.5):
    return assign_distinct(average_repetitions(prob, members, scene, weight), members)
