"""Align incomplete repeated recordings using model probabilities only."""
import argparse
import json
from collections import defaultdict

import numpy as np


def align(short, long):
    """Maximum-similarity monotone injection; no labels enter alignment."""
    sim = np.sqrt(short) @ np.sqrt(long).T
    n, m = sim.shape
    assert n <= m
    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0] = 0
    take = np.zeros((n + 1, m + 1), bool)
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            match = dp[i - 1, j - 1] + np.log(max(sim[i - 1, j - 1], 1e-12))
            if match >= dp[i, j - 1]:
                dp[i, j] = match
                take[i, j] = True
            else:
                dp[i, j] = dp[i, j - 1]
    pairs = []
    i, j = n, m
    while i:
        if take[i, j]:
            pairs.append((i - 1, j - 1))
            i -= 1
        j -= 1
    pairs.reverse()
    return np.array(pairs, int), float(np.mean([sim[a, b] for a, b in pairs]))


def links(prob, members, scene, meta, gap=250):
    records = []
    result = []
    for b, rows in members.items():
        m = meta[rows[0]]
        records.append((b, np.array(rows), m.get('date'), tuple(m.get('devices', [])),
                        m.get('t0'), meta[rows[-1]].get('t1', meta[rows[-1]].get('t0'))))
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            old = scene[a[0]] == scene[b[0]]
            if old:
                assert len(a[1]) == len(b[1])
                similarity = float(np.sqrt(prob[a[1]] * prob[b[1]]).sum() / len(a[1]))
                result.append((a[1], b[1], similarity, True))
                continue
            if a[2:4] != b[2:4] or None in (a[4], a[5], b[4], b[5]):
                continue
            if not 0 <= max(a[4], b[4]) - min(a[5], b[5]) <= gap:
                continue
            ra, rb = a[1], b[1]
            if len(ra) > len(rb):
                ra, rb = rb, ra
            if len(ra) < 2 or len(rb) > 2 * len(ra):
                continue
            pairs, similarity = align(prob[ra], prob[rb])
            result.append((ra[pairs[:, 0]], rb[pairs[:, 1]], similarity, False))
    return result


def average(prob, edges, extra=.5, threshold=.6):
    delta = np.zeros_like(prob)
    degree = np.ones((len(prob), 1))
    for a, b, similarity, old in edges:
        weight = 1. if old else extra
        midpoint = .6 if old else threshold
        gate = weight / (1 + np.exp(-12 * (similarity - midpoint)))
        diff = gate * (prob[b] - prob[a])
        delta[a] += diff
        delta[b] -= diff
        degree[a] += weight
        degree[b] += weight
    result = prob + delta / degree
    assert np.isfinite(result).all() and result.min() >= -1e-7
    assert np.allclose(result.sum(1), 1, atol=1e-5)
    return result


def main():
    from sota_data import OUT, manifests
    from sota_finish2 import structure
    from sota_group_relax import assign_relaxed
    from sota_context_prior import fit_prior, decode
    from sota_blocks import regions
    from sota_adaptive_repeats import average as baseline_average

    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm', action='store_true')
    args = ap.parse_args()
    rows, _ = manifests()
    y = np.array([int(r['action_id']) for r in rows])
    allmeta = json.loads((OUT / 'blocks.json').read_text())
    meta = allmeta['train']
    blocks = assign_relaxed(meta, 3, 100)
    members, scene = structure(meta, blocks, list(range(len(y))))
    folds = [np.sort(np.load(OUT / f'scn{f}_reg_val.npz')['ids']) for f in range(5)]
    raw = sum(w * np.load(OUT / f'oof_{k}_b6.npy') for w, k in
              zip([.4, .2, .4], ['reg', 'r160', 'thmreg']))
    configs = [(e, t) for e in [.5, 1.] for t in [.5, .6, .7]]
    if args.confirm:
        configs = [tuple(json.loads((OUT / 'aligned_selection.json').read_text())['chosen'])]
    base = np.load(OUT / 'adaptive_repeats_b6_pred.npy')
    preds = {str(c): base.copy() for c in configs}
    for f in (range(5) if args.confirm else range(3)):
        va = folds[f]
        vm, vs = structure(meta, blocks, va.tolist())
        prior = fit_prior(y, members, scene, np.setdiff1d(np.arange(len(y)), va))
        edges = links(raw[va], vm, vs, [meta[i] for i in va])
        assert np.allclose(average(raw[va], edges, 0), baseline_average(raw[va], vm, vs, 1., .6), atol=1e-6)
        for c in configs:
            pred = decode(average(raw[va], edges, *c), vm, prior, 0., .3)
            preds[str(c)][va] = pred
            print('fold', f, c, 'accuracy', float((pred == y[va]).mean()),
                  'baseline', float((base[va] == y[va]).mean()), flush=True)
    dev, confirm = np.concatenate(folds[:3]), np.concatenate(folds[3:])
    chosen = max(configs, key=lambda c: (preds[str(c)][dev] == y[dev]).mean())
    result = dict(chosen=chosen, results={str(c): dict(development=float((p[dev] == y[dev]).mean()))
                                         for c, p in [(c, preds[str(c)]) for c in configs]})
    if args.confirm:
        pred = preds[str(chosen)]
        groups = np.array(regions(meta, allmeta['train_blocks'])[0])
        delta = (pred == y).astype(float) - (base == y)
        sums = np.array([delta[groups == g].sum() for g in np.unique(groups)])
        ns = np.array([(groups == g).sum() for g in np.unique(groups)])
        draws = np.random.default_rng(42).integers(len(ns), size=(5000, len(ns)))
        result.update(accuracy=float((pred == y).mean()), baseline=float((base == y).mean()),
                      confirmation=float((pred[confirm] == y[confirm]).mean()),
                      confirmation_base=float((base[confirm] == y[confirm]).mean()),
                      folds=[float((pred[v] == y[v]).mean()) for v in folds],
                      delta_ci95=np.quantile(sums[draws].sum(1) / ns[draws].sum(1), [.025, .975]).tolist())
        np.save(OUT / 'aligned_pred.npy', pred)
    (OUT / ('aligned_confirmed.json' if args.confirm else 'aligned_selection.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
