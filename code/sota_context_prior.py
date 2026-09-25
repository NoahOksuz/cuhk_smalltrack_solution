"""Train-only action co-occurrence/transition priors with scene-held-out checks.

No test labels or manual test predictions. Each validation fold's prior sees
only its training labels. Folds 0-2 select strength; folds 3-4 confirm it.
"""
import argparse
import json
from collections import Counter
import numpy as np
from sota_structure import assign_distinct


def fit_prior(y, members, scene, ids, smoothing=3.):
    allowed = set(map(int, ids))
    selected = {b: r for b, r in members.items() if set(r) <= allowed}
    repeats = Counter(scene[b] for b in selected)
    pair = np.zeros((40, 40))
    trans = np.zeros((40, 40))
    for b, rows in selected.items():
        labels = y[rows]
        weight = 1 / repeats[scene[b]]
        for i, a in enumerate(labels):
            for j, c in enumerate(labels):
                if i != j:
                    pair[a, c] += weight
            if i + 1 < len(labels):
                trans[a, labels[i + 1]] += weight
    def ratio(count, symmetric):
        # Shrink towards independent class frequencies, not zero probability.
        q = (count.sum(0) + 1) / (count.sum() + 40)
        p = (count + smoothing * q[None]) / (count.sum(1, keepdims=True) + smoothing)
        score = np.log(np.maximum(p / q[None], 1e-8))
        return (score + score.T) / 2 if symmetric else score
    return dict(pair=ratio(pair, True), trans=ratio(trans, False))


def decode(prob, members, prior, pair_weight=0., trans_weight=0., beam=64, topk=12):
    if pair_weight == trans_weight == 0:
        return assign_distinct(prob, members)
    result = prob.argmax(1)
    logp = np.log(np.maximum(prob, 1e-12))
    for rows in members.values():
        if len(rows) < 2:
            continue
        paths = np.empty((1, 0), dtype=np.int64)
        scores = np.zeros(1)
        for pos, row in enumerate(rows):
            candidates = np.argsort(logp[row])[-max(topk, len(rows)):]
            ext = scores[:, None] + logp[row, candidates][None]
            if pos:
                ext += pair_weight / (len(rows) - 1) * prior['pair'][paths[:, :, None], candidates[None, None]].sum(1)
                ext += trans_weight * prior['trans'][paths[:, -1, None], candidates[None]]
                ext[np.any(paths[:, :, None] == candidates[None, None], axis=1)] = -np.inf
            take = np.argsort(ext.ravel())[-min(beam, ext.size):]
            parent, choice = np.unravel_index(take, ext.shape)
            paths = np.concatenate([paths[parent], candidates[choice, None]], axis=1)
            scores = ext.ravel()[take]
        result[rows] = paths[np.argmax(scores)]
    return result


def main():
    from sota_data import OUT, manifests
    from sota_finish2 import structure
    from sota_structure import average_repetitions
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='r160')
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--selection', default='r160')
    args = ap.parse_args()
    rows, _ = manifests()
    y = np.array([int(r['action_id']) for r in rows])
    meta = json.loads((OUT / 'blocks.json').read_text())
    members, scene = structure(meta['train'], meta['train_blocks'], list(range(len(y))))
    raw = (np.load(OUT / f'oof_{args.kind}_b5.npy') + np.load(OUT / 'oof_thmreg_b5.npy')) / 2
    base = assign_distinct(average_repetitions(raw, members, scene, .7), members)
    folds = [np.sort(np.load(OUT / f'scn{f}_reg_val.npz')['ids']) for f in range(5)]
    pairs = [(p, t) for p in [0., .25, .5, 1., 2.] for t in [0., .15, .3, .6, 1.]]
    target = OUT / f'context_prior_{args.kind}.json'
    if args.confirm:
        selection = json.loads((OUT / f'context_prior_{args.selection}.json').read_text())
        pairs = [tuple(selection['chosen'])]
    predictions = {str(pt): base.copy() for pt in pairs}
    report = {str(pt): [] for pt in pairs}
    for f in (range(5) if args.confirm else range(3)):
        va = folds[f]
        tr = np.setdiff1d(np.arange(len(y)), va)
        assert not set(tr) & set(va)
        vm, vs = structure(meta['train'], meta['train_blocks'], va.tolist())
        pr = average_repetitions(raw[va], vm, vs, .7)
        prior = fit_prior(y, members, scene, tr)
        for pw, tw in pairs:
            pred = decode(pr, vm, prior, pw, tw)
            predictions[str((pw, tw))][va] = pred
            report[str((pw, tw))].append(float((pred == y[va]).mean()))
        print('fold', f, 'base', float((base[va] == y[va]).mean()), flush=True)
    ids = np.concatenate(folds if args.confirm else folds[:3])
    ranked = sorted(pairs, key=lambda pt: (predictions[str(pt)][ids] == y[ids]).mean(), reverse=True)
    for pt in ranked[:12]:
        pred = predictions[str(pt)]
        print(pt, 'accuracy', float((pred[ids] == y[ids]).mean()), 'folds', report[str(pt)], flush=True)
    if args.confirm:
        pred = predictions[str(pairs[0])]
        hold = np.concatenate(folds[3:])
        delta = (pred == y).astype(float) - (base == y).astype(float)
        # Resample whole scene regions, preserving dependence between retakes.
        from sota_blocks import regions
        groups = np.array(regions(meta['train'], meta['train_blocks'])[0])
        sums = np.array([delta[groups == g].sum() for g in np.unique(groups)])
        ns = np.array([(groups == g).sum() for g in np.unique(groups)])
        rng = np.random.default_rng(42)
        draws = rng.integers(len(ns), size=(5000, len(ns)))
        boot = sums[draws].sum(1) / ns[draws].sum(1)
        result = dict(chosen=pairs[0], accuracy=float((pred == y).mean()), baseline=float((base == y).mean()),
                      confirmation=float((pred[hold] == y[hold]).mean()), confirmation_base=float((base[hold] == y[hold]).mean()),
                      delta_ci95=np.quantile(boot, [.025, .975]).tolist(), folds=report[str(pairs[0])])
        np.save(OUT / f'context_prior_{args.kind}_pred.npy', pred)
        (OUT / f'context_prior_{args.kind}_confirmed.json').write_text(json.dumps(result, indent=2))
        print('CONFIRM', result, flush=True)
    else:
        target.write_text(json.dumps(dict(chosen=ranked[0], results=report), indent=2))


if __name__ == '__main__':
    main()
