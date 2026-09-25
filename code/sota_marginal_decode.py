"""Minimize clip errors using posterior mass over distinct sequence paths."""
import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def paths_for(prob, transition, strength=.3, beam=512, topk=12):
    paths = np.empty((1, 0), dtype=np.int64)
    scores = np.zeros(1)
    logp = np.log(np.maximum(prob, 1e-12))
    for pos, lp in enumerate(logp):
        candidates = np.argsort(lp)[-max(topk, len(prob)):]
        ext = scores[:, None] + lp[candidates][None]
        if pos:
            ext += strength * transition[paths[:, -1, None], candidates[None]]
            ext[np.any(paths[:, :, None] == candidates[None, None], axis=1)] = -np.inf
        take = np.argsort(ext.ravel())[-min(beam, ext.size):]
        parent, choice = np.unravel_index(take, ext.shape)
        paths = np.concatenate([paths[parent], candidates[choice, None]], axis=1)
        scores = ext.ravel()[take]
    finite = np.isfinite(scores)
    return paths[finite], scores[finite]


def minimum_risk(paths, scores, classes, temperature=1.):
    weights = np.exp((scores - scores.max()) / temperature)
    weights /= weights.sum()
    posterior = np.zeros((paths.shape[1], classes))
    for j in range(paths.shape[1]):
        np.add.at(posterior[j], paths[:, j], weights)
    i, c = linear_sum_assignment(-posterior)
    pred = np.empty(paths.shape[1], int)
    pred[i] = c
    return pred, posterior


def main():
    from sota_data import OUT, manifests
    from sota_finish2 import structure
    from sota_group_relax import assign_relaxed
    from sota_context_prior import fit_prior
    from sota_aligned_repeats import average, links
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm', action='store_true')
    args = ap.parse_args()
    rows, _ = manifests()
    y = np.array([int(r['action_id']) for r in rows])
    meta = json.loads((OUT / 'blocks.json').read_text())['train']
    blocks = assign_relaxed(meta, 3, 100)
    members, scene = structure(meta, blocks, list(range(len(y))))
    folds = [np.sort(np.load(OUT / f'scn{f}_reg_val.npz')['ids']) for f in range(5)]
    raw = sum(w * np.load(OUT / f'oof_{k}_b6.npy') for w, k in
              zip([.4, .2, .4], ['reg', 'r160', 'thmreg']))
    base = np.load(OUT / 'aligned_pred.npy')
    configs = ['map512', '.5', '1.0', '2.0']
    if args.confirm:
        configs = [json.loads((OUT / 'marginal_selection.json').read_text())['chosen']]
    preds = {c: base.copy() for c in configs}
    for f in (range(5) if args.confirm else range(3)):
        va = folds[f]
        vm, vs = structure(meta, blocks, va.tolist())
        prior = fit_prior(y, members, scene, np.setdiff1d(np.arange(len(y)), va))
        p = average(raw[va], links(raw[va], vm, vs, [meta[i] for i in va]), 1., .7)
        for c in configs:
            preds[c][va] = p.argmax(1)
        for r in vm.values():
            if len(r) < 2:
                continue
            paths, scores = paths_for(p[r], prior['trans'])
            for c in configs:
                result = paths[scores.argmax()] if c == 'map512' else minimum_risk(paths, scores, 40, float(c))[0]
                preds[c][va[r]] = result
        print('fold', f, {c: float((preds[c][va] == y[va]).mean()) for c in configs},
              'baseline', float((base[va] == y[va]).mean()), flush=True)
    dev, hold = np.concatenate(folds[:3]), np.concatenate(folds[3:])
    chosen = max(configs, key=lambda c: (preds[c][dev] == y[dev]).mean())
    result = dict(chosen=chosen, baseline=float((base[dev] == y[dev]).mean()),
                  results={c: float((preds[c][dev] == y[dev]).mean()) for c in configs})
    if args.confirm:
        pred = preds[chosen]
        result.update(accuracy=float((pred == y).mean()), confirmation=float((pred[hold] == y[hold]).mean()),
                      confirmation_base=float((base[hold] == y[hold]).mean()),
                      folds=[float((pred[v] == y[v]).mean()) for v in folds])
        np.save(OUT / 'marginal_pred.npy', pred)
    (OUT / ('marginal_confirmed.json' if args.confirm else 'marginal_selection.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
