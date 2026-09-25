"""The accepted test-time decoder, shared by validation, release building and inference.

1. Agreement-gated averaging across repeated recordings, including partial or
   unequal repetitions aligned by probability similarity (sota_aligned_repeats).
2. Per recording block, beam search over distinct-action paths scored by the
   train-only transition prior, then minimum expected clip-error assignment over
   the path posterior (sota_marginal_decode).
No labels enter either step; the prior is fit on training labels only.
"""
from sota_aligned_repeats import average, links
from sota_marginal_decode import paths_for, minimum_risk

DEFAULT = dict(extra=1., threshold=.7, strength=.3, beam=512, topk=12, temperature=1.)


def ensemble_decode(prob, members, scene, meta, prior, extra=1., threshold=.7, strength=.3,
                    beam=512, topk=12, temperature=1., return_marginals=False):
    """`members` maps block -> row positions of `prob`; `meta` is indexed by the same positions."""
    q = average(prob, links(prob, members, scene, meta), extra, threshold)
    pred = q.argmax(1)
    marginal = q.copy()
    for rows in members.values():
        if len(rows) < 2:
            continue
        paths, scores = paths_for(q[rows], prior['trans'], strength, beam, topk)
        pred[rows], marginal[rows] = minimum_risk(paths, scores, q.shape[1], temperature)
    return (pred, marginal) if return_marginals else pred
