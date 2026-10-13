"""冻结的双向结构粒度修正（BSKR）公共实现。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from metrics import evaluate_clustering


def support(bmu, pred, q):
    p = np.bincount(bmu[pred == q], minlength=900).astype(float) + 1e-6
    return p / p.sum()


def pair_stats(x, bmu, pred):
    labs = np.unique(pred)
    means = {q: x[pred == q].mean(0) for q in labs}
    supp = {q: support(bmu, pred, q) for q in labs}
    return [(a, b, float(np.linalg.norm(means[a] - means[b])),
             float(1 - jensenshannon(supp[a], supp[b], base=2)))
            for i, a in enumerate(labs) for b in labs[i + 1:]]


def merge_refine(x, bmu, pred0, d_thr, o_thr, max_merges=12):
    pred = pred0.copy(); steps = 0
    while len(np.unique(pred)) > 2 and steps < max_merges:
        eligible = [p for p in pair_stats(x, bmu, pred)
                    if (p[2] < d_thr and p[3] > o_thr) or p[2] < .08]
        if not eligible: break
        a, b, _, _ = max(eligible, key=lambda p: p[3] - p[2] / max(d_thr, 1e-6))
        pred[pred == b] = a
        _, pred = np.unique(pred, return_inverse=True); steps += 1
    return pred, steps


def split_candidates(x, pred, seed):
    rows = []
    for q in np.unique(pred):
        idx = np.flatnonzero(pred == q)
        if len(idx) < 60: continue
        z = x[idx]
        g1 = GaussianMixture(1, covariance_type="diag", n_init=3, reg_covar=1e-5,
                             random_state=seed + int(q)).fit(z)
        g2 = GaussianMixture(2, covariance_type="diag", n_init=3, reg_covar=1e-5,
                             random_state=seed + 100 + int(q)).fit(z)
        gain = float((g1.bic(z) - g2.bic(z)) / len(z)); local = g2.predict(z)
        if min(np.bincount(local)) < max(20, int(.02 * len(pred))): continue
        rows.append((gain, q, idx, local))
    return sorted(rows, reverse=True, key=lambda z: z[0])


def split_refine(pred0, candidates, gain_thr, max_splits):
    pred = pred0.copy(); steps = 0; next_label = int(pred.max()) + 1
    for gain, q, idx, local in candidates:
        if gain <= gain_thr or steps >= max_splits: continue
        pred[idx[local == 1]] = next_label; next_label += 1; steps += 1
    _, pred = np.unique(pred, return_inverse=True)
    return pred, steps


def regime_gate(x, bmu, pred, gate_d, gate_o):
    pairs = pair_stats(x, bmu, pred)
    return "merge" if (min((p[2] for p in pairs), default=np.inf) < gate_d or
                       max((p[3] for p in pairs), default=0.) > gate_o) else "split"


def refine(x, bmu, pred, support_enabled, seed):
    if not support_enabled:
        raise ValueError("This release includes only the main method")
    ps, ns = split_refine(pred, split_candidates(x, pred, seed), 2.0, 2)
    pm, nm = merge_refine(x, bmu, pred, .5, .2)
    route = regime_gate(x, bmu, pred, .35, .3)
    return ((pm, nm) if route == 'merge' else (ps, ns)) + (route,)

def evaluate(y, pred):
    m = evaluate_clustering(y, pred)
    m['K_abs_error'] = abs(m['K_pred'] - m['K_true'])
    return m
