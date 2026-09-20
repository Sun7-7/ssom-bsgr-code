"""Bidirectional structural granularity refinement (BSGR).

The refinement stage takes the initial sample-level partition produced by the
tied-covariance GMM grid together with the BMU index of every pulse, and corrects
its granularity in two directions:

* **merge** — join two components whose descriptor means are close *and* whose
  distributions over SSOM nodes overlap strongly. The node-overlap term is the
  structural cue: two fragments of the same emitter tend to occupy the same
  region of the map even when their descriptor means have drifted apart.
* **split** — divide a component that is better described by two Gaussians, when
  the BIC gain per sample exceeds the threshold and both halves are large enough.

A gate then chooses which direction to apply for the current partition, so the
two corrections are alternatives rather than a fixed sequence. The thresholds are
frozen at the values reported in the paper; they were fixed once on the source
development data and are not re-tuned on any evaluation window.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from metrics import evaluate_clustering

# Frozen thresholds (selected on source development data only).
MERGE_D, MERGE_O = 0.5, 0.2
SPLIT_GAIN, MAX_SPLITS = 2.0, 2
GATE_D, GATE_O = 0.35, 0.3

# Node count of the 30 x 30 SSOM, used to normalise the support distributions.
N_NODES = 900


def support(bmu, pred, q):
    """Normalised distribution of one component's pulses over the SSOM nodes."""
    p = np.bincount(bmu[pred == q], minlength=N_NODES).astype(float) + 1e-6
    return p / p.sum()


def pair_stats(x, bmu, pred):
    """Descriptor distance and node-overlap similarity for every component pair."""
    labs = np.unique(pred)
    means = {q: x[pred == q].mean(0) for q in labs}
    supp = {q: support(bmu, pred, q) for q in labs}
    return [(a, b, float(np.linalg.norm(means[a] - means[b])),
             float(1 - jensenshannon(supp[a], supp[b], base=2)))
            for i, a in enumerate(labs) for b in labs[i + 1:]]


def merge_refine(x, bmu, pred0, d_thr, o_thr, max_merges=12):
    """Merge component pairs that are close in descriptor space and on the map."""
    pred = pred0.copy()
    steps = 0
    while len(np.unique(pred)) > 2 and steps < max_merges:
        eligible = [p for p in pair_stats(x, bmu, pred)
                    if (p[2] < d_thr and p[3] > o_thr) or p[2] < 0.08]
        if not eligible:
            break
        a, b, _, _ = max(eligible, key=lambda p: p[3] - p[2] / max(d_thr, 1e-6))
        pred[pred == b] = a
        _, pred = np.unique(pred, return_inverse=True)
        steps += 1
    return pred, steps


def split_candidates(x, pred, seed):
    """Components whose two-Gaussian BIC gain per sample is largest."""
    rows = []
    for q in np.unique(pred):
        idx = np.flatnonzero(pred == q)
        if len(idx) < 60:
            continue
        z = x[idx]
        g1 = GaussianMixture(1, covariance_type='diag', n_init=3, reg_covar=1e-5,
                             random_state=seed + int(q)).fit(z)
        g2 = GaussianMixture(2, covariance_type='diag', n_init=3, reg_covar=1e-5,
                             random_state=seed + 100 + int(q)).fit(z)
        gain = float((g1.bic(z) - g2.bic(z)) / len(z))
        local = g2.predict(z)
        if min(np.bincount(local)) < max(20, int(0.02 * len(pred))):
            continue
        rows.append((gain, q, idx, local))
    return sorted(rows, reverse=True, key=lambda z: z[0])


def split_refine(pred0, candidates, gain_thr, max_splits):
    """Apply the highest-gain splits that clear the threshold and size test."""
    pred = pred0.copy()
    steps = 0
    next_label = int(pred.max()) + 1
    for gain, q, idx, local in candidates:
        if gain <= gain_thr or steps >= max_splits:
            continue
        pred[idx[local == 1]] = next_label
        next_label += 1
        steps += 1
    _, pred = np.unique(pred, return_inverse=True)
    return pred, steps


def regime_gate(x, bmu, pred, gate_d, gate_o):
    """Choose the correction direction for the current partition."""
    pairs = pair_stats(x, bmu, pred)
    return 'merge' if (min((p[2] for p in pairs), default=np.inf) < gate_d or
                       max((p[3] for p in pairs), default=0.0) > gate_o) else 'split'


def x_pairs(x, pred):
    """Descriptor-only pair distances, used by the no-support ablation."""
    labs = np.unique(pred)
    means = {q: x[pred == q].mean(0) for q in labs}
    return [(a, b, float(np.linalg.norm(means[a] - means[b])))
            for i, a in enumerate(labs) for b in labs[i + 1:]]


def merge_x_only(x, pred0, threshold=MERGE_D, max_merges=12):
    """Merge using descriptor distance alone: the feature-distance-only variant."""
    pred = pred0.copy()
    steps = 0
    while len(np.unique(pred)) > 2 and steps < max_merges:
        eligible = [p for p in x_pairs(x, pred) if p[2] < threshold]
        if not eligible:
            break
        a, b, _ = min(eligible, key=lambda p: p[2])
        pred[pred == b] = a
        _, pred = np.unique(pred, return_inverse=True)
        steps += 1
    return pred, steps


def refine(x, bmu, pred, support_enabled, seed):
    """Apply BSGR to one window.

    Args:
        x: standardized descriptor features, one row per pulse.
        bmu: BMU index per pulse, or None when ``support_enabled`` is False.
        pred: initial component assignment from the GMM grid.
        support_enabled: whether the structural (node-overlap) cue is used.
        seed: seed for the per-component split candidates.

    Returns:
        ``(labels, operations, route)`` where ``route`` is the direction the gate
        selected and ``operations`` counts the applied corrections.
    """
    candidates = split_candidates(x, pred, seed)
    ps, ns = split_refine(pred, candidates, SPLIT_GAIN, MAX_SPLITS)
    if support_enabled:
        pm, nm = merge_refine(x, bmu, pred, MERGE_D, MERGE_O)
        route = regime_gate(x, bmu, pred, GATE_D, GATE_O)
    else:
        pm, nm = merge_x_only(x, pred)
        nearest = min((p[2] for p in x_pairs(x, pred)), default=np.inf)
        route = 'merge' if nearest < GATE_D else 'split'
    return ((pm, nm) if route == 'merge' else (ps, ns)) + (route,)


def k_diagnosis(y, pred):
    """How the predicted component count departs from the true emitter count."""
    tv, pv = np.unique(y), np.unique(pred)
    mat = np.array([[(np.asarray(y) == t)[np.asarray(pred) == p].sum() for p in pv]
                    for t in tv])
    split_counts = np.array([sum((row >= 5) & (row / row.sum() >= 0.01)) for row in mat])
    merge_counts = np.array([sum((col >= 5) & (col / col.sum() >= 0.01))
                             for col in mat.T])
    si = float(np.maximum(split_counts - 1, 0).mean())
    mi = float(np.maximum(merge_counts - 1, 0).mean())
    dom = ('mixed' if si > 0.1 and mi > 0.1 else 'split' if si > 0.1
           else 'merge' if mi > 0.1 else 'none')
    return dict(K_signed_error=int(len(pv) - len(tv)), K_exact=int(len(pv) == len(tv)),
                K_over=int(len(pv) > len(tv)), K_under=int(len(pv) < len(tv)),
                split_index=si, merge_index=mi,
                pred_cluster_purity=float(mat.max(0).sum() / mat.sum()),
                effective_component_count=int((mat.sum(0) >= 0.01 * len(y)).sum()),
                dominant_error=dom)


def segment_class_f1(y, pred, classes=(2, 4, 7)):
    """Per-emitter-count F1 after a Hungarian assignment of components to emitters."""
    vals = sorted(set(y) | set(pred))
    vi = {v: i for i, v in enumerate(vals)}
    cm = np.zeros((len(vals), len(vals)), int)
    for a, b in zip(y, pred):
        cm[vi[a], vi[b]] += 1
    rr, cc = linear_sum_assignment(-cm)
    mapping = {vals[c]: vals[r] for r, c in zip(rr, cc)}
    mapped = np.array([mapping.get(v, v) for v in pred])
    return {f'F1_class_{c}': f1_score(y == c, mapped == c, zero_division=0)
            for c in classes}


def evaluate(y, pred):
    """Clustering metrics plus the count and assignment diagnostics.

    ``K_abs_error`` is the mean absolute error of the estimated emitter count,
    which the paper reports alongside ACC/ARI/NMI.
    """
    met = {**evaluate_clustering(y, pred), **k_diagnosis(y, pred),
           **segment_class_f1(y, pred)}
    met['K_abs_error'] = abs(met['K_pred'] - met['K_true'])
    return met
