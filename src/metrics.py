import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
)


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def clustering_accuracy(y_true, y_pred, ignore_label=None):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return 0.0, {}

    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)
    tmap = {v: i for i, v in enumerate(true_ids)}
    pmap = {v: i for i, v in enumerate(pred_ids)}
    conf = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        conf[pmap[yp], tmap[yt]] += 1
    r, c = linear_sum_assignment(conf.max() - conf)
    mapping = {pred_ids[ri]: true_ids[ci] for ri, ci in zip(r, c)}
    mapped = np.array([mapping.get(v, -10**9) for v in y_pred])
    return float((mapped == y_true).mean()), mapping


def majority_vote_accuracy(y_true, y_pred, ignore_label=None):
    """Legacy SSOM node-label accuracy used by the archived radar experiments.

    Each occupied BMU receives the majority label of the samples mapped to it.
    Labels are used only for post-hoc evaluation; they never affect training.
    This many-to-one mapping is appropriate for a SOM node partition, where many
    prototypes may represent the same emitter. It must not be confused with the
    one-to-one Hungarian accuracy used after unknown-K consolidation.
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return 0.0, {}
    mapping = {}
    for cluster in np.unique(y_pred):
        labels = y_true[y_pred == cluster]
        values, counts = np.unique(labels, return_counts=True)
        mapping[int(cluster)] = int(values[counts.argmax()])
    mapped = np.asarray([mapping[int(v)] for v in y_pred], dtype=int)
    return float((mapped == y_true).mean()), mapping


def macro_node_purity(y_true, y_pred, ignore_label=None):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    purities = []
    for cluster in np.unique(y_pred):
        labels = y_true[y_pred == cluster]
        if len(labels):
            _, counts = np.unique(labels, return_counts=True)
            purities.append(float(counts.max() / len(labels)))
    return float(np.mean(purities)) if purities else 0.0


def purity_score(y_true, y_pred, ignore_label=None):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return 0.0
    total = 0
    for c in np.unique(y_pred):
        labels = y_true[y_pred == c]
        if len(labels):
            _, counts = np.unique(labels, return_counts=True)
            total += counts.max()
    return float(total / len(y_true))


def evaluate_clustering(y_true, y_pred, ignore_label=None):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    acc, mapping = clustering_accuracy(y_true, y_pred)
    mapped = np.array([mapping.get(v, -10**9) for v in y_pred])
    macro_f1 = f1_score(y_true, mapped, average="macro", zero_division=0)
    return {
        "ACC": acc,
        "NMI": normalized_mutual_info_score(y_true, y_pred) if len(y_true) else 0.0,
        "ARI": adjusted_rand_score(y_true, y_pred) if len(y_true) else 0.0,
        "Purity": purity_score(y_true, y_pred),
        "F1": float(macro_f1),
        "K_pred": int(len(np.unique(y_pred[y_pred >= 0]))),
        "K_true": int(len(np.unique(y_true))) if len(y_true) else 0,
    }


def evaluate_clustering_with_rejection(y_true, y_pred, reject_label=-1):
    """Cluster metrics where rejected real samples are always incorrect for ACC/F1."""
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    accepted = y_pred != reject_label
    if accepted.any():
        _, mapping = clustering_accuracy(y_true[accepted], y_pred[accepted])
    else:
        mapping = {}
    mapped = np.asarray([mapping.get(v, -10**9) for v in y_pred], dtype=int)
    return {
        "ACC": float((mapped == y_true).mean()) if len(y_true) else 0.0,
        "NMI": normalized_mutual_info_score(y_true, y_pred) if len(y_true) else 0.0,
        "ARI": adjusted_rand_score(y_true, y_pred) if len(y_true) else 0.0,
        "Purity": purity_score(y_true, y_pred),
        "F1": float(f1_score(y_true, mapped, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "K_pred": int(len(np.unique(y_pred[accepted]))),
        "K_true": int(len(np.unique(y_true))) if len(y_true) else 0,
        "real_reject_rate": float((~accepted).mean()) if len(y_true) else 0.0,
    }


def evaluate_bmu_partition(y_true, y_pred, ignore_label=None):
    """Evaluate the fine-grained BMU partition with the archived SSOM protocol."""
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_true, y_pred = y_true[mask], y_pred[mask]
    acc, mapping = majority_vote_accuracy(y_true, y_pred)
    mapped = np.asarray([mapping[int(v)] for v in y_pred], dtype=int) if len(y_pred) else np.array([], dtype=int)
    hungarian_acc, _ = clustering_accuracy(y_true, y_pred)
    return {
        "ACC": acc,
        "Hungarian_ACC": hungarian_acc,
        "NMI": normalized_mutual_info_score(y_true, y_pred) if len(y_true) else 0.0,
        "ARI": adjusted_rand_score(y_true, y_pred) if len(y_true) else 0.0,
        "Purity": purity_score(y_true, y_pred),
        "Macro_node_purity": macro_node_purity(y_true, y_pred),
        "F1": float(f1_score(y_true, mapped, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "K_pred": int(len(np.unique(y_pred[y_pred >= 0]))),
        "K_true": int(len(np.unique(y_true))) if len(y_true) else 0,
    }


def spurious_rejection_metrics(y_true, y_pred, spurious_label=-1, reject_label=-1):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    spur = y_true == spurious_label
    real = ~spur
    return {
        "spurious_recall": float((y_pred[spur] == reject_label).mean()) if spur.any() else np.nan,
        "real_reject_rate": float((y_pred[real] == reject_label).mean()) if real.any() else np.nan,
    }
