"""The SSOM-BSGR pipeline for one observation window.

Stages, in order:

1. **Feature construction** — RF, PW, PA and the sine/cosine pair of DOA are
   standardized. TOA is deliberately excluded: absolute arrival time does not
   identify an emitter across a window, and the paper reports the TOA condition
   as a no-op control.
2. **Initial statistical hypotheses** — a grid of tied-covariance Gaussian
   mixtures with K = 4..20. K is chosen by the first point where the BIC gain per
   sample falls below a fixed fraction of the early median gain, not by a
   validation score, so no labels or downstream metric enters the choice.
3. **Neurodynamic representation** — the raw descriptors in the order
   (RF, PW, PA, DOA) drive a frozen spiking encoder and an adaptive
   self-organizing map of 30 x 30 nodes; each pulse is assigned its BMU.
4. **Bidirectional structural granularity refinement** — see ``refinement.py``.

The labels are used only to score the result, never inside a fit call.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from refinement import evaluate, refine
from spiking_som import ModelConfig, STDPSSOM, TrainSchedule

# Frozen protocol (selected once on the source development data).
GMM_K_MIN, GMM_K_MAX = 4, 20
GMM_N_INIT, GMM_MAX_ITER, GMM_REG = 5, 400, 1e-5
GMM_GAIN_RATIO = 0.08
MAP_SIDE = 30
INPUT_ORDER = [1, 2, 4, 3]  # RF, PW, PA, DOA


def default_device() -> str:
    """Use CUDA when present, otherwise fall back to CPU."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_features(a: np.ndarray) -> np.ndarray:
    """Standardized descriptor features; ``a`` columns are TOA, RF, PW, DOA, PA."""
    angle = np.deg2rad(a[:, 3])
    return StandardScaler().fit_transform(
        np.column_stack([a[:, 1], a[:, 2], a[:, 4], np.sin(angle), np.cos(angle)]))


def gmm_grid(x: np.ndarray, start: int = 0):
    """Fit the tied-covariance grid and return the selected K and its partition.

    The rule reproduces the paper: compare successive BIC gains per sample and
    stop at the first K (from the third onwards) whose gain drops below
    ``GMM_GAIN_RATIO`` times the median of the first three gains.
    """
    fits = {}
    for k in range(GMM_K_MIN, GMM_K_MAX + 1):
        g = GaussianMixture(k, covariance_type='tied', n_init=GMM_N_INIT,
                            max_iter=GMM_MAX_ITER, reg_covar=GMM_REG,
                            random_state=5100 + start + 31 * k).fit(x)
        fits[k] = (g.score(x) * len(x), g.predict(x))
    gains = np.diff([fits[k][0] for k in range(GMM_K_MIN, GMM_K_MAX + 1)])
    base = max(float(np.median(gains[:3])), 1e-9)
    k0 = next((i + GMM_K_MIN for i, g in enumerate(gains)
               if i >= 2 and g < GMM_GAIN_RATIO * base), GMM_K_MAX)
    return int(k0), fits[k0][1]


def fit_ssom(a: np.ndarray, device: str):
    """Fit the frozen spiking encoder plus adaptive SSOM and return the BMUs.

    Only the map is adapted; the encoder projection stays frozen, because the
    encoder ablation shows ten STDP updates change final ARI by 8.5e-4 on average.
    """
    model = STDPSSOM(4, ModelConfig(output_shape=(MAP_SIDE, MAP_SIDE),
                                    preprocess='legacy_zscore_clip_minmax_l2',
                                    som_update_rule='mass_normalized', seed=42),
                     device=device)
    neural = a[:, INPUT_ORDER].astype(np.float32)
    model.fit(neural, schedule=TrainSchedule(mode='random_frozen'))
    bmu = model.predict_bmu(neural).cpu().numpy()
    return bmu, model


def run_window(a: np.ndarray, y: np.ndarray, start: int = 0, seed: int = 42,
               device: str | None = None, with_baselines: bool = True) -> dict:
    """Run the full method on one window and score the result.

    Args:
        a: descriptor array with columns TOA, RF, PW, DOA, PA.
        y: true emitter label per pulse, used only for scoring.
        start: first pulse index of the window, used to seed the GMM grid exactly
            as in the paper.
        seed: SSOM seed; the refinement seed is derived from it and ``start``.
        device: ``'cuda'`` or ``'cpu'``; defaults to CUDA when available.
        with_baselines: also compute the initial partition and the
            feature-distance-only variant, which the ablation table reports.

    Returns:
        A dict with the final labels under ``pred``, the diagnostics, and the
        comparison partitions when requested.
    """
    device = device or default_device()
    x = build_features(a)
    k0, initial = gmm_grid(x, start)
    bmu, model = fit_ssom(a, device)
    pred, operations, route = refine(x, bmu, initial, True, 10100 + start + seed)

    out = dict(pred=pred, bmu=bmu, initial=initial, K_initial=k0,
               operations=int(operations), route=route, seed=seed,
               encoder_updates=model.training_stats['encoder_updates'],
               metrics=evaluate(y, pred))
    if with_baselines:
        pure, pure_ops, pure_route = refine(x, None, initial, False,
                                            10100 + start + seed)
        out.update(pure_x=pure, pure_x_operations=int(pure_ops),
                   pure_x_route=pure_route, pure_x_metrics=evaluate(y, pure))
        out['initial_metrics'] = evaluate(y, initial)
    return out
