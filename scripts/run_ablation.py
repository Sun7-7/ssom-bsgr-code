"""Component ablation for SSOM-BSGR.

Every variant shares the same feature construction, the same GMM grid and the
same SSOM fit, so the table isolates the contribution of each component rather
than re-running an independent pipeline.

Variants, matching the paper:
    ssom_only    assign each pulse its occupied BMU index and stop
    gmm_only     the initial GMM partition, no refinement
    no_ssom      refinement with the descriptor-distance cue only (BSGR w/o SSOM)
    merge_only   apply the merge direction unconditionally
    split_only   apply the split direction unconditionally
    full         the gate selects between merge and split (SSOM-BSGR)

Example:
    python scripts/run_ablation.py --pdw data/PDW_data.csv --windows 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from pipeline import build_features, default_device, fit_ssom, gmm_grid  # noqa: E402
from refinement import (MERGE_D, MERGE_O, SPLIT_GAIN, MAX_SPLITS,  # noqa: E402
                        evaluate, merge_refine, merge_x_only, refine,
                        split_candidates, split_refine)

VARIANTS = ['ssom_only', 'gmm_only', 'no_ssom', 'merge_only', 'split_only', 'full']
METRIC_KEYS = ['ACC', 'ARI', 'NMI', 'F1', 'K_abs_error']


def run_variants(a: np.ndarray, y: np.ndarray, start: int, seed: int,
                 device: str) -> dict[str, np.ndarray]:
    """Return one predicted partition per variant for a single window."""
    x = build_features(a)
    _, initial = gmm_grid(x, start)
    bmu, _ = fit_ssom(a, device)
    cand = split_candidates(x, initial, 10100 + start + seed)

    preds = {'ssom_only': bmu, 'gmm_only': initial}
    preds['no_ssom'], _, _ = refine(x, None, initial, False, 10100 + start + seed)
    preds['merge_only'], _ = merge_refine(x, bmu, initial, MERGE_D, MERGE_O)
    preds['split_only'], _ = split_refine(initial, cand, SPLIT_GAIN, MAX_SPLITS)
    preds['full'], _, _ = refine(x, bmu, initial, True, 10100 + start + seed)
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pdw', required=True, type=Path)
    p.add_argument('--window', type=int, default=10000)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--windows', type=int, default=1,
                   help='number of consecutive windows to average over')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--label-column', default='Label')
    p.add_argument('--device', default=None, choices=[None, 'cuda', 'cpu'])
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.pdw)
    a_all = df[['TOA', 'RF', 'PW', 'DOA', 'PA']].to_numpy(float)
    y_all = df[args.label_column].to_numpy()

    rows = []
    for i in range(args.windows):
        s = args.start + i * args.window
        a, y = a_all[s:s + args.window], y_all[s:s + args.window]
        assert len(a) == args.window, f'window {s} only has {len(a)} pulses'
        for name, pred in run_variants(a, y, s, args.seed,
                                       args.device or default_device()).items():
            m = evaluate(y, pred)
            rows.append(dict(variant=name, window_start=s,
                             **{k: float(m[k]) for k in METRIC_KEYS}))
        print(f'  window {s} done', flush=True)

    frame = pd.DataFrame(rows)
    table = frame.groupby('variant', as_index=False)[METRIC_KEYS].mean()
    table['variant'] = pd.Categorical(table.variant, VARIANTS, ordered=True)
    table = table.sort_values('variant')
    print('\n' + table.round(4).to_string(index=False))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out / 'component_ablation_raw.csv', index=False)
        table.to_csv(args.out / 'component_ablation_summary.csv', index=False)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
