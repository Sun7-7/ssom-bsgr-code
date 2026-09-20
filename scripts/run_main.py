"""Run SSOM-BSGR on one or more windows of a PDW record.

Example:
    python scripts/run_main.py --pdw data/PDW_data.csv --window 10000 \
        --start 10000 --seed 42 --out predictions/

The record must have columns TOA, RF, PW, DOA, PA and a label column; see
docs/data_format.md. The paper's main evaluation uses 10000-pulse windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from pipeline import default_device, run_window  # noqa: E402

METRIC_KEYS = ['ACC', 'ARI', 'NMI', 'F1', 'K_abs_error']


def load_record(path: Path, label_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a PDW record and return the descriptor matrix and the labels."""
    df = pd.read_csv(path)
    needed = ['TOA', 'RF', 'PW', 'DOA', 'PA']
    missing = [c for c in needed if c not in df.columns]
    assert not missing, f'{path.name} is missing columns {missing}'
    assert label_column in df.columns, f'{path.name} has no {label_column!r} column'
    return df[needed].to_numpy(float), df[label_column].to_numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pdw', required=True, type=Path, help='PDW record (CSV)')
    p.add_argument('--window', type=int, default=10000, help='pulses per window')
    p.add_argument('--start', type=int, default=0, help='first pulse index')
    p.add_argument('--start-step', type=int, default=0,
                   help='if >0, evaluate consecutive windows of this stride')
    p.add_argument('--count', type=int, default=1, help='number of windows')
    p.add_argument('--seed', type=int, default=42, help='SSOM seed')
    p.add_argument('--label-column', default='Label')
    p.add_argument('--device', default=None, choices=[None, 'cuda', 'cpu'])
    p.add_argument('--out', type=Path, default=None, help='directory for predictions')
    args = p.parse_args()

    a_all, y_all = load_record(args.pdw, args.label_column)
    print(f'{args.pdw.name}: {len(a_all)} pulses, '
          f'{len(np.unique(y_all))} emitter identities, device='
          f'{args.device or default_device()}')

    rows = []
    for i in range(args.count):
        s = args.start + i * (args.start_step or 0)
        a, y = a_all[s:s + args.window], y_all[s:s + args.window]
        assert len(a) == args.window, f'window {s} only has {len(a)} pulses'
        res = run_window(a, y, start=s, seed=args.seed, device=args.device)
        m = res['metrics']
        rows.append(dict(window_start=s, K_true=int(len(np.unique(y))),
                         K_initial=res['K_initial'], K_pred=int(m['K_pred']),
                         route=res['route'], operations=res['operations'],
                         **{k: float(m[k]) for k in METRIC_KEYS}))
        print(f'  window {s}: ACC {m["ACC"]:.4f}  ARI {m["ARI"]:.4f}  '
              f'NMI {m["NMI"]:.4f}  K {m["K_pred"]} (true {len(np.unique(y))})  '
              f'route {res["route"]}')
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out / f'window{s}_seed{args.seed}.npz',
                                y=y, pred=res['pred'], bmu=res['bmu'],
                                initial=res['initial'])

    frame = pd.DataFrame(rows)
    summary = frame[METRIC_KEYS + ['K_abs_error']].mean().to_dict()
    print('\nmean over windows: ' + '  '.join(
        f'{k} {summary[k]:.4f}' for k in METRIC_KEYS))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out / 'summary.csv', index=False)
        (args.out / 'summary.json').write_text(
            json.dumps(dict(windows=len(frame), mean=summary), indent=2),
            encoding='utf8')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
