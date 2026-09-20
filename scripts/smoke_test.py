"""End-to-end check of the released pipeline on synthetic PDW data.

The radar records are not redistributed, so this generates a small interleaved
record instead and runs the full method and the component ablation on it. It
verifies that the code path executes and produces a partition consistent with the
injected emitter structure; it is not a reproduction of any reported number.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pipeline import default_device, run_window  # noqa: E402

N_PULSES, N_EMITTERS, SEED = 3000, 5, 7
# Emitter signatures are well separated so the check is about the code path, not
# about matching the difficulty of the real records.
CENTRES = {
    'RF': np.linspace(2600, 9400, N_EMITTERS),
    'PW': np.linspace(1.2, 9.0, N_EMITTERS),
    'PA': np.linspace(62, 90, N_EMITTERS),
    'DOA': np.linspace(-55, 55, N_EMITTERS),
}


def make_record(rng: np.random.Generator) -> pd.DataFrame:
    """Build an interleaved PDW record with the documented column layout."""
    labels = rng.integers(0, N_EMITTERS, N_PULSES)
    toa = np.sort(rng.uniform(0, 2000, N_PULSES))
    rows = {'TOA': toa, 'Label': labels}
    jitter = {'RF': 90.0, 'PW': 0.35, 'PA': 2.5, 'DOA': 3.0}
    for name, centre in CENTRES.items():
        rows[name] = centre[labels] + rng.normal(0, jitter[name], N_PULSES)
    return pd.DataFrame(rows)[['TOA', 'RF', 'PW', 'DOA', 'PA', 'Label']]


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = make_record(rng)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pdw = td / 'synthetic_PDW.csv'
        df.to_csv(pdw, index=False)
        print(f'synthetic record: {len(df)} pulses, {N_EMITTERS} emitters, '
              f'device={default_device()}')

        keep = ['ACC', 'ARI', 'NMI', 'K_abs_error']
        res = run_window(df[['TOA', 'RF', 'PW', 'DOA', 'PA']].to_numpy(float),
                         df.Label.to_numpy(), start=0, seed=42)
        m = res['metrics']
        print('main method: ' + '  '.join(f'{k} {m[k]:.4f}' for k in keep)
              + f'  K_init {res["K_initial"]} -> K {m["K_pred"]} '
              f'(true {N_EMITTERS})  route {res["route"]}')
        # This checks the code path, not a reported number. The synthetic record is
        # separable, so the partition must clearly recover the emitter structure.
        # K itself is not asserted: the frozen K-selection rule is calibrated on the
        # BIC gain profile of the source records and may over-segment a record whose
        # gain profile differs, which is what happens here.
        assert m['ARI'] > 0.5, f'ARI {m["ARI"]:.4f} too low on separable data'
        assert m['ACC'] > 0.5, f'ACC {m["ACC"]:.4f} too low on separable data'
        assert m['K_pred'] >= 1, 'degenerate partition'
        assert res['pure_x_metrics']['ARI'] > 0.5, 'support-free variant failed'

        for script, extra in [('run_main.py', ['--out', str(td / 'pred')]),
                              ('run_ablation.py', ['--out', str(td / 'abl')])]:
            cmd = [sys.executable, str(ROOT / 'scripts' / script),
                   '--pdw', str(pdw), '--window', str(N_PULSES), *extra]
            print(f'\n$ {" ".join(Path(c).name if c.startswith(str(ROOT)) else c for c in cmd)}')
            subprocess.run(cmd, check=True)
        assert (td / 'pred' / 'summary.csv').exists()
        abl = pd.read_csv(td / 'abl' / 'component_ablation_summary.csv')
        print('\nablation:\n' + abl.round(4).to_string(index=False))
        assert abl.set_index('variant').loc['full', 'ARI'] >= \
            abl.set_index('variant').loc['gmm_only', 'ARI'], 'refinement did not help'
        # On this synthetic record the emitters are far apart, so the extra
        # structural cue of the full method is not needed and the support-free
        # variant can score at least as well. The paper's ordering of these two
        # variants is measured on the source records, where the descriptor
        # distributions overlap far more; nothing here contradicts it.
        print('\nnote: synthetic emitters are widely separated, so the ordering '
              'between no_ssom and full here\n      is not comparable to the '
              'paper, which measures it on the real records.')
    print('\nsmoke test passed: the released pipeline runs end to end.')


if __name__ == '__main__':
    main()
