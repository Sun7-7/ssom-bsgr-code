"""Exercise actual main CLI and reject repeated-window or output regressions."""
from pathlib import Path
import tempfile, subprocess, sys
import numpy as np
import pandas as pd
from generate_simulation import generate

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import run_window

def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);data=root/'data';generate(data,pulses=6000)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_main.py'),
            '--pdw',str(data/'synthetic_k6.csv'),'--window','3000','--count','2',
            '--seed','123','--out',str(root/'out')],check=True)
        frame=pd.read_csv(root/'out/summary.csv')
        assert frame.window_start.tolist()==[0,3000]
        assert frame.seed.tolist()==[123,123]
        assert frame.encoder_updates.eq(0).all()
        assert np.isfinite(frame[['ACC','ARI','NMI','F1','K_abs_error']]).all().all()
        assert frame.ARI.min()>.5 and frame.ACC.min()>.5
        for start in (0,3000):
            with np.load(root/'out'/f'window{start}_seed123.npz') as z:
                assert z['pred'].shape==(3000,) and z['bmu'].shape==(3000,)
        print('PASS: actual CLI, two distinct windows, fixed encoder, finite metrics, saved predictions')

if __name__=='__main__':main()
