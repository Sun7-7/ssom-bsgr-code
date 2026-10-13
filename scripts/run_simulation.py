"""Run the main method for all bundled simulation windows and seeds."""
from pathlib import Path
import argparse, subprocess, sys, json, hashlib, platform
import pandas as pd
import numpy as np
import torch, sklearn

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data/simulation')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/simulation')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,123,456])
    ap.add_argument('--device',choices=['cpu','cuda'],default=None)
    ap.add_argument('--window',type=int,default=10000)
    args=ap.parse_args()
    if args.window<60:ap.error('window must be >=60')
    if args.out.exists() and any(args.out.iterdir()):ap.error('Choose a new empty output directory')
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    frames=[]
    for record in manifest['records']:
        path=args.data/record['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Data hash mismatch: {path.name}')
        count=record['pulses']//args.window
        if count<1:ap.error('Not enough pulses for requested window')
        for seed in args.seeds:
            dest=args.out/path.stem/f'seed{seed}'
            cmd=[sys.executable,'-B',str(ROOT/'scripts/run_main.py'),'--pdw',str(path),
                 '--window',str(args.window),'--count',str(count),'--seed',str(seed),'--out',str(dest)]
            if args.device:cmd+=['--device',args.device]
            subprocess.run(cmd,check=True)
            frame=pd.read_csv(dest/'summary.csv');frame['scenario']=path.stem;frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(args.out/'per_run.csv',index=False)
    keys=['ACC','ARI','NMI','F1','K_abs_error']
    frame.groupby('scenario')[keys].mean().to_csv(args.out/'summary_by_scenario.csv')
    info={'scope':'Independent simulation only; not reproduction of withheld paper numbers',
          'runs':len(frame),'window':args.window,'seeds':args.seeds,'python':platform.python_version(),
          'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,
          'cuda_available':torch.cuda.is_available(),'mean':frame[keys].mean().to_dict(),
          'protocol_sha256':hashlib.sha256((ROOT/'configs/main.json').read_bytes()).hexdigest(),
          'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ['src','scripts'] for p in (ROOT/folder).glob('*.py')}}
    (args.out/'run.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
