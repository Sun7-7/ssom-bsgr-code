"""Generate independent synthetic PDW records without private data inputs."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def generate(output, pulses=20000, seed=20261013):
    output.mkdir(parents=True, exist_ok=True)
    records=[]
    for k in (4, 6, 8):
        rng=np.random.default_rng(seed+k)
        labels=rng.integers(0,k,pulses)
        centres={'RF':np.linspace(2600,9400,k), 'PW':np.linspace(1.2,9,k),
                 'DOA':np.linspace(-55,55,k), 'PA':np.linspace(62,90,k)}
        jitter={'RF':90.,'PW':.35,'DOA':3.,'PA':2.5}
        values={'TOA':np.sort(rng.uniform(0,20000,pulses))}
        for key in centres:values[key]=centres[key][labels]+rng.normal(0,jitter[key],pulses)
        values['Label']=labels
        p=output/f'synthetic_k{k}.csv'
        pd.DataFrame(values).to_csv(p,index=False,float_format='%.8f',lineterminator='\n')
        records.append({'file':p.name,'pulses':pulses,'emitters':k,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'seed':seed,'generator':'independent_gaussian_pdw_v1',
              'private_data_inputs':False,'scope':'software execution demonstration; not paper metrics',
              'records':records}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'data/simulation')
    ap.add_argument('--pulses',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=20261013)
    args=ap.parse_args()
    if args.pulses<60:ap.error('pulses must be >= 60')
    print(json.dumps(generate(args.out,args.pulses,args.seed),indent=2))
