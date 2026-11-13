"""Compare dev-only V2/V2.1 reports; never reads pickle or outer test files.

Use existing, immutable CV4 report files. No training, no new evaluation.
Reject unexpected ring IDs / mismatched CV memberships.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np

DEV=('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
OUTER={'DM07','DM08','DM11','DM15'}

def load_report(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    cfg=d['configuration'];rows=d['per_ring']
    ids=[v['ring'] for v in rows]
    if len(ids)!=8 or tuple(sorted(ids))!=DEV or len(set(ids))!=8 or OUTER.intersection(ids):
        raise ValueError(f'Rejected non-development or non-CV4 rows: {path}')
    if cfg['folds']!=4 or int(cfg['epochs'])!=10 or int(cfg['steps'])!=8 or float(cfg['C'])!=1:
        raise ValueError(f'Unexpected CV configuration: {path}')
    if d.get('outer_ids_excluded')!=list(sorted(OUTER)):
        # Original reference uses order DM07,DM08,DM11,DM15; sorted matches.
        raise ValueError(f'Outer-exclusion metadata failed: {path}')
    if 'RATE100' in Path(path).name:
        b=d.get('switch_rate_scaling',{})
        if b.get('multiplier_from_V2_rate')!=100.0:
            raise ValueError('Missing documented V2.1 scaling')
    res=np.array([next(v for v in rows if v['ring']==r)['residual_mm'] for r in DEV],dtype=float)
    if res.shape!=(8,2) or not np.isfinite(res).all():
        raise ValueError('Non-finite residuals or wrong dimensions')
    return cfg, res

def summarize(paths):
    models=[]
    for p in paths:
        cfg,e=load_report(p)
        rmse=np.sqrt(np.mean(e**2,axis=0))
        mae=np.mean(abs(e),axis=0)
        models.append((f"{cfg['variant']:2s} K={cfg['states']}" + (' rate100' if 'RATE100' in p.name else ' original'),rmse,mae,e,p))
    models.sort(key=lambda a:(int(a[0].split('K=')[1].split()[0]),a[0]))
    print(f'{len(models)} DEV-ONLY CV4 Berichte. Dies ist KEINE neue Testauswertung.')
    print(f'{"Modell":<27} {"RMSE außen":>12} {"RMSE Höhe":>12} {"MAE außen":>12} {"MAE Höhe":>12}')
    for name,r,m,_,_ in models:
        print(f'{name:<27} {r[0]:12.6f} {r[1]:12.6f} {m[0]:12.6f} {m[1]:12.6f}')
    for _,_,_,_,p in models:
        if OUTER.intersection(str(p)):raise RuntimeError('Unacceptable path name')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',default='results/dynamic_quality_v2')
    parser.add_argument('--new',default='results/dynamic_quality_v21')
    a=parser.parse_args()
    files=[]
    for root in (Path(a.old),Path(a.new)):
        if root.exists(): files.extend(root.glob('**/*FOLDS4.json'))
    if not files:raise FileNotFoundError('No CV4 development reports in the supplied folders')
    summarize(files)
if __name__=='__main__':main()
