"""Pure post-hoc comparison of six *existing* development JSON reports.

No data pickle, no training, no outer truth or predictors loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from compare_preprocessing_cv4 import EXPECTED_L,EXPECTED_U,LOCKED


def summarize(root):
    root=Path(root)
    expected={(p,m) for p in ('P0','P1','P2') for m in ('M0','M2')}
    records={}
    for p,m in sorted(expected):
        path=root/f'CV4_{p}_{m}_K3'/f'PREP_{p}_{m}_K3_C1_FOLDS4.json'
        if not path.is_file():raise FileNotFoundError(path)
        data=json.loads(path.read_text(encoding='utf8'))
        cfg=data['configuration']
        if (cfg['preprocessing'],cfg['variant'],cfg['K'],cfg['C'],cfg['epochs'],cfg['steps'],cfg['folds'])!=(p,m,3,1.,10,8,4):
            raise AssertionError(f'Unexpected configuration: {path}')
        if tuple(data['outer_ids_excluded'])!=LOCKED or tuple(data['development_labeled'])!=EXPECTED_L or tuple(data['development_unlabeled_process_only'])!=EXPECTED_U:
            raise AssertionError(f'Partition mismatch: {path}')
        if tuple(x['ring'] for x in data['per_ring'])!=EXPECTED_L:
            raise AssertionError(f'Wrong/duplicate validation rows: {path}')
        if len(data['folds'])!=4:raise AssertionError('Missing fold')
        for f in data['folds']:
            if len(f['train_L'])!=6 or len(f['train_U_process_only'])!=4 or len(f['validation'])!=2 or set(f['x_scaler']['fit_ids'])!=set(f['train_L'])|set(f['train_U_process_only']):
                raise AssertionError('Fold fit mismatch')
            if set(f['validation'])&set(f['x_scaler']['fit_ids']) or set(f['validation'])&set(f['target_scaler_fit_ids']):
                raise AssertionError('Validation leakage')
        e=np.array([x['residual_mm'] for x in data['per_ring']])
        rmse=np.sqrt(np.mean(e**2,axis=0))
        mae=np.mean(np.abs(e),axis=0)
        if not (np.allclose(rmse,data['metrics']['rmse_mm'],atol=1e-11) and np.allclose(mae,data['metrics']['mae_mm'],atol=1e-11)):
            raise AssertionError('Metric mismatch')
        records[(p,m)]={'path':str(path),'report':data,'errors':e,'rmse':rmse,'mae':mae}
    print('6/6 BERiCHTE UND SPLITS VALIDiERT (NUR ENTWICKLUNGSDATEN)')
    for p in ('P0','P1','P2'):
        for m in ('M0','M2'):
            a=records[(p,m)]; print(p,m,'RMSE mm',np.round(a['rmse'],4),'MAE mm',np.round(a['mae'],4))
    print('PAIRED P0 VS P1/P2: delta_MSE mm2 (negativ = besser)')
    for m in ('M0','M2'):
        for p in ('P1','P2'):
            e=records[(p,m)]['errors'];ref=records[('P0',m)]['errors']
            print(m,p,'Delta MSE',np.round(np.mean(e*e-ref*ref,axis=0),4),
                  'Anzahl kleinerer Einzelring-Absolutfehler',np.sum(np.abs(e)<np.abs(ref),axis=0).tolist())
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='results/preprocessing_cv4')
    a=ap.parse_args();summarize(a.root)

if __name__=='__main__':main()
