"""Aggregate strict 4-fold MANUAL state-number grid; only registered DEV data.

Reads JSON reports written by compare_states_cv4.py, never accesses pickle.
This repeated development CV is exploratory, not an independent benchmark.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from compare_states_cv4 import EXPECTED_L,EXPECTED_U,LOCKED,DATA_SHA,VALID_STATES


def read_report(p):
    r=json.loads(Path(p).read_text(encoding='utf8'))
    c=r['configuration']; name=(c['preprocessing'],c['variant'],int(c['K']))
    if name[0] not in ('P0','P2') or name[1] not in ('M0','M2') or name[2] not in VALID_STATES:
        raise ValueError(f'Unregistered config: {p}')
    if c['folds']!=4 or c['epochs']!=10 or c['steps']!=8 or c['C']!=1.:
        raise ValueError(f'Not full fixed-budget four-fold evaluation: {p}')
    if r['source_data_sha256']!=DATA_SHA:
        raise ValueError(f'Wrong data hash: {p}')
    if tuple(r['development_labeled'])!=EXPECTED_L or tuple(r['development_unlabeled_process_only'])!=EXPECTED_U:
        raise ValueError(f'Inconsistent development protocol: {p}')
    if tuple(r['outer_ids_excluded'])!=LOCKED:
        raise ValueError(f'Outer split mismatch: {p}')
    rows=r['per_ring']
    if len(rows)!=8 or set(z['ring'] for z in rows)!=set(EXPECTED_L):
        raise ValueError(f'Incomplete or repeated rings: {p}')
    folds=r['folds']
    if len(folds)!=4:
        raise ValueError(f'Incomplete folds: {p}')
    for i,f in enumerate(folds):
        expected_val=EXPECTED_L[2*i:2*i+2]
        expected_l=tuple(z for z in EXPECTED_L if z not in expected_val)
        if tuple(f['train_L'])!=expected_l or tuple(f['train_U_process_only'])!=EXPECTED_U or tuple(f['validation'])!=expected_val:
            raise ValueError(f'Fold leakage or mismatch: {p}')
        assert set(f['x_scaler']['fit_ids'])==set(expected_l)|set(EXPECTED_U)
        assert tuple(f['target_scaler_fit_ids'])==tuple(sorted(expected_l))
        diag=f['state_diagnostics']
        if len(diag['state_mass_timepoint_weighted'])!=name[2]:
            raise ValueError('Bad state diagnostic dimensions')
    for z in rows:
        if tuple(folds[z['fold']-1]['validation']).count(z['ring'])!=1:
            raise ValueError('Ring not in declared fold')
        for field in ('truth_mm','pred_mm','residual_mm','predictive_sd_mm'):
            arr=np.asarray(z[field],float)
            if arr.shape!=(2,) or not np.all(np.isfinite(arr)):
                raise ValueError(f'Invalid {field}')
        if not np.allclose(np.asarray(z['pred_mm'])-np.asarray(z['truth_mm']),z['residual_mm'],rtol=0,atol=1e-9):
            raise ValueError('Residual mismatch')
    residual=np.array([z['residual_mm'] for z in rows],float)
    rmse=np.sqrt(np.mean(residual**2,axis=0));mae=np.mean(np.abs(residual),axis=0)
    if not np.allclose(r['metrics']['rmse_mm'],rmse,atol=1e-9) or not np.allclose(r['metrics']['mae_mm'],mae,atol=1e-9):
        raise ValueError('RMSE or MAE mismatch')
    diag=[f['state_diagnostics'] for f in folds]
    return name, {'configuration':name,'source':str(p),'rmse_mm':rmse.tolist(),'mae_mm':mae.tolist(),
                  'mean_effective_states_entropy':float(np.mean([z['effective_states_entropy_exp'] for z in diag])),
                  'min_fold_state_mass':float(min(z['state_min_mass'] for z in diag)),
                  'min_fold_states_above_5pct':int(min(z['states_above_5pct'] for z in diag)),
                  'mean_posterior_self_transition':float(np.mean([z['posterior_mean_self_transition_mean'] for z in diag])),
                  'per_ring_squared_errors':{z['ring']:(np.asarray(z['residual_mm'],float)**2).tolist() for z in rows}}


def pareto_members(entries):
    result=[]
    for i,e in enumerate(entries):
        a=np.asarray(e['rmse_mm'])
        dominated=any(j!=i and np.all(np.asarray(f['rmse_mm'])<=a) and np.any(np.asarray(f['rmse_mm'])<a)
                      for j,f in enumerate(entries))
        if not dominated:result.append(tuple(e['configuration']))
    return result


def summarize(root):
    paths=sorted(Path(root).rglob('STATES_*_C1_FOLDS4.json'))
    if not paths: raise FileNotFoundError('No completed 4-fold state-number reports')
    items={}
    for p in paths:
        name,record=read_report(p)
        if name in items:raise ValueError(f'Duplicate configuration {name}')
        items[name]=record
    reference=items.get(('P0','M0',3))
    if reference:
        anchor=np.asarray(reference['rmse_mm'])**2
        for rec in items.values():
            rec['relative_mse_each_target_vs_P0M0K3']=(np.asarray(rec['rmse_mm'])**2/anchor).tolist()
            rec['symmetric_relative_mse_mean']=float(np.mean(np.asarray(rec['rmse_mm'])**2/anchor))
            rec['delta_mse_each_target_vs_P0M0K3']=(np.asarray(rec['rmse_mm'])**2-anchor).tolist()
    ordered=[items[k] for k in sorted(items)]
    return {'status':'EXPLORATORY MANUAL GRID ONLY; REPEATED DEVELOPMENT DATA','n_configs':len(items),
            'registered_n_configs':16,'outer_test_excluded':list(LOCKED),
            'objective_note':'Two distinct quality objectives; symmetric relative MSE is diagnostic only, anchored at P0/M0/K3 developer CV.',
            'pareto_configs':pareto_members(ordered),'results':ordered}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/state_grid_manual')
    p.add_argument('--out',default=None)
    a=p.parse_args();r=summarize(a.root)
    print(f"VOLLSTAENDIGE K-VERGLEICHE {r['n_configs']}/16 (NUR ENTWICKLUNGSDATEN)")
    for item in r['results']:
        prep,model,k=item['configuration']
        print(f"{prep:2} {model:2} K={k:2} RMSE {np.round(item['rmse_mm'],4)} "
              f"active>=5% minfold={item['min_fold_states_above_5pct']} "
              f"effK={item['mean_effective_states_entropy']:.2f}")
    print('PARETO:',r['pareto_configs'])
    if a.out:
        dest=Path(a.out)
        if dest.exists():raise FileExistsError(f'Refuse overwrite {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
        print('SAVED:',dest.resolve())

if __name__=='__main__':main()
