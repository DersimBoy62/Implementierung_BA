"""Train-only audit: Wishart prior scale, covariance conditioning and inner CV.

Run from Implementierung_BA after copying this script next to dm_data.py/fullcov_b1.py.
Never reads outer-test rings into model views or computes their targets.
This is a PRIOR-SCALE sensitivity test; it does NOT vary prior effective degrees-of-freedom.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import dm_data as dm
import fullcov_b1 as fc

SOURCE_SHA='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'

def cov_stats(p, X):
    K,d=p.mu.shape
    g=np.zeros(K)
    for x in X.values():
        g+=fc.infer_process_full(x,p)['fb'].gamma.sum(axis=0)
    rows=[]
    for k in range(K):
        # With Wishart precision Lambda ~ Wishart(W, nu), the expected covariance
        # E[Lambda^-1] = inv(W)/(nu-d-1) if nu>d+1.
        if p.nu[k]<=d+1: raise ValueError('Covariance posterior mean is undefined')
        cov=np.linalg.solve(p.W[k],np.eye(d))/(p.nu[k]-d-1)
        cov=(cov+cov.T)/2
        ev=np.linalg.eigvalsh(cov)
        sd=np.sqrt(np.diag(cov))
        cor=cov/np.outer(sd,sd)
        pairs=[(a,b,float(cor[a,b])) for a in range(d) for b in range(a+1,d)]
        peak=max(pairs,key=lambda t:abs(t[2]))
        rows.append({'state_index_0_based':int(k),'state_mass_fraction':float(g[k]/g.sum()),
                     'state_weighted_timepoints':float(g[k]),
                     'covariance_min_eigenvalue':float(ev[0]),'covariance_max_eigenvalue':float(ev[-1]),
                     'covariance_condition_number':float(ev[-1]/ev[0]),
                     'max_abs_covariance_correlation':float(abs(peak[2])),
                     'peak_pair_indices':list(peak[:2]),'peak_pair_correlation':peak[2],
                     'nu_posterior':float(p.nu[k]),'kappa_posterior':float(p.kappa[k])})
    return rows

def run(raw,args,fold_index,prior_precision,split):
    # Feature preparation explicitly excludes all four outer-test rings.
    dev_ids=sorted(set(split['L'])|set(split['U']))
    assert set(dev_ids).isdisjoint(split['TEST_GESPERRT'])
    dataset=dm.view({rid:raw[rid] for rid in dev_ids},args.features,args.window)
    dm.refuse_post_quality_inputs(dataset)
    inner_val=tuple(sorted(split['L'])[2*fold_index:2*fold_index+2])
    train_L=tuple(r for r in split['L'] if r not in inner_val)
    train_U=tuple(split['U'])
    train_ids=tuple(sorted(set(train_L)|set(train_U)))
    assert len(train_L)==6 and len(train_U)==4 and len(inner_val)==2
    assert not (set(inner_val)&set(train_ids))
    xscale=dm.fit_x_scaler(dataset,train_ids)
    yscale=dm.fit_y_scaler(dataset,train_L)
    X={rid:xscale.transform(dataset[rid].x) for rid in train_ids}
    Y={rid:yscale.transform(dataset[rid].y) for rid in train_L}
    val_X={rid:xscale.transform(dataset[rid].x) for rid in inner_val}
    fit=fc.train_full(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,
                      max_local_iter=args.local_steps,
                      prior_options={'prior_precision':float(prior_precision)},verbose=False)
    p=fit['process'];q=fit['quality']
    cstats=cov_stats(p,X)
    ring_rows=[]
    for rid in inner_val:
        yhat,_,_,_=fc.predict_full(val_X[rid],p,q)
        yhat=yhat*yscale.scale+yscale.mean
        y=dataset[rid].y
        ring_rows.append({'ring':rid,'y_true_mm':y.tolist(),'y_pred_mm':yhat.tolist(),
                          'y_error_mm':(yhat-y).tolist(),
                          'y_error_norm_train_sd':((yhat-y)/yscale.scale).tolist(),
                          'X_logscore_approx_per_step':float(fc.predictive_logx_full(val_X[rid],p)/len(val_X[rid]))})
    return {'fold':int(fold_index+1),'prior_precision':float(prior_precision),'train_L':list(train_L),
            'train_U':list(train_U),'inner_val':list(inner_val),
            'outer_test_locked':list(split['TEST_GESPERRT']),
            'scaler_fit_X_ids':list(xscale.fitted_ids),'scaler_fit_Y_ids':list(yscale.fitted_ids),
            'final_elbo':float(fit['history'][-1]['elbo']),'emission_covariances':cstats,
            'inner_val_rows':ring_rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/covariance_prior_stability')
    ap.add_argument('--features',default='V1_Kern_6',choices=list(dm.SETS))
    ap.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--C',type=float,default=1.)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--folds',type=str,default='all',help='all or one of 1,2,3,4')
    ap.add_argument('--prior-precisions',nargs='+',type=float,default=[.25,1.,4.])
    a=ap.parse_args()
    if any((not np.isfinite(v)) or v<=0 for v in a.prior_precisions):
        raise ValueError('Prior-präzisionen müssen positiv und endlich sein')
    fold_indices=range(4) if a.folds=='all' else [int(a.folds)-1]
    if any(f not in range(4) for f in fold_indices): raise ValueError('Fold 1-4 oder all')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Zielordner existiert und ist nicht leer: {out}')
    out.mkdir(parents=True,exist_ok=True)
    raw=dm.load_raw(a.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=a.seed)
    results=[]
    for fold in fold_indices:
        for prec in a.prior_precisions:
            print(f'Fold {fold+1}, Prior precision {prec:g}: trainiere FULL Gaussian-Wishart ...',flush=True)
            res=run(raw,a,fold,prec,split)
            results.append(res)
            (out/f'fold_{fold+1}_prior_{prec:g}.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
            for s in res['emission_covariances']:
                print(f"   Zustand {s['state_index_0_based']+1}: Masse {s['state_mass_fraction']:.3f}, "
                      f"cond {s['covariance_condition_number']:.2f}, "
                      f"max |corr(Sigma)| {s['max_abs_covariance_correlation']:.3f}",flush=True)
    compact=[]
    for r in results:
        for s in r['emission_covariances']:
            compact.append({'fold':r['fold'],'prior_precision':r['prior_precision'],**s})
    with (out/'covariance_conditioning_by_state.csv').open('w',encoding='utf-8-sig',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(compact[0]));wr.writeheader();wr.writerows(compact)
    summary=[]
    for prec in a.prior_precisions:
        matches=[r for r in results if r['prior_precision']==prec]
        errs=np.array([v['y_error_mm'] for r in matches for v in r['inner_val_rows']])
        norm=np.array([v['y_error_norm_train_sd'] for r in matches for v in r['inner_val_rows']])
        xscore=[v['X_logscore_approx_per_step'] for r in matches for v in r['inner_val_rows']]
        summary.append({'prior_precision':prec,'folds_run':len(matches),'inner_validation_rings':len(errs),
                        'RMSE_mm_per_target':np.sqrt(np.mean(errs**2,axis=0)).tolist(),
                        'NRMSE_trainSD_per_target':np.sqrt(np.mean(norm**2,axis=0)).tolist(),
                        'mean_X_approx_logscore_per_step_per_ring':float(np.mean(xscore)),
                        'condition_number_worst_all_states':float(max(s['covariance_condition_number'] for r in matches for s in r['emission_covariances']))})
        print('PRIOR',prec,'RMSE mm',np.round(summary[-1]['RMSE_mm_per_target'],4),
              'X logscore',round(summary[-1]['mean_X_approx_logscore_per_step_per_ring'],4),
              'MAX CONDITION',round(summary[-1]['condition_number_worst_all_states'],1),flush=True)
    full={'status':'EXPLORATIVE INNER-ONLY PRIOR-SCALE SENSITIVITY; OUTER TEST LOCKED',
          'config':vars(a),'note':'prior_precision variiert den Erwartungswert der Prior-Präzisionsmatrix, '
          'nicht die Wishart-Freiheitsgrade/Priorstärke. Keine finale hyperparametrische Auswahl. '
          'Zustandslabels zwischen Folds/Varianten nicht unmittelbar vergleichbar. '
          'Bewertungsbeobachtungseinheit ist Ring, nicht Zeitstempel.',
          'outer_test_locked':list(split['TEST_GESPERRT']),
          'summary':summary,'runs':results}
    (out/'PRIOR_SENSITIVITY_SUMMARY.json').write_text(json.dumps(full,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Prüfberichte:',out.resolve(),flush=True)

if __name__=='__main__':main()
