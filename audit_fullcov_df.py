"""Inneres CV4: Wishart-Freiheitsgrade bei KONSTANTEM E[Praezision]=I.

Keine alten Dateien aendern, keine aeußeren Testdaten trainieren/validieren.
Neue Modellkopie `fullcov_b1_df` unterscheidet sich vom bisherigen Full-Modell
nur im optionalen Freiheitsgrad-Offset von make_priors_full().
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import dm_data as dm
import fullcov_b1_df as fc
from audit_fullcov_stability import cov_stats, SOURCE_SHA


def run_fold(raw, split, a, f, offset):
    outer = set(split['TEST_GESPERRT'])
    labeled = tuple(split['L'])
    unlabeled = tuple(split['U'])
    val = tuple(sorted(labeled)[2*f:2*f+2])
    train_L = tuple(r for r in labeled if r not in val)
    train_U = tuple(unlabeled)
    train_ids = tuple(sorted(set(train_L)|set(train_U)))
    dev_ids = tuple(sorted(set(labeled)|set(unlabeled)))
    assert len(val) == 2 and len(train_L) == 6 and len(train_U) == 4
    assert not (outer & set(dev_ids))
    assert not (set(val) & set(train_ids))
    dataset = dm.view({rid: raw[rid] for rid in dev_ids}, a.features, a.window)
    dm.refuse_post_quality_inputs(dataset)
    xsc = dm.fit_x_scaler(dataset, train_ids)
    ysc = dm.fit_y_scaler(dataset, train_L)
    X = {rid: xsc.transform(dataset[rid].x) for rid in train_ids}
    Y = {rid: ysc.transform(dataset[rid].y) for rid in train_L}
    predX = {rid: xsc.transform(dataset[rid].x) for rid in val}
    fitted = fc.train_full(X, Y, K=a.K, C=a.C, epochs=a.epochs, seed=a.seed,
                           max_local_iter=a.local_steps,
                           prior_options={'prior_precision':a.prior_precision,
                                          'prior_df_offset':int(offset)})
    p, q, prior = fitted['process'], fitted['quality'], fitted['prior']
    d = X[train_ids[0]].shape[1]
    precision_eig = np.linalg.eigvalsh(prior.nu[0] * prior.W[0])
    assert np.allclose(precision_eig, a.prior_precision, atol=1e-12)
    cstates = cov_stats(p, X)
    valrows = []
    for rid in val:
        mean, sd, _, _ = fc.predict_full(predX[rid], p, q)
        yhat = mean * ysc.scale + ysc.mean
        y = dataset[rid].y
        logx = fc.predictive_logx_full(predX[rid], p)/len(predX[rid])
        baseline = ysc.mean
        valrows.append({'ring':rid,
                        'true_mm':y.tolist(), 'pred_mm':yhat.tolist(),
                        'predictive_sd_mm':(sd*ysc.scale).tolist(),
                        'error_mm':(yhat-y).tolist(),
                        'error_norm_train_sd':((yhat-y)/ysc.scale).tolist(),
                        'baseline_error_mm':(baseline-y).tolist(),
                        'baseline_error_norm_train_sd':((baseline-y)/ysc.scale).tolist(),
                        'X_logscore_approx_per_step':float(logx)})
    return {'fold':f+1, 'prior_df_offset':int(offset), 'nu0':float(prior.nu[0]),
            'prior_precision':float(a.prior_precision),
            'prior_expected_precision_eigenvalues':precision_eig.tolist(),
            'prior_expected_covariance_eigenvalues': np.linalg.eigvalsh(np.linalg.inv(prior.W[0])/(prior.nu[0]-d-1)).tolist(),
            'train_L':list(train_L), 'train_U':list(train_U), 'inner_val':list(val),
            'outer_test_locked':sorted(outer), 'x_scaler_fitted_ids':list(xsc.fitted_ids),
            'y_scaler_fitted_ids':list(ysc.fitted_ids), 'quality_targets':2,
            'states':cstates,'valrows':valrows,
            'elbo':float(fitted['history'][-1]['elbo']),
            'history':[{'epoch':int(h['epoch']), 'elbo':float(h['elbo']),
                        'states_effective':h['states_effective']} for h in fitted['history']]}


def summarise(results, offset):
    matches=[r for r in results if r['prior_df_offset']==offset]
    rr=[v for r in matches for v in r['valrows']]
    errors=np.asarray([r['error_mm'] for r in rr]); norme=np.asarray([r['error_norm_train_sd'] for r in rr])
    be=np.asarray([r['baseline_error_mm'] for r in rr]); bn=np.asarray([r['baseline_error_norm_train_sd'] for r in rr])
    sc=np.asarray([r['X_logscore_approx_per_step'] for r in rr])
    states=[s for r in matches for s in r['states']]
    weighted_states=[s for s in states if s['state_mass_fraction']>=.05]
    return {'prior_df_offset':int(offset),'folds':len(matches), 'val_rings':len(rr),
            'RMSE_mm':np.sqrt((errors**2).mean(axis=0)).tolist(),
            'NRMSE_trainSD':np.sqrt((norme**2).mean(axis=0)).tolist(),
            'baseline_RMSE_mm':np.sqrt((be**2).mean(axis=0)).tolist(),
            'baseline_NRMSE_trainSD':np.sqrt((bn**2).mean(axis=0)).tolist(),
            'approx_X_logscore_mean_ring_step':float(sc.mean()),
            'max_condition_all_states':float(max(s['covariance_condition_number'] for s in states)),
            'max_condition_states_ge_5pct':float(max(s['covariance_condition_number'] for s in weighted_states)) if weighted_states else None,
            'min_state_mass':float(min(s['state_mass_fraction'] for s in states))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    parser.add_argument('--out',default='results/fullcov_prior_concentration_cv4')
    parser.add_argument('--features',default='V1_Kern_6',choices=list(dm.SETS))
    parser.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    parser.add_argument('--K',type=int,default=3)
    parser.add_argument('--C',type=float,default=1.)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--local-steps',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--folds',default='all', help='all oder 1,2,3,4')
    parser.add_argument('--prior-precision',type=float,default=1.0)
    parser.add_argument('--df-offsets',nargs='+',type=int,default=[2,3,10,30])
    a=parser.parse_args()
    if any(v<2 for v in a.df_offsets) or not np.isfinite(a.prior_precision) or a.prior_precision<=0:
        parser.error('df-offsets >=2, prior-precision positiv und endlich')
    if len(set(a.df_offsets))!=len(a.df_offsets):parser.error('keine doppelten df-offsets')
    if a.folds=='all': folds=range(4)
    else:
        try: folds=[int(a.folds)-1]
        except ValueError:parser.error('folds: all oder 1-4')
        if folds[0] not in range(4):parser.error('folds: all oder 1-4')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Zielordner muss leer oder neu sein: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    raw=dm.load_raw(a.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=a.seed)
    allresults=[]
    for f in folds:
        for offset in a.df_offsets:
            print(f'Fold {f+1}, nu0=d+{offset}, Eprior[Lambda]={a.prior_precision:g} I: Training ...',flush=True)
            r=run_fold(raw,split,a,f,offset)
            allresults.append(r)
            (out/f'fold_{f+1}_df_{offset}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
            print('  ELBO',round(r['elbo'],2),'Zustandsmassen',
                  np.round([s['state_mass_fraction'] for s in r['states']],3),
                  'schlechteste Kondition',round(max(s['covariance_condition_number'] for s in r['states']),1),flush=True)
    summary=[summarise(allresults,off) for off in a.df_offsets]
    rows=[{'fold':r['fold'],'df_offset':r['prior_df_offset'],**s} for r in allresults for s in r['states']]
    with (out/'covariance_by_state.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'status':'EXPLORATORY INNER CV; FOUR OUTER RINGS NEVER USED',
            'method':'vary nu0=d+offset and W0=prior_precision*I/nu0, so E_prior[Lambda]=prior_precision*I constant',
            'warning':'Fixed expected precision does not fix expected covariance; prior concentration differs. X log-score is approximate. No nested tuning, no outer-test assessment.',
            'config':vars(a),'outer_test_locked':list(split['TEST_GESPERRT']),
            'summary':summary,'runs':allresults}
    (out/'DF_CONCENTRATION_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    for r in summary:
        print('nu0=d+',r['prior_df_offset'],'RMSE mm',np.round(r['RMSE_mm'],4),
              'X logscore',round(r['approx_X_logscore_mean_ring_step'],4),
              'max-cond-all',round(r['max_condition_all_states'],1),
              'max-cond-mass>=5%',round(r['max_condition_states_ge_5pct'],1) if r['max_condition_states_ge_5pct'] else 'NA',flush=True)
    print('Berichte:',out.resolve())
    print('OUTER-TEST GESPERRT',sorted(split['TEST_GESPERRT']),flush=True)

if __name__=='__main__':main()
