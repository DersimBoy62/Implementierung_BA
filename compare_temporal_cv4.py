"""CV4 auf acht inneren Entwicklungsringen: Bayesian-HMM mit dynamischem Emissionsmittelwert.

Trainiert AUSSCHLIESSLICH auf den 6 L + 4 U des jeweiligen inneren Folds.
Aeussere vier Testringe werden weder fuer Preprocessing noch Training noch Scoring verwendet.
Die bereits existierende Referenz-CV4 ist OPTIONAL (nur fuer expliziten Vergleich).
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import dm_data as dm
import temporal_b1 as tm

SOURCE_SHA='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/temporal_cv4_e10')
    ap.add_argument('--folds',type=int,nargs='+',choices=[1,2,3,4],default=[1,2,3,4])
    ap.add_argument('--degrees',type=int,nargs='+',choices=[0,1,2],default=[1])
    ap.add_argument('--features',choices=list(dm.SETS),default='V1_Kern_6')
    ap.add_argument('--window',choices=['F0','F1','F2'],default='F2')
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--C',type=float,default=1.)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--prior-df-offset',type=int,default=3)
    ap.add_argument('--prior-precision',type=float,default=1.)
    ap.add_argument('--slope-kappa',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--reference-cv4',default=None)
    a=ap.parse_args()
    if len(set(a.folds))!=len(a.folds) or len(set(a.degrees))!=len(a.degrees):
        raise ValueError('Keine doppelten Folds oder Grade')
    raw=dm.load_raw(a.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=a.seed)
    dev=tuple(sorted(set(split['L'])|set(split['U'])))
    assert len(dev)==12 and len(split['TEST_GESPERRT'])==4 and not(set(dev)&set(split['TEST_GESPERRT']))
    # Inner validation data are present ONLY in view, never in any fitted scaler or training data.
    ds=dm.view({r:raw[r] for r in dev},a.features,a.window)
    dm.refuse_post_quality_inputs(ds)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Ordner nicht leer: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    report={'scope':'4 FOLDS INNER CV; NO OUTER TEST; TIME NORMALIZATION uses full batch duration',
            'warning':'Explorative, bereits mehrfach ausgewertete Entwicklungsringe. Keine finale Modellwahl. '
                     'Posterior-predictive HMM X score approximiert Dirichlet-Zufallsvariablen durch ihren Mittelwert. '
                     'Zeitbasis am Batchende, NICHT fuer Online-Prognosen ohne Kenntnis der Ringdauer.',
            'config':vars(a),'outer_test_locked':list(split['TEST_GESPERRT']),
            'models':{},'folds':{}}
    for deg in a.degrees:
        rows=[];fold_reports=[]
        for i in a.folds:
            val=tuple(sorted(split['L'])[2*(i-1):2*i]);trL=tuple(r for r in split['L'] if r not in val)
            trU=tuple(split['U']);tr=tuple(sorted(set(trL)|set(trU)))
            assert len(trL)==6 and len(trU)==4 and len(val)==2 and not(set(tr)&set(val))
            assert not(set(tr)&set(split['TEST_GESPERRT']))
            sx=dm.fit_x_scaler({r:ds[r] for r in tr},tr)
            sy=dm.fit_y_scaler({r:ds[r] for r in trL},trL)
            X={r:sx.transform(ds[r].x) for r in tr}
            T={r:ds[r].t.copy() for r in tr}
            Y={r:sy.transform(ds[r].y) for r in trL}
            print(f'\n=== Fold {i}, degree={deg} === TRAIN L={trL} U={trU} VAL={val} OUTER-TEST GESPERRT',split['TEST_GESPERRT'],flush=True)
            f=tm.train(X,T,Y,degree=deg,K=a.K,C=a.C,epochs=a.epochs,local_steps=a.local_steps,
                       seed=a.seed,prior_options={'prior_df_offset':a.prior_df_offset,
                                                'prior_precision':a.prior_precision,
                                                'slope_kappa':a.slope_kappa},verbose=False)
            fr=[]
            for rid in val:
                x=sx.transform(ds[rid].x)
                mean,sd,_,_=tm.predict(x,ds[rid].t,f['process'],f['quality'])
                pred=mean*sy.scale+sy.mean
                err=(pred-ds[rid].y)/sy.scale
                score=tm.predictive_logx(x,ds[rid].t,f['process'])/len(x)
                entry={'ring':rid,'y_true_mm':ds[rid].y.tolist(),'y_pred_mm':pred.tolist(),
                       'error_norm_train_sd':err.tolist(),'rmse_error_mm':(pred-ds[rid].y).tolist(),
                       'X_approx_logscore_per_timepoint':float(score),'T':len(x)}
                rows.append(entry);fr.append(entry)
                print('  VAL',rid,'error_norm',np.round(err,3),'logscore X',round(score,3),flush=True)
            fold_reports.append({'fold':i,'train_L':trL,'train_U':trU,'inner_val':val,
                                 'xscaler_fitted':sx.fitted_ids,'yscaler_fitted':sy.fitted_ids,
                                 'history':f['history'],'validation':fr})
            print('  FINAL ELBO',round(f['history'][-1]['elbo'],2),'states',np.round(f['history'][-1]['states_effective'],3),flush=True)
        er=np.array([r['rmse_error_mm'] for r in rows]);er_norm=np.array([r['error_norm_train_sd'] for r in rows])
        result={'n_rings':len(rows),'rmse_mm_per_target':np.sqrt(np.mean(er**2,axis=0)).tolist(),
                'normalized_rmse_per_target':np.sqrt(np.mean(er_norm**2,axis=0)).tolist(),
                'mean_X_logp_approx_per_timepoint_per_ring':float(np.mean([r['X_approx_logscore_per_timepoint'] for r in rows])),
                'folds':fold_reports,'per_ring':rows}
        report['models'][f'degree_{deg}']=result
        print('DEGREE',deg,'CV RMSE mm',np.round(result['rmse_mm_per_target'],4),
              'X score',round(result['mean_X_logp_approx_per_timepoint_per_ring'],4),flush=True)
    if a.reference_cv4:
        reference=json.loads(Path(a.reference_cv4).read_text(encoding='utf-8'))
        c=reference['config']
        for key in ('K','epochs','C','seed','features','window','local_steps'):
            if key not in c or c[key]!=getattr(a,key):
                # previously CV4 uses hyphen-normalized key for local-steps in JSON via args.local_steps
                raise ValueError(f'Reference config mismatch: {key}, {c.get(key)} != {getattr(a,key)}')
        if set(reference['outer_test_locked'])!=set(split['TEST_GESPERRT']):
            raise ValueError('Test-Splits mismatch')
        report['reference_CV4_baseline_and_stationary']={'baseline':reference['baseline'],
                                                         'diagonal':{k:v for k,v in reference['metrics']['diag'].items() if k!='per_ring'},
                                                         'full':{k:v for k,v in reference['metrics']['full'].items() if k!='per_ring'}}
        print('Referenzkonfiguration und Test-Split stimmen ueberein.',flush=True)
    (out/'TEMPORAL_CV4_SUMMARY.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Bericht:',(out/'TEMPORAL_CV4_SUMMARY.json').resolve(),flush=True)
    print('OUTER TEST GESPERRT',split['TEST_GESPERRT'],flush=True)

if __name__=='__main__':main()
