"""Ringwise 4-fold CV of FULL Gaussian-Wishart Bayesian HMM + physical quality covariates.
External test stays sealed. Only training L+U used for ring-feature standardization.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import hybrid_b1 as hb
from quality_link_challenge import extract_descriptors,DATA_SHA,FIXED_SEED,LOCKED

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/hybrid_b1_cv4')
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--physics-prior-variance',type=float,default=.25)
    ap.add_argument('--reference-cv4',default=None)
    a=ap.parse_args()
    raw=dm.load_raw(a.data,expected_sha=DATA_SHA)
    spl=dm.ring_split(raw,seed=FIXED_SEED)
    if tuple(spl['TEST_GESPERRT'])!=LOCKED:raise AssertionError('Wrong outer holdout')
    dev=tuple(sorted(set(spl['L'])|set(spl['U'])))
    rings=dm.view({i:raw[i] for i in dev},'V1_Kern_6','F2')
    dm.refuse_post_quality_inputs(rings)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory nonempty '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; folds=[]
    # 6 physical covariates fixed before seeing labels: duration, avg forces, deformation D/s/h.
    columns=[0,1,2,5,6,7]
    for k in range(4):
        val=tuple(sorted(spl['L'])[2*k:2*k+2]);L=tuple(r for r in spl['L'] if r not in val)
        U=tuple(spl['U']);tr=tuple(sorted(set(L)|set(U)))
        assert len(L)==6 and len(U)==4 and len(val)==2 and not set(tr)&set(val)
        assert not set(spl['TEST_GESPERRT'])&(set(tr)|set(val))
        sx=dm.fit_x_scaler({r:rings[r] for r in tr},tr)
        sy=dm.fit_y_scaler({r:rings[r] for r in L},L)
        x={r:sx.transform(rings[r].x) for r in tr}
        yf={r:sy.transform(rings[r].y) for r in L}
        cov={r:extract_descriptors(sx.transform(rings[r].x),rings[r].t,'physical')[columns] for r in tr+val}
        sc=StandardScaler().fit(np.array([cov[r] for r in tr]))
        sf={r:sc.transform(cov[r][None,:])[0] for r in cov}
        assert set(L)==set(yf) and set(tr)==set(x)
        print(f'FOLD {k+1} L={L} U={U} VAL={val} OUTER TEST={spl["TEST_GESPERRT"]}',flush=True)
        model=hb.train(x,yf,{r:sf[r] for r in tr},K=a.K,epochs=a.epochs,
                       steps=a.local_steps,seed=FIXED_SEED,
                       quality_physics_prior_var=a.physics_prior_variance)
        va=[]
        for r in val:
            pred,sd=hb.predict(sx.transform(rings[r].x),sf[r],model)
            predmm=pred*sy.scale+sy.mean
            residual=predmm-rings[r].y
            row={'ring':r,'y_true_mm':rings[r].y.tolist(),'y_pred_mm':predmm.tolist(),
                 'residual_mm':residual.tolist(),'residual_norm':(residual/sy.scale).tolist(),
                 'predictive_sd_mm':(sd*sy.scale).tolist()}
            rows.append(row);va.append(row)
            print('  VAL',r,'residual mm',np.round(residual,3),flush=True)
        folds.append({'fold':k+1,'trainL':L,'trainU':U,'val':val,
                      'x_scaler_fit_ids':sx.fitted_ids,'y_scaler_fit_ids':sy.fitted_ids,
                      'physics_scaler_fit_ids':tr,'history':model['history'],'rows':va})
        print('  FINAL ELBO',round(model['history'][-1]['elbo'],2),'states',np.round(model['history'][-1]['states_effective'],3),flush=True)
    err=np.array([r['residual_mm'] for r in rows]);errn=np.array([r['residual_norm'] for r in rows])
    metrics={'rmse_mm':np.sqrt(np.mean(err**2,axis=0)).tolist(),
             'normalized_rmse':np.sqrt(np.mean(errn**2,axis=0)).tolist(),
             'mae_mm':np.mean(np.abs(err),axis=0).tolist(),'n_val_rings':len(rows)}
    report={'status':'EXPLORATORY inner CV4; no outer holdout accessed',
            'description':'Full Gaussian-Wishart HMM with joint Markov VI and Bayesian ring-quality head: 3 state occupancies + six physical batch summaries; full h second moment; fixed prior variance .25 for physical covariates',
            'config':vars(a),'quality_physics_indices':columns,
            'outer_test_locked':spl['TEST_GESPERRT'],'folds':folds,'metrics':metrics,'per_ring':rows}
    if a.reference_cv4:
        ref=json.loads(Path(a.reference_cv4).read_text())
        if tuple(ref['outer_test_locked'])!=LOCKED:raise ValueError('Reference outer holdout mismatch')
        report['reference']={'baseline':ref['baseline']['rmse_mm_per_target'],
                             'full':ref['metrics']['full']['rmse_mm_per_target']}
    (out/'HYBRID_CV4.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('HYBRID FULL-JOINT CV4 RMSE mm',np.round(metrics['rmse_mm'],4))
    print('OUTER TEST SEALED:',spl['TEST_GESPERRT'])
    print('REPORT:',(out/'HYBRID_CV4.json').resolve())
if __name__=='__main__':main()
