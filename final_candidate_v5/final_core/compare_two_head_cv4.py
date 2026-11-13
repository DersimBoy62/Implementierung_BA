"""Exploratory two-target quality head architecture test in fixed inner 4-fold DM CV.

Pre-registered candidates, neither nested optimization nor independent validation.
Outer 4 test rings are not loaded into view, preprocessor, train or validation.
Already reused development rings: DO NOT claim out-of-sample confirmation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import two_head_b1 as th
from quality_link_challenge import extract_descriptors,DATA_SHA,FIXED_SEED,LOCKED

DESCRIPTOR_NAMES=(
    'log_duration','avg_radial_force','avg_axial_force','std_radial_force',
    'std_axial_force','delta_outer_diameter','delta_wall_thickness','delta_height',
    'avg_radial_speed','std_radial_speed','max_radial_force','max_axial_force'
)
# Fixed without model selection: exterior roundness -> deformation and radial effort;
# axial runout -> axial loading and deformation (physically plausible, not verified).
OUTER=(0,1,3,5,6,10)
HEIGHT_AXIAL=(0,2,4,7,11)
OPTIONS={'SEPARATE_AXIAL':(OUTER,HEIGHT_AXIAL),
         'HEIGHT_STATE_ONLY':(OUTER,())}

def evaluate(data,out,epochs=10,local_steps=8,K=3,variant='SEPARATE_AXIAL',C=1.,reference=None):
    if variant not in OPTIONS or K!=3 or C!=1.:
        raise ValueError('Registered configuration only: K3,C1, one of two specific head designs')
    raw=dm.load_raw(data,expected_sha=DATA_SHA)
    split=dm.ring_split(raw,seed=FIXED_SEED)
    assert tuple(split['TEST_GESPERRT'])==LOCKED
    dev=tuple(sorted(set(split['L'])|set(split['U'])))
    rv=dm.view({r:raw[r] for r in dev},'V1_Kern_6','F2')
    dm.refuse_post_quality_inputs(rv)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output directory not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    feature_groups=OPTIONS[variant]
    all_rows=[];folds=[]
    for k in range(4):
        val=tuple(sorted(split['L'])[2*k:2*k+2])
        lab=tuple(r for r in split['L'] if r not in val)
        U=tuple(split['U']);train=tuple(sorted(set(lab)|set(U)))
        assert len(lab)==6 and len(U)==4 and len(val)==2
        assert not set(val)&set(train) and not set(split['TEST_GESPERRT'])&(set(train)|set(val))
        sx=dm.fit_x_scaler({r:rv[r] for r in train},train)
        sy=dm.fit_y_scaler({r:rv[r] for r in lab},lab)
        x={r:sx.transform(rv[r].x) for r in train}
        y={r:sy.transform(rv[r].y) for r in lab}
        desc={r:extract_descriptors(sx.transform(rv[r].x),rv[r].t,'physical') for r in train+val}
        head_feat={r:[None,None] for r in train+val}
        for j,idx in enumerate(feature_groups):
            if idx:
                fs=StandardScaler().fit(np.array([desc[r][list(idx)] for r in train]))
                for r in train+val:
                    head_feat[r][j]=fs.transform(desc[r][list(idx)][None,:])[0]
            else:
                for r in train+val:head_feat[r][j]=np.zeros(0)
        assert all(set(train).isdisjoint({r}) for r in val)
        print('FOLD',k+1,'VARIANT',variant,'L',lab,'U',U,'VAL',val,'OUTER-TEST GESPERRT',LOCKED,flush=True)
        model=th.train(x,y,{r:head_feat[r] for r in train},K=K,epochs=epochs,steps=local_steps,C=C,seed=FIXED_SEED)
        fold_rows=[]
        for r in val:
            pred,sd=th.predict(sx.transform(rv[r].x),head_feat[r],model)
            pred_mm=pred*sy.scale+sy.mean
            residual=pred_mm-rv[r].y
            row={'ring':r,'truth_mm':rv[r].y.tolist(),'pred_mm':pred_mm.tolist(),
                 'residual_mm':residual.tolist(),'residual_norm':(residual/sy.scale).tolist(),
                 'predictive_sd_mm':(sd*sy.scale).tolist(),
                 'interval_95_approx_covered':(np.abs(residual)<=1.96*sd*sy.scale).tolist()}
            all_rows.append(row);fold_rows.append(row)
            print('   VAL',r,'residual mm',np.round(residual,4),'SD mm',np.round(sd*sy.scale,4),flush=True)
        folds.append({'fold':k+1,'trainL':lab,'trainU':U,'val':val,
                      'x_fit_ids':sx.fitted_ids,'y_fit_ids':sy.fitted_ids,
                      'physical_fit_ids':train,'history':model['history'],'rows':fold_rows})
        print('   FINAL ELBO',round(model['history'][-1]['elbo'],2),'state masses',np.round(model['history'][-1]['state_mass'],3),flush=True)
    e=np.array([r['residual_mm'] for r in all_rows])
    en=np.array([r['residual_norm'] for r in all_rows])
    metr={'rmse_mm':np.sqrt(np.mean(e*e,axis=0)).tolist(),'mae_mm':np.mean(np.abs(e),axis=0).tolist(),
          'normalized_rmse':np.sqrt(np.mean(en*en,axis=0)).tolist(),
          'approx_95_coverage_8_rings':np.mean(np.array([r['interval_95_approx_covered'] for r in all_rows]),axis=0).tolist(),
          'n_val_rings':len(all_rows)}
    report={'status':'EXPLORATORY REUSED INNER VALIDATION FOLDS; NOT FINAL MODEL SELECTION',
            'model':'full Gaussian-Wishart process HMM, two separately dimensioned Bayesian NIG quality heads with joint Markov VI',
            'warning':'Approximate 95% intervals use 1.96*moment-based sd, not guaranteed calibrated predictive quantiles.',
            'config':{'variant':variant,'K':K,'C':C,'epochs':epochs,'local_steps':local_steps,
                      'features':'V1_Kern_6','window':'F2','seed':FIXED_SEED,'data':data,
                      'heads':[[DESCRIPTOR_NAMES[i] for i in ids] for ids in feature_groups]},
            'outer_test_locked':LOCKED,'folds':folds,'per_ring':all_rows,'metrics':metr}
    if reference:
        ref=json.loads(Path(reference).read_text())
        if tuple(ref['outer_test_locked'])!=LOCKED:raise ValueError('Reference outer test not sealed')
        rc=ref['config']
        if any(rc[x]!=report['config'][x] for x in ('seed','features','window')):
            raise ValueError('Reference mismatch')
        report['reference']={'full_hmm':ref['metrics']['full']['rmse_mm_per_target'],
                             'baseline':ref['baseline']['rmse_mm_per_target']}
    filename=f'TWO_HEAD_{variant}_CV4.json'
    (out/filename).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('FINAL',variant,'RMSE mm',np.round(metr['rmse_mm'],4),'normalized',np.round(metr['normalized_rmse'],4))
    print('OUTER-TEST gesperrt:',LOCKED)
    print('Bericht:',(out/filename).resolve())
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--variant',choices=list(OPTIONS),default='SEPARATE_AXIAL')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--local-steps',type=int,default=8)
    p.add_argument('--out',default='results/two_head_cv4')
    p.add_argument('--reference-cv4',default=None)
    a=p.parse_args()
    evaluate(a.data,a.out,a.epochs,a.local_steps,variant=a.variant,reference=a.reference_cv4)
if __name__=='__main__':main()
