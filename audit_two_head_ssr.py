"""Confirmatory diagnostic within REUSED DM development CV folds: two-head SSR U ablation.

Three uses of U, matching the previously tested SSL ablation protocol:
 SUP_L_ONLY: only L enters X/physical scaler, KMeans and training.
 C0_SHARED_PREPROC: L+U enters X/physical scaler and KMeans, C=0 global U ELBO.
 SSL_U_C1: L+U enters preprocessing/init and C=1 global U ELBO.

There is NO new structure search here. Identical heads (SEPARATE_AXIAL) across modes.
The same 8 development validation rings have been inspected repeatedly: evaluation
is exploratory, NOT an unbiased generalization assessment.
4 external test rings never enter any processing or metrics.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import two_head_b1 as th
from compare_two_head_cv4 import OPTIONS
from quality_link_challenge import DATA_SHA,FIXED_SEED,LOCKED,extract_descriptors

MODES=('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')
HEADS=OPTIONS['SEPARATE_AXIAL']

def fold_split(split, fold):
    if fold not in (1,2,3,4):raise ValueError('fold must be 1..4')
    val=tuple(sorted(split['L'])[2*(fold-1):2*fold])
    lab=tuple(r for r in split['L'] if r not in val)
    U=tuple(split['U']);outer=tuple(split['TEST_GESPERRT'])
    if len(val)!=2 or len(lab)!=6 or len(U)!=4 or outer!=LOCKED:
        raise ValueError('Unexpected ring split')
    if set(val)&(set(lab)|set(U)|set(outer)) or (set(lab)|set(U))&set(outer):
        raise ValueError('Validation or outer TEST leakage')
    return lab,U,val

def mode_data(rings,lab,U,val,mode):
    if mode not in MODES:raise ValueError('Unknown training mode')
    tr=lab if mode=='SUP_L_ONLY' else tuple(sorted(set(lab)|set(U)))
    sx=dm.fit_x_scaler({r:rings[r] for r in tr},tr)
    sy=dm.fit_y_scaler({r:rings[r] for r in lab},lab)
    X={r:sx.transform(rings[r].x) for r in tr+val}
    Y={r:sy.transform(rings[r].y) for r in lab}
    d={r:extract_descriptors(X[r],rings[r].t,'physical') for r in tr+val}
    feats={r:[None,None] for r in tr+val}
    for j,idx in enumerate(HEADS):
        standard=StandardScaler().fit(np.array([d[r][list(idx)] for r in tr]))
        for r in tr+val:
            feats[r][j]=standard.transform(d[r][list(idx)][None,:])[0]
    C=1. if mode=='SSL_U_C1' else 0.
    assert set(Y)==set(lab) and not (set(U)|set(val)|set(LOCKED))&set(Y)
    assert not (set(val)|set(LOCKED))&set(tr)
    return tr,X,Y,feats,sx,sy,C

def rmse(rows):
    e=np.array([r['residual_mm'] for r in rows],float)
    return np.sqrt(np.mean(e*e,axis=0))

def paired_comparison(before,after):
    a={r['ring']:np.array(r['residual_mm'],float) for r in before}
    b={r['ring']:np.array(r['residual_mm'],float) for r in after}
    if set(a)!=set(b):raise ValueError('Ring sets must match')
    ids=sorted(a)
    old=np.vstack([a[r] for r in ids]);new=np.vstack([b[r] for r in ids])
    d=new**2-old**2
    loo=[np.mean(np.delete(d,i,axis=0),axis=0).tolist() for i in range(len(ids))] if len(ids)>1 else []
    return {'ring_ids':ids,'delta_mse_new_minus_old_mm2':d.mean(0).tolist(),
            'rings_better_per_target':np.sum(d<0,axis=0).tolist(),
            'loo_mean_delta_mse_mm2':loo,
            'loo_all_improvements':(np.all(np.array(loo)<0,axis=0)).tolist() if loo else [False,False]}

def validate_reference(reference,ssl_rows,tol=5e-6):
    if reference is None:return None
    ref=json.loads(Path(reference).read_text(encoding='utf8'))
    if tuple(ref['outer_test_locked'])!=LOCKED or ref['config']['variant']!='SEPARATE_AXIAL':
        raise ValueError('Reference split/config mismatch')
    base={r['ring']:r for r in ref['per_ring']}
    actual={r['ring']:r for r in ssl_rows}
    if set(base)!=set(actual):raise ValueError('Reference ring list mismatch')
    max_delta=0.
    for rid in actual:
        dif=np.max(np.abs(np.array(base[rid]['pred_mm'])-np.array(actual[rid]['pred_mm'])))
        max_delta=max(max_delta,float(dif))
    if max_delta>tol:raise ValueError(f'C1 reproduction mismatch max Δ={max_delta} mm')
    return {'reference_file':str(reference),'max_absolute_prediction_difference_mm':max_delta,'tolerance_mm':tol}

def evaluate(data,out,folds=(1,2,3,4),epochs=10,steps=8,reference=None):
    if epochs<1 or steps<0 or not folds or len(set(folds))!=len(folds):raise ValueError('Invalid run parameters')
    raw=dm.load_raw(data,expected_sha=DATA_SHA)
    split=dm.ring_split(raw,seed=FIXED_SEED)
    if tuple(split['TEST_GESPERRT'])!=LOCKED:raise ValueError('Outer split changed')
    dev=tuple(sorted(set(split['L'])|set(split['U'])))
    rings=dm.view({r:raw[r] for r in dev},'V1_Kern_6','F2')
    dm.refuse_post_quality_inputs(rings)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory is not empty: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows={m:[] for m in MODES};report_folds=[]
    for fold in folds:
        L,U,val=fold_split(split,fold)
        fold_report={'fold':fold,'train_labeled':L,'train_unlabeled':U,'inner_val':val,'outer_test_locked':LOCKED,'modes':{}}
        for mode in MODES:
            tr,X,Y,feat,sx,sy,C=mode_data(rings,L,U,val,mode)
            print('FOLD',fold,'MODE',mode,'train',len(tr),'label',len(Y),'unlabeled ELBO weight',C,flush=True)
            model=th.train({r:X[r] for r in tr},Y,{r:feat[r] for r in tr},K=3,C=C,
                           epochs=epochs,steps=steps,seed=FIXED_SEED)
            val_rows=[]
            for rid in val:
                pred,sd=th.predict(X[rid],feat[rid],model)
                pred_mm=pred*sy.scale+sy.mean
                resid=pred_mm-rings[rid].y
                row={'ring':rid,'fold':fold,'truth_mm':rings[rid].y.tolist(),'pred_mm':pred_mm.tolist(),
                     'residual_mm':resid.tolist(),'predictive_sd_mm':(sd*sy.scale).tolist()}
                rows[mode].append(row);val_rows.append(row)
                print('  VAL',rid,'error mm',np.round(resid,4),flush=True)
            fold_report['modes'][mode]={'x_scaler_fit':list(sx.fitted_ids),
                 'y_scaler_fit':list(sy.fitted_ids),
                 'physics_scaler_fit':list(tr),'quality_train_ids':list(Y),
                 'unlabeled_elbo_weight':C,'rows':val_rows,
                 'last_elbo':float(model['history'][-1]['elbo'])}
        report_folds.append(fold_report)
    metric={m:{'rmse_mm':rmse(rows[m]).tolist(),
               'mae_mm':np.mean(np.abs(np.array([r['residual_mm'] for r in rows[m]])),axis=0).tolist(),
               'coverage_95_moment_normal_approx':np.mean(
                   np.abs(np.array([r['residual_mm'] for r in rows[m]]))<=1.96*np.array([r['predictive_sd_mm'] for r in rows[m]]),axis=0).tolist(),
               'n_rings':len(rows[m])} for m in MODES}
    diffs={'U_PREPROCESS_AND_INIT':paired_comparison(rows['SUP_L_ONLY'],rows['C0_SHARED_PREPROC']),
           'U_ELBO_ONLY':paired_comparison(rows['C0_SHARED_PREPROC'],rows['SSL_U_C1']),
           'OVERALL_U':paired_comparison(rows['SUP_L_ONLY'],rows['SSL_U_C1'])}
    matched=validate_reference(reference,rows['SSL_U_C1']) if tuple(folds)==(1,2,3,4) else None
    rep={'status':'EXPLORATORY, SAME REPEATED 8 DEV VALIDATION RINGS; NOT FINAL GENERALIZATION',
         'mode_definitions':{'SUP_L_ONLY':'Only labeled train in all steps',
                             'C0_SHARED_PREPROC':'L+U for scaling and initialization, C=0 for unlabeled ELBO',
                             'SSL_U_C1':'L+U for scaling/init and C=1 for unlabeled ELBO'},
         'config':{'features':'V1_Kern_6','window':'F2','K':3,'head_variant':'SEPARATE_AXIAL',
                   'epochs':epochs,'local_steps':steps,'seed':FIXED_SEED,'data':str(data)},
         'outer_test_locked':LOCKED,'folds':report_folds,
         'metrics':metric,'paired_comparisons':diffs,'per_ring':rows,'C1_reference_match':matched,
         'uncertainty_note':'1.96*SD uses moment-based predictive SD, not exact posterior quantile'}
    fname=out/'TWO_HEAD_SSR_ABLATION_CV4.json'
    fname.write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf8')
    for mode in MODES:print(mode,'RMSE mm',np.round(metric[mode]['rmse_mm'],4),flush=True)
    for name,diff in diffs.items():print(name,'ΔMSE mm²',np.round(diff['delta_mse_new_minus_old_mm2'],6),
          'rings better',diff['rings_better_per_target'],'LOO gain all',diff['loo_all_improvements'],flush=True)
    print('OUTER TEST SEALED',LOCKED,'REPORT',fname.resolve(),flush=True)
    return rep

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    a.add_argument('--out',default='results/two_head_ssr_ablation_cv4')
    a.add_argument('--folds',nargs='+',type=int,choices=[1,2,3,4],default=[1,2,3,4])
    a.add_argument('--epochs',type=int,default=10)
    a.add_argument('--local-steps',type=int,default=8)
    a.add_argument('--reference-twohead',default=None)
    q=a.parse_args()
    evaluate(q.data,q.out,tuple(q.folds),q.epochs,q.local_steps,q.reference_twohead)
if __name__=='__main__':main()
