"""Development-only Bayesian-HMM input-preprocessing comparison, 2026-11-11.

Exactly three *process-channel* preprocessing strategies:
  P0 pooled observation-wise mean/std (unchanged legacy dm_data.fit_x_scaler)
  P1 equal-ring mixture mean/std (each process sequence weighted 1/N)
  P2 pooled observation-wise median/IQR / 1.3489795 (normal consistency)

The HMM likelihood is not reweighted. All head input features are recomputed
from transformed process channels, then head-wise StandardScaler is fitted
on L+U training rings only, as in V2. Only M0/M2 and K=3 by design here.

WARNING: External ring labels for DM07/08/11/15 were accessed by an earlier,
completed experiment. This script never accesses their data and produces
exploratory reused development CV, NOT independent confirmatory results.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import compare_dynamic_quality_cv4 as original
from compare_two_head_cv4 import OPTIONS, DESCRIPTOR_NAMES
from quality_link_challenge import DATA_SHA, FIXED_SEED, LOCKED, extract_descriptors
import two_head_b1 as old
import dynamic_quality_v2 as v2

VALID_MODES = ('P0','P1','P2')
VALID_MODELS = ('M0','M2')
EXPECTED_L = original.EXPECTED_L
EXPECTED_U = original.EXPECTED_U
NORMAL_IQR = 1.3489795003921634  # Phi^{-1}(0.75)-Phi^{-1}(0.25)


def fit_process_scaler(rings, allowed_ids, mode):
    """Fit ONLY from allowed training process sequences; never access Y.

    For P1, each ring provides total weight one, regardless of T_i.
    P2 uses pooled timepoint median and IQR; it is not a ring-equal robust fit.
    """
    if mode not in VALID_MODES:
        raise ValueError('Expected P0/P1/P2')
    ids=tuple(sorted(allowed_ids))
    if not ids or len(ids)!=len(set(ids)) or set(ids)-set(rings):
        raise ValueError('Missing/duplicate/empty fit IDs')
    if set(ids) & set(LOCKED):
        raise AssertionError('External test ring requested for preprocessor fit')
    xs=[np.asarray(rings[i].x,dtype=float) for i in ids]
    d=xs[0].shape[1]
    if any(x.ndim!=2 or x.shape[1]!=d or not len(x) or not np.isfinite(x).all() for x in xs):
        raise ValueError('Invalid process sequences')
    if mode=='P0':
        # Explicit exact reference to existing preprocessing: reference parity.
        return dm.fit_x_scaler(rings,ids)
    if mode=='P1':
        # Equal-ring mixture of empirical distributions (not simply the
        # unweighted mean of individual within-ring standard deviations).
        ring_means=np.stack([x.mean(axis=0) for x in xs])
        ring_second=np.stack([np.mean(x*x,axis=0) for x in xs])
        center=ring_means.mean(axis=0)
        var=np.maximum(0.,ring_second.mean(axis=0)-center*center)
        scale=np.sqrt(var)
    else:
        pooled=np.vstack(xs)
        q25,center,q75=np.percentile(pooled,[25.,50.,75.],axis=0)
        scale=(q75-q25)/NORMAL_IQR
    scale=np.where(scale>1e-10,scale,1.)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ArithmeticError('Nonfinite scaler parameters')
    return dm.Scaler(np.array(center,float),np.array(scale,float),ids)


def scaler_diagnostics(scaler,rv,train,val,mode):
    """Only descriptive X statistics; never consult ring quality values."""
    reports=[]
    for rid in tuple(sorted(train))+tuple(val):
        xs=scaler.transform(rv[rid].x)
        reports.append({'ring':rid,'partition':'train' if rid in train else 'validation',
                        'n_timepoints':len(xs),'fraction_abs_z_gt_3':(np.abs(xs)>3.).mean(axis=0).tolist(),
                        'max_abs_z':np.max(np.abs(xs),axis=0).tolist()})
    return {'mode':mode,'input_feature_names':list(rv[train[0]].names),
            'fit_ids':list(scaler.fitted_ids),
            'x_center':scaler.mean.tolist(),'x_scale':scaler.scale.tolist(),
            'ring_metrics':reports}


def evaluate(data='input/DM_16_aktive_Ringe.pkl', preprocessing='P0', variant='M0',
             states=3,epochs=10,steps=8,folds=4,C=1.,out='results/preprocessing_comparison/P0_M0_K3',verbose=True):
    if preprocessing not in VALID_MODES or variant not in VALID_MODELS:
        raise ValueError('Registered study: P0/P1/P2 and M0/M2 only')
    if states!=3 or C!=1. or folds not in (1,2,3,4) or epochs<1 or steps<0:
        raise ValueError('Registered study: K=3, C=1, valid epochs/steps/folds')
    target=Path(out)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Output directory already populated: {target}')
    if dm.sha256(data)!=DATA_SHA:
        raise RuntimeError('Input data hash mismatch; no evaluation')
    rv=original.load_development(data)
    assert set(rv)==set(EXPECTED_L)|set(EXPECTED_U)
    assert not set(rv)&set(LOCKED)
    assert all(np.isnan(rv[r].y).all() for r in EXPECTED_U)
    target.mkdir(parents=True,exist_ok=True)
    per_ring=[]
    fold_records=[]
    feature_groups=OPTIONS['SEPARATE_AXIAL']
    for f in range(folds):
        val=EXPECTED_L[2*f:2*f+2]
        trL=tuple(r for r in EXPECTED_L if r not in val)
        trU=EXPECTED_U
        train=tuple(sorted((*trL,*trU)))
        assert len(val)==2 and len(trL)==6 and len(trU)==4
        assert not (set(train)&set(val)) and not (set(LOCKED)&(set(train)|set(val)))
        sx=fit_process_scaler({rid:rv[rid] for rid in train},train,preprocessing)
        sy=dm.fit_y_scaler({rid:rv[rid] for rid in trL},trL)
        assert sx.fitted_ids==train and sy.fitted_ids==tuple(sorted(trL))
        x={rid:sx.transform(rv[rid].x) for rid in train}
        y={rid:sy.transform(rv[rid].y) for rid in trL}
        ids=train+val
        desc={rid:extract_descriptors(sx.transform(rv[rid].x),rv[rid].t,'physical') for rid in ids}
        heads={rid:[None,None] for rid in ids}
        for j,idx in enumerate(feature_groups):
            fit_desc=np.array([desc[rid][list(idx)] for rid in train])
            fs=StandardScaler().fit(fit_desc)
            for rid in ids:
                heads[rid][j]=fs.transform(desc[rid][list(idx)][None,:])[0]
        if verbose:
            print(f'FOLD {f+1} PREP {preprocessing} VARIANT {variant} K3 C1',
                  'TRAIN_L',trL,'TRAIN_U',trU,'VAL',val,flush=True)
        sdata={rid:heads[rid] for rid in train}
        if variant=='M0':
            mdl=old.train(x,y,sdata,K=3,C=1.,epochs=epochs,steps=steps,seed=FIXED_SEED)
            pred_fn=old.predict
        else:
            mdl=v2.train(x,y,sdata,variant='M2',K=3,C=1.,epochs=epochs,steps=steps,seed=FIXED_SEED)
            pred_fn=v2.predict
        for rid in val:
            p,sd=pred_fn(sx.transform(rv[rid].x),heads[rid],mdl)
            pred_mm=p*sy.scale+sy.mean
            sd_mm=sd*sy.scale
            residual=pred_mm-rv[rid].y  # First allowed val-quality access after prediction
            per_ring.append({'ring':rid,'fold':f+1,'truth_mm':rv[rid].y.tolist(),
                             'pred_mm':pred_mm.tolist(),'residual_mm':residual.tolist(),
                             'predictive_sd_mm':sd_mm.tolist()})
            if verbose:
                print(' VAL',rid,'err_mm',np.round(residual,4),'SD_mm',np.round(sd_mm,4),flush=True)
        fold_records.append({'fold':f+1,'train_L':trL,'train_U_process_only':trU,'validation':val,
                             'x_scaler':scaler_diagnostics(sx,rv,train,val,preprocessing),
                             'target_scaler_fit_ids':sy.fitted_ids,
                             'physical_scaler_fit_ids':train,
                             'training_history':mdl['history']})
    err=np.asarray([r['residual_mm'] for r in per_ring],dtype=float)
    rmse=np.sqrt(np.mean(err*err,axis=0))
    mae=np.mean(np.abs(err),axis=0)
    report={
        'status':'EXPLORATORY REUSED DEVELOPMENT FOLDS; NO INDEPENDENT VALIDATION',
        'warning':'External DM07/08/11/15 were used in an earlier completed project test. No external data accessed here.',
        'source_data_sha256':DATA_SHA,
        'configuration':{'preprocessing':preprocessing,'variant':variant,'K':3,'C':1.,'epochs':epochs,'steps':steps,
                         'folds':folds,'seed':FIXED_SEED,'feature_set':'V1_Kern_6','window':'F2',
                         'normal_iqr_constant':NORMAL_IQR if preprocessing=='P2' else None,
                         'physical_head_groups':[[DESCRIPTOR_NAMES[i] for i in group] for group in feature_groups],
                         'HMM_likelihood_timepoint_weighting':'unchanged across P0/P1/P2',
                         'normal_wishart_and_head_priors':'unchanged numeric values in transformed space',
                         'prior_warning':'Input scaling alters prior implied on physical scale'},
        'development_labeled':EXPECTED_L,'development_unlabeled_process_only':EXPECTED_U,
        'outer_ids_excluded':LOCKED,
        'folds':fold_records,'per_ring':per_ring,
        'metrics':{'rmse_mm':rmse.tolist(),'mae_mm':mae.tolist(),'n_validation_predictions':len(per_ring)}
    }
    filename=f'PREP_{preprocessing}_{variant}_K3_C1_FOLDS{folds}.json'
    p=target/filename
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('DEV-ONLY PREPROCESSING',preprocessing,variant,'K3 FOLDS',folds,
          'RMSE mm',np.round(rmse,4),'MAE mm',np.round(mae,4),flush=True)
    print('AUSSERER TEST NICHT ANGEFASST; ERGEBNIS:',p.resolve(),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--preprocessing',choices=VALID_MODES,default='P0')
    p.add_argument('--variant',choices=VALID_MODELS,default='M0')
    p.add_argument('--states',type=int,default=3)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--folds',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--C',type=float,default=1.)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    evaluate(data=a.data,preprocessing=a.preprocessing,variant=a.variant,states=a.states,
             epochs=a.epochs,steps=a.steps,folds=a.folds,C=a.C,out=a.out)

if __name__=='__main__':
    main()
