"""Exploratory development-only sensitivity: switch rates per 100 intervals.

Only 8 previously reused L development rings are validated; U={DM01,02,10,14}
are process-only. Four external rings are neither featurized nor evaluated.
The old outer test was already opened in a DIFFERENT, completed experiment.
No result produced here is an unbiased independent confirmation.

Requires: dm_data.py, quality_link_challenge.py, compare_two_head_cv4.py,
          two_head_b1.py, vb_b1.py, full_markov_b1.py, fullcov_b1.py
and the three new V2 scripts in the project root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
from quality_link_challenge import extract_descriptors, DATA_SHA, FIXED_SEED, LOCKED
from compare_two_head_cv4 import OPTIONS,DESCRIPTOR_NAMES
import two_head_b1 as old
import dynamic_quality_v21 as new

EXPECTED_L=('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
EXPECTED_U=('DM01','DM02','DM10','DM14')
assert LOCKED==('DM07','DM08','DM11','DM15')
assert tuple(sorted(set(EXPECTED_L)|set(EXPECTED_U)|set(LOCKED)))==tuple(sorted([f'DM{k:02d}' for k in range(1,19) if k not in (3,13)]))


def load_development(path):
    """Inspect trusted pickle hash, then restrict view WITHOUT ACCESSING outer/U truths.

    Pickle deserialization does bring outer ring Python objects into memory;
    neither their label fields nor process signals are read by V2. This is
    not a sandbox against an untrusted pickle. The first-class split is by ID.
    """
    if dm.sha256(path)!=DATA_SHA:raise RuntimeError('Unexpected source pickle SHA256')
    with open(path,'rb') as f:raw=pickle.load(f)  # original, locally trusted artifact only
    if tuple(sorted(raw))!=tuple(sorted(set(EXPECTED_L)|set(EXPECTED_U)|set(LOCKED))):
        raise RuntimeError('Source ring IDs do not match the pre-existing fixed split')
    s=dm.ring_split(raw,seed=FIXED_SEED)
    if tuple(s['L'])!=EXPECTED_L or tuple(s['U'])!=EXPECTED_U or tuple(s['TEST_GESPERRT'])!=LOCKED:
        raise RuntimeError('Split mismatch / sealed test protection')
    # Copy only development dict entries. Do NOT access raw[r]['zielwerte'] for U or TEST.
    devraw={r:raw[r] for r in EXPECTED_L}
    for r in EXPECTED_U:
        devraw[r]={'logger':raw[r]['logger'],
                   'zielwerte':{target:float('nan') for target in dm.TARGETS}}
    del raw
    viewed=dm.view(devraw,'V1_Kern_6','F2')
    dm.refuse_post_quality_inputs(viewed)
    assert set(viewed)==set(EXPECTED_L)|set(EXPECTED_U)
    assert all(np.isnan(viewed[r].y).all() for r in EXPECTED_U)
    return viewed


def evaluate(data,variant='M3',states=3,epochs=10,steps=8,folds=4,C=1.,out='results/dynamic_quality_v2/M3_K3_CV4',verbose=True):
    if variant not in ('M1','M3') or states not in (2,3,5,8) or folds not in (1,2,3,4):
        raise ValueError('V2.1 scaled-switch study supports M1/M3, K in 2/3/5/8, folds 1-4')
    if C not in (0.,1.):raise ValueError('Only controlled C=0/C=1 here; avoid unregistered sweeps')
    if epochs<1 or steps<0:raise ValueError('Wrong training configuration')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing to overwrite: {out}')
    # No destination created and no results written until source split / hash pass.
    rv=load_development(data)
    out.mkdir(parents=True,exist_ok=True)
    per_ring=[];fold_reports=[]
    feature_groups=OPTIONS['SEPARATE_AXIAL']
    for f in range(folds):
        val=EXPECTED_L[2*f:2*f+2]
        trL=tuple(r for r in EXPECTED_L if r not in val)
        trU=EXPECTED_U
        train=tuple(sorted((*trL,*trU)))
        assert len(val)==2 and len(trL)==6 and len(trU)==4
        assert not set(train)&set(val)
        assert not set(train)&set(LOCKED)
        assert not set(val)&set(LOCKED)
        sx=dm.fit_x_scaler({r:rv[r] for r in train},train)
        sy=dm.fit_y_scaler({r:rv[r] for r in trL},trL)
        # The only quality labels that enter fitting are exactly train-L.
        x={r:sx.transform(rv[r].x) for r in train}
        y={r:sy.transform(rv[r].y) for r in trL}
        ids=train+val
        desc={r:extract_descriptors(sx.transform(rv[r].x),rv[r].t,'physical') for r in ids}
        headfeat={r:[None,None] for r in ids}
        for j,idx in enumerate(feature_groups):
            fs=StandardScaler().fit(np.array([desc[r][list(idx)] for r in train]))
            for r in ids:
                headfeat[r][j]=fs.transform(desc[r][list(idx)][None,:])[0]
        assert sx.fitted_ids==tuple(sorted(train))
        assert sy.fitted_ids==tuple(sorted(trL))
        if verbose:
            print('FOLD',f+1,'VARIANT',variant,'K',states,'C',C,'TRAIN_L',trL,'TRAIN_U',trU,'VAL',val,flush=True)
        sdata={r:headfeat[r] for r in train}
        if variant=='M0':
            mdl=old.train(x,y,sdata,K=states,C=C,epochs=epochs,steps=steps,seed=FIXED_SEED)
            pred_fn=old.predict
        else:
            mdl=new.train(x,y,sdata,variant=variant,K=states,C=C,epochs=epochs,steps=steps,seed=FIXED_SEED)
            pred_fn=new.predict
        for rid in val:
            p,sd=pred_fn(sx.transform(rv[rid].x),headfeat[rid],mdl)
            pred_mm=p*sy.scale+sy.mean
            sd_mm=sd*sy.scale
            # The validation label is accessed strictly here, after prediction.
            residual=pred_mm-rv[rid].y
            row={'ring':rid,'fold':f+1,'truth_mm':rv[rid].y.tolist(),
                 'pred_mm':pred_mm.tolist(),'residual_mm':residual.tolist(),
                 'predictive_sd_mm':sd_mm.tolist()}
            per_ring.append(row)
            if verbose:print(' VAL',rid,'err_mm',np.round(residual,4),'SD_mm',np.round(sd_mm,4),flush=True)
        fold_reports.append({'fold':f+1,'train_L':trL,'train_U_process_only':trU,'val':val,
                             'process_scaler_fit_ids':sx.fitted_ids,
                             'target_scaler_fit_ids':sy.fitted_ids,
                             'physical_scaler_fit_ids':train,
                             'history':mdl['history']})
    residuals=np.asarray([r['residual_mm'] for r in per_ring],float)
    rmse=np.sqrt(np.mean(residuals**2,axis=0))
    mae=np.mean(np.abs(residuals),axis=0)
    report={'status':'EXPLORATORY DEVELOPMENT-ONLY; REUSED CV; NOT NEW CONFIRMATORY TEST',
            'warning':'The 4 external ring labels were opened in the old, completed experiment. They are NOT accessed here, and may not be used in V2 selection.',
            'research_branch':'V2.1-SWITCH-PER-100-SENSITIVITY; DO NOT MERGE WITH V2',
            'switch_rate_scaling':{'unit':'expected transitions per 100 adjacent measurement intervals in each half',
                                   'multiplier_from_V2_rate':100.0,
                                   'coefficient_prior_variance_retained':0.15,
                                   'note':'This changes the effective prior on the original switch-rate scale'},
            'source_data_sha256':DATA_SHA,
            'configuration':{'variant':variant,'states':states,'epochs':epochs,'steps':steps,'folds':folds,'C':C,
                             'feature_set':'V1_Kern_6','window':'F2','seed':FIXED_SEED,'quality_head_groups':[[DESCRIPTOR_NAMES[i] for i in group] for group in feature_groups],
                             'latent_quality_features':(['occupancy'] if variant=='M0' else
                                                         ['occupancy']+(['early_late_switch_rates_per_100_intervals'] if variant in ('M1','M3') else [])+
                                                         (['late_minus_early_occupancy'] if variant in ('M2','M3') else []))},
            'development_labeled':EXPECTED_L,'development_unlabeled':EXPECTED_U,
            'outer_ids_excluded':LOCKED,'folds':fold_reports,'per_ring':per_ring,
            'metrics':{'rmse_mm':rmse.tolist(),'mae_mm':mae.tolist(), 'n_validation_predictions':len(per_ring)}}
    outfile=out/f'{variant}_RATE100_K{states}_C{int(C)}_FOLDS{folds}.json'
    outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('DEV-ONLY RATE100',variant,'K',states,'FOLDS',folds,'RMSE mm',np.round(rmse,4),'MAE mm',np.round(mae,4))
    print('KEIN AEUSSERER TEST. Dokument:',outfile.resolve())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--variant',choices=('M1','M3'),default='M3')
    p.add_argument('--states',type=int,choices=(2,3,5,8),default=3)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--folds',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--C',type=float,choices=(0.,1.),default=1.)
    p.add_argument('--out',default='results/dynamic_quality_v21/M3_RATE100_K3_CV4')
    a=p.parse_args()
    evaluate(a.data,a.variant,a.states,a.epochs,a.steps,a.folds,a.C,a.out)
if __name__=='__main__':main()
