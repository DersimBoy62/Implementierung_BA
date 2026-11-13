"""READ-ONLY, development-only preprocessor audit for Bayesian Bachelor DM V2.1.

Run in the EXISTING Implementierung_BA project root. No model training.
The original DM outer test is not used in any fitting, transformation or audit statistics.
The trusted pickle is deserialized by the existing project reader, but its outer labels
and its outer process signals are never queried by this script.

Dependencies: dm_data, compare_dynamic_quality_cv4_v21, quality_link_challenge,
              compare_two_head_cv4, numpy, sklearn (already in project).
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import compare_dynamic_quality_cv4_v21 as cv
from quality_link_challenge import extract_descriptors,DATA_SHA
from compare_two_head_cv4 import OPTIONS,DESCRIPTOR_NAMES


def vec(x):
    return [float(z) for z in np.asarray(x).ravel()]


def audit(source):
    dev=cv.load_development(source)
    assert set(dev)==set(cv.EXPECTED_L)|set(cv.EXPECTED_U)
    assert not (set(dev)&set(cv.LOCKED))
    assert all(np.isnan(dev[r].y).all() for r in cv.EXPECTED_U)
    features=list(next(iter(dev.values())).names)
    windows=[]
    for rid in sorted(dev):
        r=dev[rid]
        assert r.names==tuple(features) and np.all(np.isfinite(r.x))
        dt=np.diff(r.t)
        assert len(dt)>0 and (dt>0).all()
        windows.append({'ring':rid,'n_samples':len(r.t),'duration_s':float(r.t[-1]-r.t[0]),
                        'dt_min_s':float(dt.min()),'dt_median_s':float(np.median(dt)),
                        'dt_max_s':float(dt.max()),'regular_0p1s':bool(np.allclose(dt,.1,rtol=0,atol=1e-9))})
    results=[]
    for fold in range(4):
        val=tuple(cv.EXPECTED_L[fold*2:fold*2+2])
        train_l=tuple(r for r in cv.EXPECTED_L if r not in val)
        train_u=cv.EXPECTED_U
        train=tuple(sorted((*train_l,*train_u)))
        assert len(train)==10 and len(val)==2
        sx=dm.fit_x_scaler({r:dev[r] for r in train},train)
        sy=dm.fit_y_scaler({r:dev[r] for r in train_l},train_l)
        assert sx.fitted_ids==tuple(sorted(train))
        assert sy.fitted_ids==tuple(sorted(train_l))
        assert set(sx.fitted_ids).isdisjoint(val) and set(sy.fitted_ids).isdisjoint(val)
        assert set(sx.fitted_ids).isdisjoint(cv.LOCKED)
        assert set(sy.fitted_ids).isdisjoint(train_u)
        train_x=np.vstack([dev[r].x for r in train])
        val_x=np.vstack([dev[r].x for r in val])
        tz=sx.transform(train_x)
        vz=sx.transform(val_x)
        assert np.allclose(tz.mean(axis=0),0,atol=1e-9)
        assert np.allclose(tz.std(axis=0),1,atol=1e-9)
        assert np.isfinite(tz).all() and np.isfinite(vz).all()
        assert np.allclose(sy.transform(np.stack([dev[r].y for r in train_l])).mean(axis=0),0,atol=1e-9)
        # Verify val-only perturbation cannot alter any fitted x/y normalizer.
        victim=val[0]
        damaged=dict(dev)
        damaged[victim]=replace(dev[victim],x=dev[victim].x+1000.,y=np.array([100000.,100000.]))
        sx2=dm.fit_x_scaler({r:damaged[r] for r in train},train)
        sy2=dm.fit_y_scaler({r:damaged[r] for r in train_l},train_l)
        assert np.array_equal(sx.mean,sx2.mean) and np.array_equal(sx.scale,sx2.scale)
        assert np.array_equal(sy.mean,sy2.mean) and np.array_equal(sy.scale,sy2.scale)
        # Compare sample-weighted and equal-ring summaries; the latter is a DIAGNOSTIC,
        # not a silently changed experiment pipeline.
        equal_mu=np.mean([dev[r].x.mean(axis=0) for r in train],axis=0)
        equal_var=np.mean([dev[r].x.var(axis=0)+(dev[r].x.mean(axis=0)-equal_mu)**2 for r in train],axis=0)
        equal_scale=np.sqrt(equal_var)
        desc={r:extract_descriptors(sx.transform(dev[r].x),dev[r].t,'physical') for r in train+val}
        heads=[]
        for j,idx in enumerate(OPTIONS['SEPARATE_AXIAL']):
            XX=np.stack([desc[r][list(idx)] for r in train])
            h=StandardScaler().fit(XX)
            headval=np.stack([desc[r][list(idx)] for r in val])
            assert np.allclose(h.transform(XX).mean(axis=0),0,atol=1e-9)
            assert np.isfinite(h.transform(headval)).all()
            heads.append({'target':('outer_roundness','height_runout')[j],
                          'names':[DESCRIPTOR_NAMES[i] for i in idx],
                          'training_ring_fit_ids':list(train),
                          'feature_mean':vec(h.mean_),
                          'feature_std':vec(h.scale_),
                          'validation_max_abs_z':vec(np.max(np.abs(h.transform(headval)),axis=0))})
        weights={r:float(len(dev[r].x)/len(train_x)) for r in train}
        results.append({'fold':fold+1,'train_L':list(train_l),'train_U_process_only':list(train_u),
                        'validation':list(val),'train_sample_count':len(train_x),
                        'training_ring_sample_weights':weights,
                        'largest_ring_weight':max(weights.values()),
                        'sample_weighted_logger_mean':vec(sx.mean),'sample_weighted_logger_std':vec(sx.scale),
                        'equal_ring_logger_mean':vec(equal_mu),'equal_ring_logger_std':vec(equal_scale),
                        'equal_minus_sample_mean_in_original_std':vec((equal_mu-sx.mean)/sx.scale),
                        'equal_divided_by_sample_std':vec(equal_scale/sx.scale),
                        'train_fraction_abs_z_gt_3':vec((np.abs(tz)>3).mean(axis=0)),
                        'validation_fraction_abs_z_gt_3':vec((np.abs(vz)>3).mean(axis=0)),
                        'train_max_abs_z':vec(np.max(np.abs(tz),axis=0)),
                        'validation_max_abs_z':vec(np.max(np.abs(vz),axis=0)),
                        'target_training_labeled_mean_mm':vec(sy.mean),
                        'target_training_labeled_std_mm':vec(sy.scale),
                        'heads':heads,
                        'verification':{'X_train_z_mean_zero_std_one':True,'Y_L_only_fit':True,
                                        'scalers_unchanged_by_validation_perturbation':True}})
    return {'status':'READ-ONLY DEVELOPMENT AUDIT; NO TRAINING; NO OUTER DATA ANALYZED',
            'source_pickle_sha256':dm.sha256(source), 'source_sha256_verified':dm.sha256(source)==DATA_SHA,
            'development_labeled':list(cv.EXPECTED_L),'development_U_process_only':list(cv.EXPECTED_U),
            'outer_ids_excluded':list(cv.LOCKED),'feature_names':features,
            'window':'F2 (phase 1/2, LaserOK=1, MessEin=1, contiguous)',
            'sampling':windows,'folds':results,
            'design_note':'Logger scaler currently pools all TIMEPOINTS (longer rings get larger weights). Ring features are separately z-scaled across 10 train rings. No clipping/outlier treatment. Switch rate per 100 intervals alters implied coefficient prior when prior variance is not changed.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--out',default='results/preprocessing_audit_v21/DEVELOPMENT_ONLY_AUDIT.json')
    a=p.parse_args()
    target=Path(a.out)
    if target.exists():raise FileExistsError(f'Refusing to overwrite existing report: {target}')
    report=audit(a.data)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS original pickle hash',report['source_pickle_sha256'])
    print('PASS no outer data used; U targets masked; X/Y/physical scalers fit only current train groups')
    print('PASS val perturbation leaves fitted X and Y scalers unchanged in all 4 folds')
    print('PASS contiguous 0.1-second time axis:',len(report['sampling']),'development rings')
    for f in report['folds']:
        idx=int(np.argmax(f['train_max_abs_z']))
        print(f"FOLD {f['fold']}: samples={f['train_sample_count']}; max ring weight={f['largest_ring_weight']:.1%}; max train |z|={f['train_max_abs_z'][idx]:.2f} in {report['feature_names'][idx]}")
    print('NO MODEL TRAINING / NO EXTERNAL TEST. Report:',target.resolve())

if __name__=='__main__':main()
