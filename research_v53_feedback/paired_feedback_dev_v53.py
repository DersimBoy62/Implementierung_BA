#!/usr/bin/env python3
"""V5.3: fixed-configuration, paired DEV-ONLY local feedback ablation.

Two distinct checks:
(a) q(z) shift for local VI steps=0 vs 8 under IDENTICAL final globals;
(b) four-fold paired validation of two independently fitted models, with no tuning.

NEVER uses old test IDs DM07, DM08, DM11, DM15, or U quality labels.
Original 10 mathematical core modules are imported read-only, byte unchanged.
This is exploratory repeated-development analysis, NOT external confirmation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys
import platform
import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
V52 = ROOT / 'final_candidate_v52'
CORE = V52 / 'final_core'
if str(V52) not in sys.path: sys.path.insert(0, str(V52))
if str(CORE) not in sys.path: sys.path.insert(0, str(CORE))

import main_bayesian_ssr_v52 as main
import release_integrity_v52 as lock
import compare_dynamic_quality_cv4 as dev
import dm_data as dm
import dynamic_quality_v2 as dyn
from quality_link_challenge import extract_descriptors
from compare_two_head_cv4 import OPTIONS

STEPS = (0, 8)
FOLDS = (1, 2, 3, 4)
L = ('DM04', 'DM05', 'DM06', 'DM09', 'DM12', 'DM16', 'DM17', 'DM18')
U = ('DM01', 'DM02', 'DM10', 'DM14')
OLD_TEST = ('DM07', 'DM08', 'DM11', 'DM15')
PROTOCOL = 'V53_PAIRED_LOCAL_0_VS_8_DEV_CV4_2026-11-12_v1'
TARGETS = ('outside_ovality_mm', 'height_runout_mm')


def fail(message):
    raise RuntimeError(message)


def hard_guard(data=None):
    cfg = main.load_config()  # SHA-lock all 10 original core modules + exact live config
    if tuple(cfg['labeled_ring_ids']) != L or tuple(cfg['unlabeled_ring_ids']) != U:
        fail('Locked development split differs from registered protocol')
    if tuple(cfg['never_use_for_selection_ids']) != OLD_TEST:
        fail('Historical test split has changed')
    exact = {'K': 2, 'C': 0.2, 'physical_var': 0.38526108747593474,
             'epochs': 10, 'local_steps': 8, 'random_seed': 20261111}
    if any(cfg[k] != v for k, v in exact.items()):
        fail('Model parameter changed from registered candidate')
    if cfg['input_feature_set'] != 'V1_Kern_6' or cfg['process_window'] != 'F2':
        fail('Feature set / F2 changed')
    if tuple(cfg['quality_targets']) != dm.TARGETS:
        fail('Quality target mapping changed')
    if tuple(tuple(g) for g in OPTIONS['SEPARATE_AXIAL']) != ((0,1,3,5,6,10),(0,2,4,7,11)):
        fail('Descriptor groups changed')
    if data is not None and lock.sha_file(data) != cfg['dataset_sha256']:
        fail('Unknown source bytes: SHA256 mismatch (no pickle load)')
    return cfg


def fold_plan(f):
    if f not in FOLDS: fail('Unregistered fold')
    val = L[2*(f-1):2*f]
    trl = tuple(r for r in L if r not in val)
    train = tuple(sorted((*trl, *U)))
    if len(trl) != 6 or len(val) != 2 or len(train) != 10:
        fail('Fold dimensions changed')
    if set(train) & set(val) or (set(train)|set(val)) & set(OLD_TEST):
        fail('Leakage in ring allocation')
    return trl, U, val, train


def load_allowed(data):
    cfg = hard_guard(data)
    rv = dev.load_development(data)  # fixed 8L + 4 masked U, no use of old test X/y
    if set(rv) != set(L)|set(U): fail('Development-only view mismatch')
    if not all(np.isfinite(rv[r].y).all() for r in L): fail('L target invalid')
    if not all(np.isnan(rv[r].y).all() for r in U): fail('U target unexpectedly accessible')
    return cfg, rv


def prepare_fold(rv, f):
    trl, tru, val, train = fold_plan(f)
    rvtrain = {r:rv[r] for r in train}
    sx = dm.fit_x_scaler(rvtrain, train)
    sy = dm.fit_y_scaler({r:rv[r] for r in trl}, trl)
    if sx.fitted_ids != tuple(sorted(train)) or sy.fitted_ids != tuple(sorted(trl)):
        fail('Incorrect scaler fit IDs')
    xtrain = {r:sx.transform(rv[r].x) for r in train}
    ynorm = {r:sy.transform(rv[r].y) for r in trl}
    all_ids = train + val
    xall = {r:sx.transform(rv[r].x) for r in all_ids}
    phys = {r:extract_descriptors(xall[r],rv[r].t,'physical') for r in all_ids}
    hf = {r:[None,None] for r in all_ids}
    for j, ix in enumerate(OPTIONS['SEPARATE_AXIAL']):
        ix = list(ix)
        sf = StandardScaler().fit(np.stack([phys[r][ix] for r in train]))
        for r in all_ids:
            hf[r][j] = sf.transform(phys[r][ix][None,:])[0]
    return {'train_L': trl, 'train_U': tru, 'validation': val,
            'train': train, 'xtrain': xtrain, 'ynorm':ynorm,
            'head_train': {r:hf[r] for r in train},
            'head_all':hf, 'xall':xall, 'sy':sy,
            'process_scaler_fit_ids': sx.fitted_ids,
            'target_scaler_fit_ids': sy.fitted_ids,
            'physical_scaler_fit_ids': train}


def fixed_global_feedback(mdl, prep):
    """Direct effect: compare 0 vs 8 local iterations using EXACT SAME p, heads and labels."""
    rows=[]
    for rid in prep['train_L']:
        args=(prep['xtrain'][rid],mdl['process'],mdl['heads'],prep['ynorm'][rid],
              prep['head_train'][rid],mdl['spec'])
        a = dyn.local_joint(*args, steps=0)
        b = dyn.local_joint(*args, steps=8)
        delta = b['fb'].gamma - a['fb'].gamma
        rows.append({'ring_id':rid,'n_timepoints':len(prep['xtrain'][rid]),
                     'mean_abs_gamma_shift':float(np.mean(np.abs(delta))),
                     'max_abs_gamma_shift':float(np.max(np.abs(delta))),
                     'max_abs_M2_first_moment_shift':float(np.max(np.abs(b['m_g']-a['m_g']))),
                     'max_abs_M2_second_moment_shift':float(np.max(np.abs(b['H_g']-a['H_g']))),
                     'local_gain_fixed_globals':float(b['local_gain']),
                     'optimizer_accepted':bool(b['accepted']),
                     'optimizer_iterations':int(b['iterations']),
                     'occupancy_step0':np.mean(a['fb'].gamma,axis=0).tolist(),
                     'occupancy_step8':np.mean(b['fb'].gamma,axis=0).tolist()})
    return rows


def fold_run(cfg, rv, f):
    prep=prepare_fold(rv,f)
    out={'fold':f,'protocol':PROTOCOL,
         'train_L':list(prep['train_L']), 'train_U_process_only':list(prep['train_U']),
         'validation':list(prep['validation']),
         'scaler_fit_ids':{'X':list(prep['process_scaler_fit_ids']),
                           'y':list(prep['target_scaler_fit_ids']),
                           'physical_per_head':list(prep['physical_scaler_fit_ids'])},
         'arms':{},'paired_validation':[]}
    preds={}
    for steps in STEPS:
        print(f'FOLD {f} arm steps={steps}: start fixed-configuration fit',flush=True)
        model=dyn.train(prep['xtrain'],prep['ynorm'],prep['head_train'],variant='M2',
                        K=cfg['K'],C=cfg['C'],epochs=cfg['epochs'],
                        seed=cfg['random_seed'],steps=steps,physical_var=cfg['physical_var'],
                        verbose=True)
        history=model['history']
        armrows={}
        for rid in prep['validation']:
            mean,sd = dyn.predict(prep['xall'][rid],prep['head_all'][rid],model)
            pred = mean*prep['sy'].scale + prep['sy'].mean
            unc = sd*prep['sy'].scale
            if not(np.isfinite(pred).all() and np.isfinite(unc).all() and np.all(unc>0)):
                fail('Non-finite prediction or predictive SD')
            armrows[rid]={'prediction_mm':pred.tolist(), 'predictive_sd_mm':unc.tolist()}
        preds[steps]=armrows
        entry={'history':history,'final_ELBO':float(history[-1]['elbo']),
               'final_train_local_gain_sum':float(history[-1]['local_gain']),
               'validation_predictions_without_labels':armrows}
        if steps==8:
            entry['fixed_global_local_feedback_train_L']=fixed_global_feedback(model,prep)
        out['arms'][f'steps_{steps}']=entry
        print(f'FOLD {f} arm steps={steps}: finished. No val metrics until both arms end.',flush=True)
    # IMPORTANT: Only now is the validation outcome read. No val labels enter training.
    for rid in prep['validation']:
        truth=np.asarray(rv[rid].y,float)
        if truth.shape!=(2,) or not np.isfinite(truth).all():fail('Validation truth invalid')
        r={'ring_id':rid,'truth_mm':truth.tolist()}
        for steps in STEPS:
            pred=np.asarray(preds[steps][rid]['prediction_mm'])
            err=pred-truth
            r[f'error_steps_{steps}_mm']=err.tolist()
            r[f'abs_error_steps_{steps}_mm']=np.abs(err).tolist()
        r['delta_abs_error_8_minus_0_mm']=(np.abs(np.asarray(r['error_steps_8_mm']))-
                                           np.abs(np.asarray(r['error_steps_0_mm']))).tolist()
        out['paired_validation'].append(r)
    print(f'FOLD {f}: two-arm paired development diagnostic completed',flush=True)
    return out


def write_new_json(dest, payload):
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized=json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with dest.open('x',encoding='utf-8') as fh:fh.write(serialized)


def aggregate(fold_docs):
    if len(fold_docs)!=4 or set(int(d['fold']) for d in fold_docs)!=set(FOLDS):
        fail('All four unique folds must be provided for summary')
    all_rows=[]
    for d in fold_docs:
        if d['protocol'] != PROTOCOL:fail('Incompatible protocol in result')
        fold=int(d['fold'])
        trl,tru,val,tr=fold_plan(fold)
        if (tuple(d['train_L'])!=trl or tuple(d['train_U_process_only'])!=tru or
            tuple(d['validation'])!=val):fail('Fold allocation tampered')
        if set(d['arms']) != {'steps_0','steps_8'}:fail('Arms missing or changed')
        for r in d['paired_validation']:
            if r['ring_id'] not in val:fail('Ring outside fold found')
            all_rows.append(r)
    if len(all_rows)!=8 or set(r['ring_id'] for r in all_rows)!=set(L):
        fail('Missing/duplicate development ring prediction')
    e0=np.asarray([r['error_steps_0_mm'] for r in all_rows],float)
    e8=np.asarray([r['error_steps_8_mm'] for r in all_rows],float)
    if not (np.isfinite(e0).all() and np.isfinite(e8).all()):fail('Invalid residuals')
    rm0=np.sqrt(np.mean(e0**2,axis=0));rm8=np.sqrt(np.mean(e8**2,axis=0))
    mae0=np.mean(np.abs(e0),axis=0);mae8=np.mean(np.abs(e8),axis=0)
    return {'status':'EXPLORATORY repeated 8-development-label CV4, NOT independent validation',
            'protocol':PROTOCOL,'n_unique_dev_validation_rings':8,
            'outer_test_ids_NEVER_USED':list(OLD_TEST),
            'target_order':list(TARGETS),
            'RMSE_steps0_mm':rm0.tolist(),'RMSE_steps8_mm':rm8.tolist(),
            'delta_RMSE_8_minus_0_mm':(rm8-rm0).tolist(),
            'MAE_steps0_mm':mae0.tolist(),'MAE_steps8_mm':mae8.tolist(),
            'delta_MAE_8_minus_0_mm':(mae8-mae0).tolist(),
            'count_abs_error_better_with_8':(np.abs(e8)<np.abs(e0)).sum(axis=0).tolist(),
            'ringwise':sorted(all_rows,key=lambda r:r['ring_id']),
            'scientific_warning':'Do not use this comparison to optimize K, C, prior, epochs or steps retroactively; 8 labels have been repeatedly used in earlier studies.'}


def synthetic_tests():
    cfg=hard_guard()
    tests=[]
    for f in FOLDS:
        a,b,c,d=fold_plan(f)
        assert len(a)==6 and len(b)==4 and len(c)==2 and len(d)==10
    tests.append('four_pre_registered_disjoint_folds')
    assert all(r not in OLD_TEST for f in FOLDS for part in fold_plan(f) for r in part)
    tests.append('historical_outer_ring_exclusion')
    assert tuple(cfg['quality_targets'])==dm.TARGETS
    tests.append('exact_quality_target_binding')
    mock=[]
    for f in FOLDS:
        a,b,c,d=fold_plan(f)
        mock.append({'fold':f,'protocol':PROTOCOL,'train_L':list(a),
                     'train_U_process_only':list(b),'validation':list(c),
                     'arms':{'steps_0':{},'steps_8':{}},
                     'paired_validation':[{'ring_id':rid,'error_steps_0_mm':[2.,3.],
                                           'error_steps_8_mm':[1.,4.]} for rid in c]})
    sums=aggregate(mock)
    assert np.allclose(sums['RMSE_steps0_mm'],[2.,3.])
    assert np.allclose(sums['RMSE_steps8_mm'],[1.,4.])
    tests.append('paired_two_arm_aggregate_math')
    bad=json.loads(json.dumps(mock));bad[0]['validation']=['DM07','DM04']
    try: aggregate(bad)
    except RuntimeError:pass
    else:fail('Tampered validation allocation accepted')
    tests.append('tampered_fold_detected')
    print('V53 FEEDBACK PREP: '+str(len(tests))+'/'+str(len(tests))+' synthetic protocol tests PASS; NO DM PICKLE')
    for item in tests:print('PASS',item)


def main_cli():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('self-test',help='No DM data; protocol and hash sanity only')
    pp=sp.add_parser('preflight',help='Check trusted DM dataset SHA and allowed 8L+4U view; NO FIT')
    pp.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ff=sp.add_parser('run-fold',help='EXPLICIT dev-only paired 0 vs 8 local VI fit for ONE fold')
    ff.add_argument('--fold',type=int,choices=FOLDS,required=True)
    ff.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ff.add_argument('--out-dir',default='results/v53_feedback_devonly')
    aa=sp.add_parser('summarize',help='Aggregate only when all 4 original folds are available; no fitting')
    aa.add_argument('--out-dir',default='results/v53_feedback_devonly')
    a=p.parse_args()
    if a.command=='self-test':synthetic_tests();return
    if a.command=='preflight':
        cfg, rv=load_allowed(a.data)
        print(json.dumps({'status':'V53_DEVONLY_PREFLIGHT_PASS_NO_TRAINING',
                          'model_freeze_status':cfg['freeze_status'],
                          'source_sha256':cfg['dataset_sha256'],
                          'L':list(L),'U_MASKED':list(U),'OLD_TEST_NEVER_USED':list(OLD_TEST),
                          'dev_F2_timepoints':{r:int(len(rv[r].t)) for r in tuple(sorted(rv))},
                          'folds':[{ 'fold': f,'train_L':fold_plan(f)[0],
                                     'validation':fold_plan(f)[2]} for f in FOLDS]},
                         indent=2,ensure_ascii=False));return
    if a.command=='run-fold':
        cfg=hard_guard(a.data)
        dest=pathlib.Path(a.out_dir)/f'FOLD_{a.fold:01d}_PAIRED_STEPS_0_8.json'
        if dest.exists() or dest.is_symlink():fail('Refusing to overwrite existing paired fold result')
        cfg, rv=load_allowed(a.data)
        d=fold_run(cfg,rv,a.fold)
        d['source_sha256']=cfg['dataset_sha256']
        d['configuration_sha256']=lock.FREEZE_CONFIG_HASH
        d['runtime']={'python':platform.python_version(),'platform':platform.platform(),
                      'release_environment':lock.environment_snapshot()}
        write_new_json(dest,d)
        print('RESULT WRITTEN ONCE',dest)
        return
    if a.command=='summarize':
        hard_guard()
        out=pathlib.Path(a.out_dir)
        dest=out/'V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json'
        if dest.exists() or dest.is_symlink():fail('No overwrite of existing CV4 summary')
        docs=[]
        for f in FOLDS:
            path=out/f'FOLD_{f:01d}_PAIRED_STEPS_0_8.json'
            docs.append(json.loads(path.read_text(encoding='utf-8')))
            if docs[-1]['source_sha256']!=lock.SOURCE_HASH or docs[-1]['configuration_sha256']!=lock.FREEZE_CONFIG_HASH:
                fail('Fold source/config fingerprint mismatch')
        result=aggregate(docs)
        result['source_sha256']=lock.SOURCE_HASH
        result['config_sha256']=lock.FREEZE_CONFIG_HASH
        write_new_json(dest,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('ringwise','scientific_warning')},indent=2,ensure_ascii=False))
        print('SUMMARY WRITTEN ONCE',dest)


if __name__=='__main__':main_cli()
