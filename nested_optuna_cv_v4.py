"""Development-only nested, ring-wise 4x3 CV with independent Optuna tuning per outer fold.

IMPORTANT: The same eight L rings have been used during exploratory model development.
This is an improved *within-run* separation of tuning and evaluation, but is
NOT fresh independent confirmation. U data never provide labels, prior outer
TEST_GESPERRT DM07/DM08/DM11/DM15 are never featurized/evaluated.

Expected unchanged project modules: optuna_ssr_development.py, dm_data.py,
compare_states_cv4.py, compare_two_head_cv4.py, quality_link_challenge.py,
dynamic_quality_v2.py and their original dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler

import optuna_ssr_development as source
import compare_states_cv4 as state
import dynamic_quality_v2 as dynamic
from compare_two_head_cv4 import OPTIONS
from quality_link_challenge import extract_descriptors

PROTOCOL='DM-V4-NESTED-OPTUNA-4x3-2026-11-12-v1'
EXPECTED_L=tuple(source.EXPECTED_L)
EXPECTED_U=tuple(source.EXPECTED_U)
EXPECTED_TEST=('DM07','DM08','DM11','DM15')
FOLDS_OUT=4
FOLDS_IN=3
ARM_MODES=('strict_L_only','shared_U_optuna')
ALLOWED_K=(2,3)
TRAIN_SEED=20261111
TPE_STARTUP=6
MAX_TRIALS=60


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fd:
        for chunk in iter(lambda:fd.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_new(path,obj):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as fd:
        json.dump(obj,fd,indent=2,ensure_ascii=False,allow_nan=False)
        fd.write('\n')


def validate_boundaries():
    if (len(EXPECTED_L)!=8 or len(set(EXPECTED_L))!=8
        or len(EXPECTED_U)!=4 or len(set(EXPECTED_U))!=4
        or tuple(source.LOCKED)!=EXPECTED_TEST):
        raise AssertionError('Changed registered development ring boundary')
    if set(EXPECTED_L)&set(EXPECTED_U) or (set(EXPECTED_L)|set(EXPECTED_U))&set(EXPECTED_TEST):
        raise AssertionError('Data leakage boundary')


def outer_order(pairing):
    validate_boundaries()
    if pairing=='canonical':return EXPECTED_L
    if pairing not in ('20261114','20261115'):
        raise ValueError('Unregistered validation grouping')
    return tuple(map(str,np.random.default_rng(int(pairing)).permutation(EXPECTED_L)))


def outer_pairs(pairing):
    order=outer_order(pairing)
    groups=tuple(tuple(order[i:i+2]) for i in range(0,8,2))
    assert set().union(*map(set,groups))==set(EXPECTED_L)
    assert len(set().union(*map(set,groups)))==8
    return groups


def inner_pairs(outer_train):
    """Six OUTER-train labels appear once each in three INNER validation folds.

    Preserve registered chronological/sorted ordering, no external randomness or
    influence of outer validation labels. Each inner training fold has four L.
    """
    if len(outer_train)!=6 or len(set(outer_train))!=6 or not set(outer_train)<=set(EXPECTED_L):
        raise ValueError('Inner CV requires precisely six outer-training L rings')
    rr=tuple(sorted(outer_train))
    return tuple(tuple(rr[i:i+2]) for i in range(0,6,2))


def validate_groups(trainL,val,arm):
    trainL=tuple(trainL);val=tuple(val)
    if arm not in ARM_MODES:raise ValueError('Unknown training arm')
    if (len(trainL) not in (4,6) or len(val)!=2 or not trainL
        or len(set(trainL))!=len(trainL) or len(set(val))!=len(val)):
        raise AssertionError('Wrong outer/inner fold cardinalities')
    if not set(trainL+val)<=set(EXPECTED_L) or set(trainL)&set(val):
        raise AssertionError('Validation leakage, unknown or old outer ring')
    if (set(trainL)|set(val))&set(EXPECTED_TEST):raise AssertionError('Old outer test prohibited')


def validate_config(cfg,arm):
    if arm not in ARM_MODES or set(cfg)!={'K','C','physical_var','preprocessing','variant'}:
        raise ValueError('Invalid registered arm/configuration')
    if (type(cfg['K']) is not int or cfg['K'] not in ALLOWED_K or
        cfg['preprocessing']!='P0' or cfg['variant']!='M2'):
        raise ValueError('Only K2/K3, P0/M2 registered')
    if not np.isfinite(cfg['C']) or not 0<=float(cfg['C'])<=2:
        raise ValueError('Invalid C')
    if not np.isfinite(cfg['physical_var']) or not 0.2<=float(cfg['physical_var'])<=0.8:
        raise ValueError('Invalid prior variance')
    if arm=='strict_L_only' and cfg['C']!=0:
        raise ValueError('Strict arm must use C=0 with ZERO U rings')


def suggest(trial,arm):
    if arm not in ARM_MODES:raise ValueError('Unknown arm')
    k=trial.suggest_categorical('K',list(ALLOWED_K))
    pv=trial.suggest_float('physical_var',0.2,0.8,log=True)
    c=0.0
    if arm=='shared_U_optuna':
        # Explicit zero point and continuous positive branch. C=0 uses U
        # for process scaler / KMeans, but not process ELBO updates.
        switch=trial.suggest_categorical('U_ELBO_branch',['zero','positive'])
        if switch=='positive':
            c=trial.suggest_float('C_positive',0.05,2.0,log=True)
    cfg={'K':int(k),'C':float(c),'physical_var':float(pv),
         'preprocessing':'P0','variant':'M2'}
    validate_config(cfg,arm)
    return cfg


def train_predict_fold(rv,trainL,val,arm,cfg,*,epochs=10,steps=8,seed=TRAIN_SEED):
    """Fit ALL transformations within current train set. Test ring Y only at scoring.

    For strict: process scaler, descriptor scaler, KMeans and VB see only L.
    For SSL: process/physical scalers and KMeans see L+U; quality scaler L only.
    """
    validate_groups(trainL,val,arm)
    validate_config(cfg,arm)
    trainL=tuple(sorted(trainL));val=tuple(val)
    trainX=tuple(sorted(trainL if arm=='strict_L_only' else (*trainL,*EXPECTED_U)))
    assert set(trainX)&set(val)==set()
    if arm=='strict_L_only':assert set(trainX)==set(trainL)
    else:assert set(trainX)==set(trainL)|set(EXPECTED_U)
    sx=state.fit_process_scaler({r:rv[r] for r in trainX},trainX,'P0')
    sy=source.dm.fit_y_scaler({r:rv[r] for r in trainL},trainL)
    if set(sx.fitted_ids)!=set(trainX) or set(sy.fitted_ids)!=set(trainL):
        raise AssertionError('Fold scaling leakage')
    x={rid:sx.transform(rv[rid].x) for rid in trainX}
    y={rid:sy.transform(rv[rid].y) for rid in trainL}
    all_used=(*trainX,*val)
    desc={rid:extract_descriptors(sx.transform(rv[rid].x),rv[rid].t,'physical') for rid in all_used}
    ss={rid:[None,None] for rid in all_used}
    groups=OPTIONS['SEPARATE_AXIAL']
    for j,idx in enumerate(groups):
        fs=StandardScaler().fit(np.asarray([desc[rid][list(idx)] for rid in trainX]))
        for rid in all_used:
            ss[rid][j]=fs.transform(desc[rid][list(idx)][None,:])[0]
    model=dynamic.train(x,y,{r:ss[r] for r in trainX},variant='M2',
                        K=cfg['K'],C=(0.0 if arm=='strict_L_only' else cfg['C']),
                        epochs=epochs,steps=steps,seed=seed,
                        physical_var=cfg['physical_var'])
    rows=[]
    for rid in val:
        # Validation input is only transformed, never used for any fitting.
        mu,sd=dynamic.predict(sx.transform(rv[rid].x),ss[rid],model)
        mu=np.asarray(mu)*sy.scale+sy.mean
        sd=np.asarray(sd)*sy.scale
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or not np.all(sd>0):
            raise ArithmeticError('Bad quality prediction')
        # The held-out label is read here ONLY, after the entire model was fitted.
        err=np.asarray(mu-rv[rid].y,dtype=float)
        rows.append({'ring':rid,'prediction_mm':mu.tolist(),
                     'residual_mm':err.tolist(),'predictive_sd_mm':sd.tolist()})
    diag={'train_labeled':list(trainL),
          'train_U_process_only':list(EXPECTED_U) if arm!='strict_L_only' else [],
          'validation':list(val),
          'x_scaler_fit_ids':list(sx.fitted_ids),
          'physical_scaler_fit_ids':list(trainX),
          'y_scaler_fit_ids':list(sy.fitted_ids),
          'hmm_train_ids':list(trainX),
          'u_used':bool(arm!='strict_L_only'),
          'state_usage':state.summarize_states(model,cfg['K']),
          'elbo_last':model['history'][-1]['elbo']}
    return {'per_ring':rows,'diagnostics':diag}


def metrics(rows):
    if not rows or len({r['ring'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate ring-level held-out predictions')
    err=np.asarray([r['residual_mm'] for r in rows],float)
    if err.shape!=(len(rows),2) or not np.isfinite(err).all():
        raise ArithmeticError('Invalid held-out residual matrix')
    return {'rmse_mm':np.sqrt(np.mean(err**2,axis=0)).tolist(),
            'mae_mm':np.mean(np.abs(err),axis=0).tolist(),
            'n_rings':len(rows)}


def inner_baseline(rv,outer_train):
    pp=inner_pairs(outer_train)
    rows=[]
    for val in pp:
        train=tuple(r for r in outer_train if r not in val)
        mu=np.mean(np.asarray([rv[r].y for r in train]),axis=0)
        for rid in val:
            rows.append({'ring':rid,'residual_mm':(mu-rv[rid].y).tolist()})
    result=metrics(rows)['rmse_mm']
    if any((not np.isfinite(v)) or v<=1e-10 for v in result):
        raise ArithmeticError('Non-positive inner mean-baseline RMSE')
    return np.asarray(result,dtype=float)


def evaluate_inner(rv,outer_train,arm,cfg,*,epochs=10,steps=8,max_inner=3):
    validate_config(cfg,arm)
    pairs=inner_pairs(outer_train)
    if max_inner not in (1,3):raise ValueError('Only smoke 1 or full 3 inner folds')
    rows=[];details=[]
    for no,val in enumerate(pairs[:max_inner],start=1):
        tr=tuple(r for r in outer_train if r not in val)
        fit=train_predict_fold(rv,tr,val,arm,cfg,epochs=epochs,steps=steps)
        for rr in fit['per_ring']:
            rows.append(dict(rr,inner_fold=no))
        details.append(dict(fit['diagnostics'],inner_fold=no))
    result=metrics(rows)
    if set(r['ring'] for r in rows)!=set().union(*[set(p) for p in pairs[:max_inner]]):
        raise AssertionError('Inner ring coverage incomplete')
    return {'configuration':cfg,'arm':arm,'inner_metrics':result,
            'per_ring':rows,'inner_fold_diagnostics':details,
            'outer_validation_truth_never_accessed_in_inner':True}


def inner_baseline_for_mode(rv,outer_train,max_inner):
    if max_inner==3:return inner_baseline(rv,outer_train)
    val=inner_pairs(outer_train)[0]
    tr=tuple(r for r in outer_train if r not in val)
    mu=np.mean(np.asarray([rv[r].y for r in tr]),axis=0)
    return np.asarray(metrics([{'ring':r,'residual_mm':(mu-rv[r].y).tolist()} for r in val])['rmse_mm'])


def choose_pareto(study):
    """Single predetermined compromise; strictly uses INNER Optuna objectives.

    Score minimizes worst relative RMSE to the fold-local training-mean
    reference, then sum of both ratios, K, C and trial number for ties.
    """
    pts=list(study.best_trials)
    if not pts:raise ValueError('No completed Pareto trial')
    def key(t):
        v=tuple(float(x) for x in t.values)
        if len(v)!=2 or not all(np.isfinite(x) for x in v):
            raise ArithmeticError('Malformed two-target Optuna objectives')
        return (max(v),sum(v),int(t.params['K']),float(t.user_attrs['config']['C']),t.number)
    return min(pts,key=key)


def make_manifest(args):
    if args.smoke:
        if (args.max_outer,args.max_inner,args.epochs,args.steps)!=(1,1,1,1) or args.trials>2:
            raise ValueError('Smoke must use one outer/one inner, one epoch/one step, <=2 trials')
    else:
        if (args.max_outer,args.max_inner,args.epochs,args.steps)!=(4,3,10,8):
            raise ValueError('Full nested protocol fixed at outer4, inner3, epochs10, steps8')
        if args.trials<8:raise ValueError('Full nested optimization needs >=8 trials per arm/outer fold')
    if args.trials<1 or args.trials>MAX_TRIALS:raise ValueError('Invalid trial budget')
    validate_boundaries()
    if source.dm.sha256(args.data)!=source.DATA_SHA:
        raise RuntimeError('Input pickle hash changed. Abort before data access.')
    src_root=Path(source.__file__).resolve().parent
    return {'protocol':PROTOCOL,'status':'EXPLORATORY NESTED DEVELOPMENT CV: prior dev rings repeatedly inspected',
            'outer_pairs':[list(x) for x in outer_pairs(args.pairing)],
            'pairing':args.pairing,'inner_folds':3,'max_inner':args.max_inner,
            'max_outer':args.max_outer,'trials_per_outer_per_arm':args.trials,
            'epochs':args.epochs,'steps':args.steps,'smoke':args.smoke,
            'seed':TRAIN_SEED,'tpe_startup':TPE_STARTUP,
            'arms':list(ARM_MODES),'K_choices':list(ALLOWED_K),
            'preprocessing':'P0','coupling':'M2',
            'physical_var_range':[0.2,0.8],'C_range':[0.,2.],
            'U_C_zero':'U features used for P0 scaler and KMeans, not U-ELBO; not strict supervised',
            'selection':'Pareto candidate with min worst inner RMSE/inner-train-mean-RMSE; deterministic tie-break',
            'input_sha256':source.DATA_SHA,'source_hashes':source._hash_dependencies(src_root),
            'prior_optuna_source_hash':sha(source.__file__),
            'new_source_hash':sha(__file__),
            'optuna_version':optuna.__version__,'python_version':platform.python_version(),
            'previous_outer_TEST_GESPERRT':list(EXPECTED_TEST),
            'warning':'Neither historical C-selected candidates nor these reused 8 L rings constitute an untouched external test.'}


def safe_out(out,data):
    p=Path(out).resolve()
    d=Path(data).resolve()
    root=Path.cwd().resolve()
    if p==root or p==d or p in d.parents or p==Path.home().resolve():
        raise ValueError('Unsafe output path')
    if p in Path(source.__file__).resolve().parents:
        raise ValueError('Cannot write above original Python module')
    return p


def trial_folder(out,fold,arm):
    return out/f'OUTER_{fold:02d}'/arm


def optimize_arm(rv,trainL,fold,arm,out,args):
    path=trial_folder(out,fold,arm)
    path.mkdir(parents=True,exist_ok=True)
    db=(path/'study.sqlite3').resolve()
    url=f'sqlite:///{db.as_posix()}'
    sampler=optuna.samplers.TPESampler(seed=TRAIN_SEED+1000*fold+(0 if arm=='strict_L_only' else 31),
                                       n_startup_trials=TPE_STARTUP,multivariate=False,constant_liar=False)
    study=optuna.create_study(directions=['minimize','minimize'],sampler=sampler,
                              storage=url,study_name=f'DM_V4_{args.pairing}_OUTER_{fold}_{arm}_{"smoke" if args.smoke else "cv"}',
                              load_if_exists=bool(args.resume))
    baseline=inner_baseline_for_mode(rv,trainL,args.max_inner)
    def objective(trial):
        cfg=suggest(trial,arm)
        r=evaluate_inner(rv,trainL,arm,cfg,epochs=args.epochs,steps=args.steps,
                         max_inner=args.max_inner)
        val=np.asarray(r['inner_metrics']['rmse_mm'],dtype=float)/baseline
        r['trial_number']=trial.number
        r['inner_mean_baseline_rmse_mm']=baseline.tolist()
        r['normalized_objectives']=val.tolist()
        r['outer_fold']=fold
        r['inner_outer_train_labeled']=list(trainL)
        fname=f'TRIAL_{trial.number:04d}.json'
        # A REPORT is immutable. An interrupted trial may need manual repair.
        write_new(path/fname,r)
        trial.set_user_attr('report_file',fname)
        trial.set_user_attr('report_sha256',sha(path/fname))
        trial.set_user_attr('config',cfg)
        trial.set_user_attr('rmse_mm',r['inner_metrics']['rmse_mm'])
        print(f'OUTER {fold} {arm} INNER TRIAL {trial.number} K={cfg["K"]} '
              f'C={cfg["C"]:.5f} prior={cfg["physical_var"]:.5f} '
              f'RMSE={np.round(r["inner_metrics"]["rmse_mm"],4)}',flush=True)
        return tuple(map(float,val))
    done=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done>args.trials:raise ValueError('Existing study exceeds registered trial budget')
    if done<args.trials:
        study.optimize(objective,n_trials=args.trials-done,n_jobs=1,gc_after_trial=True,catch=())
    else:
        print(f'OUTER {fold} {arm} already has {done} completed trials',flush=True)
    done=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done!=args.trials:raise RuntimeError('Study incomplete; do not score outer fold')
    selected_file=path/'SELECTED_INNER.json'
    if selected_file.is_file():
        previous=json.loads(selected_file.read_text(encoding='utf-8'))
        tnum=previous['selected_trial_number']
        sel=choose_pareto(study)
        if tnum!=sel.number or previous['configuration']!=sel.user_attrs['config']:
            raise RuntimeError('Resumed selected candidate has changed')
        return previous
    sel=choose_pareto(study)
    rfile=path/sel.user_attrs['report_file']
    if sha(rfile)!=sel.user_attrs['report_sha256']:
        raise RuntimeError('Chosen inner report does not match stored hash')
    selected={'selected_trial_number':sel.number,'configuration':sel.user_attrs['config'],
              'inner_rmse_mm':sel.user_attrs['rmse_mm'],
              'inner_mean_baseline_rmse_mm':baseline.tolist(),
              'normalized_objectives':list(map(float,sel.values)),
              'selection_uses_outer_validation_labels':False,
              'total_complete_trials':done,
              'study_path':str(db.relative_to(out))}
    write_new(selected_file,selected)
    return selected


def evaluate_outer_arm(rv,trainL,val,arm,cfg,args):
    fit=train_predict_fold(rv,trainL,val,arm,cfg,epochs=args.epochs,steps=args.steps)
    for r in fit['per_ring']:r['outer_fold']=args._current_fold
    return {'arm':arm,'configuration':cfg,'per_ring':fit['per_ring'],
            'metrics':metrics(fit['per_ring']),'fold_diagnostics':fit['diagnostics']}


def evaluate_outer_baseline(rv,trainL,val):
    mu=np.mean(np.asarray([rv[r].y for r in trainL]),axis=0)
    rows=[]
    for rid in val:
        rows.append({'ring':rid,'prediction_mm':mu.tolist(),
                     'residual_mm':(mu-rv[rid].y).tolist(),
                     'predictive_sd_mm':[None,None]})
    return {'arm':'labeled_train_mean','per_ring':rows,'metrics':metrics(rows),
            'train_labeled':list(trainL),'validation':list(val)}


def aggregate(out,args):
    groups=outer_pairs(args.pairing)
    records={arm:[] for arm in (*ARM_MODES,'shared_C0_fixed_SSL_structure','labeled_train_mean')}
    selections=[]
    for f,val in enumerate(groups[:args.max_outer],start=1):
        saved=out/f'OUTER_{f:02d}'/'OUTER_EVALUATION.json'
        rep=json.loads(saved.read_text(encoding='utf-8'))
        if rep['validation']!=list(val) or set(rep['outer_train_labeled']) != set(EXPECTED_L)-set(val):
            raise ValueError('Outer evaluation split metadata mismatch')
        selections.append({'outer_fold':f,'validation':list(val),
                           'strict_config':rep['inner_selected']['strict_L_only']['configuration'],
                           'ssl_config':rep['inner_selected']['shared_U_optuna']['configuration'],
                           'strict_inner_rmse_mm':rep['inner_selected']['strict_L_only']['inner_rmse_mm'],
                           'ssl_inner_rmse_mm':rep['inner_selected']['shared_U_optuna']['inner_rmse_mm']})
        for arm,rr in rep['outer_results'].items():
            if arm not in records:raise ValueError('Unexpected training arm')
            if {r['ring'] for r in rr['per_ring']} != set(val):
                raise AssertionError('Wrong held-out rings in outer report')
            records[arm].extend(rr['per_ring'])
    outer_expected=set().union(*[set(g) for g in groups[:args.max_outer]])
    out_metrics={}
    for arm,rows in records.items():
        if {r['ring'] for r in rows}!=outer_expected or len(rows)!=len(outer_expected):
            raise AssertionError('Incomplete outer evaluation')
        out_metrics[arm]=metrics(rows)
    contrasts={}
    for a,b in [('shared_U_optuna','strict_L_only'),('shared_C0_fixed_SSL_structure','strict_L_only'),
                ('shared_U_optuna','shared_C0_fixed_SSL_structure'),
                ('shared_U_optuna','labeled_train_mean')]:
        ea={r['ring']:np.asarray(r['residual_mm'],float) for r in records[a]}
        eb={r['ring']:np.asarray(r['residual_mm'],float) for r in records[b]}
        order=sorted(ea)
        if set(ea)!=set(eb):raise AssertionError('Paired outer ring mismatch')
        diff=np.asarray([ea[r]**2-eb[r]**2 for r in order])
        better=np.asarray([np.abs(ea[r])<np.abs(eb[r]) for r in order])
        loo=np.asarray([np.mean(np.delete(diff,i,axis=0),axis=0) for i in range(len(order))])
        contrasts[a+'_minus_'+b]={
            'delta_mse_mm2':np.mean(diff,axis=0).tolist(),
            'rings_with_lower_abs_error':np.sum(better,axis=0).tolist(),
            'n_rings':len(order),
            'leave_one_ring_out_direction_stable':[bool(np.all(loo[:,j]<0) or np.all(loo[:,j]>0)) for j in (0,1)],
            'sign_convention':'negative delta MSE means first arm better'}
    agg={'protocol':PROTOCOL,'status':'EXPLORATORY NESTED-CV OF PREVIOUSLY REUSED 8 DEV LABELS',
         'pairing':args.pairing,'outer_folds_completed':args.max_outer,
         'total_outer_predictions_per_arm':2*args.max_outer,
         'outer_metrics':out_metrics,'paired_comparisons':contrasts,'inner_selections':selections,
         'outer_per_ring':records,
         'note':'The OLD previously opened four outer test rings are not accessed here. Do not call this independent final test.',
         'causal_note':'Strict and SSL tuned separately with equal trial counts; shared_C0_fixed_SSL_structure is a controlled removal of U-ELBO but retains U preprocessing.'}
    write_new(out/'NESTED_SUMMARY.json',agg)
    with (out/'OUTER_PREDICTIONS.csv').open('x',newline='',encoding='utf-8-sig') as fh:
        w=csv.writer(fh,delimiter=';')
        w.writerow(['arm','ring','outer_fold','residual_outer_mm','residual_height_mm',
                    'prediction_outer_mm','prediction_height_mm','sd_outer_mm','sd_height_mm'])
        for arm,rows in records.items():
            for rr in rows:
                w.writerow([arm,rr['ring'],rr.get('outer_fold',''),*rr['residual_mm'],
                            *rr['prediction_mm'],*rr.get('predictive_sd_mm',[None,None])])
    with (out/'INNER_SELECTED_BY_OUTER_FOLD.csv').open('x',newline='',encoding='utf-8-sig') as fh:
        w=csv.writer(fh,delimiter=';')
        w.writerow(['outer_fold','validation','arm','K','C','physical_var','inner_rmse_outer_mm','inner_rmse_height_mm'])
        for rec in selections:
            for arm in ARM_MODES:
                c=rec['strict_config' if arm=='strict_L_only' else 'ssl_config']
                v=rec['strict_inner_rmse_mm' if arm=='strict_L_only' else 'ssl_inner_rmse_mm']
                w.writerow([rec['outer_fold'],','.join(rec['validation']),arm,c['K'],c['C'],c['physical_var'],*v])
    for arm,m in out_metrics.items():
        print('NESTED HELD-OUT',arm,'RMSE mm',np.round(m['rmse_mm'],4),
              'MAE mm',np.round(m['mae_mm'],4),flush=True)
    print('NESTED CV SUMMARY:',out/'NESTED_SUMMARY.json')
    print('DEV REUSED; NO NEW INDEPENDENT TEST',flush=True)
    return agg


def run(args):
    safe_out(args.out,args.data)
    out=Path(args.out).resolve()
    if out.exists() and not args.resume:raise FileExistsError('Output exists; never overwrite. Use --resume for SAME interrupted protocol.')
    if args.resume and not out.is_dir():raise FileNotFoundError('No study to resume')
    manifest=make_manifest(args)
    if args.resume:
        stored=json.loads((out/'NESTED_PROTOCOL.json').read_text(encoding='utf-8'))
        if stored!=manifest:raise RuntimeError('Resume blocked: source/seed/split/hyperparams changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_new(out/'NESTED_PROTOCOL.json',manifest)
    rv=source.dev.load_development(args.data)
    source.validate_development(rv)
    if set(rv)!=set(EXPECTED_L)|set(EXPECTED_U):raise AssertionError('Unexpected development view')
    for f,val in enumerate(outer_pairs(args.pairing)[:args.max_outer],start=1):
        trainL=tuple(r for r in EXPECTED_L if r not in val)
        assert len(trainL)==6 and not set(trainL)&set(val)
        folder=out/f'OUTER_{f:02d}'
        evalfile=folder/'OUTER_EVALUATION.json'
        if evalfile.is_file():
            print(f'OUTER {f} already completed; reuse immutable evaluation',flush=True)
            continue
        print(f'OUTER {f}/{args.max_outer} TRAIN L={trainL} VAL={val} U process only={EXPECTED_U}',flush=True)
        selected={}
        for arm in ARM_MODES:
            selected[arm]=optimize_arm(rv,trainL,f,arm,out,args)
        args._current_fold=f
        results={}
        for arm in ARM_MODES:
            results[arm]=evaluate_outer_arm(rv,trainL,val,arm,selected[arm]['configuration'],args)
        sslcfg=dict(selected['shared_U_optuna']['configuration'])
        c0cfg=dict(sslcfg,C=0.)
        results['shared_C0_fixed_SSL_structure']=evaluate_outer_arm(rv,trainL,val,
                                                'shared_U_optuna',c0cfg,args)
        results['shared_C0_fixed_SSL_structure']['arm']='shared_C0_fixed_SSL_structure'
        results['labeled_train_mean']=evaluate_outer_baseline(rv,trainL,val)
        for mode,r in results.items():
            for item in r['per_ring']:item['outer_fold']=f
        write_new(evalfile,{'outer_fold':f,'outer_train_labeled':list(trainL),
                            'validation':list(val),'inner_selected':selected,
                            'outer_results':results,
                            'notice':'Outer validation labels only scored after selection and refit.'})
        print(f'OUTER {f} scored after inner tuning; strict RMSE={np.round(results["strict_L_only"]["metrics"]["rmse_mm"],4)} '
              f'SSL RMSE={np.round(results["shared_U_optuna"]["metrics"]["rmse_mm"],4)}',flush=True)
    summary=out/'NESTED_SUMMARY.json'
    if not summary.exists():return aggregate(out,args)
    return json.loads(summary.read_text(encoding='utf-8'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--pairing',choices=('canonical','20261114','20261115'),default='canonical')
    p.add_argument('--trials',type=int,default=16,help='Trials PER OUTER FOLD AND PER ARM (strict/SSL)')
    p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-outer',type=int,default=4)
    p.add_argument('--max-inner',type=int,default=3)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps',type=int,default=8)
    args=p.parse_args(argv)
    return run(args)

if __name__=='__main__':main()
