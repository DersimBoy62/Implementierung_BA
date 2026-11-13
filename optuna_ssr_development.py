"""Reproducible *exploratory* multi-objective Optuna search for Bayesian-SSR V2.

Strict 12-ring development-only evaluation: 8 L, 4 U with hidden quality Y.
External DM07/08/11/15 were already opened in a previous final experiment;
no data from those rings are used or evaluated in this research branch.

Two stages:
  structural: K x P0/P2 x M0/M2, C=1 and physical_var=.25 fixed.
  extended:   as above, plus C and physical_var search.

Individual results, study manifest and SQLite storage can be audited independently.
Requires installed optuna >=4, existing unchanged 2026-11-11 source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler

import dm_data as dm
import compare_dynamic_quality_cv4 as dev
import compare_states_cv4 as states
from compare_two_head_cv4 import OPTIONS, DESCRIPTOR_NAMES
from quality_link_challenge import DATA_SHA, FIXED_SEED, LOCKED, extract_descriptors
import two_head_b1 as basic
import dynamic_quality_v2 as dynamic

ALLOWED_K = (2, 3, 5, 8)
ALLOWED_PREP = ('P0', 'P2')
ALLOWED_MODEL = ('M0', 'M2')
EXPECTED_L = dev.EXPECTED_L
EXPECTED_U = dev.EXPECTED_U
NORM_REFERENCE = (0.5528856197, 3.2026)  # PREDECLARED original P0/M0/K3 CV4 numbers (mm)
SOURCE_NAMES = (
    'dm_data.py', 'quality_link_challenge.py', 'compare_two_head_cv4.py',
    'two_head_b1.py', 'dynamic_quality_v2.py', 'dynamic_markov_v2.py',
    'compare_dynamic_quality_cv4.py', 'compare_states_cv4.py',
    'vb_b1.py', 'fullcov_b1.py', 'full_markov_b1.py',
)


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _write_new(path, content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        f.write(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _hash_dependencies(root):
    root=Path(root)
    hashes={}
    for fname in SOURCE_NAMES:
        p=root/fname
        if not p.is_file(): raise FileNotFoundError(f'Missing registered dependency: {p}')
        hashes[fname]=_sha256(p)
    return hashes


def validate_development(rv):
    # No attribute access to raw U/test Y except the intentionally NaN U field.
    assert set(rv)==set(EXPECTED_L)|set(EXPECTED_U), 'Development-only split required'
    assert not set(rv)&set(LOCKED), 'External rings are forbidden'
    assert len(EXPECTED_L)==8 and len(EXPECTED_U)==4 and len(LOCKED)==4
    for rid in EXPECTED_U:
        if not np.isnan(rv[rid].y).all():
            raise AssertionError(f'U-quality is accessible: {rid}')
    for rid in EXPECTED_L:
        if not np.isfinite(rv[rid].y).all():
            raise AssertionError(f'Invalid L-quality: {rid}')
    dm.refuse_post_quality_inputs(rv)


def validate_parameters(config):
    required={'K','preprocessing','variant','C','physical_var'}
    if set(config)!=required: raise ValueError(f'Expected configuration keys {required}')
    if config['K'] not in ALLOWED_K: raise ValueError('K outside registered set')
    if config['preprocessing'] not in ALLOWED_PREP: raise ValueError('Invalid preprocessing')
    if config['variant'] not in ALLOWED_MODEL: raise ValueError('Invalid quality coupling')
    if not np.isfinite(config['C']) or not 0. <= config['C'] <= 2.: raise ValueError('C out of range')
    if not np.isfinite(config['physical_var']) or not 0.05 <= config['physical_var'] <= 1.: raise ValueError('physical prior variance out of range')


def evaluate_configuration(rv,config,*,folds=4,epochs=10,steps=8):
    """One candidate, exactly the same fold-local transforms as manual K grid.

    Only training rings fit X/sum/Y scalers. U targets are NaN. Validation
    quality is read ONLY after trained model has predicted each validation ring.
    """
    validate_parameters(config)
    validate_development(rv)
    if folds not in (1,2,3,4) or epochs<1 or steps<0: raise ValueError('Invalid training budget')
    K=int(config['K']); C=float(config['C']); pv=float(config['physical_var'])
    mode=str(config['preprocessing']); variant=str(config['variant'])
    per_ring=[]; fold_diagnostics=[]
    groups=OPTIONS['SEPARATE_AXIAL']
    for f in range(folds):
        val=EXPECTED_L[2*f:2*f+2]
        trL=tuple(r for r in EXPECTED_L if r not in val)
        trU=EXPECTED_U
        train=tuple(sorted((*trL,*trU)))
        assert len(train)==10 and len(trL)==6 and len(val)==2 and not set(train)&set(val)
        assert not (set(LOCKED)&(set(train)|set(val)))
        sx=states.fit_process_scaler({rid:rv[rid] for rid in train},train,mode)
        sy=dm.fit_y_scaler({rid:rv[rid] for rid in trL},trL)
        if sx.fitted_ids!=train or sy.fitted_ids!=tuple(sorted(trL)):
            raise AssertionError('Scaler leakage')
        x={rid:sx.transform(rv[rid].x) for rid in train}
        y={rid:sy.transform(rv[rid].y) for rid in trL}
        ids=train+val
        desc={rid:extract_descriptors(sx.transform(rv[rid].x),rv[rid].t,'physical') for rid in ids}
        ss={rid:[None,None] for rid in ids}
        for j,ix in enumerate(groups):
            fs=StandardScaler().fit(np.array([desc[rid][list(ix)] for rid in train]))
            for rid in ids:
                ss[rid][j]=fs.transform(desc[rid][list(ix)][None,:])[0]
        sdata={rid:ss[rid] for rid in train}
        if variant=='M0':
            model=basic.train(x,y,sdata,K=K,C=C,epochs=epochs,steps=steps,seed=FIXED_SEED,physical_var=pv)
            pred_fn=basic.predict
        else:
            model=dynamic.train(x,y,sdata,variant='M2',K=K,C=C,epochs=epochs,steps=steps,
                                seed=FIXED_SEED,physical_var=pv)
            pred_fn=dynamic.predict
        for rid in val:
            mu,sd=pred_fn(sx.transform(rv[rid].x),ss[rid],model)
            mu_mm=mu*sy.scale+sy.mean
            sd_mm=sd*sy.scale
            if not np.isfinite(mu_mm).all() or not np.isfinite(sd_mm).all() or (sd_mm<=0).any():
                raise ArithmeticError(f'Nonfinite predicted moments for {rid}')
            # Access to validation truth ONLY at final metrics stage.
            resid=np.asarray(mu_mm-rv[rid].y,dtype=float)
            per_ring.append({'ring':rid,'fold':f+1,'prediction_mm':mu_mm.tolist(),
                             'residual_mm':resid.tolist(),'predictive_sd_mm':sd_mm.tolist()})
        sd=states.summarize_states(model,K)
        fold_diagnostics.append({'fold':f+1,'train_labeled':list(trL),'train_U_process_only':list(trU),
                                 'validation':list(val),'x_scaler_fit_ids':list(sx.fitted_ids),
                                 'y_scaler_fit_ids':list(sy.fitted_ids),
                                 'physical_scaler_fit_ids':list(train),
                                 'effective_states_entropy_exp':sd['effective_states_entropy_exp'],
                                 'states_above_5pct':sd['states_above_5pct'],
                                 'state_masses':sd['state_mass_timepoint_weighted'],
                                 'elbo_last':model['history'][-1]['elbo']})
    errors=np.array([r['residual_mm'] for r in per_ring],dtype=float)
    rmse=np.sqrt(np.mean(errors*errors,axis=0))
    mae=np.mean(np.abs(errors),axis=0)
    if errors.shape!=(folds*2,2) or not np.isfinite(rmse).all():
        raise ArithmeticError('Incomplete trial metrics')
    return {'configuration':dict(config),'folds_count':folds,'epochs':epochs,'steps':steps,
            'development_labeled_ids':list(EXPECTED_L),'development_unlabeled_process_only':list(EXPECTED_U),
            'outer_ids_explicitly_excluded':list(LOCKED),'source_data_sha256':DATA_SHA,
            'rmse_mm':rmse.tolist(),'mae_mm':mae.tolist(),'per_ring':per_ring,
            'fold_diagnostics':fold_diagnostics,
            'status':'EXPLORATORY REUSED DEVELOPMENT CV; NOT INDEPENDENT VALIDATION'}


def suggest_configuration(trial,phase):
    if phase not in ('structural','extended'):raise ValueError('Unknown search phase')
    config={'K':trial.suggest_categorical('K',list(ALLOWED_K)),
            'preprocessing':trial.suggest_categorical('preprocessing',list(ALLOWED_PREP)),
            'variant':trial.suggest_categorical('variant',list(ALLOWED_MODEL)),
            'C':1.0,'physical_var':0.25}
    if phase=='extended':
        config['C']=trial.suggest_float('C',0.25,2.,log=True)
        config['physical_var']=trial.suggest_float('physical_var',0.05,1.,log=True)
    validate_parameters(config)
    return config


def _make_manifest(args,origin_root):
    if dm.sha256(args.data)!=DATA_SHA:
        raise RuntimeError('Data file hash changed: NO SEARCH')
    if tuple(LOCKED)!=('DM07','DM08','DM11','DM15'):
        raise AssertionError('Wrong locked outer IDs')
    return {'protocol':'DM-V3-OPTUNA-EXPLORATORY-v1',
            'search_phase':args.phase,'smoke':bool(args.smoke),
            'trial_sample_seed':FIXED_SEED,'TPE_n_startup_trials':8,
            'allowed_K':list(ALLOWED_K),'allowed_preprocessing':list(ALLOWED_PREP),
            'allowed_quality_coupling':list(ALLOWED_MODEL),
            'extended_C_range':[0.25,2.0] if args.phase=='extended' else [1.,1.],
            'extended_physical_var_range':[0.05,1.] if args.phase=='extended' else [.25,.25],
            'objectives':'RMSE per target / fixed prior CV RMSE; two MINIMIZE objectives',
            'objective_normalizers_rmse_mm':list(NORM_REFERENCE),
            'folds':args.folds,'epochs':args.epochs,'local_steps':args.steps,
            'input_pickle_sha256':DATA_SHA,
            'outer_test_ids_already_evaluated_in_previous_study_and_excluded':list(LOCKED),
            'code_sha256':_hash_dependencies(origin_root),
            'optuna_source_sha256':_sha256(__file__),
            'optuna_version_runtime':optuna.__version__,
            'python_runtime':platform.python_version(),
            'research_notice':'These repeated development CV results are exploratory; never tune from old outer test.'}


def run(args):
    if args.trials<=0: raise ValueError('Require positive number of trials')
    if args.smoke:
        if args.folds!=1 or args.epochs!=1 or args.steps!=1 or args.trials>2:
            raise ValueError('SMOKE REQUIRES folds=1 epochs=1 steps=1 and trials<=2')
    elif (args.folds,args.epochs,args.steps)!=(4,10,8):
        raise ValueError('Production study requires EXACTLY 4 folds, 10 epochs and 8 local steps')
    if args.phase=='structural' and args.trials>40:
        raise ValueError('Structural budget limited to 40 additional trials; prevent accidental huge run')
    if args.phase=='extended' and args.trials>80:
        raise ValueError('Extended budget limited to 80 additional trials')
    out=Path(args.out)
    if out.resolve()==Path('.').resolve():raise ValueError('Do not use project root as output folder')
    manifest=_make_manifest(args,Path(__file__).resolve().parent)
    mfile=out/'STUDY_MANIFEST.json'
    if args.resume:
        if not mfile.is_file():raise FileNotFoundError('No manifest to resume')
        prev=json.loads(mfile.read_text(encoding='utf-8'))
        if prev!=manifest:raise RuntimeError('Study settings/source mismatch on resume')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Output directory already populated. Use --resume only to continue SAME study.')
        out.mkdir(parents=True,exist_ok=True)
        _write_new(mfile,manifest)
    # Check source and development split BEFORE creating/continuing any trial.
    rv=dev.load_development(args.data)
    validate_development(rv)
    db=(out/'optuna_study.sqlite3').resolve()
    # pathlib/as_posix yields SQLAlchemy-compatible sqlite:///C:/... on Windows
    storage=f'sqlite:///{db.as_posix()}'
    sampler=optuna.samplers.TPESampler(seed=FIXED_SEED,n_startup_trials=8,
                                       multivariate=False,constant_liar=False)
    study=optuna.create_study(directions=['minimize','minimize'],
                               sampler=sampler,storage=storage,
                               study_name=f'DM_V3_{args.phase}_{"smoke" if args.smoke else "cv4"}',
                               load_if_exists=bool(args.resume))
    def objective(trial):
        config=suggest_configuration(trial,args.phase)
        report=evaluate_configuration(rv,config,folds=args.folds,epochs=args.epochs,steps=args.steps)
        # Each trial writes a separate immutable report before Optuna completes.
        report['trial_number']=trial.number
        report['source_manifest_sha256']=_sha256(mfile)
        file_name=f'TRIAL_{trial.number:04d}.json'
        _write_new(out/file_name,report)
        trial.set_user_attr('report',file_name)
        trial.set_user_attr('report_sha256',_sha256(out/file_name))
        trial.set_user_attr('rmse_mm',report['rmse_mm'])
        trial.set_user_attr('mae_mm',report['mae_mm'])
        objectives=tuple(float(report['rmse_mm'][j]/NORM_REFERENCE[j]) for j in (0,1))
        print(f'TRIAL {trial.number} {config} RMSE mm {np.round(report["rmse_mm"],4)} '
              f'NORMALIZED {np.round(objectives,4)}',flush=True)
        return objectives
    study.optimize(objective,n_trials=args.trials,n_jobs=1,gc_after_trial=True,
                   catch=(),show_progress_bar=False)
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    print('OPTUNA DEVELOPMENT STUDY:',out.resolve())
    print('Total completed trials:',len(done),'Pareto solutions:',len(study.best_trials))
    for t in sorted(study.best_trials,key=lambda q:q.number):
        print(' PARETO',t.number,t.params,'RMSE mm',np.round(t.user_attrs['rmse_mm'],4))
    print('KEIN AEUSSERER TEST; NICHT UNABHAENGIG BESTAETIGT.')
    return study


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    parser.add_argument('--phase',choices=['structural','extended'],default='structural')
    parser.add_argument('--trials',type=int,default=16)
    parser.add_argument('--folds',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--out',required=True)
    a=parser.parse_args(argv)
    run(a)

if __name__=='__main__':main()
