"""Post-hoc, DEVELOPMENT-ONLY controlled 3x2 ablation of Optuna V3 choices.

For each of two preselected Pareto candidates (Optuna trial 13 and 22), keep
K/preprocessing/quality coupling fixed; vary C in {0, selected, 1} and physical
prior variance in {0.25, selected}. Exactly 6 full CV4 runs per candidate.

This is EXPLORATORY because development validation labels were reused for model
selection. At C=0 U rings still affect X/preprocessor and KMeans initialization.
It is NOT a strictly supervised baseline. No external rings are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import optuna_ssr_development as optdev
import summarize_optuna_development as summarizer

REGISTERED = (13, 22)
REGISTERED_OUTER = ('DM07','DM08','DM11','DM15')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def create_new_json(path, data):
    p=Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    return sha(p)


def selected_trial(report, number):
    rows = [r for r in report['trials'] if r['trial']==number]
    if len(rows)!=1:raise ValueError(f'Expected exactly one complete trial {number}')
    r=rows[0]
    c=r['config']
    if c['preprocessing']!='P0' or c['variant']!='M2' or c['K'] not in (2,3):
        raise ValueError('Registered V3 trial structure changed')
    if number==13 and c['K']!=3 or number==22 and c['K']!=2:
        raise ValueError('Unexpected state count')
    if not 0.25<=float(c['C'])<=0.4 or not 0.30<=float(c['physical_var'])<=0.6:
        raise ValueError('Unexpected selected hyperparameters')
    return r


def factorial_configs(candidate):
    """Ordering is fixed, irrespective of Optuna trial ordering."""
    selected=float(candidate['C']); physical=float(candidate['physical_var'])
    if abs(selected-1)<1e-8 or abs(physical-.25)<1e-8:
        raise ValueError('Factorial values must be distinct')
    return [dict(candidate,C=c,physical_var=p)
            for p in (.25,physical) for c in (0.0, selected, 1.0)]


def error_array(report):
    entries=report['per_ring']
    rings=tuple(sorted(x['ring'] for x in entries))
    if len(rings)!=8 or len(set(rings))!=8 or set(rings)!=set(optdev.EXPECTED_L):
        raise ValueError('Bad evaluation ring IDs')
    err=np.array([next(x['residual_mm'] for x in entries if x['ring']==r) for r in rings],float)
    if err.shape!=(8,2) or not np.isfinite(err).all():raise ValueError('Bad residual array')
    return rings,err


def paired(new, old):
    rings_a,e1=error_array(new);rings_b,e0=error_array(old)
    if rings_a!=rings_b:raise ValueError('Paired ring mismatch')
    diff=e1**2-e0**2
    delta=np.mean(diff,axis=0)
    better=np.sum(np.abs(e1)<np.abs(e0),axis=0)
    loo=np.array([np.mean(np.delete(diff,i,axis=0),axis=0) for i in range(len(rings_a))])
    return {'paired_delta_mse_mm2':delta.tolist(),
            'n_rings_better_abs_error':better.astype(int).tolist(),
            'loo_delta_mse_ranges_mm2':np.stack([loo.min(0),loo.max(0)],axis=1).tolist(),
            'loo_sign_stable':[(bool(np.all(loo[:,j]<0)) or bool(np.all(loo[:,j]>0))) for j in range(2)],
            'direction':'negative delta is better for first config'}


def contrasts(reports):
    """reports map (C_level,prior_level) to evaluated report."""
    pairs={
        'C_selected_minus_C0_at_prior_selected':(('selected','selected'),('zero','selected')),
        'C_selected_minus_C1_at_prior_selected':(('selected','selected'),('one','selected')),
        'C_selected_minus_C0_at_prior_manual':(('selected','manual'),('zero','manual')),
        'prior_selected_minus_prior_manual_at_C_selected':(('selected','selected'),('selected','manual')),
        'prior_selected_minus_prior_manual_at_C0':(('zero','selected'),('zero','manual')),
        'prior_selected_minus_prior_manual_at_C1':(('one','selected'),('one','manual')),
    }
    return {name:paired(reports[a],reports[b]) for name,(a,b) in pairs.items()}


def run(study_dir,outdir,*,data='input/DM_16_aktive_Ringe.pkl',trials=REGISTERED,
        folds=4,epochs=10,steps=8):
    if tuple(trials)!=REGISTERED:raise ValueError('Requires both registered trials 13 and 22; no cherry-picking')
    if (folds,epochs,steps)!=(4,10,8):
        raise ValueError('Final analysis requires exactly 4 folds, 10 epochs, 8 local updates')
    study_dir=Path(study_dir);out=Path(outdir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory occupied; never overwrite')
    loaded=summarizer.summarize(study_dir)
    manifest=json.loads((study_dir/'STUDY_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['search_phase']!='extended' or manifest['smoke'] or loaded['completed_trials']!=24:
        raise ValueError('Require complete original 24-trial V3 extended CV4 study')
    if (manifest['folds'],manifest['epochs'],manifest['local_steps'])!=(4,10,8):
        raise ValueError('Original study not same CV budget')
    original=manifest['code_sha256']
    current=optdev._hash_dependencies(Path(optdev.__file__).resolve().parent)
    if original!=current:raise ValueError('Core code changed from original Optuna study; abort')
    if optdev.dm.sha256(data)!=manifest['input_pickle_sha256']:
        raise ValueError('Original dataset hash mismatch')
    selected={i:selected_trial(loaded,i) for i in trials}
    protocol={'protocol':'DM-V3-FIXED-FACTORIAL-ABLATION-v1',
              'source_optuna_manifest_sha256':sha(study_dir/'STUDY_MANIFEST.json'),
              'source_trial_report_sha256':{str(t):sha(study_dir/f'TRIAL_{t:04}.json') for t in trials},
              'registered_trial_ids':list(trials),
              'selected_configs':{str(i):selected[i]['config'] for i in trials},
              'core_code_sha256':current,'input_sha256':manifest['input_pickle_sha256'],
              'ablation_code_sha256':sha(__file__),
              'folds':folds,'epochs':epochs,'steps':steps,
              'method':'3 levels C x 2 levels physical_var; K, P0, M2 held fixed per candidate',
              'warning':'C=0 still uses U process data for X scaler and KMeans initialisation; not STRICT SUPERVISED.',
              'status':'EXPLORATORY reused development CV, no new independent test, outer never read'}
    create_new_json(out/'ABLATION_PROTOCOL.json',protocol)
    rv=optdev.dev.load_development(data)
    optdev.validate_development(rv)
    reports={}
    for trial in trials:
        cand=selected[trial]['config']; reports[trial]={}
        cs=(('zero',0.),('selected',float(cand['C'])),('one',1.))
        vs=(('manual',.25),('selected',float(cand['physical_var'])))
        for pname,p in vs:
            for cname,c in cs:
                config=dict(cand,C=float(c),physical_var=float(p))
                t0=time.monotonic()
                rep=optdev.evaluate_configuration(rv,config,folds=folds,epochs=epochs,steps=steps)
                rep['source_trial']=trial;rep['c_level']=cname;rep['prior_level']=pname
                rep['origin_protocol_sha256']=sha(out/'ABLATION_PROTOCOL.json')
                name=f'TRIAL_{trial:04d}_C_{cname}_P_{pname}.json'
                create_new_json(out/name,rep)
                reports[trial][(cname,pname)]=rep
                print(f'TRIAL {trial:02d} C={c:.5g} prior={p:.5g} CV4 RMSE mm '
                      f'{np.round(rep["rmse_mm"],4)}  duration={time.monotonic()-t0:.1f}s',flush=True)
                if cname==pname=='selected':
                    target=np.asarray(selected[trial]['rmse_mm'])
                    if not np.allclose(rep['rmse_mm'],target,atol=1e-5,rtol=1e-5):
                        raise AssertionError('Selected Optuna config did not reproduce saved RMSE')
    summary={'protocol':protocol['protocol'],'source_protocol_sha256':sha(out/'ABLATION_PROTOCOL.json'),
             'results':{str(i):{
                 'selected_config':selected[i]['config'],
                 'cells':{f'C_{a}_P_{b}':{'C':r['configuration']['C'],'physical_var':r['configuration']['physical_var'],
                                        'rmse_mm':r['rmse_mm'],'mae_mm':r['mae_mm']}
                          for (a,b),r in reports[i].items()},
                 'paired_contrasts':contrasts(reports[i])} for i in trials},
             'caution':'Exploratory repeated CV and post-hoc configs; NOT independent performance evidence.'}
    create_new_json(out/'ABLATION_SUMMARY.json',summary)
    print('ABLATION FINISHED:',out.resolve())
    print('NO OUTER TEST READ; C=0 IS NOT STRICT SUPERVISED.')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    a=p.parse_args();run(a.study,a.out,data=a.data)

if __name__=='__main__': main()
