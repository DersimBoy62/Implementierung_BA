"""Exploratory seeded robustness study on the 12 DEVELOPMENT rings only.

V3 Optuna trial 13 (K3) and trial 22 (K2), each STRICT vs selected-C SSL.
Only KMeans random_state changes; CV splits, model/priors, C, scaling,
quality likelihood and training budget are exactly identical for all seeds.

No code in locked original studies is edited. No outer rings are accessed.
Old outer results already viewed in another branch, so this is NOT a fresh test.
"""
from __future__ import annotations
import argparse, json, hashlib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np

import optuna_ssr_development as opt
import compare_strict_supervised_v3 as strict
import ablate_optuna_v3 as ab

TRIALS=(13,22)
SEEDS=(20261111,20261112,20261113)
MODES=('strict_L_only','ssl_Cselected')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def refuse_unsafe(out, study, ablation, strict_dir, data):
    dst=Path(out).resolve()
    if dst==Path('.').resolve() or dst==Path(data).resolve():
        raise ValueError('Invalid output directory')
    for x in (study,ablation,strict_dir):
        root=Path(x).resolve()
        if dst==root or root in dst.parents:
            raise ValueError('Output must be outside previous immutable study folders')
    if Path(out).exists():
        raise FileExistsError('Output path exists; never overwrite or re-run in same folder')


def verify_old_sources(study,ablation,strict_dir,data):
    """Verify source metadata and paired stored per-ring controls, without training."""
    manifest,protocol,stored=strict.validate_ablation(study,ablation,data)
    root=Path(strict_dir)
    sm=json.loads((root/'STRICT_CONTROL_MANIFEST.json').read_text(encoding='utf-8'))
    if sm['protocol']!='DM-V3-STRICT-L-CONTROL-v1' or sm['trials']!=list(TRIALS):
        raise ValueError('Strict control protocol mismatch')
    if sm['source_ablation_protocol_sha256']!=sha(Path(ablation)/'ABLATION_PROTOCOL.json'):
        raise ValueError('Strict control provenance changed')
    if sm['source_ablation_summary_sha256']!=sha(Path(ablation)/'ABLATION_SUMMARY.json'):
        raise ValueError('Strict source summary changed')
    if sm['source_code_sha256']!=sha(strict.__file__):
        raise ValueError('Original strict code changed')
    if sm['data_sha256']!=manifest['input_pickle_sha256']:
        raise ValueError('Data source mismatch')
    if sm['core_code_sha256']!=opt._hash_dependencies(Path(opt.__file__).resolve().parent):
        raise ValueError('Model core changed')
    old_summary=json.loads((root/'STRICT_COMPARISON_SUMMARY.json').read_text(encoding='utf-8'))
    if old_summary['protocol']!='DM-V3-STRICT-L-CONTROL-v1':
        raise ValueError('Old strict results mismatch')
    old={}
    for trial in TRIALS:
        cfg=protocol['selected_configs'][str(trial)]
        if not (cfg['preprocessing']=='P0' and cfg['variant']=='M2' and cfg['K']==(3 if trial==13 else 2)):
            raise ValueError('Trial configuration not registered')
        strict_report=json.loads((root/f'STRICT_L_ONLY_TRIAL_{trial:04}.json').read_text(encoding='utf-8'))
        old[trial]={'strict_L_only':strict_report,'ssl_Cselected':stored[trial]['ssl_Cselected']}
        for mode,rep in old[trial].items():
            rings,errors=ab.error_array(rep)
            rmse=np.sqrt(np.mean(errors**2,axis=0))
            if not np.allclose(rmse,rep['rmse_mm'],rtol=1e-9,atol=1e-9):
                raise ValueError('Old stored RMSE inconsistent')
            expected=old_summary['results'][str(trial)]['models'][mode]['rmse_mm']
            if not np.allclose(rmse,expected,rtol=1e-9,atol=1e-9):
                raise ValueError('Old stored summary inconsistent')
    return manifest,protocol,old


def train_one(rv,cfg,seed,mode,*,smoke=False):
    if seed not in SEEDS or mode not in MODES:
        raise ValueError('Unregistered seed or mode')
    opt.validate_development(rv)
    args={'folds':1,'epochs':1,'steps':1} if smoke else {'folds':4,'epochs':10,'steps':8}
    if mode=='ssl_Cselected':
        # Original unchanged function passes opt.FIXED_SEED to original train.
        # Patch ONLY this module-global during this synchronous call.
        with patch.object(opt,'FIXED_SEED',seed):
            rep=opt.evaluate_configuration(rv,cfg,**args)
    else:
        # Original unchanged strict_fold passes strict.FIXED_SEED to train.
        if smoke:
            with patch.object(strict,'FIXED_SEED',seed):
                rows,diag=strict.strict_fold(rv,cfg,0,epochs=1,steps=1)
            e=np.asarray([z['residual_mm'] for z in rows],float)
            rep={'per_ring':rows,'rmse_mm':np.sqrt(np.mean(e*e,axis=0)).tolist(),
                 'mae_mm':np.mean(np.abs(e),axis=0).tolist(),'fold_diagnostics':[diag]}
        else:
            with patch.object(strict,'FIXED_SEED',seed):
                rep=strict.strict_cv4(rv,cfg,epochs=10,steps=8)
    if rep['per_ring'] and not set(x['ring'] for x in rep['per_ring']).issubset(opt.EXPECTED_L):
        raise AssertionError('Outer/other ring appeared in validation')
    return rep


def baseline_agrees(new,old,atol=2e-5):
    """Compare EVERY held-out ring and both targets at original seed."""
    ri,ei=ab.error_array(new);rj,ej=ab.error_array(old)
    if ri!=rj: raise AssertionError('Reference ring IDs differ')
    delta=float(np.max(np.abs(ei-ej)))
    if delta>atol:
        raise AssertionError(f'Original seed does not reproduce old results: maximum delta {delta:.9g} mm')
    return delta


def summarize_pair(a,b):
    ra,ea=ab.error_array(a);rb,eb=ab.error_array(b)
    if ra!=rb:raise AssertionError('Mismatch in paired ring IDs')
    d=ea*ea-eb*eb
    return {'delta_mse_ssl_minus_strict_mm2':np.mean(d,axis=0).tolist(),
            'delta_rmse_ssl_minus_strict_mm':(np.asarray(a['rmse_mm'])-np.asarray(b['rmse_mm'])).tolist(),
            'ring_count_better_ssl_abs_error':np.sum(np.abs(ea)<np.abs(eb),axis=0).astype(int).tolist(),
            'loo_sign_stable':ab.paired(a,b)['loo_sign_stable']}


def aggregate(reports):
    result={}
    for trial in TRIALS:
        r=reports[trial]
        block={}
        for mode in MODES:
            x=np.array([r[s][mode]['rmse_mm'] for s in SEEDS],float)
            block[mode]={'mean_rmse_mm':x.mean(0).tolist(),
                         'sample_sd_rmse_mm':x.std(0,ddof=1).tolist(),
                         'min_rmse_mm':x.min(0).tolist(),'max_rmse_mm':x.max(0).tolist()}
        deltas=np.array([summarize_pair(r[s]['ssl_Cselected'],r[s]['strict_L_only'])['delta_mse_ssl_minus_strict_mm2'] for s in SEEDS])
        block['paired']={'delta_mse_per_seed_mm2':deltas.tolist(),
                         'number_seeds_ssl_better_per_target':np.sum(deltas<0,axis=0).astype(int).tolist(),
                         'mean_delta_mse_mm2':deltas.mean(0).tolist()}
        result[str(trial)]=block
    return result


def write_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def run(study,ablation,strict_dir,out,*,data='input/DM_16_aktive_Ringe.pkl',smoke=False):
    refuse_unsafe(out,study,ablation,strict_dir,data)
    manifest,protocol,old=verify_old_sources(study,ablation,strict_dir,data)
    rv=opt.dev.load_development(data)
    opt.validate_development(rv)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={t:{} for t in TRIALS}
    metadata={'protocol':'DM-V3-SEED-ROBUSTNESS-2026-11-11-v1',
              'test_type':'TECHNICAL_SMOKE' if smoke else 'EXPLORATORY_CV4',
              'trial_ids':list(TRIALS),'seeds':list(SEEDS if not smoke else SEEDS[:1]),
              'training_modes':list(MODES),'folds':1 if smoke else 4,
              'epochs':1 if smoke else 10,'local_steps':1 if smoke else 8,
              'data_sha256':manifest['input_pickle_sha256'],
              'core_code_sha256':manifest['code_sha256'],
              'optuna_study_manifest_sha256':sha(Path(study)/'STUDY_MANIFEST.json'),
              'ablation_protocol_sha256':sha(Path(ablation)/'ABLATION_PROTOCOL.json'),
              'strict_manifest_sha256':sha(Path(strict_dir)/'STRICT_CONTROL_MANIFEST.json'),
              'runner_sha256':sha(__file__),
              'reason':'Only KMeans random_state changes; model and CV data unchanged',
              'warning':'Reuse of same 8 development labels is exploratory; no independent confirmation. NO OUTER TEST.'}
    write_new(out/'SEED_PROTOCOL.json',metadata)
    for t in TRIALS:
        cfg=protocol['selected_configs'][str(t)]
        for seed in (SEEDS[:1] if smoke else SEEDS):
            report[t][seed]={}
            for mode in MODES:
                rep=train_one(rv,cfg,seed,mode,smoke=smoke)
                if seed==SEEDS[0] and not smoke:
                    rep['max_abs_delta_to_original_seed_mm']=baseline_agrees(rep,old[t][mode])
                rep['source_seed_protocol_sha256']=sha(out/'SEED_PROTOCOL.json')
                rep['trial']=t;rep['seed']=seed;rep['mode']=mode
                report[t][seed][mode]=rep
                write_new(out/f'TRIAL_{t:04d}_{mode}_SEED_{seed}.json',rep)
                print(f'TRIAL {t} SEED {seed} {mode} RMSE mm {np.round(rep["rmse_mm"],4)}',flush=True)
            pair=summarize_pair(report[t][seed]['ssl_Cselected'],report[t][seed]['strict_L_only'])
            print('  SSL-minus-strict delta MSE:',np.round(pair['delta_mse_ssl_minus_strict_mm2'],4),flush=True)
    if not smoke:
        overall=aggregate(report)
        write_new(out/'SEED_ROBUSTNESS_SUMMARY.json',{'protocol':metadata['protocol'],
                 'source_protocol_sha256':sha(out/'SEED_PROTOCOL.json'),'results':overall})
        print('ALL SEED COMPARISONS FINISHED:',out.resolve(),flush=True)
    else:
        print('TECHNICAL SMOKE FINISHED:',out.resolve(),flush=True)
    print('DEV ONLY. OUTER TEST NOT ACCESSED.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--ablation',required=True)
    p.add_argument('--strict',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.study,a.ablation,a.strict,a.out,data=a.data,smoke=a.smoke)

if __name__=='__main__':main()
