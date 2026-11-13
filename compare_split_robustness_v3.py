"""Registered exploratory 4-fold SPLIT sensitivity, Bayesian-SSR V3.

DO NOT USE OUTER RINGS. ONLY 8 labeled development rings in validation.
Reuses original training/evaluation functions via an ephemeral change of the
registered validation ring ORDER, not the model or feature code.

Two fixed alternative pairings plus the source canonical grouping. Two selected
Optuna configurations, three U data modes. This is NOT independent validation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

import optuna_ssr_development as opt
import compare_strict_supervised_v3 as strict
import compare_seed_robustness_v3 as seeds
import ablate_optuna_v3 as ab

SPLIT_SEEDS=(20261114,20261115)
ORIGINAL_ORDER=tuple(opt.EXPECTED_L)
TRIALS=(13,22)
MODES=('strict_L_only','shared_C0','ssl_Cselected')
BASE_SEED=20261111
EXPECTED_OUTER=('DM07','DM08','DM11','DM15')


def pairs(order):
    return tuple(tuple(sorted(order[i:i+2])) for i in range(0,len(order),2))


def new_orders():
    if len(ORIGINAL_ORDER)!=8 or len(set(ORIGINAL_ORDER))!=8:
        raise AssertionError('Expected eight unique development labeled rings')
    if set(ORIGINAL_ORDER)&set(opt.LOCKED) or tuple(opt.LOCKED)!=EXPECTED_OUTER:
        raise AssertionError('Outer ring boundary changed')
    result={}
    for seed in SPLIT_SEEDS:
        order=tuple(str(s) for s in np.random.default_rng(seed).permutation(ORIGINAL_ORDER))
        if set(order)!=set(ORIGINAL_ORDER) or len(order)!=8:
            raise AssertionError('Bad split permutation')
        result[seed]=order
    sets=[frozenset(pairs(v)) for v in (ORIGINAL_ORDER,*result.values())]
    if len(set(sets))!=3:
        raise AssertionError('Pairings are not distinct')
    return result


@contextmanager
def temporary_fold_order(order):
    order=tuple(order)
    if len(order)!=8 or set(order)!=set(ORIGINAL_ORDER) or set(order)&set(EXPECTED_OUTER):
        raise ValueError('Illegal development fold grouping')
    with patch.object(opt,'EXPECTED_L',order):
        yield


def check_report(report,order,mode):
    if not isinstance(report,dict):raise ValueError('Report missing')
    rings,err=ab.error_array(report)
    if len(rings)!=8 or set(rings)!=set(ORIGINAL_ORDER) or len(set(rings))!=8:
        raise AssertionError('Wrong held-out rings or duplicate ring')
    if not np.isfinite(err).all():raise AssertionError('Invalid residuals')
    if not np.allclose(np.sqrt(np.mean(err*err,axis=0)),report['rmse_mm'],rtol=1e-9,atol=1e-9):
        raise AssertionError('RMSE mismatch')
    rpairs=pairs(order)
    for rr in report['per_ring']:
        if rr['ring'] not in rpairs[rr['fold']-1]:
            raise AssertionError('Ring assigned to wrong fold')
    for f,d in enumerate(report['fold_diagnostics']):
        val=tuple(sorted(d['validation']))
        if val!=rpairs[f]: raise AssertionError('Fold validation IDs mismatch')
        tr=set(ORIGINAL_ORDER)-set(val)
        if mode=='strict_L_only':
            for k in ('x_scaler_fit_ids','y_scaler_fit_ids','physical_scaler_fit_ids','hmm_initialisation_uses_only'):
                if set(d[k])!=tr: raise AssertionError(f'Strict mode U use or val leakage: {k}')
            if d.get('u_used') is not False: raise AssertionError('Strict not marked U-free')
        else:
            shared=tr|set(opt.EXPECTED_U)
            if set(d['x_scaler_fit_ids'])!=shared or set(d['physical_scaler_fit_ids'])!=shared:
                raise AssertionError('Shared mode scaler data boundary violation')
            if set(d['y_scaler_fit_ids'])!=tr:
                raise AssertionError('Validation/U labels in target scaler')
    return err


def fit_pairing(rv,cfg,order,mode):
    if mode not in MODES:raise ValueError('Unknown training mode')
    with temporary_fold_order(order), patch.object(opt,'FIXED_SEED',BASE_SEED),patch.object(strict,'FIXED_SEED',BASE_SEED):
        opt.validate_development(rv)
        if mode=='strict_L_only':
            result=strict.strict_cv4(rv,cfg,epochs=10,steps=8)
        else:
            c=dict(cfg)
            if mode=='shared_C0':c['C']=0.
            result=opt.evaluate_configuration(rv,c,folds=4,epochs=10,steps=8)
    check_report(result,order,mode)
    return result


def ensure_new_output(out,study,ablation,strict_dir,seed_dir,data):
    p=Path(out).resolve()
    if p.exists():raise FileExistsError('Results exist. Never rerun into an occupied output folder')
    if p in (Path('.').resolve(),Path(data).resolve()):raise ValueError('Unsafe output')
    for x in (study,ablation,strict_dir,seed_dir):
        r=Path(x).resolve()
        if p==r or r in p.parents:raise ValueError('Do not write inside historical results')


def load_canonical(study,ablation,strict_dir,seed_dir,data):
    # Previous integrity chain validates the frozen source data and core code.
    manifest,protocol,old=seeds.verify_old_sources(study,ablation,strict_dir,data)
    meta=json.loads((Path(seed_dir)/'SEED_PROTOCOL.json').read_text(encoding='utf-8'))
    summ=json.loads((Path(seed_dir)/'SEED_ROBUSTNESS_SUMMARY.json').read_text(encoding='utf-8'))
    if meta['protocol']!='DM-V3-SEED-ROBUSTNESS-2026-11-11-v1' or meta['test_type']!='EXPLORATORY_CV4':
        raise ValueError('Wrong source seed study')
    if meta['core_code_sha256']!=manifest['code_sha256'] or meta['data_sha256']!=manifest['input_pickle_sha256']:
        raise ValueError('Seed study source version changed')
    if summ['source_protocol_sha256']!=seeds.sha(Path(seed_dir)/'SEED_PROTOCOL.json'):
        raise ValueError('Seed study report hash mismatch')
    _,_,stored=strict.validate_ablation(study,ablation,data)
    canonical={}
    for trial in TRIALS:
        canonical[trial]={'strict_L_only':old[trial]['strict_L_only'],
                          'shared_C0':stored[trial]['shared_C0'],
                          'ssl_Cselected':old[trial]['ssl_Cselected']}
        for mode,report in canonical[trial].items():
            check_report(report,ORIGINAL_ORDER,mode)
    return manifest,protocol,canonical


def summarize(per_partition):
    summary={}
    for trial, partitions in per_partition.items():
        x={}
        for mode in MODES:
            rm=np.asarray([partitions[p][mode]['rmse_mm'] for p in ('canonical',*SPLIT_SEEDS)],float)
            x[mode]={'rmse_by_split_mm':rm.tolist(),'mean_rmse_mm':rm.mean(0).tolist(),
                     'min_rmse_mm':rm.min(0).tolist(),'max_rmse_mm':rm.max(0).tolist(),
                     'sample_sd_rmse_mm':rm.std(0,ddof=1).tolist()}
        contrasts={}
        for a,b in [('shared_C0','strict_L_only'),('ssl_Cselected','shared_C0'),('ssl_Cselected','strict_L_only')]:
            differences=[]
            for p in ('canonical',*SPLIT_SEEDS):
                differences.append(seeds.summarize_pair(partitions[p][a],partitions[p][b]))
            d=np.array([z['delta_mse_ssl_minus_strict_mm2'] for z in differences],float)
            contrasts[a+'_minus_'+b]={
                'delta_mse_by_split_mm2':d.tolist(),
                'n_splits_better_per_target':(d<0).sum(0).tolist(),
                'better_rings_by_split':[z['ring_count_better_ssl_abs_error'] for z in differences],
                'loo_sign_stable_by_split':[z['loo_sign_stable'] for z in differences]}
        summary[str(trial)]={'models':x,'contrasts':contrasts}
    return summary


def write_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as fh:
        json.dump(obj,fh,indent=2,ensure_ascii=False,allow_nan=False);fh.write('\n')


def run(args):
    ensure_new_output(args.out,args.study,args.ablation,args.strict,args.seed_results,args.data)
    manifest,protocol,canonical=load_canonical(args.study,args.ablation,args.strict,args.seed_results,args.data)
    orders=new_orders()
    rv=opt.dev.load_development(args.data)
    opt.validate_development(rv)
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    metadata={'protocol':'DM-V3-SPLIT-SENSITIVITY-2026-11-12-v1',
              'candidate_trials':list(TRIALS),'modes':list(MODES),
              'source_data_sha256':manifest['input_pickle_sha256'],
              'source_model_code_sha256':manifest['code_sha256'],
              'source_optuna_manifest_sha256':seeds.sha(Path(args.study)/'STUDY_MANIFEST.json'),
              'source_ablation_protocol_sha256':seeds.sha(Path(args.ablation)/'ABLATION_PROTOCOL.json'),
              'source_strict_manifest_sha256':seeds.sha(Path(args.strict)/'STRICT_CONTROL_MANIFEST.json'),
              'source_seed_protocol_sha256':seeds.sha(Path(args.seed_results)/'SEED_PROTOCOL.json'),
              'script_sha256':seeds.sha(__file__),'process_init_seed':BASE_SEED,
              'original_pairing':list(map(list,pairs(ORIGINAL_ORDER))),
              'alternative_splits':{str(s):list(map(list,pairs(v))) for s,v in orders.items()},
              'epochs':10,'steps':8,'folds_per_split':4,
              'warning':'Existing 8 developmental ring labels reused; split sensitivity is not an independent test.'}
    write_new(root/'SPLIT_PROTOCOL.json',metadata)
    results={t:{'canonical':canonical[t]} for t in TRIALS}
    for trial in TRIALS:
        cfg=protocol['selected_configs'][str(trial)]
        for ss,order in orders.items():
            results[trial][ss]={}
            for mode in MODES:
                rep=fit_pairing(rv,cfg,order,mode)
                rep['provenance']={'trial':trial,'pairing_seed':ss,'training_mode':mode,
                                   'split_protocol_sha256':seeds.sha(root/'SPLIT_PROTOCOL.json')}
                write_new(root/f'TRIAL_{trial:04d}_SPLIT_{ss}_{mode}.json',rep)
                results[trial][ss][mode]=rep
                print(f'TRIAL {trial} SPLIT {ss} {mode}: RMSE {np.round(rep["rmse_mm"],4)}',flush=True)
    s=summarize(results)
    write_new(root/'SPLIT_SENSITIVITY_SUMMARY.json',{'protocol':metadata['protocol'],
              'source_protocol_sha256':seeds.sha(root/'SPLIT_PROTOCOL.json'),
              'split_names':['canonical',*list(SPLIT_SEEDS)],
              'results':s,'warning':metadata['warning']})
    for trial,v in s.items():
        for contrast,c in v['contrasts'].items():
            print('TRIAL',trial,contrast,'beneficial split counts',c['n_splits_better_per_target'],flush=True)
    print('SPLIT SENSITIVITY FINISHED',root.resolve(),flush=True)
    print('DEVELOPMENT ONLY. PREVIOUS OUTER RINGS NOT OPENED.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--ablation',required=True)
    p.add_argument('--strict',required=True)
    p.add_argument('--seed-results',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    run(p.parse_args())

if __name__=='__main__': main()
