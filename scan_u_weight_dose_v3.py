"""DEVELOPMENT ONLY: fixed C dose-response, two frozen Optuna candidates.

Two candidates (trials 13 and 22) and one fixed 4-fold pairing per invocation.
No refit of original C=0 or selected-C anchors: load previously validated reports.
Each C is assessed with the same Bayesian HMM, 10 epochs, 8 local steps,
fixed candidate-specific physical_var, fixed K/P0/M2, and same 4-fold ring split.

The eight development labels have already been repeatedly used for selection;
this is an EXPLORATORY, not an independent performance assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

import compare_split_robustness_v3 as split
import optuna_ssr_development as opt
import ablate_optuna_v3 as ab
import compare_seed_robustness_v3 as seedmod

PROTOCOL = 'DM-V3-U-WEIGHT-DOSE-2026-11-12-v1'
TRIALS=(13,22)
FIXED_C=(0.0,0.05,0.10,0.20,0.35,0.50,0.75,1.0,2.0)
PAIRS=('canonical','20261114','20261115')
BUDGET=(4,10,8)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fh:
        for chunk in iter(lambda:fh.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_new(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as fh:
        json.dump(data,fh,indent=2,ensure_ascii=False,allow_nan=False)
        fh.write('\n')


def token(c):
    if not 0.<=float(c)<=2. or not np.isfinite(c):raise ValueError('Illegal C')
    return f'{float(c):.5f}'.replace('.','p')


def requested_levels(selected):
    s=float(selected)
    if not 0.<s<2.:raise ValueError('Optuna C outside registered levels')
    levels=sorted(set((*FIXED_C,s)))
    if len(levels)!=len(FIXED_C)+1:
        raise ValueError('Selected C duplicates registered grid')
    if len({token(c) for c in levels})!=len(levels):
        raise ValueError('C filename collision')
    return levels


def validate_output_path(out, roots):
    out=Path(out).resolve()
    if out==Path('.').resolve():raise ValueError('No output into project root')
    for root in roots:
        p=Path(root).resolve()
        if out==p or p in out.parents or out in p.parents:
            raise ValueError('Output location intersects immutable source folder')
    return out


def source_bundle(args):
    manifest,orig,cached=split.load_canonical(args.study,args.ablation,args.strict,args.seed_results,args.data)
    root=Path(args.split_results)
    p=root/'SPLIT_PROTOCOL.json'
    s=root/'SPLIT_SENSITIVITY_SUMMARY.json'
    proto=json.loads(p.read_text(encoding='utf-8'))
    summ=json.loads(s.read_text(encoding='utf-8'))
    if proto['protocol']!='DM-V3-SPLIT-SENSITIVITY-2026-11-12-v1':
        raise ValueError('Expected registered split sensitivity source')
    if summ['source_protocol_sha256']!=sha(p):raise ValueError('Split summary/protocol mismatch')
    if proto['source_data_sha256']!=manifest['input_pickle_sha256']:
        raise ValueError('Split data hash mismatch')
    if proto['source_model_code_sha256']!=manifest['code_sha256']:
        raise ValueError('Split model source hash mismatch')
    if proto['script_sha256']!=sha(split.__file__):
        raise ValueError('Split source script changed')
    orders={'canonical':split.ORIGINAL_ORDER}
    for seed,ordr in split.new_orders().items():orders[str(seed)]=ordr
    if proto['original_pairing']!=list(map(list,split.pairs(split.ORIGINAL_ORDER))):
        raise ValueError('Canonical pairing changed')
    if proto['alternative_splits']!={str(k):list(map(list,split.pairs(v))) for k,v in split.new_orders().items()}:
        raise ValueError('Alternative pairing changed')
    anchors={}
    hashes={}
    for t in TRIALS:
        anchors[t]={'canonical':cached[t]}
        hashes[str(t)]={'canonical':{'shared_C0':sha(Path(args.ablation)/f'TRIAL_{t:04d}_C_zero_P_selected.json'),
                                     'ssl_Cselected':sha(Path(args.ablation)/f'TRIAL_{t:04d}_C_selected_P_selected.json')}}
        for name in PAIRS[1:]:
            anchors[t][name]={};hashes[str(t)][name]={}
            for mode in ('strict_L_only','shared_C0','ssl_Cselected'):
                q=root/f'TRIAL_{t:04d}_SPLIT_{name}_{mode}.json'
                rep=json.loads(q.read_text(encoding='utf-8'))
                split.check_report(rep,orders[name],mode)
                pv=rep.get('provenance')
                if not isinstance(pv,dict) or (pv.get('trial'),str(pv.get('pairing_seed')),pv.get('training_mode'))!=(t,name,mode):
                    raise ValueError('Split report provenance invalid')
                if pv.get('split_protocol_sha256')!=sha(p):
                    raise ValueError('Split report protocol SHA invalid')
                anchors[t][name][mode]=rep;hashes[str(t)][name][mode]=sha(q)
    if args.pairing not in orders:raise ValueError('Unregistered pairing')
    return manifest,orig,orders,anchors,hashes,{'split_protocol':sha(p),'split_summary':sha(s)}


def profile_metrics(reports,selected):
    if len(reports)<2:raise ValueError('No C-profile')
    ordered=sorted(reports)
    anchor=reports[0.0]
    if selected not in reports:raise ValueError('Missing selected-C reference')
    series=[]
    for c in ordered:
        r=reports[c]
        comparison=ab.paired(r,anchor)
        series.append({'C':float(c),'rmse_mm':r['rmse_mm'],'mae_mm':r['mae_mm'],
                       'delta_mse_vs_shared_C0_mm2':comparison['paired_delta_mse_mm2'],
                       'better_rings_abs_vs_shared_C0':comparison['n_rings_better_abs_error'],
                       'loo_sign_stable_vs_shared_C0':comparison['loo_sign_stable'],
                       'anchor': c==0.0 or c==selected})
    best=[min(series,key=lambda x:x['rmse_mm'][j])['C'] for j in (0,1)]
    return {'dose_curve':series,'minimum_descriptive_rmse_C_by_target':best,
            'warning':'Selected C is post-hoc on reused 8-ring development labels. Minima are exploratory, NOT proven optimal.'}


def cached_or_fit(rv,c,selected,cache,order,cfg):
    if c==0.0:return cache['shared_C0'], 'immutable_historical_reference'
    if c==selected:return cache['ssl_Cselected'], 'immutable_historical_reference'
    params=dict(cfg,C=float(c))
    # The old modules must be patched only synchronously and must be restored.
    with split.temporary_fold_order(order),patch.object(opt,'FIXED_SEED',split.BASE_SEED):
        out=opt.evaluate_configuration(rv,params,folds=4,epochs=10,steps=8)
    split.check_report(out,order,'ssl_Cselected')
    return out,'new_cv4_training'


def make_protocol(args,manifest,orig,orders,hashes,source_hashes):
    return {'protocol':PROTOCOL,'status':'EXPLORATORY_POST_SELECTION_DEV_CV',
            'pairing':args.pairing,'pairing_order':list(orders[args.pairing]),
            'validation_pairs':[list(p) for p in split.pairs(orders[args.pairing])],
            'trials':list(TRIALS),
            'configurations':{str(t):orig['selected_configs'][str(t)] for t in TRIALS},
            'fixed_C_grid':list(FIXED_C),'plus_exact_original_selected_C':True,
            'fixed_epochs':10,'fixed_steps':8,'fixed_initialization_seed':split.BASE_SEED,
            'data_sha256':manifest['input_pickle_sha256'],
            'core_code_sha256':manifest['code_sha256'],
            'source_reference_file_hashes':hashes,'source_protocol_sha256':source_hashes,
            'script_sha256':sha(__file__),
            'no_outer_ids':list(opt.LOCKED),
            'research_question':'Effect of weighted unlabeled process ELBO holding preprocessing, K and prior fixed',
            'important_C0_note':'C=0 retains U process observations in X and descriptor scaling and HMM initialization; compare separately against strict_L_only.',
            'multiple_testing_warning':'No independent test; these same 8 labeled rings have been repeatedly examined.'}


def summary_file_name(trial,c):
    return f'TRIAL_{trial:04d}_C_{token(c)}.json'


def check_cached_and_outputs(args,out,protocol,orig,orders,anchors):
    """Pure integrity / validation of the already present reports."""
    result={}
    for trial in TRIALS:
        cfg=orig['selected_configs'][str(trial)]
        selected=float(cfg['C'])
        all_reports={}
        for c in requested_levels(selected):
            if c in (0.0,selected):
                r=anchors[trial][args.pairing]['shared_C0' if c==0.0 else 'ssl_Cselected']
            else:
                path=out/summary_file_name(trial,c)
                rr=json.loads(path.read_text(encoding='utf-8'))
                if rr.get('dose_protocol_sha256')!=sha(out/'C_DOSE_PROTOCOL.json'):
                    raise ValueError('New dose report protocol checksum mismatch')
                if rr.get('trial')!=trial or not np.isclose(float(rr.get('C')),c,atol=1e-12,rtol=0):
                    raise ValueError('New dose report metadata mismatch')
                if rr.get('kind')!='new_cv4_training':raise ValueError('Report kind changed')
                if rr['configuration']!=dict(cfg,C=float(c)):
                    raise ValueError('New dose report hyperparameters changed')
                r=rr['cv4_report']
            split.check_report(r,orders[args.pairing],'shared_C0' if c==0.0 else 'ssl_Cselected')
            all_reports[c]=r
        info=profile_metrics(all_reports,selected)
        info['trial_configuration']=cfg
        info['strict_supervised_rmse_mm']=anchors[trial][args.pairing]['strict_L_only']['rmse_mm']
        info['selected_C']=selected
        info['pairing']=args.pairing
        result[str(trial)]=info
    return result


def run(args):
    roots=(args.study,args.ablation,args.strict,args.seed_results,args.split_results,args.data)
    out=validate_output_path(args.out,roots)
    if out.exists() and not args.resume:
        raise FileExistsError('Output exists. Use --resume only for interrupted SAME protocol; never overwrite.')
    if not out.exists() and args.resume:
        raise FileNotFoundError('No protocol exists to resume')
    manifest,orig,orders,anchors,hashes,source_hashes=source_bundle(args)
    protocol=make_protocol(args,manifest,orig,orders,hashes,source_hashes)
    rv=opt.dev.load_development(args.data)
    opt.validate_development(rv)
    if args.resume:
        old=json.loads((out/'C_DOSE_PROTOCOL.json').read_text(encoding='utf-8'))
        if old!=protocol:raise RuntimeError('Cannot resume: code, data, sources or protocol changed')
        if (out/'C_DOSE_SUMMARY.json').exists():
            raise FileExistsError('Already completed. Never rerun an already summarized profile.')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_new(out/'C_DOSE_PROTOCOL.json',protocol)
    for trial in TRIALS:
        cfg=orig['selected_configs'][str(trial)]
        selected=float(cfg['C'])
        for c in requested_levels(selected):
            if c in (0.,selected):
                print(f'TRIAL {trial} SPLIT {args.pairing} C={c:.6f} reuse verified previous report',flush=True)
                continue
            path=out/summary_file_name(trial,c)
            if path.exists():
                old=json.loads(path.read_text(encoding='utf-8'))
                if (old.get('dose_protocol_sha256')!=sha(out/'C_DOSE_PROTOCOL.json') or
                    old.get('trial')!=trial or old.get('C')!=c or
                    old.get('configuration')!=dict(cfg,C=c) or old.get('kind')!='new_cv4_training'):
                    raise ValueError('Cannot resume: existing trial metadata mismatch')
                split.check_report(old['cv4_report'],orders[args.pairing],'ssl_Cselected')
                print(f'TRIAL {trial} SPLIT {args.pairing} C={c:.4f} reuse completed new report',flush=True)
                continue
            report,kind=cached_or_fit(rv,c,selected,anchors[trial][args.pairing],orders[args.pairing],cfg)
            record={'kind':kind,'trial':trial,'C':c,'configuration':dict(cfg,C=c),
                    'pairing':args.pairing,
                    'dose_protocol_sha256':sha(out/'C_DOSE_PROTOCOL.json'),
                    'cv4_report':report}
            write_new(out/summary_file_name(trial,c),record)
            print(f'TRIAL {trial} SPLIT {args.pairing} C={c:.4f} RMSE mm {np.round(report["rmse_mm"],4)}',flush=True)
    complete=check_cached_and_outputs(args,out,protocol,orig,orders,anchors)
    summary={'protocol':PROTOCOL,'source_protocol_sha256':sha(out/'C_DOSE_PROTOCOL.json'),
             'results':complete,'warning':'Repeated previously selected development rings; no independent test.'}
    write_new(out/'C_DOSE_SUMMARY.json',summary)
    for trial,group in complete.items():
        print(f'TRIAL {trial} SPLIT {args.pairing} C-profile complete: {len(group["dose_curve"])} levels; '
              f'minima descriptive only {group["minimum_descriptive_rmse_C_by_target"]}',flush=True)
    print('DOSE RESPONSE SAVED',out)
    print('NO OUTER TEST; NO NEW INDEPENDENT EVALUATION; NEVER TUNE FROM OLD OUTER TEST')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--ablation',required=True)
    p.add_argument('--strict',required=True)
    p.add_argument('--seed-results',required=True)
    p.add_argument('--split-results',required=True)
    p.add_argument('--pairing',required=True,choices=PAIRS)
    p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true',help='Resume only interrupted run with exact same source/config; never overwrite reports')
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    args=p.parse_args();run(args)


if __name__=='__main__':main()
