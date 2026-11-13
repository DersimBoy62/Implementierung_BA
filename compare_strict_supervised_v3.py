"""Registered DEVELOPMENT-ONLY strict-supervised control for Optuna V3 trials 13/22.

The existing factorial ablation remains IMMUTABLE. Only two additional strict-L-only
CV4 configurations are fitted, with zero U-process contribution to scalers,
KMeans initialisation, Bayesian updates or quality heads. All four old outer
ring IDs are excluded. Reuses stored C0/Cselected/C1 reports unchanged.

IMPORTANT: all estimates are exploratory; same eight validation rings reused.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import ablate_optuna_v3 as ab
import optuna_ssr_development as opt
import compare_states_cv4 as state
import dynamic_quality_v2 as dynamic
from compare_two_head_cv4 import OPTIONS
from quality_link_challenge import extract_descriptors, FIXED_SEED

TRIALS=(13,22)
MODES=('strict_L_only','shared_C0','ssl_Cselected','ssl_C1')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def strict_fold(rv,config,f,*,epochs=10,steps=8):
    """No U feature or quality is accessed after loading development view.

    Only training L entries are used to fit process, target and physical scalers,
    and only L are passed to dynamic.train and its KMeans initialisation.
    """
    opt.validate_parameters(config)
    opt.validate_development(rv)
    if config['preprocessing']!='P0' or config['variant']!='M2' or config['K'] not in (2,3):
        raise ValueError('Only pre-registered structures 13/22')
    if not 0<=f<4:raise ValueError('Expected fold 0-3')
    val=opt.EXPECTED_L[2*f:2*f+2]
    trainL=tuple(r for r in opt.EXPECTED_L if r not in val)
    assert len(trainL)==6 and not (set(trainL)|set(val))&set(opt.LOCKED)
    # The ONLY training data visible below comes from the six L rings.
    labeled={rid:rv[rid] for rid in trainL}
    sx=state.fit_process_scaler(labeled,trainL,'P0')
    sy=opt.dm.fit_y_scaler(labeled,trainL)
    x={rid:sx.transform(rv[rid].x) for rid in trainL}
    y={rid:sy.transform(rv[rid].y) for rid in trainL}
    used=trainL+val
    desc={rid:extract_descriptors(sx.transform(rv[rid].x),rv[rid].t,'physical') for rid in used}
    groups=OPTIONS['SEPARATE_AXIAL']
    ss={rid:[None,None] for rid in used}
    for j,idx in enumerate(groups):
        fs=StandardScaler().fit(np.asarray([desc[r][list(idx)] for r in trainL]))
        for rid in used:ss[rid][j]=fs.transform(desc[rid][list(idx)][None,:])[0]
    sdata={rid:ss[rid] for rid in trainL}
    # C=0 here only as a redundant safeguard; zero U rings are supplied.
    model=dynamic.train(x,y,sdata,variant='M2',K=config['K'],C=0.,
                        epochs=epochs,steps=steps,seed=FIXED_SEED,
                        physical_var=float(config['physical_var']))
    pred=[]
    for rid in val:
        p,sd=dynamic.predict(sx.transform(rv[rid].x),ss[rid],model)
        pmm=p*sy.scale+sy.mean; sdmm=sd*sy.scale
        if not np.isfinite(pmm).all() or not np.isfinite(sdmm).all() or (sdmm<=0).any():
            raise ArithmeticError('Invalid strict supervised prediction')
        # Validation labels are accessed only after prediction is finished.
        res=np.asarray(pmm-rv[rid].y,dtype=float)
        pred.append({'ring':rid,'fold':f+1,'prediction_mm':pmm.tolist(),
                     'residual_mm':res.tolist(),'predictive_sd_mm':sdmm.tolist()})
    return pred, {'fold':f+1,'training_labeled_only':list(trainL),'validation':list(val),
                  'x_scaler_fit_ids':list(sx.fitted_ids),
                  'y_scaler_fit_ids':list(sy.fitted_ids),
                  'physical_scaler_fit_ids':list(trainL),
                  'hmm_initialisation_uses_only':list(trainL),
                  'u_used':False,'states':state.summarize_states(model,config['K'])}


def strict_cv4(rv,config,*,epochs=10,steps=8):
    if (epochs,steps)!=(10,8):raise ValueError('Registered full CV requires 10 epochs, 8 steps')
    rows=[]; diagnostics=[]
    for f in range(4):
        rr,dd=strict_fold(rv,config,f,epochs=epochs,steps=steps)
        rows+=rr; diagnostics.append(dd)
        print(f'STRICT FOLD {f+1} K={config["K"]} P0/M2 VALIDATION {dd["validation"]}',flush=True)
    errs=np.asarray([x['residual_mm'] for x in rows],dtype=float)
    if errs.shape!=(8,2) or {x['ring'] for x in rows}!=set(opt.EXPECTED_L):
        raise AssertionError('Strict CV rings inconsistent')
    return {'configuration':dict(config, C=0.0),'selected_optuna_C_not_used':float(config['C']),'mode':'strict_L_only',
            'definition':'Only six labeled train rings used in EACH fold; no U ring in preprocessing or initialization or ELBO',
            'epochs':epochs,'steps':steps,'folds_count':4,'per_ring':rows,'fold_diagnostics':diagnostics,
            'rmse_mm':np.sqrt(np.mean(errs**2,axis=0)).tolist(),
            'mae_mm':np.mean(np.abs(errs),axis=0).tolist()}


def validate_ablation(study_dir,ablation_dir,data):
    study=Path(study_dir);a=Path(ablation_dir)
    manifest=json.loads((study/'STUDY_MANIFEST.json').read_text(encoding='utf-8'))
    protocol=json.loads((a/'ABLATION_PROTOCOL.json').read_text(encoding='utf-8'))
    if manifest['search_phase']!='extended' or manifest['smoke'] or (manifest['folds'],manifest['epochs'],manifest['local_steps'])!=(4,10,8):
        raise ValueError('Expected extended CV4 original study')
    if protocol['protocol']!='DM-V3-FIXED-FACTORIAL-ABLATION-v1' or protocol['registered_trial_ids']!=[13,22]:
        raise ValueError('Unexpected controlled ablation')
    if protocol['source_optuna_manifest_sha256']!=sha(study/'STUDY_MANIFEST.json'):
        raise ValueError('Optuna manifest mismatch')
    if protocol['core_code_sha256']!=opt._hash_dependencies(Path(opt.__file__).resolve().parent):
        raise ValueError('Core files changed')
    if protocol['input_sha256']!=manifest['input_pickle_sha256'] or opt.dm.sha256(data)!=manifest['input_pickle_sha256']:
        raise ValueError('Source pickle changed')
    summary=json.loads((a/'ABLATION_SUMMARY.json').read_text(encoding='utf-8'))
    if summary['source_protocol_sha256']!=sha(a/'ABLATION_PROTOCOL.json'):
        raise ValueError('Ablation summary and protocol mismatch')
    reports={}
    for trial in TRIALS:
        c=protocol['selected_configs'][str(trial)]
        if ab.selected_trial({'trials':[{'trial':trial,'config':c}]},trial)['config']!=c:
            raise ValueError('Candidate metadata changed')
        reports[trial]={}
        for key,cname in [('shared_C0','zero'),('ssl_Cselected','selected'),('ssl_C1','one')]:
            filename=f'TRIAL_{trial:04d}_C_{cname}_P_selected.json'
            rp=json.loads((a/filename).read_text(encoding='utf-8'))
            if (rp.get('source_trial')!=trial or rp.get('c_level')!=cname or rp.get('prior_level')!='selected'
                or rp.get('origin_protocol_sha256')!=sha(a/'ABLATION_PROTOCOL.json')):
                raise ValueError('Ablation cell metadata does not match')
            expected=summary['results'][str(trial)]['cells'][f'C_{cname}_P_selected']['rmse_mm']
            if not np.allclose(rp['rmse_mm'],expected,atol=1e-12,rtol=1e-12):
                raise ValueError('Ablation stored RMSE mismatch')
            ab.error_array(rp)
            reports[trial][key]=rp
    return manifest,protocol,reports


def summarize_comparisons(reports):
    return {trial:{'models':{k:{'rmse_mm':r['rmse_mm'],'mae_mm':r['mae_mm']} for k,r in group.items()},
                   'contrasts':{
                       'shared_C0_minus_strict':ab.paired(group['shared_C0'],group['strict_L_only']),
                       'ssl_Cselected_minus_shared_C0':ab.paired(group['ssl_Cselected'],group['shared_C0']),
                       'ssl_Cselected_minus_strict':ab.paired(group['ssl_Cselected'],group['strict_L_only']),
                       'ssl_C1_minus_ssl_Cselected':ab.paired(group['ssl_C1'],group['ssl_Cselected'])
                   }} for trial,group in reports.items()}


def run(study_dir,ablation_dir,outdir,*,data='input/DM_16_aktive_Ringe.pkl'):
    a=Path(ablation_dir);out=Path(outdir)
    if out.exists():
        raise FileExistsError('Output path already exists; never overwrite, use a new name')
    manifest,protocol,reports=validate_ablation(study_dir,a,data)
    rv=opt.dev.load_development(data)
    opt.validate_development(rv)
    final={}
    for trial in TRIALS:
        candidate=protocol['selected_configs'][str(trial)]
        result=strict_cv4(rv,candidate)
        final[trial]={**reports[trial],'strict_L_only':result}
    summary=summarize_comparisons(final)
    out.mkdir(parents=True,exist_ok=False)
    ab.create_new_json(out/'STRICT_CONTROL_MANIFEST.json',{
        'protocol':'DM-V3-STRICT-L-CONTROL-v1',
        'source_ablation_protocol_sha256':sha(a/'ABLATION_PROTOCOL.json'),
        'source_ablation_summary_sha256':sha(a/'ABLATION_SUMMARY.json'),
        'data_sha256':manifest['input_pickle_sha256'],
        'core_code_sha256':opt._hash_dependencies(Path(opt.__file__).resolve().parent),
        'source_code_sha256':sha(__file__),
        'trials':list(TRIALS),
        'modes':list(MODES),
        'warning':'Post-hoc exploratory; same 8 validation rings reused many times; no old outer test'} )
    for trial in TRIALS:
        ab.create_new_json(out/f'STRICT_L_ONLY_TRIAL_{trial:04d}.json',final[trial]['strict_L_only'])
        r=summary[trial]
        print(f'TRIAL {trial}:',flush=True)
        for mode in MODES:
            print(f'  {mode:17s} RMSE mm {np.round(r["models"][mode]["rmse_mm"],4)}',flush=True)
        for k,v in r['contrasts'].items():
            print('  ',k,'paired ΔMSE',np.round(v['paired_delta_mse_mm2'],4),
                  'better rings',v['n_rings_better_abs_error'],
                  'LOO stable',v['loo_sign_stable'],flush=True)
    ab.create_new_json(out/'STRICT_COMPARISON_SUMMARY.json',
                       {'protocol':'DM-V3-STRICT-L-CONTROL-v1',
                        'results':{str(t):summary[t] for t in TRIALS},
                        'warning':'Exploratory repeated development CV; no independent confirmation'})
    print('STRICT COMPARISON SAVED:',out.resolve())
    print('NO OUTER TEST; NO U LABELS; ORIGINAL STUDY AND FACTORIAL REPORTS NOT MODIFIED')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--ablation',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    x=p.parse_args()
    run(x.study,x.ablation,x.out,data=x.data)

if __name__=='__main__':main()
