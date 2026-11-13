"""Freeze the pre-external-test DM two-head SSR experiment protocol.

Read only development CV reports, local Python source bytes, and hash of the source
pickle. Does NOT deserialize the pickle, fit anything, infer latent states, or
calculate labels/errors of the four external test rings.
Standard library only; Python >= 3.9.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import math
import platform
from pathlib import Path
from datetime import datetime, timezone

EXPECTED_DATA_SHA = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'
OUTER = ('DM07','DM08','DM11','DM15')
LABELED = ('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
UNLABELED = ('DM01','DM02','DM10','DM14')
FOLD_VALS = (('DM04','DM05'),('DM06','DM09'),('DM12','DM16'),('DM17','DM18'))
MODES = ('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')
CODE_FILES = ('dm_data.py','vb_b1.py','full_markov_b1.py','fullcov_b1.py',
              'quality_link_challenge.py','hybrid_b1.py',
              'two_head_b1.py','compare_two_head_cv4.py','audit_two_head_ssr.py')
HEADS = {
    'Außenunrundheit': ('log_duration','avg_radial_force','std_radial_force',
                       'delta_outer_diameter','delta_wall_thickness','max_radial_force'),
    'Höhenschlag': ('log_duration','avg_axial_force','std_axial_force',
                   'delta_height','max_axial_force')
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(report,reference):
    assert tuple(report['outer_test_locked'])==OUTER, 'Wrong outer ring list in SSR ablation'
    assert tuple(reference['outer_test_locked'])==OUTER, 'Wrong outer ring list in two-head CV'
    cfg=report['config'];rcfg=reference['config']
    expected={'features':'V1_Kern_6','window':'F2','K':3,'epochs':10,
              'local_steps':8,'seed':20261111,'head_variant':'SEPARATE_AXIAL'}
    for k,v in expected.items():
        assert cfg[k]==v, f'Wrong SSR hyperparameter: {k}'
    for k,v in {'variant':'SEPARATE_AXIAL','K':3,'C':1.,'epochs':10,'local_steps':8,
                 'seed':20261111,'features':'V1_Kern_6','window':'F2'}.items():
        assert rcfg[k]==v, f'Wrong two-head hyperparameter: {k}'
    assert list(rcfg['heads'])==[list(HEADS['Außenunrundheit']),list(HEADS['Höhenschlag'])], 'Quality head features changed'
    assert len(report['folds'])==4 and len(reference['folds'])==4,'Need all four inner folds'
    for i,fold in enumerate(report['folds']):
        assert fold['fold']==i+1 and tuple(fold['inner_val'])==FOLD_VALS[i],f'Unexpected fold {i+1}'
        L=tuple(fold['train_labeled']); U=tuple(fold['train_unlabeled'])
        assert set(L)==set(LABELED)-set(FOLD_VALS[i]) and set(U)==set(UNLABELED)
        assert tuple(fold['outer_test_locked'])==OUTER
        assert set(fold['modes'])==set(MODES)
        for mode in MODES:
            mr=fold['modes'][mode]
            train=set(L) if mode=='SUP_L_ONLY' else (set(L)|set(U))
            assert set(mr['x_scaler_fit'])==train and set(mr['physics_scaler_fit'])==train
            assert set(mr['y_scaler_fit'])==set(L) and set(mr['quality_train_ids'])==set(L)
            assert mr['unlabeled_elbo_weight']==(1. if mode=='SSL_U_C1' else 0.)
            assert {x['ring'] for x in mr['rows']}==set(FOLD_VALS[i]);
            assert not train.intersection(set(FOLD_VALS[i])|set(OUTER))
    for mode in MODES:
        rows=report['per_ring'][mode]
        assert len(rows)==8 and {r['ring'] for r in rows}==set(LABELED)
        for r in rows:
            assert len(r['residual_mm'])==2 and len(r['predictive_sd_mm'])==2
            assert all(math.isfinite(float(x)) for x in r['residual_mm']+r['predictive_sd_mm'])
        calc=[(sum(float(r['residual_mm'][j])**2 for r in rows)/8)**.5 for j in (0,1)]
        assert max(abs(a-b) for a,b in zip(calc,report['metrics'][mode]['rmse_mm']))<1e-9
    original={r['ring']:r for r in reference['per_ring']}
    now={r['ring']:r for r in report['per_ring']['SSL_U_C1']}
    assert set(original)==set(now)==set(LABELED)
    maxdiff=max(abs(float(now[r]['pred_mm'][j])-float(original[r]['pred_mm'][j]))
                for r in original for j in (0,1))
    assert maxdiff<=5e-6, f'Abandon freeze; SSR reference mismatch {maxdiff}'
    return maxdiff


def create(data,ablation,twohead,root,out):
    data=Path(data);ablation=Path(ablation);twohead=Path(twohead);root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(f'Freeze manifest already exists: {out}')
    for p in (data,ablation,twohead):
        if not p.is_file():raise FileNotFoundError(p)
    source_sha=sha(data)
    if source_sha!=EXPECTED_DATA_SHA:
        raise ValueError(f'Original pickle SHA256 differs; expected {EXPECTED_DATA_SHA}, actual {source_sha}')
    code_hash={}
    for filename in CODE_FILES:
        f=root/filename
        if not f.is_file():raise FileNotFoundError(f'Expected model source missing: {f}')
        code_hash[filename]=sha(f)
    report=json.loads(ablation.read_text(encoding='utf8'))
    reference=json.loads(twohead.read_text(encoding='utf8'))
    difference=verify(report,reference)
    versions={}
    for pkg in ('numpy','scipy','scikit-learn','numba','optuna'):
        try:versions[pkg]=md.version(pkg)
        except md.PackageNotFoundError:versions[pkg]='not installed'
    protocol={
        'protocol_id':'DM-2HEAD-SSR-FREEZE-2026-11-11-v1',
        'status':'FROZEN BEFORE EXTERNAL TEST; NO EXTERNAL TEST EVALUATION HERE',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'packages':versions,
        'source_data_sha256':source_sha,'source_data_name':data.name,
        'source_python_sha256':code_hash,
        'dev_ablation_sha256':sha(ablation),'dev_twohead_sha256':sha(twohead),
        'outer_test_locked':list(OUTER),'dev_labeled_ids':list(LABELED),'dev_unlabeled_ids':list(UNLABELED),
        'previous_inner_val_folds':[list(x) for x in FOLD_VALS],
        'training':{
            'process_emissions':'Full Normal-Wishart Gaussian with d=6 (not diagonal)',
            'features':'V1_Kern_6', 'window':'F2','K':3,'epochs':10,'local_steps':8,
            'seed':20261111,'FULL_JOINT':True,'unlabeled_C':1.,
            'head_variant':'SEPARATE_AXIAL','head_features':HEADS,
            'head_quality_priors':{'state_var':2.0,'physical_var':0.25,'a':2.5,'b':1.5},
            'Wishart_prior':{'nu0':'d+3','E_prior_Precision':'I','kappa0':0.2},
            'final_train_L':'All 8 development labeled rings',
            'final_train_U':'All 4 development unlabeled rings',
            'scaling':'X and physical features fit 12 development rings; Y scaler fit 8 labeled development rings',
            'quality_targets':['outer_roundness_mm','height_runout_mm'],
            'prediction':'New ring X-only state inference; NO use of its quality values'
        },
        'confirmatory_models':{
            'primary':'Two-head, FULL_JOINT, C=1; both quality outputs, NO post-test change',
            'secondary_supervised':'Same heads, K, priors, epoch count; use ONLY labeled dev rings for scaling/initialization/training',
            'baseline':'Training-label mean, fitted from all 8 development labeled quality pairs',
        },
        'evaluation_protocol':{
            'primary_metrics':'Two individual outer-test RMSE in physical mm; report BOTH (no after-the-fact metric switching)',
            'secondary_metrics':['MAE in mm per target','errors per test ring','paired MSE change vs supervised','paired MSE change vs baseline'],
            'uncertainty':'Per-ring predicted SD and coverage of ±1.96*SD only as APPROXIMATIVE descriptive metrics; no exact 95% guarantee',
            'N_outer_test_rings':4,
            'allowed_after_test':'Reporting and scientifically interpreting fixed results; NO tuning, selection, or retesting on same four rings',
            'forbidden':'Use outer ring quality values in fitting, scaling, HMM posterior for prediction, hyperparameter/model selection',
            'limitations':'Earlier inner eight validation rings repeatedly inspected; 4 outer test rings only; no trustworthy general population significance claim'
        },
        'development_results_EXPLORATORY':{k:report['metrics'][k] for k in MODES},
        'dev_ssl_matches_previous_run_max_difference_mm':difference
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--ablation',default='results/two_head_ssr_ablation_cv4/TWO_HEAD_SSR_ABLATION_CV4.json')
    p.add_argument('--twohead',default='results/two_head_separate_axial_cv4/TWO_HEAD_SEPARATE_AXIAL_CV4.json')
    p.add_argument('--code-root',default='.')
    p.add_argument('--out',default='results/experiment_freeze/DM_TWO_HEAD_PROTOCOL_FREEZE.json')
    a=p.parse_args()
    r=create(a.data,a.ablation,a.twohead,a.code_root,a.out)
    print('PROTOKOLL EINGEFROREN:',Path(a.out).resolve())
    print('Originalpickles SHA256',r['source_data_sha256'])
    print('Innere 4 Folds, 8 Entwicklungsringe und Software-SHA256 geprüft.')
    print('ÄUSSERER TEST NICHT EVALUIERT:',r['outer_test_locked'])
    print('Nächster Schritt: unveränderliche Auswertungsroutine prüfen; erst danach einmalige äußere Bewertung.')

if __name__=='__main__':main()
