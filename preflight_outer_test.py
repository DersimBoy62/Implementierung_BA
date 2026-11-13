"""Preflight of the frozen DM external-test protocol. NO model fitting, no pickle deserialization.

Only SHA256 read of the source pickle, development reports, and frozen source files;
plus synthetic-only prediction/label separation and metric calculations.
This script intentionally has NO option for real outer-test evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_ID = 'DM-2HEAD-SSR-FREEZE-2026-11-11-v1'
EXPECTED_DATA_SHA = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'
L = ('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
U = ('DM01','DM02','DM10','DM14')
OUTER = ('DM07','DM08','DM11','DM15')
MODELS = ('primary_ssl', 'strict_supervised', 'mean_baseline')
EXPECTED_SOURCE_FILES = (
    'dm_data.py','vb_b1.py','full_markov_b1.py','fullcov_b1.py',
    'quality_link_challenge.py','hybrid_b1.py','two_head_b1.py',
    'compare_two_head_cv4.py','audit_two_head_ssr.py'
)


def sha256_file(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def preflight_manifest(protocol_path, data_path, code_root, ablation_path, twohead_path):
    """Fail closed on any change to registered data, code, dev reports, or model selection.

    Reading the pickle is limited to a streaming cryptographic hash. No unpickling.
    """
    protocol = read_json(protocol_path)
    if protocol.get('protocol_id') != PROTOCOL_ID or 'FROZEN BEFORE EXTERNAL TEST' not in protocol.get('status',''):
        raise ValueError('Wrong or non-frozen experiment protocol')
    if protocol.get('source_data_sha256') != EXPECTED_DATA_SHA:
        raise ValueError('Frozen data SHA not recognized')
    if sha256_file(data_path) != EXPECTED_DATA_SHA:
        raise ValueError('Source DM pickle has changed')
    if protocol.get('source_data_name') != Path(data_path).name:
        raise ValueError('Source pickle filename mismatch')
    if tuple(protocol.get('outer_test_locked',[])) != OUTER:
        raise ValueError('Outer ring IDs changed')
    if tuple(protocol.get('dev_labeled_ids',[])) != L or tuple(protocol.get('dev_unlabeled_ids',[])) != U:
        raise ValueError('Development ring IDs changed')
    if protocol.get('previous_inner_val_folds') != [list(L[i:i+2]) for i in range(0,8,2)]:
        raise ValueError('Inner folds changed')
    t = protocol['training']
    required = {'features':'V1_Kern_6','window':'F2','K':3,'epochs':10,
                'local_steps':8,'seed':20261111,'FULL_JOINT':True,
                'unlabeled_C':1.0,'head_variant':'SEPARATE_AXIAL'}
    for k,v in required.items():
        if t.get(k) != v:raise ValueError('Frozen parameter changed: '+k)
    if t.get('quality_targets') != ['outer_roundness_mm','height_runout_mm']:
        raise ValueError('Target order or identity changed')
    if t.get('Wishart_prior') != {'nu0':'d+3','E_prior_Precision':'I','kappa0':0.2}:
        raise ValueError('Frozen process prior changed')
    if t.get('head_quality_priors') != {'state_var':2.,'physical_var':.25,'a':2.5,'b':1.5}:
        raise ValueError('Frozen head priors changed')
    if t.get('head_features') != {
        'Außenunrundheit':['log_duration','avg_radial_force','std_radial_force',
                          'delta_outer_diameter','delta_wall_thickness','max_radial_force'],
        'Höhenschlag':['log_duration','avg_axial_force','std_axial_force',
                        'delta_height','max_axial_force']}:
        raise ValueError('Quality head feature definitions changed')
    c = protocol['confirmatory_models']
    if not all(k in c for k in ('primary','secondary_supervised','baseline')):
        raise ValueError('Control models missing')
    ep = protocol['evaluation_protocol']
    if ep.get('N_outer_test_rings') != 4 or 'RMSE' not in ep.get('primary_metrics',''):
        raise ValueError('Frozen primary evaluation rule changed')
    if protocol.get('dev_ablation_sha256') != sha256_file(ablation_path):
        raise ValueError('Frozen development ablation report has changed')
    if protocol.get('dev_twohead_sha256') != sha256_file(twohead_path):
        raise ValueError('Frozen two-head development report has changed')
    recorded=protocol.get('source_python_sha256',{})
    if tuple(recorded) != EXPECTED_SOURCE_FILES:
        raise ValueError('Frozen code file list/order changed')
    code_root=Path(code_root)
    for file in EXPECTED_SOURCE_FILES:
        if not (code_root/file).is_file() or recorded[file]!=sha256_file(code_root/file):
            raise ValueError('Frozen source code changed or missing: '+file)
    return {'protocol_id':PROTOCOL_ID,
            'source_data_sha256':EXPECTED_DATA_SHA,
            'frozen_python_files_checked':len(EXPECTED_SOURCE_FILES),
            'frozen_development_reports_checked':2,
            'outer_test_locked':list(OUTER),
            'only_streamed_pickle_bytes_for_sha256':True}


def validate_predictions(predictions):
    """Exact schema gate. Throws before a label is made accessible for scoring."""
    if set(predictions) != set(MODELS):raise ValueError('Missing/extra frozen models')
    for model,ring_predictions in predictions.items():
        if set(ring_predictions) != set(OUTER):
            raise ValueError('Predictions must cover exactly four fixed test rings: '+model)
        for rid, vec in ring_predictions.items():
            if not isinstance(vec,(list,tuple)) or len(vec)!=2:
                raise ValueError(f'Two outputs required: {model}/{rid}')
            if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in vec):
                raise ValueError('Nonfinite/non-numeric prediction')
    return True


class SyntheticSealedLabels:
    """Contract test only: labels cannot be read until all predictions are committed."""
    def __init__(self,labels):
        self.__labels = {k:list(v) for k,v in labels.items()}
        self.__released=False
        if set(self.__labels)!=set(OUTER):raise ValueError('Wrong synthetic label rings')
    def get(self,ring):
        if not self.__released:
            raise PermissionError('Quality labels are sealed until ALL predictions are committed')
        return list(self.__labels[ring])
    def commit_predictions(self,predictions):
        validate_predictions(predictions)
        self.__released=True


def calculate_scores(predictions,truth):
    """Exact equal-ring weighting for the 2 frozen targets; no metric selection."""
    validate_predictions(predictions)
    if set(truth)!=set(OUTER) or any(len(truth[r])!=2 for r in OUTER):
        raise ValueError('Exactly 4 truth pairs required')
    residuals={}
    metrics={}
    for name in MODELS:
        e=[[predictions[name][r][j]-truth[r][j] for j in range(2)] for r in OUTER]
        residuals[name]=e
        metrics[name]={'rmse_mm':[
            math.sqrt(sum(q[j]*q[j] for q in e)/len(OUTER)) for j in (0,1)],
            'mae_mm':[sum(abs(q[j]) for q in e)/len(OUTER) for j in (0,1)]}
    paired={}
    for comp in ('strict_supervised','mean_baseline'):
        paired[comp] = [sum(residuals['primary_ssl'][i][j]**2-residuals[comp][i][j]**2
                            for i in range(len(OUTER)))/len(OUTER) for j in (0,1)]
    return {'targets':['outer_roundness_mm','height_runout_mm'],
            'ring_count':len(OUTER),'model_metrics':metrics,
            'paired_delta_mse_primary_minus_control_mm2':paired}


def synthetic_dry_run():
    """Artificial numbers unrelated to real DM labels; tests scoring interface only."""
    fake_y={rid:[float(i),float(3-i)] for i,rid in enumerate(OUTER)}
    fake_preds={
        'primary_ssl':{rid:[float(i)+0.1,float(3-i)-0.2] for i,rid in enumerate(OUTER)},
        'strict_supervised':{rid:[float(i)+0.4,float(3-i)+0.5] for i,rid in enumerate(OUTER)},
        'mean_baseline':{rid:[1.5,1.5] for rid in OUTER},
    }
    vault=SyntheticSealedLabels(fake_y)
    blocked=False
    try:vault.get(OUTER[0])
    except PermissionError:blocked=True
    if not blocked:raise AssertionError('Synthetic labels were visible before prediction completion')
    vault.commit_predictions(fake_preds)
    truth={r:vault.get(r) for r in OUTER}
    score=calculate_scores(fake_preds,truth)
    if any(abs(x-y)>1e-12 for x,y in zip(score['model_metrics']['primary_ssl']['rmse_mm'],[.1,.2])):
        raise AssertionError('Synthetic scoring RMSE is wrong')
    return {'synthetic_labels_sealed_until_all_predictions_committed':blocked,
            'synthetic_predictions_have_two_targets':True,
            'synthetic_equal_ring_weighting_and_rmses_verified':True,
            'synthetic_primary_rmse_mm':score['model_metrics']['primary_ssl']['rmse_mm'],
            'real_outer_test_labels_accessed':False,
            'real_models_trained_or_evaluated':False}


def run(protocol, data, code_root, ablation, twohead, out):
    out=Path(out)
    if out.exists():raise FileExistsError('Preflight output already exists: '+str(out))
    audit=preflight_manifest(protocol,data,code_root,ablation,twohead)
    syn=synthetic_dry_run()
    result={'status':'PREFLIGHT ONLY, SYNTHETIC; OUTER TEST SEALED',
            'frozen_contract':audit,'synthetic_contract_test':syn,
            'real_outer_test_run_authorized':False,
            'preflight_python_sha256':sha256_file(__file__)}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='results/experiment_freeze/DM_TWO_HEAD_PROTOCOL_FREEZE.json')
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--code-root',default='.')
    p.add_argument('--ablation',default='results/two_head_ssr_ablation_cv4/TWO_HEAD_SSR_ABLATION_CV4.json')
    p.add_argument('--twohead',default='results/two_head_separate_axial_cv4/TWO_HEAD_SEPARATE_AXIAL_CV4.json')
    p.add_argument('--out',default='results/experiment_freeze/OUTER_PREFLIGHT_SYNTHETIC.json')
    a=p.parse_args()
    result=run(a.protocol,a.data,a.code_root,a.ablation,a.twohead,a.out)
    print('PREFLIGHT BESTANDEN:',Path(a.out).resolve())
    print('9 Code-Hashes, 2 Berichthashes und Original-Pickle-Hash geprüft.')
    print('Künstliche Vorhersagen und gesperrte künstliche Labels: PASS')
    print('KEIN echtes Training. KEINE realen Testlabels geöffnet. KEINE externe Testbewertung.')
    print('ÄUSSERE TESTRINGE WEITER GESPERRT:', result['frozen_contract']['outer_test_locked'])

if __name__=='__main__':main()
