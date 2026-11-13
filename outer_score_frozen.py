"""Read four real external quality labels only AFTER all three predictions are committed.

Runs in a separate process AFTER outer_predict_frozen.py. NO fit, NO inference,
NO model selection. No output overwrites. Must be run once, after human review.
"""
from __future__ import annotations
import argparse,json,pickle,os
from pathlib import Path
import numpy as np
import dm_data as dm
import preflight_outer_test as preflight
from preflight_outer_test import (preflight_manifest,validate_predictions,calculate_scores,
                                  EXPECTED_DATA_SHA,OUTER,sha256_file)


def verify_commit(folder,protocol_path,data_path):
    folder=Path(folder)
    pred_file=folder/'OUTER_PREDICTIONS.json'
    commit_file=folder/'PREDICTIONS_COMMITTED.json'
    if not pred_file.is_file() or not commit_file.is_file():
        raise ValueError('Uncommitted or incomplete predictions')
    commit=json.loads(commit_file.read_text(encoding='utf8'))
    pred=json.loads(pred_file.read_text(encoding='utf8'))
    prot=json.loads(Path(protocol_path).read_text(encoding='utf8'))
    prot_sha=sha256_file(protocol_path)
    if commit['status']!='COMMITTED FROZEN PREDICTIONS — DO NOT RE-TRAIN AFTER SCORE':
        raise ValueError('Wrong commit status')
    if (commit['protocol_sha256']!=prot_sha or pred['protocol_sha256']!=prot_sha
        or commit['prediction_sha256']!=sha256_file(pred_file)
        or commit['source_pickle_sha256']!=EXPECTED_DATA_SHA
        or pred['source_pickle_sha256']!=EXPECTED_DATA_SHA
        or pred['protocol_id']!=prot['protocol_id']):
        raise ValueError('Frozen prediction/manifest integrity failure')
    if (commit['prediction_program_sha256']!=sha256_file(Path(__file__).with_name('outer_predict_frozen.py'))
        or commit['preflight_program_sha256']!=sha256_file(preflight.__file__)):
        raise ValueError('Prediction or preflight helper modified after commitment')
    if pred['target_order']!=['outer_roundness_mm','height_runout_mm'] or pred['outer_ring_ids']!=list(OUTER):
        raise ValueError('Output target order/ring set changed')
    validate_predictions(pred['model_outputs'])
    return pred,commit


def read_external_truth_only_after_commit(data_path,commit_checked):
    if commit_checked is not True:raise PermissionError('Prediction commit not verified')
    if sha256_file(data_path)!=EXPECTED_DATA_SHA:raise ValueError('Source integrity mismatch')
    with Path(data_path).open('rb') as f:raw=pickle.load(f)
    if set(raw)!=set(OUTER)|{'DM01','DM02','DM04','DM05','DM06','DM09','DM10','DM12','DM14','DM16','DM17','DM18'}:
        raise ValueError('Unexpected source rings')
    truth={rid:[float(raw[rid]['zielwerte'][t]) for t in dm.TARGETS] for rid in OUTER}
    if any(not np.isfinite(v).all() for v in (np.array(truth[r],float) for r in OUTER)):
        raise ValueError('Nonfinite test labels')
    return truth


def score_once(folder,protocol,data,code_root,ablation,twohead,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Outer score already exists, never overwrite')
    preflight_manifest(protocol,data,code_root,ablation,twohead)
    pred,commit=verify_commit(folder,protocol,data)
    # Irreversible, exclusive release marker. Blocks a second run even if --out differs.
    # If the machine crashes after this marker, do NOT silently repeat evaluation.
    release_marker=Path(folder)/'OUTER_LABELS_RELEASED_ONCE.json'
    marker_data={'status':'OUTER LABEL RELEASE AUTHORIZED; NO RETEST',
                 'prediction_sha256':commit['prediction_sha256'],
                 'protocol_sha256':commit['protocol_sha256']}
    fd=os.open(str(release_marker),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as stream:
        json.dump(marker_data,stream,indent=2,ensure_ascii=False)
    # ALL hashes and predictions validated before explicit access to outer labels.
    truth=read_external_truth_only_after_commit(data,True)
    metrics=calculate_scores(pred['model_outputs'],truth)
    rows={}
    for rid in OUTER:
        rows[rid]={'truth_mm':truth[rid],
          'pred_mm':{m:pred['model_outputs'][m][rid] for m in pred['model_outputs']},
          'residual_mm':{m:[pred['model_outputs'][m][rid][j]-truth[rid][j] for j in (0,1)]
                         for m in pred['model_outputs']},
          'predictive_sd_mm':{m:pred['predictive_sd_mm'][m][rid] for m in pred['model_outputs']}}
    metrics['per_ring']=rows
    metrics['coverage_95_moment_normal_approx']={}
    for m in ('primary_ssl','strict_supervised'):
        cover=[]
        for rid in OUTER:
            error=np.array(rows[rid]['residual_mm'][m],float)
            sd=np.array(rows[rid]['predictive_sd_mm'][m],float)
            cover.append(np.abs(error)<=1.96*sd)
        metrics['coverage_95_moment_normal_approx'][m]=np.mean(np.array(cover),axis=0).tolist()
    metrics['uncertainty_warning']='Moment-based ±1.96*SD is only a descriptive approximation, N=4, NOT calibrated 95% coverage.' 
    metrics['source_predictions_sha256']=commit['prediction_sha256']
    metrics['protocol_sha256']=commit['protocol_sha256']
    metrics['status']='ONCE-ONLY FROZEN OUTER TEST; NO FURTHER MODEL TUNING'
    from outer_predict_frozen import atomic_json
    out.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(out,metrics)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',default='results/experiment_freeze/OUTER_FROZEN_PREDICTIONS_V1')
    p.add_argument('--protocol',default='results/experiment_freeze/DM_TWO_HEAD_PROTOCOL_FREEZE.json')
    p.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    p.add_argument('--code-root',default='.')
    p.add_argument('--ablation',default='results/two_head_ssr_ablation_cv4/TWO_HEAD_SSR_ABLATION_CV4.json')
    p.add_argument('--twohead',default='results/two_head_separate_axial_cv4/TWO_HEAD_SEPARATE_AXIAL_CV4.json')
    p.add_argument('--out',default='results/experiment_freeze/OUTER_FINAL_SCORE_V1.json')
    a=p.parse_args()
    r=score_once(a.predictions,a.protocol,a.data,a.code_root,a.ablation,a.twohead,a.out)
    print('AEUSSERER TEST EINMALIG AUSGEWERTET:',Path(a.out).resolve())
    for model,m in r['model_metrics'].items():
        print(model, 'RMSE mm',np.round(m['rmse_mm'],4),'MAE mm',np.round(m['mae_mm'],4))
    print('WICHTIG: Testlabels nun geoeffnet. Keine Modellanpassungen oder erneute Tests.')

if __name__=='__main__':main()
