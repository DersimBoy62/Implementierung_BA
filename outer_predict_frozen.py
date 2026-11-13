"""Frozen DM outer-test PREDICTIONS only. NEVER reads outer quality fields.

The original trusted source pickle physically contains outer quality fields; pickle.load
necessarily deserializes them, but THIS module does not access their values. It
immediately copies process-only time series and releases raw records. The only
read of 'zielwerte' here is on the exactly eight pre-registered dev labeled rings.
The scoring is deliberately in a SEPARATE program/process.

This file is an append-only experiment utility; none of the nine frozen model
sources or the frozen protocol is modified.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import two_head_b1 as th
from compare_two_head_cv4 import OPTIONS, DESCRIPTOR_NAMES
from quality_link_challenge import extract_descriptors
from preflight_outer_test import (preflight_manifest,validate_predictions,MODELS,L,U,OUTER,
     EXPECTED_DATA_SHA,sha256_file)

FEATURE_NAMES=tuple(dm.SETS['V1_Kern_6'])
HEAD_IDS=OPTIONS['SEPARATE_AXIAL']
SEED=20261111

@dataclass(frozen=True)
class XOnlyRing:
    ring_id: str
    t: np.ndarray
    x: np.ndarray


def _x_only(rid,record):
    """A process-only construction that must never reference target fields."""
    if record.get('ring_id') != rid or not record.get('train_eligible'):
        raise ValueError('Wrong ring identity/status: '+rid)
    logger=record['logger']
    a=np.asarray(logger['matrix'],dtype=float)
    cols=tuple(logger['matrix_columns'])
    idx={name:j for j,name in enumerate(cols)}
    if a.ndim!=2 or a.shape[1]!=115 or len(cols)!=115 or not np.isfinite(a).all():
        raise ValueError('Invalid logger matrix: '+rid)
    if cols!=( 'time', *logger['signal_names']):
        raise ValueError('Logger columns: '+rid)
    if not np.all(np.diff(a[:,0])>0):
        raise ValueError('Nonmonotonic time: '+rid)
    if not all(n in idx for n in (*FEATURE_NAMES,'phase804','LaserOK','MessEin')):
        raise ValueError('Missing process signals: '+rid)
    mask=np.isin(a[:,idx['phase804']],(1.,2.))
    mask&=(a[:,idx['LaserOK']]==1)&(a[:,idx['MessEin']]==1)
    used=np.flatnonzero(mask)
    if len(used)<6 or not np.all(np.diff(used)==1):
        raise ValueError('Invalid F2 process support: '+rid)
    x=np.array(a[np.ix_(used,[idx[n] for n in FEATURE_NAMES])],dtype=float,copy=True)
    t=np.array(a[used,0],dtype=float,copy=True)
    return XOnlyRing(rid,t,x)


def get_process_and_dev_labels(data_path):
    """Known SHA source; NEVER access target fields for U or OUTER."""
    if sha256_file(data_path)!=EXPECTED_DATA_SHA:
        raise ValueError('Source pickle SHA256 mismatch')
    with Path(data_path).open('rb') as f:
        raw=pickle.load(f)     # Trusted known SHA; the pickle contains ALL labels.
    if set(raw)!=set(L)|set(U)|set(OUTER) or len(raw)!=16:
        raise ValueError('Unexpected 16-ring split')
    process={rid:_x_only(rid,raw[rid]) for rid in (*L,*U,*OUTER)}
    labels={}
    for rid in L:
        labels[rid]=np.array([float(raw[rid]['zielwerte'][t]) for t in dm.TARGETS],float)
        if labels[rid].shape!=(2,) or not np.isfinite(labels[rid]).all():
            raise ValueError('Invalid DEVELOPMENT label: '+rid)
    # Outer and U labels never indexed, even by a convenience data-view function.
    del raw
    return process,labels


def make_heads(process,train_ids,pred_ids,sx):
    """Exact same transform as frozen inner two-head CV, train-only feature scaling."""
    all_ids=tuple(dict.fromkeys((*train_ids,*pred_ids)))
    desc={rid:extract_descriptors(sx.transform(process[rid].x),process[rid].t,'physical')
          for rid in all_ids}
    features={rid:[None,None] for rid in all_ids}
    for j,which in enumerate(HEAD_IDS):
        fs=StandardScaler().fit(np.stack([desc[r][list(which)] for r in train_ids]))
        for rid in all_ids:
            features[rid][j]=fs.transform(desc[rid][list(which)][None,:])[0]
    return features


def run_model(mode,process,dev_labels,epochs=10,steps=8):
    if mode not in ('primary_ssl','strict_supervised') or (epochs,steps)!=(10,8):
        raise ValueError('Frozen model or hyperparameters changed')
    train_ids=tuple(sorted((*L,*U))) if mode=='primary_ssl' else tuple(sorted(L))
    sx=dm.fit_x_scaler(process,train_ids)
    sy=dm.Scaler(np.mean(np.stack([dev_labels[r] for r in sorted(L)]),axis=0),
                 np.std(np.stack([dev_labels[r] for r in sorted(L)]),axis=0),tuple(sorted(L)))
    if (sy.scale<=1e-10).any():raise ValueError('Degenerate frozen Y scaler')
    x_train={rid:sx.transform(process[rid].x) for rid in train_ids}
    y_train={rid:sy.transform(dev_labels[rid]) for rid in sorted(L)}
    features=make_heads(process,train_ids,OUTER,sx)
    c=1. if mode=='primary_ssl' else 0.
    model=th.train(x_train,y_train,{rid:features[rid] for rid in train_ids},
                   K=3,C=c,epochs=epochs,steps=steps,seed=SEED,
                   state_var=2.,physical_var=.25)
    result={};sds={}
    for rid in OUTER:
        m,s=th.predict(sx.transform(process[rid].x),features[rid],model)
        predicted=(m*sy.scale+sy.mean).tolist()
        sd=(s*sy.scale).tolist()
        if len(predicted)!=2 or not np.isfinite(predicted).all() or not np.isfinite(sd).all() or min(sd)<=0:
            raise ArithmeticError('Nonfinite posterior prediction')
        result[rid]=[float(v) for v in predicted]
        sds[rid]=[float(v) for v in sd]
    return result,sds,{'mode':mode,'C':c,'x_scaler_fit_ids':list(sx.fitted_ids),
            'head_scaler_fit_ids':list(train_ids),'y_scaler_fit_ids':list(sy.fitted_ids),
            'quality_train_ids':list(y_train),'epochs':epochs,'local_steps':steps,
            'end_elbo':float(model['history'][-1]['elbo']),
            'state_mass':model['history'][-1]['state_mass']}


def build_predictions(process,dev_labels):
    if set(process)!=set(L)|set(U)|set(OUTER) or set(dev_labels)!=set(L):
        raise ValueError('Frozen split or label boundary changed')
    preds={};sds={};models={}
    for mode in ('primary_ssl','strict_supervised'):
        preds[mode],sds[mode],models[mode]=run_model(mode,process,dev_labels)
    mean=np.mean(np.stack([dev_labels[r] for r in L]),axis=0)
    preds['mean_baseline']={rid:[float(x) for x in mean] for rid in OUTER}
    sds['mean_baseline']={rid:None for rid in OUTER}
    models['mean_baseline']={'quality_train_ids':list(L),'fixed_training_mean_mm':mean.tolist(),
                             'no_test_labels_used':True}
    validate_predictions(preds)
    if any(set(models[m].get('quality_train_ids',[]))!=set(L) for m in MODELS):
        raise AssertionError('Quality fit leakage')
    return preds,sds,models


def atomic_json(path,obj):
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite committed output: '+str(path))
    temp=path.with_name(path.name+'.writing')
    if temp.exists():raise FileExistsError('Unfinished partial file: '+str(temp))
    temp.write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.rename(temp,path)


def persist_prediction(out,protocol,source_hash,pred,sd,model_info,environment):
    out=Path(out)
    if out.exists():raise FileExistsError('Frozen prediction output directory already exists: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    prediction={'status':'FROZEN OUTER PREDICTIONS, LABELS NOT CONSULTED',
        'protocol_id':protocol['protocol_id'],
        'protocol_sha256':environment['protocol_sha256'],
        'source_pickle_sha256':source_hash,
        'model_outputs':pred,'predictive_sd_mm':sd,'training_audit':model_info,
        'target_order': ['outer_roundness_mm','height_runout_mm'],
        'outer_ring_ids':list(OUTER),'test_target_values_in_this_file':False,
        'read_real_outer_labels':False}
    atomic_json(out/'OUTER_PREDICTIONS.json',prediction)
    pred_sha=sha256_file(out/'OUTER_PREDICTIONS.json')
    commit={'status':'COMMITTED FROZEN PREDICTIONS — DO NOT RE-TRAIN AFTER SCORE',
        'protocol_sha256':environment['protocol_sha256'],
        'prediction_sha256':pred_sha,'source_pickle_sha256':source_hash,
        'prediction_program_sha256':sha256_file(__file__),
        'preflight_program_sha256':environment['preflight_sha256'],
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'outer_test_labels_read':False,'predictions_committed_before_label_access':True}
    atomic_json(out/'PREDICTIONS_COMMITTED.json',commit)
    return commit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='results/experiment_freeze/DM_TWO_HEAD_PROTOCOL_FREEZE.json')
    parser.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    parser.add_argument('--code-root',default='.')
    parser.add_argument('--ablation',default='results/two_head_ssr_ablation_cv4/TWO_HEAD_SSR_ABLATION_CV4.json')
    parser.add_argument('--twohead',default='results/two_head_separate_axial_cv4/TWO_HEAD_SEPARATE_AXIAL_CV4.json')
    parser.add_argument('--out',default='results/experiment_freeze/OUTER_FROZEN_PREDICTIONS_V1')
    a=parser.parse_args()
    import preflight_outer_test as preflight
    manifest=preflight_manifest(a.protocol,a.data,a.code_root,a.ablation,a.twohead)
    proto=json.loads(Path(a.protocol).read_text(encoding='utf8'))
    if Path(a.out).exists():raise FileExistsError('Output already exists; no overwriting')
    process,dev_y=get_process_and_dev_labels(a.data)
    pred,sd,fitinfo=build_predictions(process,dev_y)
    committed=persist_prediction(a.out,proto,EXPECTED_DATA_SHA,pred,sd,fitinfo,
          {'protocol_sha256':sha256_file(a.protocol),'preflight_sha256':sha256_file(preflight.__file__)})
    print('VORHERSAGEN VERSIEGELT:',(Path(a.out)/'OUTER_PREDICTIONS.json').resolve())
    print('Vorhersage SHA256',committed['prediction_sha256'])
    print('Modelle',list(MODELS),'Ringe',list(OUTER))
    print('AUSSCHLIESSLICH TEST-PROZESSSIGNALE FUER INFERENZ VERWENDET.')
    print('KEINE TEST-QUALITAETSWERTE EXPLIZIT ABGERUFEN. KEINE METRIKEN.')
    print('Jetzt noch NICHT weitere Modelle trainieren oder Testlabels bewerten.')

if __name__=='__main__':main()
