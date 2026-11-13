"""Synthetic-only tests for FINAL frozen outer prediction/scoring programs. NO real labels."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch
import numpy as np
import outer_predict_frozen as op
import outer_score_frozen as sc
from preflight_outer_test import L,U,OUTER,EXPECTED_DATA_SHA,validate_predictions
from dm_data import TARGETS

class ForbiddenLabels(dict):
    def __getitem__(self,key):
        if key=='zielwerte':raise AssertionError('Forbid U/OUTER quality label access')
        return super().__getitem__(key)
    def get(self,key,default=None):
        if key=='zielwerte':raise AssertionError('Forbid U/OUTER quality label access via .get')
        return super().get(key,default)


def fake_logger(i):
    # simulate all 114 channel names with exact expected selected channels
    names=list(op.FEATURE_NAMES)+['phase804','LaserOK','MessEin']
    names+=['dummy_'+str(j) for j in range(114-len(names))]
    columns=('time',*names)
    t=np.arange(30)*0.1
    a=np.zeros((len(t),len(columns)))
    a[:,0]=t
    for j,n in enumerate(op.FEATURE_NAMES):
        a[:,j+1]=0.7*j+np.sin(t*(j+1)+i*.2)+.4*t
    index={v:j for j,v in enumerate(columns)}
    a[:,index['phase804']]=2
    a[:,index['LaserOK']]=1
    a[:,index['MessEin']]=1
    return {'matrix':a,'matrix_columns':columns,'signal_names':tuple(names)}


def fake_records():
    raw={}
    for i,rid in enumerate((*L,*U,*OUTER)):
        d={'ring_id':rid,'train_eligible':True,'logger':fake_logger(i)}
        if rid in L:
            d['zielwerte']={TARGETS[0]:0.2+i*.1,TARGETS[1]:2.+i*.3}
        else:
            d=ForbiddenLabels(d)
        raw[rid]=d
    return raw


def test_no_test_or_unlabeled_label_lookup():
    with tempfile.TemporaryDirectory() as t:
        tmp=Path(t)/'fake.pkl';tmp.write_bytes(b'fake synthetic bytes')
        with patch.object(op,'sha256_file',return_value=EXPECTED_DATA_SHA),patch.object(op.pickle,'load',return_value=fake_records()):
            rings,labels=op.get_process_and_dev_labels(tmp)
    assert len(rings)==16 and set(labels)==set(L)
    assert all(r.x.shape==(30,6) for r in rings.values())
    assert all(np.isfinite(r.t).all() for r in rings.values())
    print('PASS no_U_or_outer_label_field_access')
    return rings,labels


def test_frozen_train_modes_use_correct_labels(rings,labels):
    calls=[]
    def stub_train(X,Y,sdata,**kw):
        calls.append({'X':set(X),'Y':set(Y),'sdata':set(sdata),'kwargs':kw})
        return {'history':[{'elbo':-9.0,'state_mass':[.3,.4,.3]}]}
    def stub_predict(x,ss,model):
        assert x.shape[1]==6 and len(ss)==2 and len(ss[0])==6 and len(ss[1])==5
        return np.array([.1,.2]),np.array([.7,1.2])
    with patch.object(op.th,'train',side_effect=stub_train),patch.object(op.th,'predict',side_effect=stub_predict):
        pred,sd,train=op.build_predictions(rings,labels)
    validate_predictions(pred)
    assert len(calls)==2
    assert calls[0]['X']==set(L)|set(U) and calls[1]['X']==set(L)
    assert calls[0]['Y']==calls[1]['Y']==set(L)
    assert calls[0]['sdata']==set(L)|set(U) and calls[1]['sdata']==set(L)
    assert calls[0]['kwargs']['C']==1. and calls[1]['kwargs']['C']==0.
    assert all(train[m]['quality_train_ids']==list(sorted(L)) for m in ('primary_ssl','strict_supervised'))
    print('PASS exact_frozen_supervised_vs_SSL_train_boundary')
    return pred,sd,train


def test_commit_hash_and_tamper(pred,sd,train):
    with tempfile.TemporaryDirectory() as t:
        home=Path(t)
        proto=home/'proto.json';proto.write_text(json.dumps({'protocol_id':'synthetic-only'}))
        env={'protocol_sha256':op.sha256_file(proto),'preflight_sha256':op.sha256_file(__import__('preflight_outer_test').__file__)}
        # invoke real signature with synthetic protocol
        committed=op.persist_prediction(home/'pred',{'protocol_id':'synthetic-only'},EXPECTED_DATA_SHA,pred,sd,train,env)
        payload,seal=sc.verify_commit(home/'pred',proto,home/'fake_not_needed')
        assert payload['model_outputs']==pred and seal['prediction_sha256']==committed['prediction_sha256']
        try:op.persist_prediction(home/'pred',{'protocol_id':'synthetic-only'},EXPECTED_DATA_SHA,pred,sd,train,env)
        except FileExistsError:pass
        else:raise AssertionError('Overwriting committed predictions allowed')
        p=home/'pred'/'OUTER_PREDICTIONS.json'
        p.write_text(p.read_text(encoding='utf8')+'\n',encoding='utf8')
        try:sc.verify_commit(home/'pred',proto,home/'fake_not_needed')
        except ValueError:pass
        else:raise AssertionError('Tampered predictions accepted')
    print('PASS predictions_committed_sha_and_tamper_rejected')


def test_label_release_after_commit_only():
    # no actual source file, just exercise fail-closed access gate
    try:sc.read_external_truth_only_after_commit('/does-not-exist',False)
    except PermissionError:pass
    else:raise AssertionError('Released truth without commit')
    print('PASS truth_access_rejected_before_commit')


def test_synthetic_scoring():
    pred={m:{rid:[float(i)+.2,float(i)-.1] for i,rid in enumerate(OUTER)} for m in
          ('primary_ssl','strict_supervised','mean_baseline')}
    label={rid:[float(i),float(i)] for i,rid in enumerate(OUTER)}
    scores=sc.calculate_scores(pred,label)
    assert np.allclose(scores['model_metrics']['primary_ssl']['rmse_mm'],[.2,.1])
    print('PASS separate_frozen_scoring_equal_ring_weights')


def test_scoring_once_with_synthetic_truth(pred,sd,train):
    with tempfile.TemporaryDirectory() as root:
        home=Path(root);proto=home/'proto.json'
        proto.write_text(json.dumps({'protocol_id':'synthetic-only'}))
        env={'protocol_sha256':op.sha256_file(proto),
             'preflight_sha256':op.sha256_file(__import__('preflight_outer_test').__file__)}
        op.persist_prediction(home/'pred',{'protocol_id':'synthetic-only'},EXPECTED_DATA_SHA,pred,sd,train,env)
        fake_truth={rid:[float(i),float(i+.5)] for i,rid in enumerate(OUTER)}
        with patch.object(sc,'preflight_manifest',return_value={}),patch.object(sc,'read_external_truth_only_after_commit',return_value=fake_truth) as rdr:
            score=sc.score_once(home/'pred',proto,home/'unused.pkl',home,home/'unused_dev.json',home/'unused_ref.json',home/'score.json')
            assert rdr.call_count==1 and score['ring_count']==4
            assert (home/'pred'/'OUTER_LABELS_RELEASED_ONCE.json').exists()
            try:
                sc.score_once(home/'pred',proto,home/'unused.pkl',home,home/'unused_dev.json',home/'unused_ref.json',home/'different_score.json')
            except FileExistsError:pass
            else:raise AssertionError('Second score release was permitted')
    print('PASS synthetic_score_once_and_release_marker_block')


def main():
    rings,labels=test_no_test_or_unlabeled_label_lookup()
    predictions,sd,train=test_frozen_train_modes_use_correct_labels(rings,labels)
    test_commit_hash_and_tamper(predictions,sd,train)
    test_label_release_after_commit_only()
    test_synthetic_scoring()
    test_scoring_once_with_synthetic_truth(predictions,sd,train)
    print('ALLE 6 SYNTHETISCHEN TESTS BESTANDEN; AUSSERER TEST NICHT GELESEN.')

if __name__=='__main__':main()
