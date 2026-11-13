"""Standard-library adversarial tests for the synthetic-only external test preflight."""
from __future__ import annotations
import copy
import math
import tempfile
from pathlib import Path
import preflight_outer_test as g


def test_sealed_label_access():
    val=g.SyntheticSealedLabels({r:[0.,0.] for r in g.OUTER})
    try:val.get(g.OUTER[0])
    except PermissionError:pass
    else:raise AssertionError('Label access before prediction commitment should be blocked')
    p={n:{r:[0.,0.] for r in g.OUTER} for n in g.MODELS}
    p['primary_ssl'].pop(g.OUTER[0])
    try:val.commit_predictions(p)
    except ValueError:pass
    else:raise AssertionError('Incomplete predictions accepted')
    try:val.get(g.OUTER[0])
    except PermissionError:pass
    else:raise AssertionError('Faulty commitment released test labels')
    p['primary_ssl'][g.OUTER[0]]=[0.,0.]
    val.commit_predictions(p)
    assert val.get(g.OUTER[0])==[0.,0.]


def test_metric_arithmetic_and_equal_ring_weight():
    truth={r:[0.,0.] for r in g.OUTER}
    p={name:{r:[0.,0.] for r in g.OUTER} for name in g.MODELS}
    p['primary_ssl'][g.OUTER[0]]=[2.,4.]
    p['strict_supervised'][g.OUTER[0]]=[4.,6.]
    score=g.calculate_scores(p,truth)
    assert score['ring_count']==4
    assert score['model_metrics']['primary_ssl']['rmse_mm']==[1.,2.]
    assert score['model_metrics']['primary_ssl']['mae_mm']==[.5,1.]
    assert score['paired_delta_mse_primary_minus_control_mm2']['strict_supervised']==[-3.,-5.]


def test_wrong_model_ring_target_and_nan_rejected():
    p={name:{r:[1.,2.] for r in g.OUTER} for name in g.MODELS}
    alterations=[]
    a=copy.deepcopy(p); a.pop('mean_baseline');alterations.append(a)
    a=copy.deepcopy(p); a['primary_ssl']['DM12']=[1.,2.];alterations.append(a)
    a=copy.deepcopy(p); a['primary_ssl'][g.OUTER[0]]=[1.];alterations.append(a)
    a=copy.deepcopy(p); a['primary_ssl'][g.OUTER[0]]=[math.nan,2.];alterations.append(a)
    a=copy.deepcopy(p); a['primary_ssl'][g.OUTER[0]]=[True,2.];alterations.append(a)
    for a in alterations:
        try:g.validate_predictions(a)
        except ValueError:pass
        else:raise AssertionError('Invalid predictions accepted')


def test_manifest_change_and_source_hash_rejected():
    root=Path(__file__).resolve().parent
    freeze=Path('/mnt/data/DM_Protokoll_Freeze_2026_11_11/REFERENCE_FREEZE.json')
    if not freeze.is_file():
        # This test remains portable in the user folder using the user's freeze file.
        freeze=root/'results/experiment_freeze/DM_TWO_HEAD_PROTOCOL_FREEZE.json'
    if not freeze.is_file():
        raise FileNotFoundError('Missing frozen experiment manifest')
    x=g.read_json(freeze)
    assert x['source_data_sha256']==g.EXPECTED_DATA_SHA
    x['training']['K']=4
    with tempfile.TemporaryDirectory() as temp:
        t=Path(temp)/'bad.json'
        import json
        t.write_text(json.dumps(x),encoding='utf-8')
        with tempfile.TemporaryDirectory() as other:
            bad_pickle=Path(other)/'DM_16_aktive_Ringe.pkl'
            bad_pickle.write_bytes(b'fake-not-our-pickle')
            try:g.preflight_manifest(t,bad_pickle,root,'unknown','unknown')
            except (ValueError,FileNotFoundError):pass
            else:raise AssertionError('Changed pickle accepted')
    assert g.synthetic_dry_run()['real_outer_test_labels_accessed'] is False

if __name__=='__main__':
    tests=(('sealed_labels_reject_premature_access',test_sealed_label_access),
           ('exact_metrics_and_equal_ring_weight',test_metric_arithmetic_and_equal_ring_weight),
           ('missing_extra_nan_targets_rejected',test_wrong_model_ring_target_and_nan_rejected),
           ('source_integrity_and_synthetic_dry_run',test_manifest_change_and_source_hash_rejected))
    for title,test in tests:
        test()
        print('PASS',title)
