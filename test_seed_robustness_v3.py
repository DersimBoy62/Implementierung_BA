"""Synthetic contract tests for seed-robustness runner. Never opens real data."""
import tempfile
from pathlib import Path
import numpy as np
import compare_seed_robustness_v3 as m


def fake(x):
    e=np.asarray(x,float)
    ids=sorted(m.opt.EXPECTED_L)
    assert e.shape==(8,2)
    return {'per_ring':[{'ring':rid,'residual_mm':v.tolist()} for rid,v in zip(ids,e)],
            'rmse_mm':np.sqrt(np.mean(e**2,axis=0)).tolist(),
            'mae_mm':np.mean(np.abs(e),axis=0).tolist()}


def test_fixed_protocol():
    assert m.TRIALS==(13,22)
    assert m.SEEDS==(20261111,20261112,20261113)
    assert m.MODES==('strict_L_only','ssl_Cselected')
    assert m.opt.LOCKED==('DM07','DM08','DM11','DM15')


def test_paired_math():
    a=fake([[1,2]]*8);b=fake([[2,3]]*8)
    v=m.summarize_pair(a,b)
    assert np.allclose(v['delta_mse_ssl_minus_strict_mm2'],[-3,-5])
    assert v['ring_count_better_ssl_abs_error']==[8,8]
    assert v['loo_sign_stable']==[True,True]


def test_original_seed_parity():
    a=fake([[1,2]]*8);b=fake([[1+1e-6,2+1e-6]]*8)
    assert m.baseline_agrees(a,b)<2e-5
    try:m.baseline_agrees(fake([[1.1,2]]*8),a)
    except AssertionError:pass
    else:raise AssertionError('Wrong historical baseline accepted')


def test_safety_out_before_inputs():
    with tempfile.TemporaryDirectory() as td:
        occupied=Path(td)/'used';occupied.mkdir();(occupied/'x').write_text('x')
        try:m.run('no-study','no-ablation','no-strict',occupied,data='no-data')
        except FileExistsError:pass
        else:raise AssertionError('Occupied output not refused before input access')


def test_old_study_cannot_be_output():
    with tempfile.TemporaryDirectory() as td:
        try:m.refuse_unsafe(Path(td)/'study'/'inside',Path(td)/'study',Path(td)/'a',Path(td)/'s','data.pkl')
        except ValueError:pass
        else:raise AssertionError('Study folder protected against accidental output')


def test_other_rings_rejected():
    a=fake([[1,2]]*8)
    a['per_ring'][0]['ring']='DM07'
    try:m.summarize_pair(a,fake([[1,2]]*8))
    except ValueError:pass
    else:raise AssertionError('Outer ring slipped into per-ring summary')


def test_immutable_json_and_aggregation():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'json';m.write_new(p,{'hello':1})
        try:m.write_new(p,{'hello':2})
        except FileExistsError:pass
        else:raise AssertionError('Overwrote previous JSON')
    report={t:{s:{mode:fake([[1+(mode=='strict_L_only'),2]]*8) for mode in m.MODES} for s in m.SEEDS} for t in m.TRIALS}
    for t in m.TRIALS:
        g=m.aggregate(report)[str(t)]
        assert g['paired']['number_seeds_ssl_better_per_target']==[3,0]


if __name__=='__main__':
    tests=[test_fixed_protocol,test_paired_math,test_original_seed_parity,
           test_safety_out_before_inputs,test_old_study_cannot_be_output,
           test_other_rings_rejected,test_immutable_json_and_aggregation]
    for fn in tests:
        fn();print('PASS',fn.__name__,flush=True)
    print('7 SYNTHETIC TESTS PASSED; NO PICKLE LOADED.',flush=True)
