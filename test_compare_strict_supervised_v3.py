"""Synthetic-only contract tests: never open the ring pickle."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import compare_strict_supervised_v3 as m


def fake(e):
    return {'per_ring':[{'ring':rid,'residual_mm':list(v)} for rid,v in zip(sorted(m.opt.EXPECTED_L),e)],
            'rmse_mm':np.sqrt(np.mean(np.asarray(e)**2,axis=0)).tolist(),
            'mae_mm':np.mean(np.abs(np.asarray(e)),axis=0).tolist()}


def test_fixed_candidates_and_modes():
    assert m.TRIALS==(13,22)
    assert m.MODES==('strict_L_only','shared_C0','ssl_Cselected','ssl_C1')
    assert m.opt.LOCKED==('DM07','DM08','DM11','DM15')

def test_validated_strict_no_training_on_U():
    # Source inspection of the local-only function, backed by integration smoke test.
    import inspect
    src=inspect.getsource(m.strict_fold)
    assert 'trainL=tuple(' in src
    assert "labeled={rid:rv[rid] for rid in trainL}" in src
    assert "x={rid:sx.transform(rv[rid].x) for rid in trainL}" in src
    assert "sdata={rid:ss[rid] for rid in trainL}" in src
    assert 'C=0.' in src
    assert 'rv[rid].y' in src

def test_paired_contrasts():
    a=fake([[1.,2.]]*8);b=fake([[2.,3.]]*8)
    groups={13:{key:a if key=='ssl_Cselected' else b for key in m.MODES}}
    x=m.summarize_comparisons(groups)[13]
    assert np.allclose(x['contrasts']['ssl_Cselected_minus_strict']['paired_delta_mse_mm2'],[-3,-5])

def test_reject_outer_per_ring():
    a=fake([[1.,2.]]*8)
    a['per_ring'][0]['ring']='DM07'
    try:m.ab.error_array(a)
    except ValueError:pass
    else:raise AssertionError('Accepted outer ring in reported errors')

def test_immutable_new_json():
    with TemporaryDirectory() as t:
        p=Path(t)/'x.json';m.ab.create_new_json(p,{'s':1})
        try:m.ab.create_new_json(p,{'s':2})
        except FileExistsError:pass
        else:raise AssertionError('Overwrote immutable report')

def test_refuse_occupied_output_before_data_io():
    with TemporaryDirectory() as t:
        x=Path(t)/'already';x.mkdir();(x/'a.txt').write_text('x')
        try:m.run('unknown_study','unknown_ablation',x,data='unknown.pkl')
        except FileExistsError:pass
        else:raise AssertionError('Did not stop before touching source files')

def test_invalid_K_rejected():
    conf={'K':9,'preprocessing':'P0','variant':'M2','C':0.3,'physical_var':.4}
    try:m.opt.validate_parameters(conf)
    except ValueError:pass
    else:raise AssertionError('Invalid K accepted')

if __name__=='__main__':
    tests=[test_fixed_candidates_and_modes,test_validated_strict_no_training_on_U,
           test_paired_contrasts,test_reject_outer_per_ring,test_immutable_new_json,
           test_refuse_occupied_output_before_data_io,test_invalid_K_rejected]
    for f in tests:f();print('PASS',f.__name__)
    print('7 SYNTHETIC CONTRACT TESTS PASSED. NO RING DATA FILE OPENED.')
