"""Synthetic-only tests for fixed Optuna C/prior factorial, without ring pickle."""
import json
import numpy as np
import ablate_optuna_v3 as m
from tempfile import TemporaryDirectory
from pathlib import Path


def fake(resids):
    return {'per_ring':[{'ring':r,'residual_mm':e} for r,e in zip(sorted(m.optdev.EXPECTED_L),resids)]}


def test_registered_selection():
    ref={'trials':[{'trial':13,'config':{'K':3,'preprocessing':'P0','variant':'M2','C':.251,'physical_var':.473},'rmse_mm':[.53,3.1]},
                   {'trial':22,'config':{'K':2,'preprocessing':'P0','variant':'M2','C':.368,'physical_var':.385},'rmse_mm':[.54,2.93]}]}
    assert m.selected_trial(ref,13)['config']['K']==3
    assert m.selected_trial(ref,22)['config']['K']==2
    try:m.selected_trial(ref,4)
    except ValueError:pass
    else:raise AssertionError('Unregistered trial accepted')


def test_factorial_grid_and_C0_note():
    config={'K':2,'preprocessing':'P0','variant':'M2','C':.368,'physical_var':.385}
    rows=m.factorial_configs(config)
    assert len(rows)==6 and len(set((r['C'],r['physical_var']) for r in rows))==6
    assert set(r['C'] for r in rows)=={0.,.368,1.}
    assert set(r['physical_var'] for r in rows)=={.25,.385}
    assert all(r['K']==2 and r['preprocessing']=='P0' and r['variant']=='M2' for r in rows)


def test_paired_metrics_and_loocv():
    a=fake([[1.,2.]]*8);b=fake([[2.,3.]]*8)
    res=m.paired(a,b)
    assert np.allclose(res['paired_delta_mse_mm2'],[-3,-5])
    assert res['n_rings_better_abs_error']==[8,8]
    assert res['loo_sign_stable']==[True,True]


def test_bad_ring_rejected():
    a=fake([[1.,1.]]*8);b=fake([[1.,1.]]*8);b['per_ring'][0]['ring']='DM07'
    try:m.paired(a,b)
    except ValueError:pass
    else:raise AssertionError('External ring accepted')


def test_immutable_writer():
    with TemporaryDirectory() as tmp:
        p=Path(tmp)/'x.json';m.create_new_json(p,{'c':0})
        try:m.create_new_json(p,{'c':1})
        except FileExistsError:pass
        else:raise AssertionError('Output overwritten')


def test_rejects_parameter_changes_before_io():
    try:m.run('not_existing','not_existing_out',trials=(13,),folds=4,epochs=10,steps=8)
    except ValueError:pass
    else:raise AssertionError('Trial cherry-picking accepted')
    try:m.run('not_existing','not_existing_out',folds=1,epochs=10,steps=8)
    except ValueError:pass
    else:raise AssertionError('Changed fold budget accepted')


def test_optuna_parameter_validation():
    x={'K':2,'preprocessing':'P0','variant':'M2','C':0.,'physical_var':.25}
    m.optdev.validate_parameters(x)
    x['K']=11
    try:m.optdev.validate_parameters(x)
    except ValueError:pass
    else:raise AssertionError('Invalid state count accepted')

if __name__=='__main__':
    tests=[test_registered_selection,test_factorial_grid_and_C0_note,
           test_paired_metrics_and_loocv,test_bad_ring_rejected,
           test_immutable_writer,test_rejects_parameter_changes_before_io,
           test_optuna_parameter_validation]
    for f in tests:
        f();print('PASS',f.__name__)
    print('7 SYNTHETIC TESTS PASSED. NO DATA PICKLE OPENED.')
