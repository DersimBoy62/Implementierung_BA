"""7 safety/algorithm tests, no real data loading unless invoked separately."""
from __future__ import annotations
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import optuna
from dm_data import Ring
import optuna_ssr_development as dev
from summarize_optuna_development import pareto_indices


def raises(exc,callback):
    try: callback()
    except exc:return
    raise AssertionError(f'Expected {exc}')


def fake_dev_rings():
    rv={}
    for i,rid in enumerate(dev.EXPECTED_L+dev.EXPECTED_U):
        t=np.arange(16,dtype=float)*0.1
        x=np.column_stack([np.sin(t*2+i*.1+j*.23)+(i%3)*.1+j*.1 for j in range(6)])
        y=np.array([1.1+i*.16,10.0+i*.53]) if rid in dev.EXPECTED_L else np.full(2,np.nan)
        rv[rid]=Ring(rid,t,x,y,('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55'))
    return rv


def test_registered_domains_and_U_missing():
    rv=fake_dev_rings();dev.validate_development(rv)
    rv[dev.EXPECTED_U[0]].y=np.array([1.,2.])
    raises(AssertionError,lambda:dev.validate_development(rv))
    rv=fake_dev_rings();rv['DM07']=rv.pop(dev.EXPECTED_L[0])
    raises(AssertionError,lambda:dev.validate_development(rv))


def test_wrong_parameters_rejected():
    p={'K':3,'preprocessing':'P0','variant':'M0','C':1.,'physical_var':.25}
    dev.validate_parameters(p)
    for k,v in [('K',10),('preprocessing','P1'),('variant','M3'),('C',-1.),('physical_var',0.)]:
        q=dict(p);q[k]=v;raises(ValueError,lambda q=q:dev.validate_parameters(q))


def test_structural_and_extended_suggestion():
    study=optuna.create_study(directions=['minimize','minimize'],sampler=optuna.samplers.RandomSampler(seed=25))
    def obj(t):
        phase='structural' if t.number==0 else 'extended'
        p=dev.suggest_configuration(t,phase)
        if phase=='structural': assert p['C']==1. and p['physical_var']==.25
        else: assert .25<=p['C']<=2. and .05<=p['physical_var']<=1.
        return 1.,2.
    study.optimize(obj,n_trials=2)
    assert len(study.trials)==2


def test_synthetic_end_to_end_no_outer():
    rv=fake_dev_rings()
    p={'K':2,'preprocessing':'P0','variant':'M0','C':1.,'physical_var':.25}
    r=dev.evaluate_configuration(rv,p,folds=1,epochs=1,steps=1)
    assert [s['ring'] for s in r['per_ring']]==list(dev.EXPECTED_L[:2])
    assert np.isfinite(r['rmse_mm']).all() and len(r['fold_diagnostics'])==1
    assert set(r['fold_diagnostics'][0]['y_scaler_fit_ids'])==set(dev.EXPECTED_L[2:])


def test_synthetic_M2_p2_extended_prior():
    rv=fake_dev_rings()
    p={'K':2,'preprocessing':'P2','variant':'M2','C':.5,'physical_var':.1}
    r=dev.evaluate_configuration(rv,p,folds=1,epochs=1,steps=1)
    assert np.isfinite(r['rmse_mm']).all()
    assert np.isfinite(r['mae_mm']).all()


def test_pareto_selection():
    rows=[{'rmse_mm':[1.,3.]},{'rmse_mm':[2.,2.]},{'rmse_mm':[3.,4.]},{'rmse_mm':[1.,4.]}]
    assert pareto_indices(rows)==[0,1]


def test_no_unsafe_outer_training_source_and_immutable_write():
    source=inspect.getsource(dev.evaluate_configuration)
    assert 'dev.load_development' not in source # external access only in guarded run()
    assert 'trU=EXPECTED_U' in source and 'sy=dm.fit_y_scaler' in source
    with TemporaryDirectory() as d:
        p=Path(d)/'report.json';dev._write_new(p,{'a':1})
        raises(FileExistsError,lambda:dev._write_new(p,{'a':2}))


if __name__=='__main__':
    for fn in (test_registered_domains_and_U_missing,test_wrong_parameters_rejected,
               test_structural_and_extended_suggestion,test_synthetic_end_to_end_no_outer,
               test_synthetic_M2_p2_extended_prior,test_pareto_selection,
               test_no_unsafe_outer_training_source_and_immutable_write):
        fn();print('PASS',fn.__name__)
    print('7 SYNTHETISCHE TESTS BESTANDEN. KEINE AEUSSEREN TESTDATEN.')
