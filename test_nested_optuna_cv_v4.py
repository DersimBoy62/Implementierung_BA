"""Synthetic, no ring pickle opened; scientific contracts for nested Optuna CV."""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import optuna
import nested_optuna_cv_v4 as m


def test_outer_ring_id_boundary():
    m.validate_boundaries()
    groups=m.outer_pairs('canonical')
    assert len(groups)==4 and all(len(x)==2 for x in groups)
    assert set().union(*map(set,groups))==set(m.EXPECTED_L)
    assert not set(m.EXPECTED_TEST)&set().union(*map(set,groups))
    try:m.outer_pairs('other')
    except ValueError:pass
    else:raise AssertionError('Arbitrary split accepted')


def test_inner_folds_exclude_outer_validation():
    for outer_val in m.outer_pairs('canonical'):
        train=tuple(x for x in m.EXPECTED_L if x not in outer_val)
        pp=m.inner_pairs(train)
        assert len(pp)==3 and all(len(p)==2 for p in pp)
        assert set().union(*map(set,pp))==set(train)
        for val in pp:
            inner_tr=tuple(r for r in train if r not in val)
            assert len(inner_tr)==4
            m.validate_groups(inner_tr,val,'strict_L_only')
            m.validate_groups(inner_tr,val,'shared_U_optuna')
            assert not set(outer_val)&set(inner_tr+val)


def test_reject_data_leakage_or_U_as_validation():
    l=m.EXPECTED_L
    for bad in ((l[0],m.EXPECTED_TEST[0]),(l[0],m.EXPECTED_U[0])):
        try:m.validate_groups(l[1:5],bad,'shared_U_optuna')
        except AssertionError:pass
        else:raise AssertionError('Illegal outer/U ring allowed into validation')
    try:m.validate_groups(l[:4],(l[3],l[4]),'strict_L_only')
    except AssertionError:pass
    else:raise AssertionError('Train-val overlap accepted')


def test_parameters_and_zero_meaning():
    cfg={'K':3,'C':0.,'physical_var':.4,'preprocessing':'P0','variant':'M2'}
    m.validate_config(cfg,'shared_U_optuna')
    m.validate_config(cfg,'strict_L_only')
    for change in [{'C':2.1},{'physical_var':.1},{'K':5},{'variant':'M3'}]:
        bad=dict(cfg,**change)
        try:m.validate_config(bad,'shared_U_optuna')
        except ValueError:pass
        else:raise AssertionError('Invalid hyperparameters accepted')
    try:m.validate_config(dict(cfg,C=.2),'strict_L_only')
    except ValueError:pass
    else:raise AssertionError('Strict C>0 accepted')


def test_multi_objective_selection_uses_inner_only():
    d=[SimpleNamespace(number=0,values=(.9,1.15),params={'K':2},user_attrs={'config':{'C':.2}}),
       SimpleNamespace(number=1,values=(1.0,.98),params={'K':3},user_attrs={'config':{'C':.3}}),
       SimpleNamespace(number=2,values=(.95,1.00),params={'K':2},user_attrs={'config':{'C':.1}})]
    selected=m.choose_pareto(SimpleNamespace(best_trials=d))
    assert selected.number==2


def test_synthetic_loss_and_inner_baseline():
    assert m.metrics([{'ring':'DM04','residual_mm':[.1,2.]},{'ring':'DM05','residual_mm':[-.1,-2.]}])['rmse_mm']==[.1,2.0]
    try:m.metrics([{'ring':'DM04','residual_mm':[0,1]},{'ring':'DM04','residual_mm':[0,1]}])
    except ValueError:pass
    else:raise AssertionError('Duplicate heldout ring accepted')
    rv={r:SimpleNamespace(y=np.array([float(i+1),float((i+1)*10)])) for i,r in enumerate(m.EXPECTED_L)}
    base=m.inner_baseline(rv,m.EXPECTED_L[:6])
    assert base.shape==(2,) and np.all(base>0)


def test_optuna_arms_differ_and_pv_tuned():
    assert optuna.__version__
    for arm in m.ARM_MODES:
        study=optuna.create_study(directions=['minimize','minimize'],sampler=optuna.samplers.RandomSampler(seed=4))
        trial=study.ask()
        cfg=m.suggest(trial,arm)
        assert cfg['K'] in (2,3) and .2<=cfg['physical_var']<=.8
        if arm=='strict_L_only':assert cfg['C']==0 and 'U_ELBO_branch' not in trial.params
        else:assert 'U_ELBO_branch' in trial.params
        study.tell(trial,[1.,1.])


def test_output_immutable_and_resume_gate():
    with TemporaryDirectory() as td:
        p=Path(td)/'sample.json'
        m.write_new(p,{'a':1})
        try:m.write_new(p,{'a':2})
        except FileExistsError:pass
        else:raise AssertionError('Overwrote existing result')
        assert json.loads(p.read_text())=={'a':1}
        try:m.safe_out('.', 'input.pkl')
        except ValueError:pass
        else:raise AssertionError('Project root accepted as result folder')


def test_frozen_multi_fold_protocol():
    args=SimpleNamespace(smoke=False,max_outer=4,max_inner=3,epochs=10,steps=8,trials=8)
    with patch.object(m.source.dm,'sha256',return_value=m.source.DATA_SHA):
        with patch.object(m.source,'_hash_dependencies',return_value={'dummy':'sha'}):
            mf=m.make_manifest(SimpleNamespace(**vars(args),data='dummy',pairing='canonical'))
    assert len(mf['outer_pairs'])==4 and mf['max_inner']==3
    assert mf['arms']==['strict_L_only','shared_U_optuna']
    assert mf['previous_outer_TEST_GESPERRT']==list(m.EXPECTED_TEST)
    bad=SimpleNamespace(**vars(args),data='dummy',pairing='canonical',max_inner_wrong=2)
    bad.max_inner=2
    try:m.make_manifest(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid full CV budget accepted')


def main():
    tests=[test_outer_ring_id_boundary,test_inner_folds_exclude_outer_validation,
           test_reject_data_leakage_or_U_as_validation,test_parameters_and_zero_meaning,
           test_multi_objective_selection_uses_inner_only,test_synthetic_loss_and_inner_baseline,
           test_optuna_arms_differ_and_pv_tuned,test_output_immutable_and_resume_gate,
           test_frozen_multi_fold_protocol]
    for fn in tests:
        fn();print('PASS',fn.__name__,flush=True)
    print(f'{len(tests)} SYNTHETIC TESTS PASSED. NO RING PICKLE LOADED.',flush=True)

if __name__=='__main__':main()
