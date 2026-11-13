"""Synthetic contract tests, no real label access and no outer evaluation."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from compare_states_cv4 import VALID_STATES,VALID_MODES,summarize_states,fit_process_scaler,EXPECTED_L,EXPECTED_U,LOCKED
from summarize_state_grid import pareto_members, read_report

def raises(exc,fn):
    try:fn()
    except exc:return
    raise AssertionError('Expected rejection')

def test_registered_study_only():
    assert VALID_STATES==(2,3,5,8)
    assert VALID_MODES==('P0','P2')
    assert len(EXPECTED_L)==8 and len(EXPECTED_U)==4 and len(LOCKED)==4
    assert not set(EXPECTED_L)&set(LOCKED) and not set(EXPECTED_U)&set(LOCKED)

def test_equal_prior_p0_and_robust_p2_scaler():
    class Data:
        x=np.array([[1.],[2.],[3.],[4.],[100.]],float)
        @property
        def y(self):raise AssertionError('NO QUALITY ACCESS')
    rr={'DM01':Data()}
    p0=fit_process_scaler(rr,('DM01',),'P0')
    p2=fit_process_scaler(rr,('DM01',),'P2')
    assert np.isclose(p0.mean[0],22.)
    assert np.isclose(p2.mean[0],3.)
    rr['DM07']=Data()
    raises(AssertionError,lambda:fit_process_scaler(rr,('DM07',),'P0'))

def test_state_mass_entropy_formula():
    for K in VALID_STATES:
        p=np.zeros(K);p[0]=.5;p[1]=.5
        A=np.eye(K)*3+np.ones((K,K))
        mdl={'history':[{'state_mass':p.tolist()}],'process':SimpleNamespace(A=A)}
        d=summarize_states(mdl,K)
        assert np.isclose(d['effective_states_entropy_exp'],2.)
        assert np.isclose(d['effective_states_inverse_simpson'],2.)
        assert d['states_above_5pct']==2
        assert len(d['posterior_mean_self_transition_per_state'])==K

def test_invalid_state_mass_rejected():
    mdl={'history':[{'state_mass':[0.3,0.7]}],'process':SimpleNamespace(A=np.ones((2,2)))}
    raises(ValueError,lambda:summarize_states(mdl,3))
    mdl['history'][0]['state_mass']=[np.nan,.2]
    raises(ValueError,lambda:summarize_states(mdl,2))

def test_pareto_2_targets():
    rows=[{'rmse_mm':[1.,3.],'configuration':['P0','M0',2]},
          {'rmse_mm':[2.,2.],'configuration':['P0','M0',3]},
          {'rmse_mm':[2.5,3.5],'configuration':['P0','M0',5]}]
    assert pareto_members(rows)==[('P0','M0',2),('P0','M0',3)]

def test_report_rejects_bad_fold_and_bad_metrics():
    from compare_states_cv4 import DATA_SHA
    train0=tuple(r for r in EXPECTED_L if r not in EXPECTED_L[:2])
    folds=[];per=[]
    for i in range(4):
        val=EXPECTED_L[2*i:2*i+2]
        train_l=tuple(r for r in EXPECTED_L if r not in val)
        folds.append({'train_L':train_l,'train_U_process_only':EXPECTED_U,'validation':val,
                      'x_scaler':{'fit_ids':tuple(sorted((*train_l,*EXPECTED_U)))},
                      'target_scaler_fit_ids':tuple(sorted(train_l)),
                      'state_diagnostics':{'state_mass_timepoint_weighted':[.5,.5],
                        'effective_states_entropy_exp':2.,'state_min_mass':.5,
                        'states_above_5pct':2,'posterior_mean_self_transition_mean':.7}})
        for rid in val:per.append({'ring':rid,'fold':i+1,'truth_mm':[1.,2.],
                 'pred_mm':[2.,4.],'residual_mm':[1.,2.],'predictive_sd_mm':[.1,.2]})
    r={'configuration':{'preprocessing':'P0','variant':'M0','K':2,'C':1.,'folds':4,'epochs':10,'steps':8},
       'source_data_sha256':DATA_SHA,'development_labeled':EXPECTED_L,
       'development_unlabeled_process_only':EXPECTED_U,'outer_ids_excluded':LOCKED,
       'per_ring':per,'folds':folds,'metrics':{'rmse_mm':[1.,2.],'mae_mm':[1.,2.]}}
    with TemporaryDirectory() as root:
        p=Path(root)/'STATES_P0_M0_K2_C1_FOLDS4.json'
        p.write_text(json.dumps(r))
        name, summary=read_report(p)
        assert name==('P0','M0',2) and np.allclose(summary['rmse_mm'],[1.,2.])
        bad=deepcopy(r);bad['metrics']['rmse_mm']=[2.,2.]
        p.write_text(json.dumps(bad));raises(ValueError,lambda:read_report(p))
        bad=deepcopy(r);bad['folds'][0]['x_scaler']['fit_ids']=list(LOCKED)
        p.write_text(json.dumps(bad));raises(AssertionError,lambda:read_report(p))


def test_no_automatic_outer_or_pickle_scoring():
    import inspect,compare_states_cv4,summarize_state_grid
    txt=inspect.getsource(summarize_state_grid)
    assert 'pickle.load' not in txt and 'dm.view' not in txt
    txt=inspect.getsource(compare_states_cv4)
    assert 'load_development' in txt and 'EXPECTED_U' in txt and 'sy.transform' in txt

if __name__=='__main__':
    for f in (test_registered_study_only,test_equal_prior_p0_and_robust_p2_scaler,
              test_state_mass_entropy_formula,test_invalid_state_mass_rejected,
              test_pareto_2_targets,test_report_rejects_bad_fold_and_bad_metrics,
              test_no_automatic_outer_or_pickle_scoring):
        f();print('PASS',f.__name__)
    print('7 TESTS BESTANDEN; KEIN ECHTES TRAINING; KEINE TESTLABELS.')
