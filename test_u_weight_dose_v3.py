"""Synthetic-only mathematical/contract tests; never load any ring pickle."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import numpy as np
import scan_u_weight_dose_v3 as d
import summarize_u_weight_dose_v3 as summ
import optuna_ssr_development as opt
import compare_split_robustness_v3 as split


def fake(v):
    r=[]
    for i,rid in enumerate(opt.EXPECTED_L):
        r.append({'ring':rid,'fold':i//2+1,'residual_mm':[v+0.03*i,2*v+0.05*i]})
    a=np.array([q['residual_mm'] for q in r]);return {'per_ring':r,'rmse_mm':np.sqrt((a*a).mean(0)).tolist(),
                                                     'mae_mm':np.abs(a).mean(0).tolist()}


def test_fixed_registered_C_levels():
    a=d.requested_levels(.25053356993048204)
    assert len(a)==len(d.FIXED_C)+1 and 0 in a and 2. in a and .25053356993048204 in a
    assert a==sorted(a) and len({d.token(v) for v in a})==len(a)


def test_bad_dose_rejected():
    for x in (float('nan'),-1.,2.1):
        try:d.token(x)
        except ValueError:pass
        else:raise AssertionError(f'Invalid C accepted: {x}')
    try:d.requested_levels(.35)
    except ValueError:pass
    else:raise AssertionError('Duplicated selected C accepted')


def test_paired_profile_metrics_exact():
    r={0.:fake(.5),.25:fake(.2),1.:fake(.8)}
    p=d.profile_metrics(r,.25)
    assert p['minimum_descriptive_rmse_C_by_target']==[.25,.25]
    assert p['dose_curve'][0]['delta_mse_vs_shared_C0_mm2']==[0.,0.]
    assert np.all(np.array(p['dose_curve'][1]['delta_mse_vs_shared_C0_mm2'])<0)
    assert p['dose_curve'][1]['better_rings_abs_vs_shared_C0']==[8,8]


def test_reject_outer_or_duplicate_ring():
    bad=fake(.5);bad['per_ring'][0]['ring']='DM07'
    try:d.profile_metrics({0.:bad,.25:fake(.2)},.25)
    except ValueError:pass
    else:raise AssertionError('Outer ring not refused')


def test_temporary_pair_order_restoration():
    old=tuple(opt.EXPECTED_L)
    new=split.new_orders()[20261114]
    with split.temporary_fold_order(new):assert tuple(opt.EXPECTED_L)==new
    assert tuple(opt.EXPECTED_L)==old


def test_immutable_json_writer():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'one.json';d.write_new(p,{'one':1})
        try:d.write_new(p,{'one':2})
        except FileExistsError:pass
        else:raise AssertionError('Overwrite accepted')
        assert json.loads(p.read_text())=={'one':1}


def test_disjoint_source_output_path():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'old';root.mkdir()
        d.validate_output_path(Path(tmp)/'new',(root,))
        try:d.validate_output_path(root/'illegal',(root,))
        except ValueError:pass
        else:raise AssertionError('Writing into old study permitted')


def test_three_split_aggregator():
    p={'data_sha256':'DATA','core_code_sha256':{'a':'b'},'configurations':{'13':{'C':.251},'22':{'C':.368}}}
    c1=[{'C':0.,'rmse_mm':[1.,3.],'delta_mse_vs_shared_C0_mm2':[0.,0.]},
        {'C':.25,'rmse_mm':[.8,3.2],'delta_mse_vs_shared_C0_mm2':[-.36,1.24]}]
    c2=[{'C':0.,'rmse_mm':[.9,3.],'delta_mse_vs_shared_C0_mm2':[0.,0.]},
        {'C':.25,'rmse_mm':[.7,2.8],'delta_mse_vs_shared_C0_mm2':[-.32,-1.16]}]
    def pack(curve):return {'results':{'13':{'dose_curve':curve},'22':{'dose_curve':curve}}}
    out=summ.aggregate({'canonical':(p,pack(c1)),'20261114':(p,pack(c2))})
    a=out['candidate_C_profiles']['13'][1]
    assert a['splits_better_than_C0_by_target']==[2,1]
    assert np.allclose(a['mean_of_split_RMSE_mm'],[.75,3.])


if __name__=='__main__':
    tests=[test_fixed_registered_C_levels,test_bad_dose_rejected,
           test_paired_profile_metrics_exact,test_reject_outer_or_duplicate_ring,
           test_temporary_pair_order_restoration,test_immutable_json_writer,
           test_disjoint_source_output_path,test_three_split_aggregator]
    for t in tests:t();print('PASS',t.__name__)
    print('8 SYNTHETIC TESTS PASSED. NO RING PICKLE LOADED. NO OUTER TEST.')
