"""Pure synthetic, no data pickle or outer labels: split sensitivity contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import compare_split_robustness_v3 as s


def fake_report(order,mode,amount=1.0):
    rows=[];diags=[]
    for f in range(4):
        val=order[2*f:2*f+2]
        tr=set(s.ORIGINAL_ORDER)-set(val)
        fit=tr if mode=='strict_L_only' else tr|set(s.opt.EXPECTED_U)
        for rid in val:
            rows.append({'ring':rid,'fold':f+1,'prediction_mm':[1,1],
                         'residual_mm':[amount,2*amount],'predictive_sd_mm':[1,1]})
        d={'validation':list(val),'x_scaler_fit_ids':list(fit),
           'y_scaler_fit_ids':list(tr),'physical_scaler_fit_ids':list(fit)}
        if mode=='strict_L_only':d.update(hmm_initialisation_uses_only=list(tr),u_used=False)
        diags.append(d)
    return {'per_ring':rows,'fold_diagnostics':diags,'rmse_mm':[amount,2*amount],
            'mae_mm':[amount,2*amount]}


def test_permutations():
    orders=s.new_orders()
    assert list(orders)==list(s.SPLIT_SEEDS)
    assert all(set(o)==set(s.ORIGINAL_ORDER) for o in orders.values())
    assert all(set(v).isdisjoint(set(s.EXPECTED_OUTER)) for v in orders.values())


def test_ephemeral_order_restored():
    original=s.opt.EXPECTED_L
    order=s.new_orders()[s.SPLIT_SEEDS[0]]
    with s.temporary_fold_order(order):
        assert s.opt.EXPECTED_L==order
        assert s.strict.opt.EXPECTED_L==order
    assert s.opt.EXPECTED_L==original


def test_reject_illegal_order():
    for order in ((s.ORIGINAL_ORDER[0],)*8,tuple(s.ORIGINAL_ORDER[:7])+('DM07',)):
        try:
            with s.temporary_fold_order(order):pass
        except ValueError:pass
        else:raise AssertionError('Unsafe fold order accepted')


def test_report_boundaries():
    order=s.new_orders()[s.SPLIT_SEEDS[0]]
    for mode in s.MODES:
        r=fake_report(order,mode)
        e=s.check_report(r,order,mode)
        assert e.shape==(8,2)
    r=fake_report(order,'strict_L_only')
    r['fold_diagnostics'][0]['x_scaler_fit_ids'].append('DM01')
    try:s.check_report(r,order,'strict_L_only')
    except AssertionError:pass
    else:raise AssertionError('U leakage in strict accepted')


def test_bad_validation_and_duplicate_rejected():
    order=s.new_orders()[s.SPLIT_SEEDS[0]]
    r=fake_report(order,'shared_C0')
    r['per_ring'][0]['ring']='DM07'
    try:s.check_report(r,order,'shared_C0')
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Outer validation accepted')
    r=fake_report(order,'shared_C0')
    r['per_ring'][0]['fold']=2
    try:s.check_report(r,order,'shared_C0')
    except AssertionError:pass
    else:raise AssertionError('Fold mismatch accepted')


def test_immutable_output_before_inputs():
    with TemporaryDirectory() as t:
        base=Path(t);(base/'occupied').mkdir()
        try:s.ensure_new_output(base/'occupied','old','abl','strict','seed','data')
        except FileExistsError:pass
        else:raise AssertionError('Occupied output accepted')
        try:s.ensure_new_output(base/'old'/'new',base/'old','abl','strict','seed','data')
        except ValueError:pass
        else:raise AssertionError('Writing into historical root accepted')


def test_fit_pairing_patches_and_summary():
    order=s.new_orders()[s.SPLIT_SEEDS[0]]
    cfg={'K':3,'preprocessing':'P0','variant':'M2','C':0.25,'physical_var':0.47}
    seen=[]
    def fake_evaluate(rv,cfg,*,folds,epochs,steps):
        seen.append((s.opt.EXPECTED_L, cfg['C'],s.opt.FIXED_SEED))
        return fake_report(order,'ssl_Cselected' if cfg['C']>0 else 'shared_C0',.5)
    def fake_strict(rv,cfg,*,epochs,steps):
        seen.append((s.opt.EXPECTED_L,'strict',s.strict.FIXED_SEED))
        return fake_report(order,'strict_L_only',1.)
    with patch.object(s.opt,'evaluate_configuration',side_effect=fake_evaluate),\
         patch.object(s.strict,'strict_cv4',side_effect=fake_strict),\
         patch.object(s.opt,'validate_development',return_value=None):
        strict=s.fit_pairing({},cfg,order,'strict_L_only')
        shared=s.fit_pairing({},cfg,order,'shared_C0')
        ssl=s.fit_pairing({},cfg,order,'ssl_Cselected')
    assert [x[1] for x in seen]==['strict',0.0,.25]
    assert all(x[0]==order and x[2]==s.BASE_SEED for x in seen)
    assert s.opt.EXPECTED_L==s.ORIGINAL_ORDER
    d={13:{'canonical':{m:fake_report(s.ORIGINAL_ORDER,m) for m in s.MODES},
           s.SPLIT_SEEDS[0]:{m:fake_report(order,m,.5) for m in s.MODES},
           s.SPLIT_SEEDS[1]:{m:fake_report(s.new_orders()[s.SPLIT_SEEDS[1]],m,.8) for m in s.MODES}}}
    summ=s.summarize(d)
    assert len(summ['13']['models']['ssl_Cselected']['rmse_by_split_mm'])==3
    assert summ['13']['contrasts']['ssl_Cselected_minus_strict_L_only']['n_splits_better_per_target']==[0,0]


def main():
    funcs=[test_permutations,test_ephemeral_order_restored,test_reject_illegal_order,
           test_report_boundaries,test_bad_validation_and_duplicate_rejected,
           test_immutable_output_before_inputs,test_fit_pairing_patches_and_summary]
    for fn in funcs:
        fn();print('PASS',fn.__name__)
    print('7 SYNTHETIC TESTS PASSED. NO DATA PICKLE OPENED. NO OUTER TEST.')

if __name__=='__main__':main()
