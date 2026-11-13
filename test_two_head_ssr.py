"""Small, deterministic guards for two-head SSR ablation. Run in project root."""
import numpy as np
import audit_two_head_ssr as m

def t(name,fn):
    fn()
    print('PASS',name)

def test_paired():
    before=[{'ring':'A','residual_mm':[2.,3.]},{'ring':'B','residual_mm':[1.,0.]}]
    after=[{'ring':'B','residual_mm':[0.,0.]},{'ring':'A','residual_mm':[1.,2.]}]
    z=m.paired_comparison(before,after)
    assert np.allclose(z['delta_mse_new_minus_old_mm2'],[-2.,-2.5])
    assert z['rings_better_per_target']==[2,1]
    assert not all(z['loo_all_improvements'])

def test_folds():
    s={'L':('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18'),
       'U':('DM01','DM02','DM10','DM14'), 'TEST_GESPERRT':m.LOCKED}
    seen=[]
    for f in (1,2,3,4):
        L,U,V=m.fold_split(s,f)
        assert len(L)==6 and len(U)==4 and len(V)==2
        seen+=list(V)
        assert not set(V)&(set(L)|set(U)|set(m.LOCKED))
    assert sorted(seen)==sorted(s['L'])

def test_folds_reject_leakage():
    s={'L':('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18'),
       'U':('DM01','DM02','DM10','DM14'), 'TEST_GESPERRT':m.LOCKED}
    for changed in [dict(s,U=('DM01','DM04','DM10','DM14')),
                    dict(s,TEST_GESPERRT=('DM07','DM08','DM11','DM16'))]:
        try:m.fold_split(changed,1)
        except ValueError:pass
        else:raise AssertionError('Leakage not rejected')

def test_modes():
    assert m.MODES==('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')
    assert tuple(len(h) for h in m.HEADS)==(6,5)

if __name__=='__main__':
    t('paired_mse_and_lOO_not_refitting',test_paired)
    t('ringwise_fourfold_isolation',test_folds)
    t('leakage_rejected',test_folds_reject_leakage)
    t('fixed_two_head_structures_and_three_modes',test_modes)
