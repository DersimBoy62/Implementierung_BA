"""Self-contained tests for the descriptive ringwise SSR robustness audit."""
from copy import deepcopy
import numpy as np
from audit_ssr_robustness import (
    MODES,MODELS,OUTER,paired_comparison,validate_report,audit
)

L_ALL=('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
U=('DM01','DM02','DM10','DM14')


def make_fixture():
    folds=[]
    for fold in range(1,5):
        val=L_ALL[2*(fold-1):2*fold]
        label=tuple(r for r in L_ALL if r not in val)
        modes={}
        for mode in MODES:
            tr=label if mode=='SUP_L_ONLY' else tuple(sorted((*label,*U)))
            predictions={}
            for model in MODELS:
                err=[1.,2.] if mode=='SUP_L_ONLY' else ([.5,1.5] if mode=='C0_SHARED_PREPROC' else [.4,1.4])
                if model=='PHYS_NIG_TWELVE' and mode=='SSL_U_C1':err=[.5,1.5]
                predictions[model]=[{'ring':r,'residual_mm':err} for r in val]
            modes[mode]={'quality_fit_ids':label,'x_fit_ids':tr,
                         'unlabeled_training_weight':1. if mode=='SSL_U_C1' else 0.,
                         'per_model':predictions}
        folds.append({'fold':fold,'train_L':label,'train_U':U,'inner_val':val,'modes':modes})
    return {'outer_test_locked':OUTER,'folds':folds}


def test_exact_paired_delta_and_ring_counts():
    A=np.repeat([[1.,2.]],8,axis=0)
    B=np.repeat([[.5,3.]],8,axis=0)
    r=paired_comparison(A,B,tuple(str(i) for i in range(8)))
    assert np.allclose(r['change_mse_mm2'],[-.75,5.])
    assert r['improved_ring_count']==[8,0]
    assert r['worse_ring_count']==[0,8]
    assert r['scoring_loo_gain_has_same_sign_as_all']==[True,True]
    print('PASS exact_paired_delta_and_ring_counts')


def test_loo_scoring_only_and_outlier_identified():
    A=np.zeros((8,2)); B=np.zeros((8,2));B[0]=[-8.,9.];B[1:]=[1.,-1.]
    d=paired_comparison(A,B,tuple('ABCDEFGH'))
    assert np.allclose(d['change_mse_mm2'],[(64+7)/8,(81+7)/8])
    assert d['scoring_loo_delta_mse_range_mm2']['min'][0]==1.
    assert d['scoring_loo_delta_mse_range_mm2']['max'][0]==(64+6)/7
    print('PASS scoring_loo_not_retraining')


def test_fixed_fold_structure_and_direct_nig_invariance():
    rep=make_fixture()
    assert validate_report(rep)==tuple(sorted(L_ALL))
    d=audit(rep)
    assert d['model_audits']['PHYS_NIG_TWELVE']['comparisons']['U_ELBO_ONLY']['change_mse_mm2']==[0.,0.]
    assert d['model_audits']['HYBRID_SIX']['comparisons']['U_ELBO_ONLY']['improved_ring_count']==[8,8]
    print('PASS fixed_fold_structure_and_direct_nig_invariance')


def test_leakage_rejected():
    rep=make_fixture()
    for corruption in ('outer','overlap','quality_scale','x_scale'):
        q=deepcopy(rep)
        if corruption=='outer':q['outer_test_locked']=['DM07','DM08','DM11','OTHER']
        elif corruption=='overlap':q['folds'][0]['train_L']=(*q['folds'][0]['train_L'],'DM04')
        elif corruption=='quality_scale':q['folds'][0]['modes']['SSL_U_C1']['quality_fit_ids']=(*q['folds'][0]['train_L'],'DM04')
        else:q['folds'][0]['modes']['C0_SHARED_PREPROC']['x_fit_ids']=(*q['folds'][0]['train_L'],'DM04')
        try:validate_report(q)
        except ValueError:pass
        else:raise AssertionError(f'Expected {corruption} leakage refusal')
    print('PASS outer_test_and_fold_leakage_rejected')


if __name__=='__main__':
    test_exact_paired_delta_and_ring_counts()
    test_loo_scoring_only_and_outlier_identified()
    test_fixed_fold_structure_and_direct_nig_invariance()
    test_leakage_rejected()
