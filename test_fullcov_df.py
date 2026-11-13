"""Schnelle mathematische Kontrollen ohne reale Ringdaten."""
import numpy as np
import fullcov_b1 as old
import fullcov_b1_df as new


def test_default_identical():
    for d in [2,6,12]:
        a=old.make_priors_full(3,d,prior_precision=1.3)
        b=new.make_priors_full(3,d,prior_precision=1.3)
        assert np.allclose(a.nu,b.nu) and np.allclose(a.W,b.W)
    print('PASS default_matches_existing_model_exactly')


def test_precision_mean_fixed():
    d=6;values=[]
    for off in [2,3,10,30]:
        pr=new.make_priors_full(3,d,prior_precision=.7,prior_df_offset=off)
        assert np.allclose(pr.nu, d+off)
        for k in range(3):
            assert np.allclose(pr.nu[k]*pr.W[k],np.eye(d)*.7,atol=1e-12)
            assert np.isfinite(np.linalg.inv(pr.W[k])/(pr.nu[k]-d-1)).all()
            assert abs(new.kl_normal_wishart(pr.mu[k],pr.kappa[k],pr.nu[k],pr.W[k],
                pr.mu[k],pr.kappa[k],pr.nu[k],pr.W[k]))<1e-9
        values.append(2*.7**2/pr.nu[0])
    assert all(x>y for x,y in zip(values,values[1:])),values
    print('PASS prior_expected_precision_fixed_and_variance_decreases')


def test_reject_invalid_offset():
    for offset in [0,1,-1,1.5]:
        try: new.make_priors_full(2,6,prior_df_offset=offset)
        except ValueError:pass
        else: raise AssertionError(offset)
    print('PASS invalid_prior_df_rejected')


def test_training_one_epoch():
    rng=np.random.default_rng(23)
    rings={'A':rng.normal(size=(18,3)),'B':rng.normal(size=(13,3))}
    labels={'A':np.array([.2,-.1])}
    for df in [2,10]:
        fit=new.train_full(rings,labels,K=2,epochs=1,max_local_iter=2,
                           prior_options={'prior_df_offset':df})
        assert np.isfinite(fit['history'][-1]['elbo'])
        assert np.all([np.linalg.eigvalsh(w).min()>0 for w in fit['process'].W])
    print('PASS one_epoch_synthetic_training_positive_definite')


if __name__=='__main__':
    test_default_identical();test_precision_mean_fixed();test_reject_invalid_offset();test_training_one_epoch()
