"""T5: posterior, KL and predictive invariance under exact prior pushforward."""
import numpy as np
import pytest
from bayesian_ring_ssr.nig_heads import head_prior, augment
from bayesian_ring_ssr.priors import m2_lift


@pytest.mark.parametrize('K',[2,3,4])
@pytest.mark.parametrize('nphys',[0,2,6])
def test_induced_prior_conjugate_posterior_kl_prediction(K,nphys):
    rng = np.random.default_rng(123+K+nphys)
    A = m2_lift(K,nphys)
    pfull,pred = head_prior(K,nphys,.385,False),head_prior(K,nphys,.385,True)
    np.testing.assert_allclose(pred.V,A.T@pfull.V@A,atol=1e-15)
    np.testing.assert_allclose(pred.V[K:2*K-1,K:2*K-1],.15*(np.eye(K-1)+np.ones((K-1,K-1))),atol=1e-15)
    red_moments,full_moments = [],[]
    # Exact finite distribution over state-feature vectors, not H=mm'.
    for i in range(5):
        draws = rng.normal(size=(7,2*K-1+nphys))
        draws[:,:K] = rng.dirichlet(np.ones(K),size=7)
        m,H = draws.mean(0),draws.T@draws/7
        red_moments.append((m,H))
        full_moments.append((A@m,A@H@A.T))
    y = rng.normal(size=5)
    qf,qr = pfull.update(y,full_moments),pred.update(y,red_moments)
    np.testing.assert_allclose(qr.mean,A.T@qf.mean,atol=3e-13)
    np.testing.assert_allclose(qr.V,A.T@qf.V@A,atol=3e-13)
    assert qr.a == qf.a
    assert qr.b == pytest.approx(qf.b,abs=3e-13)
    assert qr.kl(pred) == pytest.approx(qf.kl(pfull),abs=2e-12)
    for yy,(m,H),(mf,Hf) in zip(y,red_moments,full_moments):
        assert qr.expected_log_likelihood(yy,m,H) == pytest.approx(qf.expected_log_likelihood(yy,mf,Hf),abs=3e-12)
        np.testing.assert_allclose(qr.predictive_moments(m,H),qf.predictive_moments(mf,Hf),atol=3e-12)

