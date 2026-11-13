"""T4: independent density/integral and hand-calculated conjugacy checks."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, invgamma, norm, dirichlet
from bayesian_ring_ssr.normal_wishart import NormalWishart
from bayesian_ring_ssr.nig_heads import NIGHead
from bayesian_ring_ssr.priors import kl_dirichlet, elog_dirichlet


def test_wishart_scale_d1_loglikelihood_and_update():
    # Lambda ~ Gamma(nu/2, SCALE=2W); E Lambda = nu W.
    p = NormalWishart(np.array([0.]),2.,5.,np.array([[.4]]))
    elog = quad(lambda v: gamma.pdf(v,2.5,scale=.8)*np.log(v),0,np.inf)[0]
    expected = .5*(elog-np.log(2*np.pi)-.5-2.*4.)
    assert p.expected_log_emission(np.array([[2.]]))[0] == pytest.approx(expected,abs=2e-10)
    # Two observations 1 and 3: n=2,sum=4,sum-square=10.
    q = p.update(2.,np.array([4.]),np.array([[10.]]))
    np.testing.assert_allclose(q.mean,[1.])
    assert q.kappa == 4 and q.nu == 7
    np.testing.assert_allclose(q.W,[[1/(2.5+10-4)]])


def test_normal_wishart_kl_against_independent_gamma_integral():
    q = NormalWishart(np.array([.7]),3.,6.,np.array([[.3]]))
    p = NormalWishart(np.array([-.2]),.5,4.,np.array([[.8]]))
    def density_integrand(lam):
        lq = gamma.logpdf(lam,q.nu/2,scale=2*q.W[0,0])
        lp = gamma.logpdf(lam,p.nu/2,scale=2*p.W[0,0])
        # KL of conditional normals from their variances 1/(kappa*Lambda).
        vq,vp = 1/(q.kappa*lam),1/(p.kappa*lam)
        conditional = .5*(np.log(vp/vq)+(vq+(.9)**2)/vp-1)
        return np.exp(lq)*(lq-lp+conditional)
    independent = quad(density_integrand,0,np.inf,epsabs=1e-10)[0]
    assert q.kl(p) == pytest.approx(independent,abs=1e-9)
    assert q.kl(q) == pytest.approx(0.,abs=1e-12)


def test_dirichlet_kl_beta_integral_and_expected_logs():
    from scipy.stats import beta
    q,p = np.array([2.,3.]),np.array([1.,4.])
    integral = quad(lambda x: beta.pdf(x,*q)*(beta.logpdf(x,*q)-beta.logpdf(x,*p)),0,1)[0]
    assert kl_dirichlet(q,p) == pytest.approx(integral,abs=1e-10)
    expected = quad(lambda x: beta.pdf(x,*q)*np.log(x),0,1)[0]
    assert elog_dirichlet(q)[0] == pytest.approx(expected,abs=1e-10)
    assert kl_dirichlet(q,q) == pytest.approx(0.,abs=1e-13)


def test_nig_inverse_gamma_exp_minus_b_over_variance_convention():
    q = NIGHead(np.array([.3]),np.array([[.5]]),4.,2.)
    p = NIGHead(np.array([-.1]),np.array([[2.]]),2.5,1.5)
    def integrand(s):
        lq,lp = invgamma.logpdf(s,q.a,scale=q.b),invgamma.logpdf(s,p.a,scale=p.b)
        conditional = .5*(np.log(4)+(.5*s+.4**2)/(2*s)-1)
        return np.exp(lq)*(lq-lp+conditional)
    assert q.kl(p) == pytest.approx(quad(integrand,0,np.inf,epsabs=1e-11)[0],abs=1e-9)
    assert q.kl(q) == pytest.approx(0.,abs=1e-13)
    Eprec = quad(lambda s: invgamma.pdf(s,q.a,scale=q.b)/s,0,np.inf)[0]
    Elogs = quad(lambda s: invgamma.pdf(s,q.a,scale=q.b)*np.log(s),0,np.inf)[0]
    # E_beta|s[(y-beta*x)^2/s] for deterministic x=2,y=1.
    expected = -.5*(np.log(2*np.pi)+Elogs+Eprec*(1-.3*2)**2+.5*4)
    assert q.expected_log_likelihood(1.,np.array([2.]),np.array([[4.]])) == pytest.approx(expected,abs=1e-9)


def test_nig_hand_computed_conjugate_update_and_predictive():
    p = NIGHead(np.array([0.]),np.array([[2.]]),2.5,1.5)
    q = p.update([1.,2.],[(np.array([1.]),np.array([[1.]]))]*2)
    np.testing.assert_allclose(q.V,[[.4]])
    np.testing.assert_allclose(q.mean,[1.2])
    assert q.a == 3.5
    assert q.b == pytest.approx(2.2)
    mean,var = q.predictive_moments(np.array([1.]),np.array([[1.]]))
    assert mean == pytest.approx(1.2)
    assert var == pytest.approx(2.2/2.5*1.4)


@pytest.mark.parametrize('bad', [np.array([[0.,1.],[1.,0.]]),np.array([[1.,.2],[0.,1.]])])
def test_spd_failures_raise(bad):
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        NormalWishart(np.zeros(2),1.,4.,bad)
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        NIGHead(np.zeros(2),bad,2.5,1.5)

