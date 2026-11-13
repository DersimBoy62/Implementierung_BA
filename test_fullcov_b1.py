"""Mathematische Tests für Normal-Wishart und Dichtevergleiche; keine Produktivdaten."""
import numpy as np
from scipy.stats import wishart, multivariate_normal
from scipy.special import logsumexp
import fullcov_b1 as fc


def test_kl_same_and_positive():
    prior=fc.make_priors_full(K=2,d=3)
    for k in range(2):
        z=fc.kl_normal_wishart(prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k],
                               prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k])
        assert abs(z)<1e-9,z
        newmu=prior.mu[k]+np.array([.1,.2,-.1]);newkappa=prior.kappa[k]+2.
        newW=prior.W[k]*1.3;newnu=prior.nu[k]+3
        z1=fc.kl_normal_wishart(newmu,newkappa,newnu,newW,prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k])
        assert z1>0 and np.isfinite(z1),z1
    print('PASS kl_same_zero_and_nonnegative')


def test_expected_log_density_monte_carlo():
    rng=np.random.default_rng(17);K=1;d=2
    pr=fc.make_priors_full(K,d)
    p=fc.FullProc(pr.pi,pr.A,np.array([[0.2,-0.3]]),np.array([3.]),np.array([12.]),
                  np.array([[[.12,.025],[.025,.14]]]))
    x=np.array([[0.4,-1.2],[-.1,.5]])
    analytic=fc.elog_emission_full(x,p)[:,0]
    # Unabhängige Monte Carlo Kontrolle von q(mu,Lambda) Erwartung.
    vals=np.zeros((3000,len(x)))
    for i in range(3000):
        lam=wishart.rvs(df=p.nu[0],scale=p.W[0],random_state=rng)
        mu=rng.multivariate_normal(p.mu[0],np.linalg.inv(lam)/p.kappa[0])
        vals[i]=multivariate_normal.logpdf(x,mean=mu,cov=np.linalg.inv(lam))
    assert np.max(np.abs(vals.mean(0)-analytic))<.12,(vals.mean(0),analytic)
    print('PASS expected_emission_matches_mc')


def test_weighted_posterior():
    pr=fc.make_priors_full(K=2,d=2)
    x=np.array([[1.,2.],[2.,1.],[3.,4.],[-1.,.7]])
    from vb_b1 import FB
    g=np.array([[.8,.2],[.9,.1],[.8,.2],[.9,.1]])
    xi=np.array([np.outer(g[i],g[i+1]) for i in range(3)])
    p=fc.update_process_full({'D1':x},{'D1':{'fb':FB(g,xi,0)}},pr,{'D1':1.})
    for k in range(2):
        n=g[:,k].sum();su=g[:,k]@x
        mu=(pr.kappa[k]*pr.mu[k]+su)/(pr.kappa[k]+n)
        assert np.allclose(p.mu[k],mu)
        assert abs(p.nu[k]-(pr.nu[k]+n))<1e-10
        assert np.linalg.eigvalsh(p.W[k]).min()>0
    print('PASS normal_wishart_weighted_posterior')


def test_student_t_scores_and_moments():
    pr=fc.make_priors_full(K=2,d=2)
    p=fc.FullProc(pr.pi,pr.A,pr.mu,pr.kappa,pr.nu,pr.W)
    x=np.array([[.0,0.],[2.,1.],[.1,-.4]])
    val=fc.predictive_logx_full(x,p)
    assert np.isfinite(val)
    ell=fc.elog_emission_full(x,p)
    assert ell.shape==(3,2) and np.isfinite(ell).all()
    print('PASS predictive_student_t_and_emission_shape')


if __name__=='__main__':
    test_kl_same_and_positive()
    test_expected_log_density_monte_carlo()
    test_weighted_posterior()
    test_student_t_scores_and_moments()
