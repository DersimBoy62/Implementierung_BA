"""Tests for two head Bayesian quality likelihood and local Markov coupling."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import two_head_b1 as th
import hybrid_b1 as hb
import vb_b1 as base
import fullcov_b1 as full
import full_markov_b1 as fm

def chk(name,fn):
    fn();print('PASS',name,flush=True)

def test_exact_compression():
    rng=np.random.default_rng(2);K=3;T=5
    q0=np.array([.2,.5,.3]);Q=rng.dirichlet(np.ones(K),size=(T-1,K))
    fb=fm.markov_marginals(q0,Q);m,H=base.moments_occupancy(fb)
    heads=[th.prior_head(K,4),th.prior_head(K,0)]
    heads[0].mu=np.arange(K+4)*.1;heads[1].mu=np.arange(K)*.2
    ss=[np.array([1.,.3,-.6,.7]),np.zeros(0)];y=[.8,-.4]
    b,G,c=th.both_compressed_coeff(y,heads,ss,K)
    direct=sum(th.quality_logterm(y[j],*th.augment(m,H,ss[j]),heads[j]) for j in range(2))
    assert abs(direct-(c+b@m+np.sum(G*H)))<2e-12

def test_fit_unequal():
    K=3;prior=[th.prior_head(K,6),th.prior_head(K,0)]
    m=np.array([.2,.3,.5]);H=np.outer(m,m)+np.diag([.02,.01,.03]);
    ss=[np.arange(6)/3.,np.zeros(0)]
    local={'DM01':{'head_moments':[th.augment(m,H,s) for s in ss]}}
    yy={'DM01':np.array([.1,-.8])}
    outs=[th.fit_head(yy,local,j,prior[j]) for j in range(2)]
    assert tuple(len(h.mu) for h in outs)==(9,3)
    assert all(np.linalg.eigvalsh(h.V).min()>0 for h in outs)

def test_state_feedback_both():
    rng=np.random.default_rng(14);K=2
    x=rng.standard_normal((10,3));p0=full.make_priors_full(K,3)
    p=full.FullProc(p0.pi,p0.A,np.zeros((K,3)),np.full(K,4.),p0.nu,p0.W)
    h1=th.prior_head(K,1);h2=th.prior_head(K,0)
    h1.mu=np.array([1.5,-1.,.2]);h2.mu=np.array([.5,-.5])
    ss=[np.array([2.]),np.zeros(0)]
    zero=th._local_augmented(x,p,[h1,h2],np.array([0.,0.]),ss,0)
    changed=th._local_augmented(x,p,[h1,h2],np.array([2.,-2.]),ss,15)
    assert changed['local_gain']>=0 and changed['accepted']
    assert np.linalg.norm(changed['m_h']-zero['m_h'])>1e-5

def test_reduction_to_shared_6():
    rng=np.random.default_rng(3);K=2
    rings={f'r{i}':rng.standard_normal((25+i*3,3)) for i in range(3)}
    y={k:rng.normal(size=2) for k in list(rings)[:2]}
    sf={k:rng.normal(size=2) for k in rings}
    m1=hb.train(rings,y,sf,K=K,C=1.,epochs=2,steps=3,seed=20261111)
    m2=th.train(rings,y,{k:[sf[k],sf[k]] for k in rings},K=K,C=1.,epochs=2,steps=3,seed=20261111)
    assert np.allclose(m1['process'].mu,m2['process'].mu,atol=1e-10)
    assert all(np.allclose(m1['quality'].mu[j],m2['heads'][j].mu,atol=1e-9) for j in range(2))
    assert abs(m1['history'][0]['elbo']-m2['history'][0]['elbo'])<1e-7

def test_no_unlabeled_y():
    rng=np.random.default_rng(12)
    rings={'L':rng.standard_normal((18,3)),'U':rng.standard_normal((20,3))}
    ss={r:[np.zeros(2),np.zeros(0)] for r in rings}
    y={'L':np.array([0.1,0.2])}
    a=th.train(rings,y,ss,K=2,epochs=1,steps=0,C=0.)
    b=th.train({'L':rings['L']},y,{'L':ss['L']},K=2,epochs=1,steps=0,C=0.)
    # scaling/init only differs; no direct U gradients at C=0 after fixed initialization.
    assert np.isfinite(a['history'][0]['elbo']) and np.isfinite(b['history'][0]['elbo'])
    try:
        th.train(rings,{'L':np.array([0.,0.]),'U':np.array([0.,0.]),'v':np.ones(2)},ss,K=2,epochs=1)
    except ValueError:pass
    else:raise AssertionError('Wrong labels silently accepted')

chk('two_heads_exact_compressed_joint_quality_likelihood',test_exact_compression)
chk('distinct_head_dimensions_and_conjugate_updates',test_fit_unequal)
chk('both_quality_heads_influence_local_markov_q',test_state_feedback_both)
chk('shared_features_reproduce_original_hybrid',test_reduction_to_shared_6)
chk('two_targets_and_labeled_data_guards',test_no_unlabeled_y)
