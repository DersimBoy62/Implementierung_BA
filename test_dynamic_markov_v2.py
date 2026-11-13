"""Tests for exact moments, local Markov feedback and Bayesian V2 model."""
from __future__ import annotations
import itertools
import numpy as np
import dynamic_markov_v2 as dm2

def all_paths_moments(q0,Q,spec):
    T=Q.shape[0]+1;K=len(q0)
    node,edge=dm2.increments(T,spec)
    m=np.zeros(spec.D);H=np.zeros((spec.D,spec.D))
    for z in itertools.product(range(K),repeat=T):
        prob=q0[z[0]]
        for t in range(1,T):prob*=Q[t-1,z[t-1],z[t]]
        g=node[0,z[0]].copy()
        for t in range(1,T):g+=node[t,z[t]]+edge[t-1,z[t-1],z[t]]
        m+=prob*g;H+=prob*np.outer(g,g)
    return m,H


def test_exact_path_moments():
    q0=np.array([.7,.3]);Q=np.array([[[.8,.2],[.3,.7]],[[.5,.5],[.1,.9]],[[.6,.4],[.2,.8]]])
    for v in ('M0','M1','M2','M3'):
        spec=dm2.Spec(2,v)
        m,H=dm2.moments(q0,Q,spec)
        mb,Hb=all_paths_moments(q0,Q,spec)
        assert np.allclose(m,mb,rtol=2e-11,atol=2e-11),(v,'m',m,mb)
        assert np.allclose(H,Hb,rtol=2e-11,atol=2e-11),(v,'H',H,Hb)
        assert np.linalg.eigvalsh(H-np.outer(m,m)).min()>-1e-10


def test_exact_gradient_all_variants():
    import full_markov_b1 as fm
    rng=np.random.default_rng(8001)
    T=5;K=2
    logpi=np.log(np.array([.65,.35]));logA=np.log(np.array([[.7,.3],[.25,.75]]));ell=rng.normal(0,.2,(T,K))
    q0=np.array([.6,.4]);Q=np.tile(np.array([[.75,.25],[.4,.6]]),(T-1,1,1))
    v=np.r_[np.log(q0),np.log(Q).ravel()]
    for name in ('M0','M1','M2','M3'):
        spec=dm2.Spec(K,name)
        b=rng.normal(0,.5,spec.D);raw=rng.normal(0,.15,(spec.D,spec.D));G=(raw+raw.T)*.5
        f,g,_,_=dm2.objective_logits(v,logpi,logA,ell,b,G,spec)
        gn=np.zeros_like(g)
        step=1e-5
        for i in range(len(v)):
            plus=v.copy();minus=v.copy();plus[i]+=step;minus[i]-=step
            f1=dm2.objective_logits(plus,logpi,logA,ell,b,G,spec)[0]
            f2=dm2.objective_logits(minus,logpi,logA,ell,b,G,spec)[0]
            gn[i]=(f1-f2)/(2*step)
        assert np.max(np.abs(g-gn))<1e-6,(name,np.max(np.abs(g-gn)))


def test_occupancy_matches_legacy():
    import vb_b1 as base
    import full_markov_b1 as fm
    rng=np.random.default_rng(4)
    for k in (2,3,5,8):
        q0=rng.dirichlet(np.ones(k));Q=rng.dirichlet(np.ones(k),size=(7,k))
        fb=fm.markov_marginals(q0,Q)
        m,H=dm2.moments(q0,Q,dm2.Spec(k,'M0'))
        mb,Hb=base.moments_occupancy(fb)
        assert np.allclose(m,mb,atol=1e-10)
        assert np.allclose(H,Hb,atol=1e-10)


def test_invariants_and_k():
    K=8;T=6
    q0=np.full(K,1./K);Q=np.full((T-1,K,K),1./K)
    for name,d in [('M0',8),('M1',10),('M2',16),('M3',18)]:
        spec=dm2.Spec(K,name);m,H=dm2.moments(q0,Q,spec)
        assert len(m)==d and H.shape==(d,d)
        assert abs(m[:K].sum()-1)<1e-12
        assert np.max(np.abs(H-H.T))<1e-12
        if spec.temporal:assert abs(m[K:2*K].sum())<1e-12
        if spec.switches:assert np.all((m[-2:]>=0)&(m[-2:]<=1))


def test_train_predict_synthetic():
    import dynamic_quality_v2 as q2
    rng=np.random.default_rng(2321)
    rings={f'L{i}':rng.normal(0,.7,(9,3)) for i in range(4)}
    rings.update({f'U{i}':rng.normal(.2,.9,(7,3)) for i in range(2)})
    y={f'L{i}':rng.normal(size=2) for i in range(4)}
    s={rid:[rng.normal(size=2),rng.normal(size=1)] for rid in rings}
    for v in ('M1','M2','M3'):
        m=q2.train(rings,y,s,variant=v,K=2,C=1,epochs=2,steps=1,seed=17)
        pred,sd=q2.predict(rings['U0'],s['U0'],m)
        assert pred.shape==(2,) and sd.shape==(2,) and np.all(sd>0)
        assert np.all(np.isfinite([z['elbo'] for z in m['history']]))


def main():
    for fn in [test_exact_path_moments,test_exact_gradient_all_variants,test_occupancy_matches_legacy,test_invariants_and_k,test_train_predict_synthetic]:
        fn();print('PASS',fn.__name__,flush=True)
if __name__=='__main__':main()
