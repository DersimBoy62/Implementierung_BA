"""Unabhängige Tests der lokalen freien Markov-B1-Optimierung."""
import itertools
import numpy as np
from scipy.optimize import check_grad
from scipy.special import logsumexp
from vb_b1 import FB,moments_occupancy,quality_logterm,make_priors,ProcPosterior,QualityPosterior,elog_emission,elog_dir,forward_backward
from full_markov_b1 import (objective_and_grad,init_logits_from_fb,unpack_logits,
                             quality_coeff,markov_marginals,local_full_joint)

def test_enumeration():
    T,K=4,3
    logpi=np.log([.45,.25,.30]);loga=np.log([[.65,.30,.05],[.1,.7,.2],[.2,.3,.5]])
    ell=np.random.default_rng(13).normal(size=(T,K))*.6
    q=QualityPosterior(np.array([[2.,-1.,.2],[-.3,.6,1.]]),np.stack([np.eye(K)*.2]*2),np.array([4.,3.]),np.array([.8,1.2]))
    y=np.array([.8,-.5]);r=np.random.default_rng(42);v=r.normal(size=K+(T-1)*K*K)*.4
    f,g=objective_and_grad(v,logpi,loga,ell,y,q)
    p0,Q=unpack_logits(v,T,K)
    probs=[]; qualities=[];proc=[]
    for z in itertools.product(range(K),repeat=T):
        lq=np.log(p0[z[0]])+sum(np.log(Q[t-1,z[t-1],z[t]]) for t in range(1,T))
        lp=logpi[z[0]]+ell[0,z[0]]+sum(loga[z[t-1],z[t]]+ell[t,z[t]] for t in range(1,T))
        h=np.bincount(z,minlength=K)/T
        probs.append(np.exp(lq));proc.append(lp-lq)
        qualities.append(quality_logterm(y,h,np.outer(h,h),q))
    truth=np.dot(probs,np.asarray(proc)+np.asarray(qualities))
    np.testing.assert_allclose(f,truth,rtol=0,atol=1e-10)
    p=markov_marginals(p0,Q);m,H=moments_occupancy(p)
    qs=np.array([np.bincount(z,minlength=K)/T for z in itertools.product(range(K),repeat=T)])
    np.testing.assert_allclose(m,np.array(probs)@qs,atol=1e-11)
    np.testing.assert_allclose(H,np.einsum('z,za,zb->ab',probs,qs,qs),atol=1e-11)
    print('PASS exact_enumeration_markov_elbo_and_moments')


def test_gradient():
    T,K=4,3
    rng=np.random.default_rng(20)
    logpi=np.log(np.array([.2,.3,.5])); loga=np.log(np.array([[.6,.2,.2],[.1,.4,.5],[.2,.3,.5]]))
    ell=rng.normal(size=(T,K))
    q=QualityPosterior(rng.normal(size=(2,K)),np.stack([np.eye(K)*.4]*2),np.array([3.,5.]),np.array([1.1,2.4]))
    y=np.array([-1.2,0.6]);v=rng.normal(size=K+(T-1)*K*K)*.2
    error=check_grad(lambda x:objective_and_grad(x,logpi,loga,ell,y,q)[0],
                     lambda x:objective_and_grad(x,logpi,loga,ell,y,q)[1],v)
    assert error<5e-6, error
    print('PASS analytic_gradient_finite_differences',error)


def test_process_posterior_and_improvement():
    K=2; x=np.zeros((15,1))
    pri=make_priors(K,1)
    p=ProcPosterior(np.array([5.,5.]),np.array([[8.,1.],[1.,8.]]),np.zeros((K,1)),np.full((K,1),10.),np.full((K,1),3.),np.full((K,1),2.))
    q=QualityPosterior(np.array([[5.,-5.],[0.,0.]]),np.stack([np.eye(K)*.05]*2),np.array([20.,20.]),np.array([.4,.4]))
    y=np.array([4.5,0.]);base=forward_backward(elog_dir(p.pi),elog_dir(p.A),elog_emission(x,p))
    init=init_logits_from_fb(base)
    a=objective_and_grad(init,elog_dir(p.pi),elog_dir(p.A),elog_emission(x,p),y,q)[0]
    z=local_full_joint(x,p,q,y,max_iter=30)
    assert z['local_objective']>=a-1e-8
    assert z['full_local_gain']>0.01,(z['full_local_gain'],a)
    print('PASS full_joint_changes_and_improves',z['full_local_gain'])

if __name__=='__main__':
    test_enumeration();test_gradient();test_process_posterior_and_improvement()
