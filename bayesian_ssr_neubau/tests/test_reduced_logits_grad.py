"""T2/T3: independent path objective and central-difference gradients."""
import itertools
import numpy as np
import pytest
from scipy.special import logsumexp
from bayesian_ring_ssr.reduced_markov import (
    MarkovChain,dimensions,unpack,pack,full_to_reduced,reduced_to_full)
from bayesian_ring_ssr.hmmmoments import m2_increments
from bayesian_ring_ssr.local_vi import objective_and_grad,optimize_local


def enumerate_objective(chain,lp,la,ell,node,b,G,c):
    value=0.
    for z in itertools.product(range(chain.K),repeat=chain.T):
        logq=np.log(chain.initial[z[0]])
        logp=lp[z[0]]+sum(ell[t,z[t]] for t in range(chain.T))
        for t in range(chain.T-1):
            logq+=np.log(chain.transitions[t,z[t],z[t+1]])
            logp+=la[z[t],z[t+1]]
        g=sum(node[t,z[t]] for t in range(chain.T))
        value+=np.exp(logq)*(logp-logq+b@g+g@G@g+c)
    return value


@pytest.mark.parametrize("T,K",[(4,2),(5,2),(4,3),(4,4)])
def test_T2_T3_objective_gradient_against_paths_and_central_differences(T,K):
    rng=np.random.default_rng(130+T+K)
    v=rng.normal(size=dimensions(T,K)[1])*.6
    lp=rng.normal(size=K)
    la=rng.normal(size=(K,K))
    ell=rng.normal(size=(T,K))
    D=2*K-1
    b=rng.normal(size=D)
    G=rng.normal(size=(D,D))*.4 # asymmetric input also has correct derivative
    c=.37
    f,gradient,m,H=objective_and_grad(v,lp,la,ell,b,G,c)
    expected=enumerate_objective(unpack(v,T,K),lp,la,ell,m2_increments(T,K),b,G,c)
    assert abs(f-expected)<4e-13
    step=2e-6
    numerical=np.empty_like(v)
    for i in range(len(v)):
        u=np.zeros_like(v);u[i]=step
        numerical[i]=(objective_and_grad(v+u,lp,la,ell,b,G,c)[0]
                      -objective_and_grad(v-u,lp,la,ell,b,G,c)[0])/(2*step)
    np.testing.assert_allclose(gradient,numerical,rtol=3e-6,atol=3e-9)


@pytest.mark.parametrize("K",[2,3,4])
def test_T3_softmax_gauge_and_gradient_chain_rule(K):
    T=4
    rng=np.random.default_rng(740+K)
    v=rng.normal(size=dimensions(T,K)[1])
    full=reduced_to_full(v,T,K).reshape(-1,K)
    shifted=full+rng.normal(size=(len(full),1))*5
    reduced=full_to_reduced(shifted.ravel(),T,K)
    np.testing.assert_allclose(reduced,v,atol=3e-15)
    probs=np.exp(shifted-logsumexp(shifted,axis=1,keepdims=True))
    direct=MarkovChain(probs[0],probs[1:].reshape(T-1,K,K))
    chain=unpack(reduced,T,K)
    np.testing.assert_allclose(chain.gamma,direct.gamma,atol=2e-15)
    np.testing.assert_allclose(chain.xi,direct.xi,atol=2e-15)
    np.testing.assert_allclose(pack(chain,floor=0),v,atol=3e-15)
    args=(rng.normal(size=K),rng.normal(size=(K,K)),rng.normal(size=(T,K)),
          rng.normal(size=2*K-1),np.eye(2*K-1)*-.2)
    f,gr,*_=objective_and_grad(v,*args)
    # Independent finite differences in each FULL logit coordinate show that
    # fixing the final coordinate selects the first K-1 gradients (not their
    # differences). Full rows have zero-sum gradients.
    gf=np.zeros_like(full)
    for row in range(len(full)):
        for col in range(K):
            perturb=np.zeros_like(full);perturb[row,col]=2e-6
            fp=objective_and_grad(full_to_reduced((full+perturb).ravel(),T,K),*args)[0]
            fm=objective_and_grad(full_to_reduced((full-perturb).ravel(),T,K),*args)[0]
            gf[row,col]=(fp-fm)/4e-6
    np.testing.assert_allclose(gr,gf[:,:-1].ravel(),atol=3e-9)
    np.testing.assert_allclose(gf.sum(axis=1),0,atol=4e-9)


def test_T6_safeguard_retains_previous_and_improves_joint_objective():
    rng=np.random.default_rng(318)
    T,K=8,2
    lp=rng.normal(size=K)
    la=rng.normal(size=(K,K))
    ell=rng.normal(size=(T,K))
    b=np.array([2.,-3.,5.])
    G=-np.eye(3)
    first=optimize_local(lp,la,ell,b,G,max_iter=8)
    second=optimize_local(lp,la,ell,b,G,previous=first.logits,max_iter=0)
    assert first.objective>=first.diagnostics['objective_fb']
    assert first.diagnostics['local_gain']>1e-5
    assert second.objective>=first.objective-1e-13
    assert second.diagnostics['selected_previous']
    np.testing.assert_array_equal(first.logits,second.logits)


def test_T9_invalid_local_shape_and_logits():
    with pytest.raises(ValueError): unpack(np.zeros(3),4,2)
    with pytest.raises(ValueError): unpack(np.full(7,np.nan),4,2)
    with pytest.raises(ValueError): MarkovChain([.4,.4],np.empty((0,2,2)))
    with pytest.raises(ValueError): pack(MarkovChain([1.,0.],np.empty((0,2,2))),floor=0)
