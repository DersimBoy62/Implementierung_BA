"""T1: path enumeration is independent of the dynamic program."""
import itertools
import numpy as np
import pytest
from scipy.special import logsumexp
from bayesian_ring_ssr.forward_backward import forward_backward


def enumerate_hmm(lp,la,ell):
    T,K=ell.shape
    paths=list(itertools.product(range(K),repeat=T))
    scores=np.array([lp[z[0]]+sum(ell[t,z[t]] for t in range(T))
                     +sum(la[z[t-1],z[t]] for t in range(1,T)) for z in paths])
    logz=logsumexp(scores)
    weights=np.exp(scores-logz)
    gamma=np.zeros((T,K))
    xi=np.zeros((T-1,K,K))
    for z,w in zip(paths,weights):
        for t,k in enumerate(z): gamma[t,k]+=w
        for t in range(T-1): xi[t,z[t],z[t+1]]+=w
    return gamma,xi,logz


@pytest.mark.parametrize("T,K",[(1,2),(2,2),(4,2),(4,3),(4,4)])
def test_T1_forward_backward_matches_all_paths(T,K):
    rng=np.random.default_rng(840+T+K)
    lp=rng.normal(size=K)-2
    la=rng.normal(size=(K,K))-3
    ell=rng.normal(size=(T,K))
    fb=forward_backward(lp,la,ell)
    g,xi,z=enumerate_hmm(lp,la,ell)
    np.testing.assert_allclose(fb.gamma,g,atol=2e-13)
    np.testing.assert_allclose(fb.xi,xi,atol=2e-13)
    assert abs(fb.log_normalizer-z)<2e-13
    np.testing.assert_allclose(fb.gamma.sum(axis=1),1,atol=1e-13)
    np.testing.assert_allclose(fb.xi.sum(axis=2),fb.gamma[:-1],atol=1e-13)
    np.testing.assert_allclose(fb.xi.sum(axis=1),fb.gamma[1:],atol=1e-13)


def test_T1_subnormalized_expected_log_potentials_are_not_renormalized():
    lp=np.array([-1.,-2.])
    la=np.array([[-.1,-4.],[-3.,-2.]])
    ell=np.array([[.2,-.4],[.1,.5],[-.7,.1]])
    fb=forward_backward(lp,la,ell)
    g,xi,z=enumerate_hmm(lp,la,ell)
    np.testing.assert_allclose(fb.gamma,g,atol=1e-14)
    assert abs(fb.log_normalizer-z)<1e-14
    normalized=forward_backward(lp-logsumexp(lp),la-logsumexp(la,axis=1,keepdims=True),ell)
    assert np.max(abs(normalized.gamma-fb.gamma))>.01


def test_T1_long_sequence_and_label_permutation():
    rng=np.random.default_rng(12)
    ell=rng.normal(size=(2800,3))*8
    lp=rng.normal(size=3)
    la=rng.normal(size=(3,3))
    fb=forward_backward(lp,la,ell)
    p=np.array([2,0,1])
    perm=forward_backward(lp[p],la[p][:,p],ell[:,p])
    assert np.isfinite(fb.log_normalizer)
    assert np.isfinite(fb.gamma).all()
    np.testing.assert_allclose(fb.gamma[:,p],perm.gamma,atol=3e-13)
    np.testing.assert_allclose(fb.log_normalizer,perm.log_normalizer,atol=1e-10)


@pytest.mark.parametrize("lp,la,ell",[
    ([0],[[0]],[[0]]),([0,0],[[0,0],[0,0]],np.empty((0,2))),
    ([0,np.nan],np.zeros((2,2)),np.zeros((2,2))),
    ([0,0],np.zeros((2,2)),[[0,np.inf]])])
def test_T9_invalid_hmm_potentials(lp,la,ell):
    with pytest.raises(ValueError): forward_backward(lp,la,ell)
