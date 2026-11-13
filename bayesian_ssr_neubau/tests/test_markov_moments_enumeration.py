"""T2: exact first/second moments from exhaustive path sums."""
import itertools
import numpy as np
import pytest
from bayesian_ring_ssr.reduced_markov import unpack,dimensions
from bayesian_ring_ssr.hmmmoments import moments,m2_increments,augment_moments


def enumerate_moments(chain,node):
    D=node.shape[-1]
    m=np.zeros(D)
    H=np.zeros((D,D))
    for path in itertools.product(range(chain.K),repeat=chain.T):
        probability=chain.initial[path[0]]
        for t in range(chain.T-1): probability*=chain.transitions[t,path[t],path[t+1]]
        feature=sum(node[t,path[t]] for t in range(chain.T))
        m+=probability*feature
        H+=probability*np.outer(feature,feature)
    return m,H


@pytest.mark.parametrize("T,K",[(4,2),(5,2),(4,3),(5,3),(4,4)])
def test_T2_markov_m2_moments_against_all_paths(T,K):
    rng=np.random.default_rng(53+T+K)
    chain=unpack(rng.normal(size=dimensions(T,K)[1]),T,K)
    node=m2_increments(T,K)
    m,H=moments(chain)
    em,eH=enumerate_moments(chain,node)
    np.testing.assert_allclose(m,em,atol=5e-14)
    np.testing.assert_allclose(H,eH,atol=5e-14)
    assert np.max(abs(H-np.outer(m,m)))>.001
    assert np.linalg.eigvalsh(H-np.outer(m,m)).min()>-2e-14
    np.testing.assert_array_equal(H,H.T)
    assert abs(m[:K].sum()-1)<2e-14


@pytest.mark.parametrize("T,K",[(4,2),(5,3),(6,4)])
def test_T2_early_late_count_and_exact_lift(T,K):
    full=m2_increments(T,K,reduced=False)
    reduced=m2_increments(T,K)
    ne=T//2
    assert full[ne-1,0,K]==-1/ne
    assert full[ne,0,K]==1/(T-ne)
    lift=np.zeros((2*K,2*K-1))
    lift[:2*K-1]=np.eye(2*K-1)
    lift[-1,K:]=-1
    # Lift identity holds for complete paths, because the discarded delta is
    # the negative sum. Per-time increments also have a deterministic offset.
    rng=np.random.default_rng(T+K)
    chain=unpack(rng.normal(size=dimensions(T,K)[1]),T,K)
    m,H=moments(chain,reduced)
    mf,Hf=moments(chain,full)
    np.testing.assert_allclose(lift@m,mf,atol=3e-14)
    np.testing.assert_allclose(lift@H@lift.T,Hf,atol=3e-14)


def test_T2_generic_additive_rewards_and_deterministic_augmentation():
    rng=np.random.default_rng(190)
    chain=unpack(rng.normal(size=dimensions(5,2)[1]),5,2)
    node=rng.normal(size=(5,2,3))
    m,H=moments(chain,node)
    em,eH=enumerate_moments(chain,node)
    np.testing.assert_allclose(m,em,atol=3e-14)
    np.testing.assert_allclose(H,eH,atol=3e-14)
    s=np.array([3.,-1.])
    ma,Ha=augment_moments(m,H,s)
    np.testing.assert_array_equal(ma[-2:],s)
    np.testing.assert_array_equal(Ha[-2:,-2:],np.outer(s,s))


def test_T9_m2_requires_four_timepoints():
    with pytest.raises(ValueError,match="T>=4"): m2_increments(3,2)
