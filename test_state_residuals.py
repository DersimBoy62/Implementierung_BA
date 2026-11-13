import numpy as np
from audit_state_residuals import state_covariances
rng=np.random.default_rng(121)

def test_ring_offset_artifact():
    # Strong anticorrelation is entirely due to offsets of the two rings.
    n=1000
    u=rng.normal(size=(n,2))
    v=rng.normal(size=(n,2))
    X={'A':u+np.array([10.,-10.]),'B':v+np.array([-10.,10.])}
    W={key:np.ones(n) for key in X}
    s=state_covariances(X,W)
    assert s['pooled_corr'][0,1]<-.95
    assert abs(s['within_corr'][0,1])<.1
    print('PASS between_ring_offset_vs_within_ring_correlation')

def test_true_within_ring_corr():
    a=rng.normal(size=2000)
    b=rng.normal(size=2000)
    X={'A':np.stack([a,2*a+.1*b],axis=1)}
    s=state_covariances(X,{'A':np.ones(2000)})
    assert s['within_corr'][0,1]>.99
    assert np.allclose(s['within_corr'],s['pooled_corr'])
    print('PASS true_within_ring_corr_preserved')

def test_unequal_state_weights():
    X={'A':np.array([[0.,0.],[1.,2.],[2.,4.],[3.,6.]]),
       'B':np.array([[100.,-100.],[99.,-98.]])}
    s=state_covariances(X,{'A':np.array([1.,.9,.8,.7]),'B':np.array([.01,.01])})
    assert np.isfinite(s['within_corr']).all() and s['mass']>3
    assert s['within_corr'][0,1]>.99
    print('PASS fractional_state_weights')

if __name__=='__main__':
    test_ring_offset_artifact()
    test_true_within_ring_corr()
    test_unequal_state_weights()
