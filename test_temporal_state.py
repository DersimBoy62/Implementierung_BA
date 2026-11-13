import numpy as np
from audit_temporal_state import temporal_state_covariances
r=np.random.default_rng(20261111)


def test_shared_linear_trend_disappears():
    n=600;t=np.linspace(0,60,n)
    a=t+0.9*r.normal(size=n)
    b=-2*t+0.9*r.normal(size=n)
    X={'R1':np.stack([a,b],axis=1)}
    z=temporal_state_covariances(X,{'R1':t},{'R1':np.ones(n)})
    assert z['corr'][0][0,1]<-0.99
    assert abs(z['corr'][1][0,1])<.12
    print('PASS independent_noise_after_linear_trend')


def test_quadratic_drift():
    n=800;t=np.linspace(0,15,n);u=(t/t[-1]*2-1)
    a=10*u*u+r.normal(size=n)
    b=-7*u*u+r.normal(size=n)
    z=temporal_state_covariances({'R1':np.c_[a,b]},{'R1':t},{'R1':np.ones(n)})
    assert z['corr'][1][0,1] < -.7
    assert abs(z['corr'][2][0,1]) < .12
    print('PASS quadratic_detrend_removes_quadratic_drift')


def test_correlated_innovations_remain():
    n=1200;t=np.linspace(0,100,n)
    u=r.normal(size=n);v=.85*u+np.sqrt(1-.85**2)*r.normal(size=n)
    X={'R1':np.c_[t/3+u,-t/2+v]}
    z=temporal_state_covariances(X,{'R1':t},{'R1':np.ones(n)})
    assert z['corr'][1][0,1]>.79 and z['corr'][2][0,1]>.79
    print('PASS correlated_innovations_remain')


def test_fractional_weights_and_fixed_ring_selection():
    n=200;t=np.linspace(0,10,n)
    X={'A':np.c_[t+.01*r.normal(size=n),r.normal(size=n)],'B':np.c_[t+.01*r.normal(size=n),r.normal(size=n)]}
    z=temporal_state_covariances(X,{'A':t,'B':t},
       {'A':np.full(n,0.8),'B':np.full(n,0.01)},min_ring_mass=10)
    assert z['included_rings']==['A'] and z['excluded_rings']==['B']
    assert z['coverage']<1 and all(np.isfinite(z['corr'][deg]).all() for deg in (0,1,2))
    print('PASS fractional_weights_and_common_ring_support')


if __name__=='__main__':
    test_shared_linear_trend_disappears()
    test_quadratic_drift()
    test_correlated_innovations_remain()
    test_fractional_weights_and_fixed_ring_selection()
