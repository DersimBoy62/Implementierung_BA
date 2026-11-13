"""Kleine rechnerische Tests, nur synthetische Daten, keine Ringqualitaeten."""
import numpy as np
from diagnose_emissions import weighted_covariance,weighted_moments_1d

def main():
    X=np.array([[0.,0.],[1.,2.],[2.,4.],[3.,6.]])
    mu,cov,corr,sd=weighted_covariance(X,np.ones(4))
    assert np.allclose(mu,[1.5,3.])
    assert np.isclose(corr[0,1],1.)
    assert np.allclose(cov,[[1.25,2.5],[2.5,5.]])
    print('PASS identical_linear_correlation')
    w=np.array([1.,0.,1.,0.])
    mu,cov,corr,sd=weighted_covariance(X,w)
    assert np.allclose(mu,[1.,2.])
    assert np.isclose(corr[0,1],1.)
    print('PASS weighted_state_correlation')
    m,std,skew,kurt=weighted_moments_1d(np.array([-1.,1.]),np.array([2.,2.]))
    assert np.isclose(m,0.) and np.isclose(std,1.) and np.isclose(skew,0.) and np.isclose(kurt,-2.)
    print('PASS skew_and_excess_kurtosis')
    X2=np.column_stack([np.arange(4.),np.array([2.,-2.,2.,-2.])])
    _,_,corr,_=weighted_covariance(X2,np.ones(4))
    assert np.isfinite(corr).all()
    print('PASS finite_correlations')

if __name__=='__main__':main()
