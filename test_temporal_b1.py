"""Tests des zeitabhaengigen voll-Bayes Emissionsmodells, vor Real-DM-Anwendung."""
import numpy as np
from scipy.stats import wishart
import temporal_b1 as tm
import fullcov_b1_df as fc

def check(c,msg):
    if not c:raise AssertionError(msg)

def test_basis():
    x=np.array([10.,15.,20.]);phi=tm.basis(x,2)
    check(np.allclose(phi,[[1,-1,1],[1,0,0],[1,1,1]]),'basis 0,1,2')
    for deg in (0,1,2): check(tm.basis(x,deg).shape==(3,deg+1),'degree shape')
    print('PASS physical_time_normalized_basis')

def test_kl_exact_identity_and_scalar_reduction():
    d=3;nu=d+3;W=np.eye(d)/nu
    B=np.ones((2,d))*.2;R=np.diag([.4,2.5])
    check(abs(tm.kl_mnw(B,R,nu,W,B,R,nu,W))<1e-8,'identical distributions KL0')
    check(tm.kl_mnw(B*1.7,R*2.,nu+5,W*.8,B,R,nu,W)>0,'KL positive')
    m=np.array([.2,-.1,.3]);k=3.;k0=.5
    a=fc.kl_normal_wishart(m,k,nu,W,np.zeros(d),k0,nu,W)
    b=tm.kl_mnw(m.reshape(1,d),np.array([[k]]),nu,W,np.zeros((1,d)),np.array([[k0]]),nu,W)
    check(abs(a-b)<1e-8,'matrix normal reduces to normal-wishart at degree0')
    print('PASS kl_same_positive_and_degree0_reduction')

def test_expected_emission_vs_mc():
    rng=np.random.default_rng(123);K=1;d=2;p=2
    R=np.array([[[3.0,0.3],[0.3,1.5]]]);nu=np.array([10.])
    W=np.array([[[.12,.018],[.018,.15]]]);B=np.array([[[.1,-.2],[.3,.5]]])
    m=tm.TemporalProc(np.ones(K),np.ones((K,K)),B,R,nu,W)
    Phi=np.array([[1.,-.9],[1.,.1],[1.,.7]]);x=np.array([[.1,.4],[.7,.2],[-.4,.5]])
    exact=tm.elog_emission(x,Phi,m)[:,0]
    draws=[]
    for _ in range(2000):
        Lam=wishart.rvs(df=nu[0],scale=W[0],random_state=rng)
        Bj=B[0]+np.linalg.cholesky(np.linalg.inv(R[0]))@rng.normal(size=(p,d))@np.linalg.cholesky(np.linalg.inv(Lam)).T
        err=x-Phi@Bj
        logp=.5*(np.linalg.slogdet(Lam)[1]-d*np.log(2*np.pi)-np.einsum('ti,ij,tj->t',err,Lam,err))
        draws.append(logp)
    avg=np.mean(draws,axis=0)
    check(np.max(abs(avg-exact))<.085,f'mc {avg} vs {exact}')
    print('PASS expected_emission_matches_independent_monte_carlo')

def test_posterior_update_pd_and_quality():
    rng=np.random.default_rng(42)
    t=np.linspace(0,5,28)
    X={'a':rng.normal(size=(28,2)),'b':rng.normal(size=(28,2))}
    Y={'a':np.array([.4,-.5])};T={'a':t,'b':t}
    out=tm.train(X,T,Y,degree=1,epochs=2,K=2,local_steps=1,seed=42)
    p=out['process']
    check(np.all(np.linalg.eigvalsh(p.W)>0),'W pos def')
    check(np.all(np.linalg.eigvalsh(p.R)>0),'R pos def')
    pred=tm.predict(X['b'],t,p,out['quality'])
    check(pred[0].shape==(2,) and np.isfinite(pred[0]).all(),'2 targets')
    check(np.isfinite(tm.predictive_logx(X['b'],t,p)),'predictive log score')
    print('PASS synthetic_training_pd_and_two_quality_targets')

def test_degree_zero_matches_existing():
    rng=np.random.default_rng(41);X={'a':rng.normal(size=(31,3)),'b':rng.normal(size=(29,3))}
    Y={'a':np.array([.3,-.4])};T={r:np.arange(len(x),dtype=float) for r,x in X.items()}
    a=fc.train_full(X,Y,K=2,epochs=1,seed=42,max_local_iter=1,verbose=False)
    b=tm.train(X,T,Y,degree=0,K=2,epochs=1,seed=42,local_steps=1,verbose=False)
    err=abs(a['history'][0]['elbo']-b['history'][0]['elbo'])
    check(err<1e-5,f'old/new ELBO differ degree0 {err}')
    check(np.allclose(a['process'].mu,b['process'].B[:,0,:],atol=1e-8),'old/new mu')
    print('PASS degree0_reproduces_original_full_wishart')

if __name__=='__main__':
    test_basis();test_kl_exact_identity_and_scalar_reduction();test_expected_emission_vs_mc()
    test_posterior_update_pd_and_quality();test_degree_zero_matches_existing()
