"""Mathematical checks for direct quality-link comparators + fully Bayesian hybrid."""
import numpy as np
from scipy.stats import t as student_t
import quality_link_challenge as qlc
import hybrid_b1 as hybrid
import vb_b1 as b
import full_markov_b1 as fm


def test_physical_inputs_are_logger_only():
    x=np.column_stack([np.linspace(0,1,31)+j for j in range(6)])
    t=np.linspace(3.0,6.0,len(x))
    p=qlc.extract_descriptors(x,t,'physical')
    s=qlc.extract_descriptors(x,t,'stats')
    path=qlc.extract_descriptors(x,t,'path')
    assert p.shape==(12,) and s.shape==(25,) and path.shape==(37,)
    assert abs(p[5]-1)<1e-12 and abs(p[7]-1)<1e-12
    assert np.isfinite(p).all() and np.isfinite(s).all() and np.isfinite(path).all()
    print('PASS physical_logger_descriptors_and_time_basis')


def test_train_only_unsupervised_pca():
    a={f'T{i}':np.sin(np.arange(5)*i/7)+i for i in range(1,9)}
    v={'VAL1':np.ones(5)*1.e6}
    u,uu,meta=qlc.make_latent(a,v,3)
    assert set(meta['fit_ids'])==set(a) and not set(v)&set(meta['fit_ids'])
    assert abs(np.mean(np.array(list(u.values())),axis=0)).max()<1e-12
    # PCA fitted on TRAIN only, extreme validation cannot change fitted PCA center.
    a2=a.copy(); v2={'VAL1':-np.ones(5)*1e9}
    u2,_,_=qlc.make_latent(a2,v2,3)
    assert np.allclose(np.array(list(u.values())),np.array(list(u2.values())))
    print('PASS train_only_pca_no_validation_fitting')


def test_conjugate_quality_nig_and_gp():
    X=np.array([[-1.,0.],[0.,1.],[1.,0.],[2.,1.],[3.,-1.]])
    Y=np.stack([X[:,0]*.7,X[:,1]*-.3],axis=1)
    p=qlc.normal_inverse_gamma(X,Y)
    mu,sd=qlc.nig_predict(p,X)
    assert mu.shape==(len(X),2) and sd.shape==(len(X),2)
    assert np.all(sd>0) and p['a']>1 and np.all(p['b']>0)
    gm,gs,length=qlc.gp_fit_predict(X,Y,X)
    assert gm.shape==Y.shape and gs.shape==Y.shape and np.all(gs>0) and length>0
    # Exact independent GP solution is equivalent to direct NumPy solve.
    kernel=qlc.matern32_k(X,X,length)
    manual=kernel@np.linalg.solve(kernel+(.5+1e-9)*np.eye(len(X)),Y)
    assert np.allclose(gm,manual,atol=1e-9)
    assert np.isfinite(student_t.logpdf(0,df=2*p['a']))
    print('PASS exact_conjugate_nig_and_fixed_gp')


def test_augmented_quality_moments_exact():
    m=np.array([.2,.3,.5]);H=np.outer(m,m)+.02*(np.eye(3)-np.ones((3,3))/3.)
    # Use deterministic one-hot draw with g=m and verify analytic second moments.
    H=np.diag(m)
    s=np.array([-.6,1.2])
    mm,HH=hybrid.augment(m,H,s)
    assert np.allclose(mm,np.r_[m,s]);assert np.allclose(HH[:3,:3],H)
    assert np.allclose(HH[:3,3:],np.outer(m,s))
    assert np.allclose(HH[3:,3:],np.outer(s,s))
    assert np.linalg.eigvalsh(HH).min()>-1.e-12
    print('PASS exact_augmented_first_second_moments')


def test_quality_joint_compression_exact():
    K=3;s=np.array([-.2,.9])
    rng=np.random.default_rng(7)
    A=rng.normal(size=(K+len(s),K+len(s)))
    q=b.QualityPosterior(mu=rng.normal(size=(2,K+len(s))),
                         V=np.stack([A.T@A*.04+np.eye(len(A))*.05 for _ in range(2)]),
                         a=np.array([4.,5.]),b=np.array([2.,2.]))
    m=np.array([.15,.3,.55]);H=np.diag(m)*.5+.5*np.outer(m,m)
    y=np.array([.3,-.5]);mm,HH=hybrid.augment(m,H,s)
    direct=b.quality_logterm(y,mm,HH,q)
    bh,Gh,c=hybrid.compressed_coefficients(y,q,s,K)
    collapsed=c+bh@m+np.sum(Gh*H)
    assert np.isclose(direct,collapsed,rtol=0,atol=1.e-10),(direct,collapsed)
    print('PASS joint_quality_likelihood_compression_exact')


if __name__=='__main__':
    test_physical_inputs_are_logger_only()
    test_train_only_unsupervised_pca()
    test_conjugate_quality_nig_and_gp()
    test_augmented_quality_moments_exact()
    test_quality_joint_compression_exact()
