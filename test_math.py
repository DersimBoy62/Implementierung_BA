"""Mathematische Validierung für Herleitungen B1/B/C vor dem echten Training."""
import itertools, numpy as np
from scipy.special import logsumexp
from vb_b1 import (forward_backward,moments_occupancy,make_priors,
                   ProcPosterior,QualityPosterior,local_inference,quality_logterm,
                   update_quality,update_process,kl_dir,kl_nig,kl_reg)


def test_exact_path_enumeration():
    """Alle K^T Zustandsfolgen: FB, E[h] und E[hh.T] muss auf 1e-10 stimmen."""
    K,T=3,4
    logpi=np.log([.2,.5,.3]);loga=np.log([[.8,.1,.1],[.25,.6,.15],[.1,.35,.55]])
    ell=np.array([[.1,-.2,.6],[-1,.5,0],[.25,-.6,.8],[.2,.2,-.4]])
    paths=list(itertools.product(range(K),repeat=T))
    lps=[]
    for z in paths:
        lps.append(logpi[z[0]]+ell[0,z[0]]+sum(loga[z[t-1],z[t]]+ell[t,z[t]] for t in range(1,T)))
    prob=np.exp(np.asarray(lps)-logsumexp(lps))
    h=np.asarray([np.bincount(z,minlength=K)/T for z in paths])
    m_true=prob@h
    H_true=np.einsum('p,pi,pj->ij',prob,h,h)
    f=forward_backward(logpi,loga,ell)
    m,H=moments_occupancy(f)
    np.testing.assert_allclose(f.logz,logsumexp(lps),atol=1e-10)
    np.testing.assert_allclose(m,m_true,atol=1e-10)
    np.testing.assert_allclose(H,H_true,atol=1e-10)
    assert np.linalg.eigvalsh(H-m[:,None]*m[None,:]).min()>-1e-10


def test_quality_uses_second_moment():
    pri=make_priors(2,1)
    gamma=np.array([[.5,.5],[.5,.5]])
    fb=forward_backward(np.log([.5,.5]),np.log([[.99,.01],[.01,.99]]),np.zeros((2,2)))
    m,H=moments_occupancy(fb)
    q=QualityPosterior(np.array([[2.,-2.],[1.,-1.]]),np.stack([np.eye(2)*.1]*2),np.ones(2)*3,np.ones(2))
    y=np.array([1.,.2])
    objective_real=quality_logterm(y,m,H,q)
    objective_fake=quality_logterm(y,m,np.outer(m,m),q)
    assert abs(objective_real-objective_fake)>0.01


def test_full_variational_kl_zero_and_positive():
    p=make_priors(2,1)
    np.testing.assert_allclose(kl_dir(p.pi,p.pi),0,atol=1e-12)
    np.testing.assert_allclose(kl_nig(p.mu,p.kappa,p.a,p.b,p.mu,p.kappa,p.a,p.b),0,atol=1e-12)
    np.testing.assert_allclose(kl_reg(p.qmu[0],p.qV[0],p.qa[0],p.qb[0],p.qmu[0],p.qV[0],p.qa[0],p.qb[0]),0,atol=1e-12)
    assert kl_dir(np.array([8.,1.]),np.ones(2))>0


def test_quality_feedback_moves_q():
    K=2;pri=make_priors(K,1)
    p=ProcPosterior(np.array([5.,5.]),np.array([[3.,3.],[3.,3.]]),np.zeros((2,1)),np.full((2,1),10.),np.full((2,1),3.),np.full((2,1),2.))
    q=QualityPosterior(np.array([[5.,-5.],[0.,0.]]),np.stack([np.eye(2)*.05]*2),np.array([20.,20.]),np.array([.4,.4]))
    x=np.zeros((10,1));y=np.array([4.5,0])
    a=local_inference(x,p,mode='OFF')
    b=local_inference(x,p,q,y,mode='TILT_JOINT',max_iter=25)
    assert b['local_objective']>local_inference(x,p,q,y,mode='OFF')['local_objective']+0.01
    assert b['m'][0]>a['m'][0]


def test_zero_C_eliminates_unlabeled_global_contributions():
    K,d=2,1;pri=make_priors(K,d)
    p=ProcPosterior(pri.pi.copy(),pri.A.copy(),pri.mu.copy(),pri.kappa.copy(),pri.a.copy(),pri.b.copy())
    x=np.zeros((4,1));l=local_inference(x,p)
    rng={'L':x,'U':x*3+100};loc={'L':l,'U':l}
    out=update_process(rng,loc,pri,{'L':1.0,'U':0.0})
    out2=update_process({'L':x},{'L':l},pri,{'L':1.0})
    np.testing.assert_allclose(out.mu,out2.mu)
    np.testing.assert_allclose(out.A,out2.A)


def run():
    functions=[test_exact_path_enumeration,test_quality_uses_second_moment,
               test_full_variational_kl_zero_and_positive,test_quality_feedback_moves_q,
               test_zero_C_eliminates_unlabeled_global_contributions]
    for fn in functions:
        fn();print('PASS',fn.__name__)
    return len(functions)

if __name__=='__main__':run()
