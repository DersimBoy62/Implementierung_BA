"""Fully joint Bayesian HMM with augmented ring-level quality likelihood.

Quality: y_r | h_r, s_r, beta, precision ~ N(beta^T [h_r,s_r], precision^-1)
s_r is a deterministic pre-defined summary of logger X_r, standardized on TRAIN X only.
The h moments [E h, E hhT] remain EXACT; augmented moments are analytic.
Fully Bayesian q(beta,precision), local FULL_JOINT Markov optimization and unlabeled X train.
No Y of unlabeled, validation or outer-test rings used for training.
This is a *new hypothesis*, not claimed to be existing Herleitung A-D.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import vb_b1 as base
import fullcov_b1 as full
import full_markov_b1 as fm


def augment(m,H,s):
    m=np.asarray(m,float);H=np.asarray(H,float);s=np.asarray(s,float)
    if H.shape!=(len(m),len(m)) or s.ndim!=1:raise ValueError('Moments shapes')
    mm=np.r_[m,s]
    HH=np.block([[H, np.outer(m,s)],[np.outer(s,m),np.outer(s,s)]])
    return mm,(HH+HH.T)/2


def compressed_coefficients(y,q,s,K):
    """Quality-dependent local objective = const + b_h*m_h + trace(G_h*H_h).
    Uses exact expected quality likelihood including coefficient uncertainty.
    """
    b,G,const=fm.quality_coeff(y,q)
    if len(b)!=K+len(s):raise ValueError('Quality coefficient dimensions')
    bh=b[:K]+2*G[:K,K:]@s
    Gh=G[:K,:K]
    c=const+b[K:]@s+s@G[K:,K:]@s
    return bh,Gh,float(c)


def local_joint_aug(x,p,q,y,s,steps=8):
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=full.elog_emission_full(x,p)
    K=len(lp)
    initfb=base.forward_backward(lp,la,ell)
    v0=fm.init_logits_from_fb(initfb)
    b,G,const=compressed_coefficients(y,q,s,K)
    def fun(v):
        q0,Q=fm.unpack_logits(v,len(x),K)
        val,d0,dq,m,H=fm._markov_objective_and_grad(q0,Q,lp,la,ell,b,G,const)
        d0=q0*(d0-np.dot(d0,q0))
        dq=Q*(dq-np.sum(dq*Q,axis=2,keepdims=True))
        return float(val),np.r_[d0,dq.ravel()]
    start,_=fun(v0)
    def f(v):
        val,grad=fun(v)
        return -val,-grad
    res=minimize(f,v0,method='L-BFGS-B',jac=True,options={'maxiter':steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
    final,_=fun(res.x)
    accepted=bool(np.isfinite(final) and final>start+1e-10)
    q0,Q=fm.unpack_logits(res.x if accepted else v0,len(x),K)
    fb=fm.markov_marginals(q0,Q)
    mh,Hh=base.moments_occupancy(fb)
    m,H=augment(mh,Hh,s)
    return {'fb':fb,'m':m,'H':H,'m_h':mh,'H_h':Hh,
            'local_gain':float(max(0,final-start)) if np.isfinite(final) else 0,
            'iterations':int(res.nit),'accepted':accepted}


def train(rings,ydata,sdata,K=3,C=1.,epochs=10,seed=20261111,steps=8,
          quality_physics_prior_var=0.25,quality_state_prior_var=2.,verbose=False):
    if not ydata or not set(ydata).issubset(rings):raise ValueError('Labeled mismatch')
    if set(rings)!=set(sdata):raise ValueError('Every ring must have precomputed s')
    d=next(iter(rings.values())).shape[1];pdim=len(next(iter(sdata.values())))
    if pdim<1 or K<2 or epochs<1 or quality_physics_prior_var<=0:raise ValueError('Invalid configuration')
    if any(len(s)!=pdim for s in sdata.values()):raise ValueError('Inconsistent s')
    prior=full.make_priors_full(K,d)
    M=2;D=K+pdim
    qV=np.diag(np.r_[np.full(K,quality_state_prior_var),np.full(pdim,quality_physics_prior_var)])
    prior=replace(prior,qmu=np.zeros((M,D)),qV=np.stack([qV.copy() for _ in range(M)]))
    sample=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    init=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sample).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),np.array(init),np.full(K,4.),prior.nu.copy(),prior.W.copy())
    q=base.QualityPosterior(prior.qmu.copy(),prior.qV.copy(),prior.qa.copy(),prior.qb.copy())
    weights={rid:1. if rid in ydata else C for rid in rings}
    history=[]
    for epoch in range(epochs):
        loc={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                loc[rid]=local_joint_aug(x,p,q,ydata[rid],sdata[rid],steps)
            else:
                l=full.infer_process_full(x,p)
                m,H=augment(l['m'],l['H'],sdata[rid]);l.update({'m':m,'H':H})
                loc[rid]=l
        p=full.update_process_full(rings,loc,prior,weights)
        q=base.update_quality(ydata,loc,prior)
        total=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            l=loc[rid];fb=l['fb'];g=fb.gamma
            entropy=-np.sum(g[0]*np.log(np.maximum(g[0],1e-300)))
            cond=fb.xi/np.maximum(g[:-1,:,None],1e-300)
            entropy-=np.sum(fb.xi*np.log(np.maximum(cond,1e-300)))
            proc=float(g[0]@base.elog_dir(p.pi)+np.sum(fb.xi*base.elog_dir(p.A)[None,:,:])
                       +np.sum(g*full.elog_emission_full(x,p))+entropy)
            if rid in ydata:total+=proc+base.quality_logterm(ydata[rid],l['m'],l['H'],q)
            else:total+=C*proc
        kl=full.global_kl_full(p,q,prior)
        if not np.isfinite(total-kl):raise ArithmeticError('Nonfinite ELBO')
        history.append({'epoch':epoch+1,'elbo':float(total-kl),
                        'states_effective':(sum(loc[r]['fb'].gamma.sum(0) for r in loc)/
                                            sum(len(rings[r]) for r in loc)).tolist(),
                        'total_local_gain':float(sum(l.get('local_gain',0.) for l in loc.values()))})
        if verbose:print('epoch',epoch+1,'ELBO',round(total-kl,3),flush=True)
    return {'process':p,'quality':q,'prior':prior,'history':history}


def predict(x,s,model):
    l=full.infer_process_full(x,model['process'])
    m,H=augment(l['m'],l['H'],s);q=model['quality']
    pred=q.mu@m
    sd=[]
    for j in range(2):
        variance=(q.b[j]/(q.a[j]-1))*(1+np.trace(q.V[j]@H))
        variance+=max(0.,q.mu[j]@H@q.mu[j]-(q.mu[j]@m)**2)
        sd.append(np.sqrt(max(variance,1e-12)))
    return pred,np.array(sd)
