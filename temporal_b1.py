"""Experimenteller voll-Bayes-HMM mit zeitabhaengigem, zustandsbedingtem Emissionsmittelwert.

X_{rt}|z=k,tau ~ Normal(phi(tau)^T B_k, Lambda_k^-1)
Lambda_k ~ Wishart(W0,nu0), B_k|Lambda_k ~ MatrixNormal(B0, R0^-1, Lambda_k^-1).
Zeitbasis phi=(1,tau,tau^2) bis Grad 2, tau aus vollstaendig beobachteter Ringsequenz.

Batch-end-quality ONLY; tau ist nicht fuer Online-Fruehprognosen gedacht.
U-Ringe tragen mit Faktor C nur zur Prozess-ELBO bei; ringweite zwei Y-Labels
und q(Z)-FULL_JOINT verwenden die unveraenderte vorhandene B1-Implementierung.
Keine Aenderung an bestehenden Dateimodulen.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.special import digamma,gammaln,logsumexp
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import vb_b1 as base
import full_markov_b1 as fm
import fullcov_b1_df as fc

@dataclass
class TemporalProc:
    pi:np.ndarray
    A:np.ndarray
    B:np.ndarray                 # K,p,d
    R:np.ndarray                 # K,p,p, row precision
    nu:np.ndarray
    W:np.ndarray                 # K,d,d Wishart precision scale

@dataclass
class TemporalPrior:
    pi:np.ndarray
    A:np.ndarray
    B:np.ndarray
    R:np.ndarray
    nu:np.ndarray
    W:np.ndarray
    qmu:np.ndarray
    qV:np.ndarray
    qa:np.ndarray
    qb:np.ndarray

def basis(t,degree):
    t=np.asarray(t,float)
    if degree not in (0,1,2):raise ValueError('degree muss 0,1,2 sein')
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('Ungueltige, nicht streng steigende Ring-Zeitachse')
    tau=2*(t-t[0])/(t[-1]-t[0])-1
    return np.stack([tau**j for j in range(degree+1)],axis=1)

def make_priors(K,d,degree,prior_df_offset=3,prior_precision=1.,intercept_kappa=.2,slope_kappa=1.,quality_prior_scale=2.,transition_diag_boost=4.):
    if degree not in (0,1,2) or K<2 or d<1 or prior_df_offset<2:
        raise ValueError('Ungueltige Struktur/Prior df')
    if min(prior_precision,intercept_kappa,slope_kappa,quality_prior_scale,transition_diag_boost)<=0:
        raise ValueError('Prior muss positiv sein')
    p=degree+1
    q=base.make_priors(K,d,M=2,transition_diag_boost=transition_diag_boost,
                       quality_prior_scale=quality_prior_scale,emission_kappa=intercept_kappa)
    nu=np.full(K,float(d+prior_df_offset));W=np.tile(np.eye(d)*prior_precision/nu[0],(K,1,1))
    R=np.tile(np.diag([intercept_kappa]+[slope_kappa]*degree),(K,1,1))
    return TemporalPrior(q.pi,q.A,np.zeros((K,p,d)),R,nu,W,q.qmu,q.qV,q.qa,q.qb)

def elog_emission(x,Phi,p):
    x=np.asarray(x,float);Phi=np.asarray(Phi,float)
    T,d=x.shape;K=len(p.nu)
    if Phi.shape!=(T,p.B.shape[1]) or p.B.shape[2]!=d:raise ValueError('Zeitbasis/X Dimension')
    out=np.empty((T,K))
    for k in range(K):
        diff=x-Phi@p.B[k]
        quad=np.einsum('ni,ij,nj->n',diff,p.W[k],diff,optimize=True)
        h=np.einsum('ni,ij,nj->n',Phi,np.linalg.inv(p.R[k]),Phi,optimize=True)
        out[:,k]=.5*(fc.expected_log_precision(p.nu[k],p.W[k])-d*np.log(2*np.pi)
                     -d*h-p.nu[k]*quad)
    return out

def update_process(rings,phis,local,prior,weights):
    K,p,d=prior.B.shape
    a_pi=prior.pi.copy();a_A=prior.A.copy()
    R=prior.R.copy();Tmat=np.einsum('kij,kjl->kil',prior.R,prior.B)
    XX=np.zeros((K,d,d));n=np.zeros(K)
    for rid,x in rings.items():
        weight=float(weights.get(rid,0.))
        if weight<=0:continue
        phi=phis[rid];g=local[rid]['fb'].gamma*weight
        a_pi+=weight*local[rid]['fb'].gamma[0]
        a_A+=weight*local[rid]['fb'].xi.sum(0)
        n+=g.sum(0)
        for k in range(K):
            R[k]+=(phi.T*g[:,k])@phi
            Tmat[k]+=(phi.T*g[:,k])@x
            XX[k]+=(x.T*g[:,k])@x
    B=np.empty((K,p,d));W=np.empty((K,d,d))
    for k in range(K):
        B[k]=np.linalg.solve(R[k],Tmat[k])
        Winv=np.linalg.inv(prior.W[k])+XX[k]+prior.B[k].T@prior.R[k]@prior.B[k]-B[k].T@R[k]@B[k]
        Winv=(Winv+Winv.T)/2
        if np.linalg.eigvalsh(Winv)[0]<=0:raise ArithmeticError('Full-Wishart Skala nicht positiv definit')
        W[k]=np.linalg.inv(Winv)
    return TemporalProc(a_pi,a_A,B,R,prior.nu+n,W)

def kl_mnw(B,R,nu,W,B0,R0,nu0,W0):
    d=B.shape[1];p=B.shape[0]
    # Wishart KL als validierter Bestandteil aus der vorherigen Implementation.
    kw=fc.kl_normal_wishart(np.zeros(d),1.,nu,W,np.zeros(d),1.,nu0,W0)
    # KL konditional MatrixNormal | Lambda: E_q[tr(Lambda D^T R0 D)].
    sign,ld=np.linalg.slogdet(R); sign0,ld0=np.linalg.slogdet(R0)
    if sign<=0 or sign0<=0:raise ArithmeticError('R nicht PD')
    D=B-B0
    km=.5*(d*(np.trace(np.linalg.solve(R,R0))-p+ld-ld0)
           +nu*np.trace(W@D.T@R0@D))
    return float(kw+km)

def global_kl(p,q,prior):
    val=base.kl_dir(p.pi,prior.pi)+sum(base.kl_dir(a,b) for a,b in zip(p.A,prior.A))
    for k in range(len(p.nu)):
        val+=kl_mnw(p.B[k],p.R[k],p.nu[k],p.W[k],prior.B[k],prior.R[k],prior.nu[k],prior.W[k])
    for i in range(len(q.a)):
        val+=base.kl_reg(q.mu[i],q.V[i],q.a[i],q.b[i],prior.qmu[i],prior.qV[i],prior.qa[i],prior.qb[i])
    return float(val)

def infer(x,Phi,p):
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=elog_emission(x,Phi,p)
    fb=base.forward_backward(lp,la,ell);m,H=base.moments_occupancy(fb)
    return {'fb':fb,'m':m,'H':H}

def local_joint(x,Phi,p,q,y,local_steps=8):
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=elog_emission(x,Phi,p)
    fb=base.forward_backward(lp,la,ell)
    init=fm.init_logits_from_fb(fb)
    v0,_=fm.objective_and_grad(init,lp,la,ell,y,q)
    def fg(v):
        vobj,grad=fm.objective_and_grad(v,lp,la,ell,y,q)
        return -vobj,-grad
    res=minimize(fg,init,method='L-BFGS-B',jac=True,options={'maxiter':local_steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
    v1,_=fm.objective_and_grad(res.x,lp,la,ell,y,q)
    use=bool(np.isfinite(v1) and v1>v0+1e-10)
    q0,Q=fm.unpack_logits(res.x if use else init,len(x),len(lp))
    fb=fm.markov_marginals(q0,Q)
    m,H=base.moments_occupancy(fb)
    return {'fb':fb,'m':m,'H':H,'full_local_gain':max(0.,v1-v0) if np.isfinite(v1) else 0.}

def train(rings,times,ydata,degree=1,K=3,C=1.,epochs=10,seed=20261111,local_steps=8,
          prior_options=None,verbose=False):
    if K<2 or C<0 or epochs<1 or local_steps<0:raise ValueError('Parameter')
    if not ydata or not set(ydata).issubset(rings) or set(rings)!=set(times):
        raise ValueError('Ungueltige Y- oder Zeitzuordnung')
    d=next(iter(rings.values())).shape[1];Phi={r:basis(times[r],degree) for r in rings}
    if any(rings[r].shape!=(len(Phi[r]),d) for r in rings):raise ValueError('Feature shape mismatch')
    prior=make_priors(K,d,degree,**(prior_options or {}))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centers=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    B=prior.B.copy();B[:,0,:]=centers
    initR=prior.R.copy();initR[:,0,0]=4.
    p=TemporalProc(prior.pi.copy(),prior.A.copy(),B,initR,prior.nu.copy(),prior.W.copy())
    q=base.QualityPosterior(prior.qmu.copy(),prior.qV.copy(),prior.qa.copy(),prior.qb.copy())
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    history=[]
    for epoch in range(epochs):
        local={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            local[rid]=(local_joint(x,Phi[rid],p,q,ydata[rid],local_steps)
                        if rid in ydata else infer(x,Phi[rid],p))
        p=update_process({rid:rings[rid] for rid in local},Phi,local,prior,weights)
        q=base.update_quality(ydata,local,prior)
        terms=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            fb=local[rid]['fb'];g=fb.gamma
            ent=-np.sum(g[0]*np.log(np.maximum(g[0],1e-300)))
            cond=fb.xi/np.maximum(g[:-1,:,None],1e-300)
            ent-=np.sum(fb.xi*np.log(np.maximum(cond,1e-300)))
            proc=float(g[0]@base.elog_dir(p.pi)+np.sum(fb.xi*base.elog_dir(p.A)[None,:,:])
                       +np.sum(g*elog_emission(x,Phi[rid],p))+ent)
            if rid in ydata:terms+=proc+base.quality_logterm(ydata[rid],local[rid]['m'],local[rid]['H'],q)
            else:terms+=C*proc
        hist={'epoch':epoch+1,'elbo':float(terms-global_kl(p,q,prior)),
              'states_effective':(sum(local[r]['fb'].gamma.sum(0) for r in local)/sum(len(rings[r]) for r in local)).tolist(),
              'sum_local_gains':float(sum(v.get('full_local_gain',0.) for v in local.values()))}
        history.append(hist)
        if verbose:print('epoch',epoch+1,'ELBO',round(hist['elbo'],2),'states',np.round(hist['states_effective'],3),flush=True)
    return {'process':p,'quality':q,'prior':prior,'local':local,'history':history,'degree':degree}

def predict(x,t,p,q):
    phi=basis(t,p.B.shape[1]-1)
    l=infer(x,phi,p);m,H=l['m'],l['H']
    mean=q.mu@m;var=[]
    for r in range(len(q.a)):
        sigma=q.b[r]/(q.a[r]-1.)
        vh=max(0.,q.mu[r]@H@q.mu[r]-(q.mu[r]@m)**2)
        var.append(sigma*(1.+np.trace(q.V[r]@H))+vh)
    return mean,np.sqrt(np.asarray(var)),m,H

def predictive_logx(x,t,p):
    # Exact Student-t-Emission pro (zeitabhängiger) phi, plug-in Dirichlet Posterior-Mittel
    Phi=basis(t,p.B.shape[1]-1)
    T,d=x.shape;K=len(p.nu);ell=np.empty((T,K))
    for k in range(K):
        df=p.nu[k]-d+1.
        h=np.einsum('ni,ij,nj->n',Phi,np.linalg.inv(p.R[k]),Phi)
        prec=np.array([df/(1.+hi)*p.W[k] for hi in h])
        ldprec=d*np.log(df)-d*np.log1p(h)+np.linalg.slogdet(p.W[k])[1]
        delta=x-Phi@p.B[k]
        mahal=np.einsum('ni,nij,nj->n',delta,prec,delta,optimize=True)
        ell[:,k]=gammaln((df+d)/2.)-gammaln(df/2.)+.5*ldprec-.5*d*np.log(df*np.pi)\
                 -.5*(df+d)*np.log1p(mahal/df)
    logpi=np.log(p.pi/p.pi.sum());logA=np.log(p.A/p.A.sum(1,keepdims=True))
    cur=logpi+ell[0]
    for it in range(1,T):cur=logsumexp(cur[:,None]+logA,axis=0)+ell[it]
    return float(logsumexp(cur))
