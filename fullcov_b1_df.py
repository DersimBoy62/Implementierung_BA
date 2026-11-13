"""Separater Kontrollzweig zur Prior-KONZENTRATION: full-covariance Normal-Wishart-Emissionen.

Zusatz zu den bestehenden Dateien vb_b1.py, full_markov_b1.py, dm_data.py.
Keine Änderung der bisherigen diagonalen Modellimplementation.
Die Qualität bleibt eine 2D ringweite Regression über (m,H) der Markov-Zustandsanteile.

Für den Prozessanteil gilt q(mu_k,Lambda_k) = N(mu_k|m_k,(kappa_k Lambda_k)^-1)
* Wishart(Lambda_k|W_k,nu_k); W ist die SKALENmatrix der Precision-Wishart.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma, gammaln, multigammaln
import vb_b1 as base

@dataclass
class FullProc:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    nu: np.ndarray
    W: np.ndarray

@dataclass
class FullPriors:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    nu: np.ndarray
    W: np.ndarray
    qmu: np.ndarray
    qV: np.ndarray
    qa: np.ndarray
    qb: np.ndarray


def make_priors_full(K,d,transition_diag_boost=4., emission_kappa=.2, quality_prior_scale=2., prior_precision=1., prior_df_offset=3):
    if min(K,d,transition_diag_boost,emission_kappa,quality_prior_scale,prior_precision)<=0:
        raise ValueError('Priorwerte müssen positiv sein')
    if not isinstance(prior_df_offset, int) or prior_df_offset < 2:
        raise ValueError('prior_df_offset muss eine ganze Zahl >= 2 sein, damit E[Sigma] existiert')
    nu0=float(d+prior_df_offset)
    nu=np.full(K,nu0)
    # E_prior[Lambda_k] = nu0 * W0 = prior_precision * I, konstant bei variierendem nu0.
    W=np.array([np.eye(d)*prior_precision/nu0 for _ in range(K)])
    q=base.make_priors(K,d,M=2,transition_diag_boost=transition_diag_boost,
                      emission_kappa=emission_kappa, quality_prior_scale=quality_prior_scale)
    return FullPriors(q.pi,q.A,np.zeros((K,d)),np.full(K,emission_kappa),nu,W,
                      q.qmu,q.qV,q.qa,q.qb)


def expected_log_precision(nu,W):
    d=W.shape[0]
    sign, ld=np.linalg.slogdet(W)
    if sign<=0: raise ArithmeticError('Wishart W nicht positiv definit')
    return (sum(digamma((nu-j)/2.) for j in range(d)) + d*np.log(2.) + ld)


def elog_emission_full(x,p):
    """E_q log N(x|mu,Lambda^-1); einschließlich Unsicherheit von mu und Lambda."""
    x=np.asarray(x,float)
    T,d=x.shape
    if p.mu.shape[1]!=d: raise ValueError('falsche Dimension')
    out=np.empty((T,len(p.nu)))
    for k in range(len(p.nu)):
        diff=x-p.mu[k]
        quad=np.einsum('ni,ij,nj->n',diff,p.W[k],diff,optimize=True)
        out[:,k]=.5*(expected_log_precision(p.nu[k],p.W[k])-d*np.log(2*np.pi)
                     -d/p.kappa[k]-p.nu[k]*quad)
    return out


def update_process_full(rings,local,prior,weights):
    K,d=prior.mu.shape
    a_pi=prior.pi.copy();a_A=prior.A.copy()
    n=np.zeros(K); sums=np.zeros((K,d)); xx=np.zeros((K,d,d))
    for rid,x in rings.items():
        w=float(weights.get(rid,0))
        if not w: continue
        fb=local[rid]['fb'];g=fb.gamma
        a_pi+=w*g[0];a_A+=w*fb.xi.sum(0)
        n+=w*g.sum(0)
        sums+=w*g.T@x
        for k in range(K):
            xx[k]+=w*(x.T*g[:,k])@x
    kappa=prior.kappa+n
    mu=(prior.kappa[:,None]*prior.mu+sums)/kappa[:,None]
    nu=prior.nu+n
    W=np.empty((K,d,d))
    for k in range(K):
        Winv=np.linalg.inv(prior.W[k])+xx[k]+prior.kappa[k]*np.outer(prior.mu[k],prior.mu[k])\
             -kappa[k]*np.outer(mu[k],mu[k])
        Winv=(Winv+Winv.T)/2.
        if np.linalg.eigvalsh(Winv)[0] <= 0:raise ArithmeticError('Inverse Wishart-Skala nicht PD')
        W[k]=np.linalg.inv(Winv)
    return FullProc(a_pi,a_A,mu,kappa,nu,W)


def kl_normal_wishart(m,k,nu,W,m0,k0,nu0,W0):
    """KL[q(mu,Lambda)||p(mu,Lambda)], Wishart precision with scale W."""
    d=len(m)
    e_log=expected_log_precision(nu,W)
    sign,ld=np.linalg.slogdet(W); sign0,ld0=np.linalg.slogdet(W0)
    if sign<=0 or sign0<=0:raise ArithmeticError('Matrix nicht positiv definit')
    klw=.5*(nu-nu0)*e_log-.5*(nu*d-nu*np.trace(np.linalg.solve(W0,W)))\
        -.5*(nu-nu0)*d*np.log(2.)-.5*nu*ld+.5*nu0*ld0\
        -multigammaln(nu/2.,d)+multigammaln(nu0/2.,d)
    delta=m-m0
    kln=.5*(d*(np.log(k/k0)+k0/k-1.)+k0*nu*(delta@W@delta))
    return float(klw+kln)


def global_kl_full(p,q,prior):
    result=base.kl_dir(p.pi,prior.pi)+sum(base.kl_dir(a,b) for a,b in zip(p.A,prior.A))
    for k in range(len(p.nu)):
        result+=kl_normal_wishart(p.mu[k],p.kappa[k],p.nu[k],p.W[k],
                                 prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k])
    for r in range(len(q.a)):
        result+=base.kl_reg(q.mu[r],q.V[r],q.a[r],q.b[r],
                            prior.qmu[r],prior.qV[r],prior.qa[r],prior.qb[r])
    return float(result)


def infer_process_full(x,p):
    fb=base.forward_backward(base.elog_dir(p.pi),base.elog_dir(p.A),elog_emission_full(x,p))
    m,H=base.moments_occupancy(fb)
    return {'fb':fb,'m':m,'H':H,'tilt':np.zeros(p.mu.shape[0]),'local_objective':float('nan')}


def local_full_joint_full(x,p,q,y,max_iter=8):
    """Vollständige C.7 Markov-B1-Optimierung mit Normal-Wishart-Emissionen."""
    from scipy.optimize import minimize
    import full_markov_b1 as fm
    lp=base.elog_dir(p.pi); la=base.elog_dir(p.A);ell=elog_emission_full(x,p)
    fb=base.forward_backward(lp,la,ell)
    init=fm.init_logits_from_fb(fb)
    score,_=fm.objective_and_grad(init,lp,la,ell,y,q)
    def fg(v):
        f,g=fm.objective_and_grad(v,lp,la,ell,y,q)
        return -f,-g
    res=minimize(fg,init,method='L-BFGS-B',jac=True,options={'maxiter':max_iter,'ftol':1e-10,'gtol':1e-5,'maxls':15})
    alt,_=fm.objective_and_grad(res.x,lp,la,ell,y,q)
    use=bool(np.isfinite(alt) and alt>score+1e-10)
    q0,Q=fm.unpack_logits(res.x if use else init,len(x),len(lp))
    fb=fm.markov_marginals(q0,Q)
    m,H=base.moments_occupancy(fb)
    return {'fb':fb,'m':m,'H':H,'tilt':np.zeros(len(lp)),
            'local_objective':float(alt if use else score),
            'full_local_gain':float(max(0,alt-score)) if np.isfinite(alt) else 0.,
            'full_local_iterations':int(res.nit),'full_local_optimizer_success':bool(res.success),
            'full_local_accepted':use}


def train_full(rings,ydata,K=3,C=1.,epochs=3,seed=20261111,max_local_iter=8,
               prior_options=None, verbose=False):
    if K<2 or C<0 or epochs<1:raise ValueError('Trainingsparameter')
    if not ydata or not set(ydata).issubset(rings):raise ValueError('Labels fehlen oder unzulässig')
    d=next(iter(rings.values())).shape[1]
    if any(x.shape[1]!=d for x in rings.values()):raise ValueError('Uneinheitliche d')
    prior=make_priors_full(K,d,**(prior_options or {}))
    from sklearn.cluster import KMeans
    sample=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    init=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sample).cluster_centers_
    # Wichtig: Initiale q(Lambda) hat E[Lambda] ~= I
    p=FullProc(prior.pi.copy(),prior.A.copy(),np.array(init),np.full(K,4.),
               prior.nu.copy(),prior.W.copy())
    q=base.QualityPosterior(prior.qmu.copy(),prior.qV.copy(),prior.qa.copy(),prior.qb.copy())
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    history=[]
    for epoch in range(epochs):
        local={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:local[rid]=local_full_joint_full(x,p,q,ydata[rid],max_iter=max_local_iter)
            else:local[rid]=infer_process_full(x,p)
        p=update_process_full({rid:rings[rid] for rid in local},local,prior,weights)
        q=base.update_quality(ydata,local,prior)
        terms=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            fb=local[rid]['fb'];g=fb.gamma
            entropy=-np.sum(g[0]*np.log(np.maximum(g[0],1e-300)))
            cond=fb.xi/np.maximum(g[:-1,:,None],1e-300)
            entropy-=np.sum(fb.xi*np.log(np.maximum(cond,1e-300)))
            proc=float(g[0]@base.elog_dir(p.pi)+np.sum(fb.xi*base.elog_dir(p.A)[None,:,:])
                       +np.sum(g*elog_emission_full(x,p))+entropy)
            if rid in ydata:terms+=proc+base.quality_logterm(ydata[rid],local[rid]['m'],local[rid]['H'],q)
            else:terms+=C*proc
        kl=global_kl_full(p,q,prior)
        record={'epoch':epoch+1,'elbo':float(terms-kl),'global_kl':float(kl),
                'states_effective':(sum(local[r]['fb'].gamma.sum(0) for r in local)/
                                    sum(len(rings[r]) for r in local)).tolist(),
                'full_local_gains':{r:float(local[r].get('full_local_gain',0)) for r in local}}
        history.append(record)
        if verbose:print('epoch',epoch+1,'ELBO',round(record['elbo'],3),
                         'states',np.round(record['states_effective'],3))
    return {'process':p,'quality':q,'prior':prior,'history':history,'local':local}


def predict_full(x,p,q):
    l=infer_process_full(x,p)
    m,H=l['m'],l['H']
    mean=q.mu@m
    var=[]
    for r in range(len(q.a)):
        sigma=q.b[r]/(q.a[r]-1.)
        vh=max(0.,q.mu[r]@H@q.mu[r]-(q.mu[r]@m)**2)
        var.append(sigma*(1.+np.trace(q.V[r]@H))+vh)
    return mean,np.sqrt(np.asarray(var)),m,H


def predictive_logx_full(x,p):
    """Ring-log score: Dirichlet posterior-mean transitions, exact emission Student-t.
    Diese ist eine APPROXIMIERTE posterior-prädiktive X-Dichte, nicht die ELBO.
    """
    from scipy.special import gammaln, logsumexp
    T,d=x.shape;K=len(p.nu)
    ell=np.empty((T,K))
    for k in range(K):
        v=p.nu[k]-d+1.
        prec=p.kappa[k]*v/(p.kappa[k]+1.)*p.W[k]
        sign,ldprec=np.linalg.slogdet(prec)
        if sign<=0:raise ArithmeticError('Prediktive Präzision nicht PD')
        df=x-p.mu[k]; maha=np.einsum('ni,ij,nj->n',df,prec,df,optimize=True)
        ell[:,k]=gammaln((v+d)/2.)-gammaln(v/2.)+.5*ldprec-.5*d*np.log(v*np.pi)\
                 -.5*(v+d)*np.log1p(maha/v)
    logpi=np.log(p.pi/p.pi.sum()); logA=np.log(p.A/p.A.sum(1,keepdims=True))
    cur=logpi+ell[0]
    for t in range(1,T):cur=logsumexp(cur[:,None]+logA,axis=0)+ell[t]
    return float(logsumexp(cur))


def predictive_logx_diag(x,p):
    """Entsprechende plug-in-HMM Prädiktivdichte mit NIG Student-t-Emissionsmarginalen."""
    from scipy.special import gammaln,logsumexp
    d=x.shape[1];K=p.mu.shape[0]
    ell=np.empty((len(x),K))
    for k in range(K):
        a=p.a[k];b=p.b[k];kap=p.kappa[k]
        v=2*a; s2=b*(kap+1.)/(a*kap)
        delta=(x-p.mu[k])/np.sqrt(v*s2)
        ell[:,k]=np.sum(gammaln((v+1.)/2.)-gammaln(v/2.)-.5*np.log(v*np.pi*s2)
                      -(v+1.)/2.*np.log1p(delta**2),axis=1)
    logpi=np.log(p.pi/p.pi.sum());logA=np.log(p.A/p.A.sum(1,keepdims=True))
    cur=logpi+ell[0]
    for t in range(1,len(x)):cur=logsumexp(cur[:,None]+logA,axis=0)+ell[t]
    return float(logsumexp(cur))
