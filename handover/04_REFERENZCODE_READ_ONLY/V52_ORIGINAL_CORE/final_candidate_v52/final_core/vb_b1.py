"""Bayesian-HMM mit zwei B1-Qualitätsausgängen und strukturierter VI.

Mathematische Quellen: Herleitungen A–D. JOINT_local='tilt' optimiert in dieser
ersten ausführbaren Stufe eine **eingeschränkte** Markov-Variationsfamilie:
 q_lambda(z) proportional exp[Elog p(z,x)] exp(lambda.T h(z)).
Das ist JOINT (Qualität wirkt auf Zustandsverteilung zurück), aber NOCH NICHT
Herleitung C.7 mit sämtlichen zeitabhängigen freien Markov-Logits!

Keine alten SFA-Parameter und keine alten SFA-Splits werden verwendet.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, digamma, gammaln
LOG2PI=np.log(2*np.pi)

@dataclass
class FB:
    gamma: np.ndarray
    xi: np.ndarray
    logz: float


def forward_backward(logpi, loga, ell):
    """Stabil in Log-Space. ell[t,k] sind erwartete Emissions-Logdichten."""
    T,K=ell.shape
    f=np.empty((T,K)); b=np.zeros((T,K))
    f[0]=logpi+ell[0]
    for t in range(1,T): f[t]=ell[t]+logsumexp(f[t-1,:,None]+loga,axis=0)
    logz=float(logsumexp(f[-1]))
    for t in range(T-2,-1,-1):b[t]=logsumexp(loga+ell[t+1,None,:]+b[t+1,None,:],axis=1)
    gamma=np.exp(f+b-logz)
    xi=np.exp(f[:-1,:,None]+loga[None,:,:]+ell[1:,None,:]+b[1:,None,:]-logz)
    # Kleine Rundungsfehler in Wahrscheinlichkeiten werden toleriert, nicht maskiert.
    if not np.allclose(gamma.sum(1),1.,atol=1e-7):raise ArithmeticError('Gamma nicht normiert')
    if not np.allclose(xi.sum((1,2)),1.,atol=1e-7):raise ArithmeticError('Xi nicht normiert')
    return FB(gamma,xi,logz)


def moments_occupancy(fb):
    """E[h] und E[hh^T] EXAKT unter der Markov-Verteilung q(z).

    Konditionierte Rückwärtsrekursion: E[Anzahl | Z_t=k],
    E[Anzahl_a Anzahl_b | Z_t=k] mittels xi/gamma (Herleitung B).
    """
    g,xi=fb.gamma,fb.xi
    T,K=g.shape
    m=np.eye(K)
    s=np.array([np.outer(m[k],m[k]) for k in range(K)])
    e=np.eye(K)
    for t in range(1,T):
        rev=xi[t-1].T/np.maximum(g[t,:,None],1e-300) # [k_new,k_old]
        mm=rev@m
        ss=np.einsum('kj,jab->kab',rev,s,optimize=True)
        s=ss+e[:,:,None]*mm[:,None,:]+mm[:,:,None]*e[:,None,:]+e[:,:,None]*e[:,None,:]
        m=mm+e
    result_m=(g[-1]@m)/T
    result_H=np.einsum('k,kab->ab',g[-1],s,optimize=True)/(T*T)
    result_H=(result_H+result_H.T)/2
    if not np.isclose(result_m.sum(),1.,atol=1e-6):raise ArithmeticError('Moment 1 nicht normiert')
    if not np.isclose(result_H.sum(),1.,atol=1e-6):raise ArithmeticError('Moment 2 nicht normiert')
    if not np.allclose(result_H@np.ones(K),result_m,atol=1e-6):
        raise ArithmeticError('B1 Moment-Konsistenz verletzt')
    return result_m,result_H


def elog_dir(alpha):return digamma(alpha)-digamma(alpha.sum(axis=-1,keepdims=True))


def kl_dir(a,b):
    return float(gammaln(a.sum())-gammaln(a).sum()-gammaln(b.sum())+gammaln(b).sum()+np.sum((a-b)*(digamma(a)-digamma(a.sum()))))


def kl_nig(m,k,a,b, m0,k0,a0,b0):
    """Summierte KL-Divergenz unabhängiger 1D-NIG-Faktoren."""
    kig=(a0*np.log(b/b0)-gammaln(a)+gammaln(a0)+(a-a0)*digamma(a)+(b0-b)*a/b)
    knorm=.5*(np.log(k/k0)+k0/k-1+k0*(m-m0)**2*a/b)
    return float(np.sum(kig+knorm))


def kl_reg(m,V,a,b,m0,V0,a0,b0):
    K=len(m); p=np.linalg.inv(V0)
    sign0,ld0=np.linalg.slogdet(V0);sign,ld=np.linalg.slogdet(V)
    if sign0<=0 or sign<=0:raise ValueError('Regression-Kovarianz nicht PD')
    d=m-m0
    g=(a0*np.log(b/b0)-gammaln(a)+gammaln(a0)+(a-a0)*digamma(a)+(b0-b)*a/b)
    kn=.5*(ld0-ld-K+np.trace(p@V)+(a/b)*(d@p@d))
    return float(g+kn)


@dataclass
class ProcPosterior:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    a: np.ndarray
    b: np.ndarray

@dataclass
class QualityPosterior:
    mu: np.ndarray  # [M,K]
    V: np.ndarray   # [M,K,K]
    a: np.ndarray   # [M]
    b: np.ndarray   # [M]

@dataclass
class Priors:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    a: np.ndarray
    b: np.ndarray
    qmu: np.ndarray
    qV: np.ndarray
    qa: np.ndarray
    qb: np.ndarray


def make_priors(K,d,M=2,transition_diag_boost=4.0,emission_kappa=0.2,quality_prior_scale=2.0):
    # Nur vorläufige, dimensionslose Priors für standardisierte Trainingsdaten.
    if min(transition_diag_boost,emission_kappa,quality_prior_scale)<=0:
        raise ValueError('Priors müssen positiv sein')
    pi=np.ones(K)
    A=np.full((K,K),0.5)+np.eye(K)*transition_diag_boost
    mu=np.zeros((K,d)); kappa=np.full((K,d),emission_kappa)
    a=np.full((K,d),2.5);b=np.full((K,d),1.5)
    qmu=np.zeros((M,K));qV=np.array([np.eye(K)*quality_prior_scale for _ in range(M)])
    qa=np.full(M,2.5);qb=np.full(M,1.5)
    return Priors(pi,A,mu,kappa,a,b,qmu,qV,qa,qb)


def elog_emission(x,p):
    T,d=x.shape;K=p.mu.shape[0]
    if p.mu.shape[1]!=d:raise ValueError('X-Dimension')
    prec=p.a/p.b
    base=.5*(digamma(p.a)-np.log(p.b)-LOG2PI-1./p.kappa)
    return np.sum(base[None,:,:]-.5*prec[None,:,:]*(x[:,None,:]-p.mu[None,:,:])**2,axis=2)


def update_process(rings,local,prior,weights):
    K,d=prior.mu.shape
    a_pi=prior.pi.copy();a_A=prior.A.copy()
    n=np.zeros(K); s=np.zeros((K,d)); s2=np.zeros((K,d))
    for rid,x in rings.items():
        w=weights.get(rid,0.)
        if w==0:continue
        fb=local[rid]['fb']
        g=fb.gamma
        a_pi+=w*g[0];a_A+=w*fb.xi.sum(axis=0)
        n+=w*g.sum(axis=0)
        s+=w*(g.T@x)
        s2+=w*(g.T@(x*x))
    k=prior.kappa+n[:,None]
    m=(prior.kappa*prior.mu+s)/k
    a=prior.a+n[:,None]/2
    b=prior.b+.5*(s2+prior.kappa*prior.mu**2-k*m*m)
    if (b<=0).any():raise ArithmeticError('Negativer NIG-Emissionsskalenparameter')
    return ProcPosterior(a_pi,a_A,m,k,a,b)


def update_quality(ydata,local,prior):
    M,K=prior.qmu.shape
    mu=np.empty((M,K));V=np.empty((M,K,K));a=np.empty(M);b=np.empty(M)
    if not ydata:return QualityPosterior(prior.qmu.copy(),prior.qV.copy(),prior.qa.copy(),prior.qb.copy())
    for r in range(M):
        p0=np.linalg.inv(prior.qV[r]); rhs=p0@prior.qmu[r]
        precision=p0.copy();y2=0.
        for rid,y in ydata.items():
            mm,H=local[rid]['m'],local[rid]['H']
            precision+=H
            rhs+=y[r]*mm
            y2+=y[r]**2
        L=np.linalg.cholesky(precision)
        V[r]=np.linalg.solve(L.T,np.linalg.solve(L,np.eye(K)))
        mu[r]=np.linalg.solve(L.T,np.linalg.solve(L,rhs))
        a[r]=prior.qa[r]+len(ydata)/2.
        b[r]=prior.qb[r]+.5*(y2+prior.qmu[r]@p0@prior.qmu[r]-mu[r]@precision@mu[r])
        if b[r]<=0:raise ArithmeticError('Negativer Qualitäts-NIG-Parameter')
    return QualityPosterior(mu,V,a,b)


def quality_logterm(y,m,H,q):
    total=0.
    for r in range(len(y)):
        precision=q.a[r]/q.b[r]
        second=q.mu[r]@H@q.mu[r]+np.trace(q.V[r]@H)/precision
        total+=-.5*LOG2PI+.5*(digamma(q.a[r])-np.log(q.b[r]))-.5*precision*(y[r]**2-2*y[r]*q.mu[r]@m+second)
    return float(total)


def local_inference(x, p, q=None, y=None, mode='OFF',max_iter=12):
    """OFF = prozessbedingte Zustände. TILT_JOINT = eingeschränkte ELBO-optimierte
    B1-Qualitätsrückkopplung über K-1 ringweit konstante Zustands-Potenziale.
    """
    from scipy.optimize import minimize
    logpi=elog_dir(p.pi);loga=elog_dir(p.A);elli=elog_emission(x,p)
    K=len(logpi);T=len(x)
    def compute(v):
        tilt=np.r_[np.asarray(v),0.] if len(v)==K-1 else np.zeros(K)
        fb=forward_backward(logpi,loga,elli+tilt[None,:]/T)
        m,H=moments_occupancy(fb)
        obj=fb.logz-tilt@m
        if y is not None and q is not None:obj+=quality_logterm(y,m,H,q)
        return float(obj),fb,m,H,tilt
    zero=np.zeros(K-1)
    start=compute(zero)
    chosen=start
    if mode=='TILT_JOINT' and y is not None and q is not None:
        # Einfache, klar begrenzte JOINT-Optimierung. B1-Moment 2 bleibt im Ziel erhalten.
        sol=minimize(lambda v:-compute(v)[0],zero,method='L-BFGS-B',
                     bounds=[(-15,15)]*(K-1), options={'maxiter':max_iter,'ftol':1e-7})
        alternative=compute(sol.x)
        if alternative[0]>start[0]+1e-9:chosen=alternative
    value,fb,m,H,tilt=chosen
    return {'fb':fb,'m':m,'H':H,'tilt':tilt,'local_objective':value}


def global_kl(p,q,prior):
    result=kl_dir(p.pi,prior.pi)
    result+=sum(kl_dir(a,b) for a,b in zip(p.A,prior.A))
    result+=kl_nig(p.mu,p.kappa,p.a,p.b,prior.mu,prior.kappa,prior.a,prior.b)
    for r in range(len(q.a)):
        result+=kl_reg(q.mu[r],q.V[r],q.a[r],q.b[r],prior.qmu[r],prior.qV[r],prior.qa[r],prior.qb[r])
    return float(result)


def train(rings, ydata, K=3, C=1.0, epochs=3, seed=123,
          mode='OFF',max_local_iter=8, prior_options=None,verbose=False):
    """rings: {id: standardisiertes X}; ydata: {id: 2 standardisierte Targets}.
    Train-/Testtrennung wird VOR dieser Funktion erzwungen; Test-X nie übergeben.
    """
    if K<2 or C<0:raise ValueError('K und C')
    if mode not in ('OFF','TILT_JOINT','FULL_JOINT'):raise ValueError('mode')
    if not ydata or not set(ydata).issubset(rings):raise ValueError('Unzulässige gelabelte Ringe')
    if len({x.shape[1] for x in rings.values()})!=1:raise ValueError('Unterschiedliche d')
    d=next(iter(rings.values())).shape[1];M=2
    prior=make_priors(K,d,M,**(prior_options or {}))
    # Training-only KMeans-Initialisierung, keine alten Modellparameter.
    from sklearn.cluster import KMeans
    rng=np.random.default_rng(seed)
    sample=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    init=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sample).cluster_centers_
    p=ProcPosterior(prior.pi.copy(),prior.A.copy(),np.array(init),np.full((K,d),4.),np.full((K,d),4.),np.full((K,d),3.))
    q=QualityPosterior(prior.qmu.copy(),prior.qV.copy(),prior.qa.copy(),prior.qb.copy())
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    history=[]
    for epoch in range(epochs):
        local={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if mode=='FULL_JOINT' and rid in ydata:
                from full_markov_b1 import local_full_joint
                local[rid]=local_full_joint(x,p,q,ydata[rid],max_iter=max_local_iter)
            else:
                local[rid]=local_inference(x,p,q,ydata.get(rid), mode=mode,
                                           max_iter=max_local_iter)
        p=update_process({rid:rings[rid] for rid in local},local,prior,weights)
        q=update_quality(ydata,local,prior)
        # ELBO NACH globalem Schritt mit fixierten lokalen Verteilungen:
        # Die lokale Markov-Entropie ist FB.logz - tilt.m minus die erwartete
        # Prozess-Likelihood; berechne diese nochmals unter neuen q(Theta).
        ell_terms=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            l=local[rid];fb=l['fb'];g=fb.gamma;tilt=l['tilt']
            el=elog_emission(x,p)
            prior_pi=elog_dir(p.pi);prior_A=elog_dir(p.A)
            # q-Entropie direkt über die Normalisierung der bei lokalem Schritt
            # benutzten Elog-Potenziale (vgl. C.14); state q bleibt fix.
            # Nutze hierfür Rekonstruktion aus FB-Paarmarginalen.
            g0=np.maximum(g[0],1e-300)
            entropy=-np.sum(g0*np.log(g0))
            conditional=fb.xi / np.maximum(g[:-1,:,None],1e-300)
            entropy-=np.sum(fb.xi*np.log(np.maximum(conditional,1e-300)))
            proc=float(g[0]@prior_pi+np.sum(fb.xi*prior_A[None,:,:])+np.sum(g*el)+entropy)
            if rid in ydata:
                ell_terms+=proc+quality_logterm(ydata[rid],l['m'],l['H'],q)
            else:ell_terms+=C*proc
        kl=global_kl(p,q,prior)
        entry={'epoch':epoch+1,'elbo':float(ell_terms-kl),'global_kl':kl,
               'full_local_gains':{rid:float(l.get('full_local_gain',0)) for rid,l in local.items()} if mode=='FULL_JOINT' else {},
               'states_effective':(np.sum([l['fb'].gamma.sum(0) for l in local.values()],axis=0)/sum(len(rings[i]) for i in local)).tolist()}
        history.append(entry)
        if verbose:print('epoch',epoch+1,'ELBO',round(entry['elbo'],2),'states',np.round(entry['states_effective'],3))
    return {'process':p,'quality':q,'prior':prior,'history':history,'local':local}


def predict(x, p, q):
    """Induktive P1-Predictive (Normalapproximation an posterior-prädiktives Gemisch).
    Der Testqualität y wird NIEMALS in die lokale Inferenz gegeben.
    """
    l=local_inference(x,p,mode='OFF')
    m,H=l['m'],l['H']
    mean=q.mu@m
    var=[]
    for r in range(len(q.a)):
        e_sigma2=q.b[r]/(q.a[r]-1.)
        s_h=np.maximum(0.,q.mu[r]@H@q.mu[r]-(q.mu[r]@m)**2)
        v=e_sigma2*(1.+np.trace(q.V[r]@H))+s_h
        var.append(v)
    return np.asarray(mean),np.sqrt(np.asarray(var)),m,H

# -----------------------------------------------------------------------------
# Beschleunigte, mathematisch äquivalente Rekursionen (Numba optional).
# Ohne Numba bleiben die vorstehenden Referenzimplementierungen verfügbar.
# -----------------------------------------------------------------------------
_reference_forward_backward=forward_backward
_reference_moments_occupancy=moments_occupancy
try:
    from numba import njit

    @njit(cache=True)
    def _logsumexp_vec(v):
        ma=np.max(v)
        return ma+np.log(np.sum(np.exp(v-ma)))

    @njit(cache=True)
    def _fb_fast(logpi,loga,ell):
        T,K=ell.shape
        f=np.empty((T,K));b=np.zeros((T,K))
        for k in range(K):f[0,k]=logpi[k]+ell[0,k]
        work=np.empty(K)
        for t in range(1,T):
            for k in range(K):
                for j in range(K):work[j]=f[t-1,j]+loga[j,k]
                f[t,k]=ell[t,k]+_logsumexp_vec(work)
        z=_logsumexp_vec(f[-1])
        for t in range(T-2,-1,-1):
            for j in range(K):
                for k in range(K):work[k]=loga[j,k]+ell[t+1,k]+b[t+1,k]
                b[t,j]=_logsumexp_vec(work)
        g=np.empty((T,K));xi=np.empty((T-1,K,K))
        for t in range(T):
            for j in range(K):g[t,j]=np.exp(f[t,j]+b[t,j]-z)
        for t in range(1,T):
            for j in range(K):
                for k in range(K):
                    xi[t-1,j,k]=np.exp(f[t-1,j]+loga[j,k]+ell[t,k]+b[t,k]-z)
        return g,xi,z

    @njit(cache=True)
    def _mom_fast(g,xi):
        T,K=g.shape
        m=np.eye(K)
        s=np.zeros((K,K,K))
        for k in range(K):s[k,k,k]=1.
        for t in range(1,T):
            mm=np.zeros((K,K));ss=np.zeros((K,K,K))
            for k in range(K):
                for j in range(K):
                    r=xi[t-1,j,k]/max(g[t,k],1e-300)
                    for a in range(K):
                        mm[k,a]+=r*m[j,a]
                        for b in range(K):ss[k,a,b]+=r*s[j,a,b]
            for k in range(K):
                for a in range(K):
                    for b in range(K):
                        s[k,a,b]=ss[k,a,b]
                        if a==k:s[k,a,b]+=mm[k,b]
                        if b==k:s[k,a,b]+=mm[k,a]
                        if a==k and b==k:s[k,a,b]+=1.
                    m[k,a]=mm[k,a]
                    if a==k:m[k,a]+=1.
        out_m=np.zeros(K);out_H=np.zeros((K,K))
        for k in range(K):
            for a in range(K):
                out_m[a]+=g[T-1,k]*m[k,a]/T
                for b in range(K):out_H[a,b]+=g[T-1,k]*s[k,a,b]/(T*T)
        return out_m,out_H

    def forward_backward(logpi,loga,ell):
        gamma,xi,z=_fb_fast(np.asarray(logpi,dtype=np.float64),np.asarray(loga,dtype=np.float64),np.asarray(ell,dtype=np.float64))
        if not (np.allclose(gamma.sum(axis=1),1.,atol=1e-6) and np.allclose(xi.sum(axis=(1,2)),1.,atol=1e-6)):
            raise ArithmeticError('Fast FB nicht normiert')
        return FB(gamma,xi,float(z))

    def moments_occupancy(fb):
        m,H=_mom_fast(fb.gamma,fb.xi)
        H=(H+H.T)/2
        if not np.isclose(np.sum(m),1.,atol=1e-6) or not np.isclose(np.sum(H),1.,atol=1e-6):
            raise ArithmeticError('Fast B1-Momente nicht normiert')
        if not np.allclose(H@np.ones(len(m)),m,atol=1e-6):raise ArithmeticError('Fast B1-Momente inkonsistent')
        return m,H
except ImportError:
    pass
