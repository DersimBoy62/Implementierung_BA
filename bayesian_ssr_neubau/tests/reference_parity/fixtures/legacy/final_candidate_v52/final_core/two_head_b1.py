"""Two target-specific fully Bayesian quality heads and a shared Gaussian-Wishart HMM.

Each output j has its own regression dimension and physical covariates s_j(X).
The states h have exactly computed first/second Markov occupancy moments.
The quality likelihood for both targets jointly feeds back into local FULL_JOINT
Markov q; unlabeled rings contribute their X likelihood only (weighted C).
No labels from U/validation/test; scalers and summary selections MUST be fitted
outside this module solely on current training rings.

This is an explicitly exploratory method; it is not a transcription of Shao et al.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import vb_b1 as base
import fullcov_b1 as full
import full_markov_b1 as fm

@dataclass
class Head:
    mu: np.ndarray
    V: np.ndarray
    a: float
    b: float


def prior_head(K, nphys, state_var=2., physical_var=.25):
    if K < 2 or nphys < 0 or min(state_var, physical_var) <= 0:
        raise ValueError('Invalid head priors')
    D=K+nphys
    return Head(np.zeros(D), np.diag(np.r_[np.full(K,state_var),np.full(nphys,physical_var)]),2.5,1.5)


def augment(m,H,s):
    m=np.asarray(m,float);H=np.asarray(H,float);s=np.asarray(s,float)
    if m.ndim!=1 or H.shape!=(len(m),len(m)) or s.ndim!=1:raise ValueError('moment dimensions')
    return np.r_[m,s],np.block([[H,np.outer(m,s)],[np.outer(s,m),np.outer(s,s)]])


def expected_quality_coeff(y,head):
    prec=head.a/head.b
    b=prec*y*head.mu
    G=-.5*(prec*np.outer(head.mu,head.mu)+head.V)
    const=-.5*base.LOG2PI+.5*(digamma(head.a)-np.log(head.b))-.5*prec*y*y
    return b,G,float(const)


def compressed_coeff(y,head,s,K):
    s=np.asarray(s,float)
    if len(head.mu)!=K+len(s):raise ValueError('Wrong feature count')
    b,G,c=expected_quality_coeff(y,head)
    return b[:K]+2*G[:K,K:]@s,G[:K,:K],float(c+b[K:]@s+s@G[K:,K:]@s)


def both_compressed_coeff(y,heads,ss,K):
    if len(heads)!=2 or len(ss)!=2 or len(y)!=2:raise ValueError('Exactly two targets')
    b=np.zeros(K);G=np.zeros((K,K));c=0.
    for j in range(2):
        bj,Gj,cj=compressed_coeff(float(y[j]),heads[j],ss[j],K)
        b+=bj;G+=Gj;c+=cj
    return b,G,float(c)


def fit_head(y,local,j,prior):
    D=len(prior.mu);P0=np.linalg.inv(prior.V)
    P=P0.copy();rhs=P0@prior.mu;y2=0.
    for rid,yy in y.items():
        m,H=local[rid]['head_moments'][j]
        if len(m)!=D:raise ValueError('Wrong moments')
        P+=H;rhs+=float(yy[j])*m;y2+=float(yy[j])**2
    ch=np.linalg.cholesky(P)
    V=np.linalg.solve(ch.T,np.linalg.solve(ch,np.eye(D)))
    mu=np.linalg.solve(ch.T,np.linalg.solve(ch,rhs))
    a=prior.a+len(y)/2
    b=prior.b+.5*(y2+prior.mu@P0@prior.mu-mu@P@mu)
    if b<=0:raise ArithmeticError('Nonpositive quality b')
    return Head(mu,(V+V.T)/2,float(a),float(b))


def quality_logterm(y,m,H,q):
    prec=q.a/q.b
    return float(-.5*base.LOG2PI+.5*(digamma(q.a)-np.log(q.b))
                 -.5*prec*(y*y-2*y*q.mu@m+q.mu@H@q.mu+np.trace(q.V@H)/prec))


def _local_augmented(x,p,heads,y,ss,max_local_iter):
    logpi=base.elog_dir(p.pi); logA=base.elog_dir(p.A); ell=full.elog_emission_full(x,p)
    K=len(p.pi)
    init=base.forward_backward(logpi,logA,ell)
    v0=fm.init_logits_from_fb(init)
    b,G,c=both_compressed_coeff(y,heads,ss,K)
    def fun(v):
        q0,Q=fm.unpack_logits(v,len(x),K)
        val,d0,dq,_,_=fm._markov_objective_and_grad(q0,Q,logpi,logA,ell,b,G,c)
        d0=q0*(d0-np.dot(q0,d0))
        dq=Q*(dq-np.sum(dq*Q,axis=2,keepdims=True))
        return float(val),np.r_[d0,dq.ravel()]
    initial,_=fun(v0)
    def loss(v):
        val,g=fun(v)
        return -val,-g
    res=minimize(loss,v0,jac=True,method='L-BFGS-B',options={'maxiter':max_local_iter,'ftol':1e-10,'gtol':1e-5,'maxls':15})
    final,_=fun(res.x)
    accepted=bool(np.isfinite(final) and final>initial+1e-10)
    q0,Q=fm.unpack_logits(res.x if accepted else v0,len(x),K)
    fb=fm.markov_marginals(q0,Q)
    m,H=base.moments_occupancy(fb)
    return {'fb':fb,'m_h':m,'H_h':H,'head_moments':[augment(m,H,s) for s in ss],
            'local_gain':float(max(0,final-initial)) if np.isfinite(final) else 0.,
            'accepted':accepted,'iterations':int(res.nit)}


def kl_all(p,heads,prior_process,priors_heads):
    kl=base.kl_dir(p.pi,prior_process.pi)
    kl+=sum(base.kl_dir(pa,pr) for pa,pr in zip(p.A,prior_process.A))
    for k in range(len(p.pi)):
        kl+=full.kl_normal_wishart(p.mu[k],p.kappa[k],p.nu[k],p.W[k],
                                  prior_process.mu[k],prior_process.kappa[k],prior_process.nu[k],prior_process.W[k])
    for q,p0 in zip(heads,priors_heads):
        kl+=base.kl_reg(q.mu,q.V,q.a,q.b,p0.mu,p0.V,p0.a,p0.b)
    return float(kl)


def train(rings, ydata, sdata, K=3, C=1., epochs=10, seed=20261111,steps=8,
          state_var=2.,physical_var=.25,verbose=False):
    if not rings or not ydata or not set(ydata).issubset(rings) or set(sdata)!=set(rings):
        raise ValueError('Labeled/summaries/rings mismatch')
    if C<0 or epochs<1 or steps<0 or K<2:raise ValueError('Training hyperparameters')
    if any(np.asarray(yy).shape!=(2,) for yy in ydata.values()):raise ValueError('Only two targets')
    if any(len(sdata[r])!=2 for r in rings):raise ValueError('Two individual head features needed')
    nphys=tuple(len(np.asarray(sdata[r][j])) for j in (0,1) for r in list(rings)[:1])
    for rid in rings:
        if tuple(len(np.asarray(sdata[rid][j])) for j in (0,1))!=nphys:
            raise ValueError('Inconsistent head feature dimensions')
    d=next(iter(rings.values())).shape[1]
    prior_process=full.make_priors_full(K,d)
    priors_heads=tuple(prior_head(K,n,state_var,physical_var) for n in nphys)
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centroids=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior_process.pi.copy(),prior_process.A.copy(),centroids,
                    np.full(K,4.),prior_process.nu.copy(),prior_process.W.copy())
    heads=list(priors_heads)
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    history=[]
    for ep in range(epochs):
        local={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                local[rid]=_local_augmented(x,p,heads,ydata[rid],sdata[rid],steps)
            else:
                l=full.infer_process_full(x,p)
                l['head_moments']=[augment(l['m'],l['H'],ss) for ss in sdata[rid]]
                local[rid]=l
        p=full.update_process_full(rings,local,prior_process,weights)
        heads=[fit_head(ydata,local,j,priors_heads[j]) for j in range(2)]
        total=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            l=local[rid];fb=l['fb'];g=fb.gamma
            entropy=-np.sum(g[0]*np.log(np.maximum(g[0],1e-300)))
            cond=fb.xi/np.maximum(g[:-1,:,None],1e-300)
            entropy-=np.sum(fb.xi*np.log(np.maximum(cond,1e-300)))
            proc=float(g[0]@base.elog_dir(p.pi)+np.sum(fb.xi*base.elog_dir(p.A)[None,:,:])
                       +np.sum(g*full.elog_emission_full(x,p))+entropy)
            if rid in ydata:
                total+=proc
                for j in range(2):
                    m,H=l['head_moments'][j]
                    total+=quality_logterm(float(ydata[rid][j]),m,H,heads[j])
            else:total+=C*proc
        elbo=total-kl_all(p,heads,prior_process,priors_heads)
        if not np.isfinite(elbo):raise ArithmeticError('ELBO nonfinite')
        history.append({'epoch':ep+1,'elbo':float(elbo),
                        'state_mass':(sum(local[r]['fb'].gamma.sum(0) for r in local)
                                     /sum(len(rings[r]) for r in local)).tolist(),
                        'local_gain':float(sum(l.get('local_gain',0) for l in local.values()))})
        if verbose:print('epoch',ep+1,'ELBO',round(elbo,3),flush=True)
    return {'process':p,'heads':heads,'priors_heads':priors_heads,'history':history}


def predict(x,ss,model):
    l=full.infer_process_full(x,model['process'])
    mu=[];sd=[]
    for j in range(2):
        m,H=augment(l['m'],l['H'],ss[j]);q=model['heads'][j]
        pred=float(q.mu@m)
        var=(q.b/(q.a-1))*(1+np.trace(q.V@H))
        var+=max(0.,q.mu@H@q.mu-pred**2)
        mu.append(pred);sd.append(np.sqrt(max(var,1e-12)))
    return np.array(mu),np.array(sd)
