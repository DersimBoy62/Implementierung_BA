"""Exploratory V2.1 sensitivity branch: scaled switch features (per 100 intervals).

Not an update of the locked 2026-11-11 final model. Works with development
rings only via separate compare_dynamic_quality_cv4.py. New model variants:
 M0 occupancy h (baseline only in CV; use old two_head_b1 for exact M0),
 M1 h + early/late off-diagonal switch rates,
 M2 h + late-minus-early state occupancies,
 M3 h + both.

First and second moments of ALL latent quality features and gradients wrt
FULL_JOINT time-inhomogeneous Markov q are computed analytically by
'dynamic_markov_v2'. No hard state decoding and no pseudo-labels.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import vb_b1 as base
import fullcov_b1 as full
import full_markov_b1 as fm
import two_head_b1 as th
from dynamic_markov_v21 import Spec,moments,objective_logits


def prior_head(spec,nphys,physical_var=.25):
    if nphys<0 or physical_var<=0: raise ValueError('Invalid priors')
    v=np.r_[spec.prior_variances,np.full(nphys,physical_var)]
    return th.Head(np.zeros(len(v)),np.diag(v),2.5,1.5)


def _fb_to_moments(fb,spec):
    logit=fm.init_logits_from_fb(fb)
    q0,Q=fm.unpack_logits(logit,len(fb.gamma),spec.K)
    return moments(q0,Q,spec)


def local_joint(x,p,heads,y,ss,spec,steps):
    logpi=base.elog_dir(p.pi);logA=base.elog_dir(p.A);ell=full.elog_emission_full(x,p)
    fb0=base.forward_backward(logpi,logA,ell)
    v0=fm.init_logits_from_fb(fb0)
    b,G,c=th.both_compressed_coeff(y,heads,ss,spec.D)
    def fg(v):
        val,g,_,_=objective_logits(v,logpi,logA,ell,b,G,spec)
        return -val-c,-g
    first,_=fg(v0)
    if steps:
        sol=minimize(fg,v0,jac=True,method='L-BFGS-B',options={'maxiter':steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
        alt,_=fg(sol.x)
        use=np.isfinite(alt) and alt<first-1e-10
        vv=sol.x if use else v0
        nit=int(sol.nit)
    else:
        vv=v0;use=False;alt=first;nit=0
    q0,Q=fm.unpack_logits(vv,len(x),spec.K)
    fb=fm.markov_marginals(q0,Q)
    m,H=moments(q0,Q,spec)
    return {'fb':fb,'m_g':m,'H_g':H,'head_moments':[th.augment(m,H,s) for s in ss],
            'local_gain':float(max(0,first-alt)) if np.isfinite(alt) else 0.,
            'accepted':bool(use),'iterations':nit}


def local_unlabeled(x,p,ss,spec):
    l=full.infer_process_full(x,p)
    m,H=_fb_to_moments(l['fb'],spec)
    l['m_g']=m;l['H_g']=H
    l['head_moments']=[th.augment(m,H,s) for s in ss]
    return l


def train(rings,ydata,sdata,variant='M3',K=3,C=1.,epochs=10,seed=20261111,steps=8,physical_var=.25,verbose=False):
    spec=Spec(K,variant)
    if not rings or not ydata or not set(ydata).issubset(rings) or set(sdata)!=set(rings):
        raise ValueError('Labeled/summaries/rings mismatch')
    if C<0 or epochs<1 or steps<0:raise ValueError('Training hyperparameters')
    if any(np.asarray(v).shape!=(2,) for v in ydata.values()):raise ValueError('Exactly 2 quality targets')
    if any(len(sdata[r])!=2 for r in rings):raise ValueError('Two feature groups needed')
    nphys=(len(np.asarray(sdata[next(iter(rings))][0])),len(np.asarray(sdata[next(iter(rings))][1])))
    if any(tuple(len(np.asarray(sdata[r][j])) for j in range(2))!=nphys for r in rings):
        raise ValueError('Inconsistent physical feature dimensions')
    d=next(iter(rings.values())).shape[1]
    if any(x.ndim!=2 or x.shape[1]!=d or len(x)<4 or not np.all(np.isfinite(x)) for x in rings.values()):
        raise ValueError('Invalid process arrays')
    prior=full.make_priors_full(K,d)
    priors_heads=tuple(prior_head(spec,nphys[j],physical_var) for j in range(2))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centroids=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),centroids,np.full(K,4.),prior.nu.copy(),prior.W.copy())
    heads=list(priors_heads)
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    history=[]
    for ep in range(epochs):
        local={}
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:local[rid]=local_joint(x,p,heads,ydata[rid],sdata[rid],spec,steps)
            else:local[rid]=local_unlabeled(x,p,sdata[rid],spec)
        p=full.update_process_full(rings,local,prior,weights)
        heads=[th.fit_head(ydata,local,j,priors_heads[j]) for j in range(2)]
        elbo=0.
        for rid,x in rings.items():
            if weights[rid]==0:continue
            fb=local[rid]['fb'];g=fb.gamma
            entropy=-np.sum(g[0]*np.log(np.maximum(g[0],1e-300)))
            cond=fb.xi/np.maximum(g[:-1,:,None],1e-300)
            entropy-=np.sum(fb.xi*np.log(np.maximum(cond,1e-300)))
            term=(g[0]@base.elog_dir(p.pi)+np.sum(fb.xi*base.elog_dir(p.A)[None,:,:])
                  +np.sum(g*full.elog_emission_full(x,p))+entropy)
            if rid in ydata:
                elbo+=term
                for j in range(2):
                    mg,Hg=local[rid]['head_moments'][j]
                    elbo+=th.quality_logterm(float(ydata[rid][j]),mg,Hg,heads[j])
            else:elbo+=C*term
        elbo-=th.kl_all(p,heads,prior,priors_heads)
        if not np.isfinite(elbo):raise ArithmeticError('Non-finite ELBO')
        wmass=np.sum([local[r]['fb'].gamma.sum(axis=0) for r in local],axis=0)
        ntime=sum(len(rings[r]) for r in local)
        history.append({'epoch':ep+1,'elbo':float(elbo),'state_mass':(wmass/ntime).tolist(),
                        'local_gain':float(sum(l.get('local_gain',0.) for l in local.values()))})
        if verbose:print(f'V2 {variant} K{K} epoch {ep+1} ELBO {elbo:.3f}',flush=True)
    return {'process':p,'heads':heads,'priors_heads':priors_heads,'history':history,
            'variant':variant,'spec':spec}


def predict(x,ss,model):
    spec=model['spec']
    l=full.infer_process_full(x,model['process'])
    m,H=_fb_to_moments(l['fb'],spec)
    means=[];sds=[]
    for j in range(2):
        a,A=th.augment(m,H,ss[j]);q=model['heads'][j]
        mean=float(q.mu@a)
        var=(q.b/(q.a-1.))*(1.+np.trace(q.V@A))
        var+=max(0.,q.mu@A@q.mu-mean*mean)
        means.append(mean);sds.append(np.sqrt(max(var,1e-12)))
    return np.array(means),np.array(sds)
