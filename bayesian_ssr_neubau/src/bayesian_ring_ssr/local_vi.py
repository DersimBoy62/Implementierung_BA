"""Safeguarded local VI using exact conditional-reward score gradients.

For F(z)=log p(X,z)-log q(z)+b'g+g'Gg+c, a normalized logit score
has E[d log q]=0. Therefore d E[F]/d logit = E[F * score]. Conditional
prefix/suffix reward moments evaluate this identity without enumerating paths
or differentiating through a fitted posterior. This is an independent
implementation of the same exact Markov expectation as the old oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.optimize import minimize
from .forward_backward import forward_backward,validate_potentials
from .reduced_markov import MarkovChain,pack,unpack
from .hmmmoments import m2_increments,validate_rewards


@njit(cache=True)
def _conditional_objective(initial,Q,node,lp,la,ell,b,G,c):
    T,K,D = node.shape
    prob = np.zeros((T,K))
    prefix_m = np.zeros((T,K,D))
    prefix_H = np.zeros((T,K,D,D))
    prefix_r = np.zeros((T,K))
    prob[0] = initial
    for i in range(K):
        prefix_m[0,i] = node[0,i]
        prefix_r[0,i] = lp[i]+ell[0,i]-np.log(max(initial[i],1e-300))
        for a in range(D):
            for d in range(D):
                prefix_H[0,i,a,d] = node[0,i,a]*node[0,i,d]
    for t in range(1,T):
        for i in range(K):
            for j in range(K):
                prob[t,j] += prob[t-1,i]*Q[t-1,i,j]
        for j in range(K):
            if prob[t,j] == 0:
                continue
            for i in range(K):
                w = prob[t-1,i]*Q[t-1,i,j]/prob[t,j]
                prefix_r[t,j] += w*(prefix_r[t-1,i]+la[i,j]+ell[t,j]
                                                    -np.log(max(Q[t-1,i,j],1e-300)))
                for a in range(D):
                    prefix_m[t,j,a] += w*(prefix_m[t-1,i,a]+node[t,j,a])
                    for d in range(D):
                        prefix_H[t,j,a,d] += w*(prefix_H[t-1,i,a,d]
                            +prefix_m[t-1,i,a]*node[t,j,d]
                            +prefix_m[t-1,i,d]*node[t,j,a]
                            +node[t,j,a]*node[t,j,d])

    m = np.zeros(D)
    H = np.zeros((D,D))
    f = c
    for i in range(K):
        f += prob[T-1,i]*prefix_r[T-1,i]
        m += prob[T-1,i]*prefix_m[T-1,i]
        H += prob[T-1,i]*prefix_H[T-1,i]
    for a in range(D):
        f += b[a]*m[a]
        for d in range(D):
            f += G[a,d]*H[a,d]

    suffix_m = np.zeros((K,D))
    suffix_H = np.zeros((K,D,D))
    suffix_r = np.zeros(K)
    grad_Q = np.zeros_like(Q)
    for t in range(T-2,-1,-1):
        shifted_m = suffix_m+node[t+1]
        shifted_H = suffix_H.copy()
        for j in range(K):
            for a in range(D):
                for d in range(D):
                    shifted_H[j,a,d] += (suffix_m[j,a]*node[t+1,j,d]
                        +suffix_m[j,d]*node[t+1,j,a]
                        +node[t+1,j,a]*node[t+1,j,d])
        edge_values = np.zeros((K,K))
        new_m = np.zeros((K,D))
        new_H = np.zeros((K,D,D))
        new_r = np.zeros(K)
        for i in range(K):
            for j in range(K):
                reward = la[i,j]+ell[t+1,j]-np.log(max(Q[t,i,j],1e-300))+suffix_r[j]
                value = prefix_r[t,i]+reward+c
                for a in range(D):
                    value += b[a]*(prefix_m[t,i,a]+shifted_m[j,a])
                    for d in range(D):
                        value += G[a,d]*(prefix_H[t,i,a,d]+shifted_H[j,a,d]
                            +prefix_m[t,i,a]*shifted_m[j,d]
                            +prefix_m[t,i,d]*shifted_m[j,a])
                edge_values[i,j] = value
                new_r[i] += Q[t,i,j]*reward
                new_m[i] += Q[t,i,j]*shifted_m[j]
                new_H[i] += Q[t,i,j]*shifted_H[j]
            baseline = np.dot(Q[t,i],edge_values[i])
            for j in range(K):
                grad_Q[t,i,j] = prob[t,i]*Q[t,i,j]*(edge_values[i,j]-baseline)
        suffix_m,suffix_H,suffix_r = new_m,new_H,new_r

    grad_initial = np.zeros(K)
    for i in range(K):
        value = prefix_r[0,i]+suffix_r[i]+c
        for a in range(D):
            value += b[a]*(node[0,i,a]+suffix_m[i,a])
            for d in range(D):
                value += G[a,d]*(node[0,i,a]*node[0,i,d]+suffix_H[i,a,d]
                    +node[0,i,a]*suffix_m[i,d]+node[0,i,d]*suffix_m[i,a])
        grad_initial[i] = initial[i]*(value-f)
    return f,grad_initial,grad_Q,m,H


def objective_and_grad(v,logpi,logA,ell,b,G,constant=0.,node=None):
    lp,la,em = validate_potentials(logpi,logA,ell)
    T,K = em.shape
    chain = unpack(v,T,K)
    rewards = validate_rewards(chain,node)
    D = rewards.shape[2]
    linear = np.asarray(b,dtype=np.float64)
    quadratic = np.asarray(G,dtype=np.float64)
    if (linear.shape!=(D,) or quadratic.shape!=(D,D)
            or not np.isfinite(linear).all() or not np.isfinite(quadratic).all()
            or not np.isfinite(constant)):
        raise ValueError("Invalid local quadratic coefficients")
    f,g0,gq,m,H = _conditional_objective(chain.initial,chain.transitions,rewards,
                                        lp,la,em,linear,quadratic,float(constant))
    grad = np.r_[g0[:-1],gq[...,:-1].ravel()]
    return float(f),grad,m,(H+H.T)*.5


@dataclass(frozen=True)
class LocalResult:
    chain: MarkovChain
    logits: np.ndarray
    m: np.ndarray
    H: np.ndarray
    objective: float
    diagnostics: dict


def optimize_local(logpi,logA,ell,b,G,constant=0.,previous=None,max_iter=8,gtol=1e-5,node=None):
    lp,la,em = validate_potentials(logpi,logA,ell)
    if int(max_iter)!=max_iter or max_iter<0 or not np.isfinite(gtol) or gtol<=0:
        raise ValueError("Invalid local optimization controls")
    T,K = em.shape
    rewards = m2_increments(T,K) if node is None else np.asarray(node,dtype=np.float64)
    process = forward_backward(lp,la,em).chain
    fb_v = pack(process)
    def evaluate(v):
        return objective_and_grad(v,lp,la,em,b,G,constant,rewards)
    fb_f,_,_,_ = evaluate(fb_v)
    old_v = None
    old_f = None
    if previous is not None:
        old_v = pack(previous) if isinstance(previous,MarkovChain) else np.asarray(previous,dtype=np.float64).copy()
        old_f = evaluate(old_v)[0]
    use_old = old_f is not None and old_f>=fb_f
    start = old_v if use_old else fb_v
    start_f,start_g,_,_ = evaluate(start)
    def minimized(v):
        f,g,_,_ = evaluate(v)
        return -f,-g
    chosen = start.copy()
    candidate_f = start_f
    accepted = False
    if max_iter:
        sol = minimize(minimized,start,method="L-BFGS-B",jac=True,
                       options={"maxiter":int(max_iter),"gtol":gtol,"ftol":1e-10,"maxls":15})
        if np.isfinite(sol.x).all():
            candidate_f = evaluate(sol.x)[0]
            if np.isfinite(candidate_f) and candidate_f>start_f+1e-10:
                chosen = sol.x.copy()
                accepted = True
        iterations,nfev,message,success = int(sol.nit),int(sol.nfev),str(sol.message),bool(sol.success)
        status = int(sol.status)
    else:
        iterations,nfev,message,success,status = 0,0,"zero local iterations",False,0
    f,g,m,H = evaluate(chosen)
    if not np.isfinite(f) or not np.isfinite(g).all() or f<start_f-1e-7:
        raise FloatingPointError("Safeguarded local update decreased objective")
    return LocalResult(unpack(chosen,T,K),chosen,m,H,f,{
        "selected_previous":bool(use_old),"accepted":accepted,
        "objective_fb":float(fb_f),"objective_previous":None if old_f is None else float(old_f),
        "objective_start":float(start_f),"objective_candidate":float(candidate_f),
        "objective_chosen":float(f),"gain_over_fb":float(f-fb_f),
        "gain_over_previous":None if old_f is None else float(f-old_f),
        "local_gain":float(f-start_f),"grad_inf":float(np.max(np.abs(g))),
        "grad_l2":float(np.linalg.norm(g)),"start_grad_inf":float(np.max(np.abs(start_g))),
        "projected_gradient_inf":float(np.max(np.abs(g))),
        "optimizer_status":status,"message":message,"nit":iterations,"nfev":nfev,
        "optimizer_success":success,"n_reduced_logits":len(chosen),
        "fb_probability_floor":1e-12})
