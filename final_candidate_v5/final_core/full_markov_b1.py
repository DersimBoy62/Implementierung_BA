"""Freie, zeitabhängige Markov-Variationsfamilie für ringweite B1-SSR.

q(z_0,...,z_{T-1}) = q0[z0] * Prod_t Q[t-1,z_{t-1},z_t]
Dabei ist jede Zeile Q[t-1,j,:] separat und zeitabhängig optimierbar.
Die lokalen Qualitätsmomente E[h] und E[hh.T] werden EXAKT rekursiv verwendet.

Lokale ELBO ist die erwartete Prozess-Logwahrscheinlichkeit + q-Entropie +
(erwartete) Qualitäts-Loglikelihood, mit globalen q(Theta) fixiert.

Numba optional zur Beschleunigung, numerische Gradientenchecks im Begleittest.
Keine Pseudolabels, keine Testsequenzen, kein Ausgabetraining in diesem Modul.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize
from vb_b1 import (FB, forward_backward, elog_dir, elog_emission, quality_logterm,
                   moments_occupancy, LOG2PI)
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

@njit(cache=True)
def _markov_objective_and_grad(prob0, qtrans, logpi, logA, ell, b, G, const):
    """Return local ELBO and derivative wrt probabilities, not raw logits.

    Gradients obtained by reverse-mode differentiation of exact forward
    recursions of unnormalized occupation moments for Markov chains.
    """
    T,K=ell.shape
    P=np.zeros((T,K));V=np.zeros((T,K,K)); W=np.zeros((T,K,K,K))
    P[0]=prob0
    for k in range(K):
        V[0,k,k]=prob0[k]
        W[0,k,k,k]=prob0[k]
    f=0.
    for k in range(K):
        f+=prob0[k]*(logpi[k]+ell[0,k]-np.log(prob0[k]))
    for t in range(1,T):
        Q=qtrans[t-1]
        for j in range(K):
            for k in range(K):
                wt=P[t-1,j]*Q[j,k]
                f+=wt*(logA[j,k]+ell[t,k]-np.log(Q[j,k]))
                P[t,k]+=wt
                for a in range(K):
                    V[t,k,a]+=Q[j,k]*V[t-1,j,a]
                    for c in range(K):
                        W[t,k,a,c]+=Q[j,k]*W[t-1,j,a,c]
        for k in range(K):
            V[t,k,k]+=P[t,k]
            for a in range(K):
                W[t,k,k,a]+=V[t,k,a]- (P[t,k] if a==k else 0.)
                W[t,k,a,k]+=V[t,k,a]- (P[t,k] if a==k else 0.)
            W[t,k,k,k]+=P[t,k]
    m=np.zeros(K);H=np.zeros((K,K))
    for k in range(K):
        for a in range(K):
            m[a]+=V[T-1,k,a]/T
            for c in range(K):
                H[a,c]+=W[T-1,k,a,c]/(T*T)
    f+=const
    for a in range(K):
        f+=b[a]*m[a]
        for c in range(K):
            f+=G[a,c]*H[a,c]

    # Reverse-mode: adjoints of P, V, W forward recursions.
    dp=np.zeros((T,K)); dv=np.zeros((T,K,K));dw=np.zeros((T,K,K,K))
    for k in range(K):
        for a in range(K):
            dv[T-1,k,a]=b[a]/T
            for c in range(K):
                dw[T-1,k,a,c]=G[a,c]/(T*T)
    dq=np.zeros((T-1,K,K))
    for t in range(T-1,0,-1):
        Q=qtrans[t-1]
        dP=np.copy(dp[t]); dB=np.copy(dv[t]); dW=np.copy(dw[t])
        for k in range(K):
            dP[k]+=dv[t,k,k]
            dP[k]+=dw[t,k,k,k]
            for a in range(K):
                dB[k,a]+=dw[t,k,k,a]+dw[t,k,a,k]
        for j in range(K):
            for k in range(K):
                qjk=Q[j,k]
                reward=logA[j,k]+ell[t,k]-np.log(qjk)
                dp[t-1,j]+=qjk*reward
                dq[t-1,j,k]+=P[t-1,j]*(reward-1.)
                # P[t,k] = sum_j P[t-1,j]*Q[j,k]
                dp[t-1,j]+=qjk*dP[k]
                dq[t-1,j,k]+=P[t-1,j]*dP[k]
                # V[t,k,a] = sum_j Q[j,k]*V[t-1,j,a] + delta(k,a)*P[t,k]
                for a in range(K):
                    dv[t-1,j,a]+=qjk*dB[k,a]
                    dq[t-1,j,k]+=V[t-1,j,a]*dB[k,a]
                    for c in range(K):
                        dw[t-1,j,a,c]+=qjk*dW[k,a,c]
                        dq[t-1,j,k]+=W[t-1,j,a,c]*dW[k,a,c]
    d0=np.zeros(K)
    for k in range(K):
        d0[k]=dp[0,k]+dv[0,k,k]+dw[0,k,k,k]
        d0[k]+=logpi[k]+ell[0,k]-np.log(prob0[k])-1.
    return f,d0,dq,m,H


def _softmax(a,axis=-1):
    return np.exp(a-logsumexp(a,axis=axis,keepdims=True))


def quality_coeff(y,q):
    """E_q[log p(y|h,...)] = constant + b.h + sum_ab G[a,b] H[a,b]."""
    K=q.mu.shape[1]
    b=np.zeros(K);G=np.zeros((K,K));const=0.
    if y is None: return b,G,const
    for r in range(len(y)):
        prec=q.a[r]/q.b[r]
        b+=prec*y[r]*q.mu[r]
        G-=.5*(prec*np.outer(q.mu[r],q.mu[r])+q.V[r])
        const+=-.5*LOG2PI+.5*(digamma(q.a[r])-np.log(q.b[r]))-.5*prec*y[r]**2
    return b,G,float(const)


def init_logits_from_fb(fb):
    g=fb.gamma; xi=fb.xi
    q0=np.maximum(g[0],1.e-12);q0/=q0.sum()
    Q=np.maximum(xi/np.maximum(g[:-1,:,None],1.e-300),1.e-12)
    Q=Q/Q.sum(axis=2,keepdims=True)
    return np.concatenate([np.log(q0), np.log(Q).ravel()])


def unpack_logits(v,T,K):
    return _softmax(v[:K]), _softmax(v[K:].reshape(T-1,K,K),axis=2)


def objective_and_grad(v,logpi,loga,ell,y,q):
    T,K=ell.shape
    prob0,Q=unpack_logits(v,T,K)
    b,G,const=quality_coeff(y,q)
    f,d0,dq,m,H=_markov_objective_and_grad(prob0,Q,logpi,loga,ell,b,G,const)
    d0=prob0*(d0-np.dot(d0,prob0))
    dq=Q*(dq-np.sum(dq*Q,axis=2,keepdims=True))
    return float(f),np.concatenate([d0,dq.ravel()])


def markov_marginals(q0,Q):
    """Convert local Markov parameters to standard gamma/xi (no path enumeration)."""
    T=Q.shape[0]+1;K=len(q0)
    g=np.empty((T,K));xi=np.empty((T-1,K,K));g[0]=q0
    for t in range(1,T):
        xi[t-1]=g[t-1,:,None]*Q[t-1]
        g[t]=xi[t-1].sum(axis=0)
    return FB(g,xi,float('nan'))


def local_full_joint(x,p,q,y,max_iter=8,gtol=1e-5):
    """Exact moment B1 objective; optimize all time-varying Markov transitions.

    Global q parameters fixed during local step. A fallback to exact process
    q is returned if the optimizer cannot improve the objective.
    """
    if y is None or q is None: raise ValueError('FULL_JOINT requires a known y and q')
    logpi=elog_dir(p.pi);loga=elog_dir(p.A);elli=elog_emission(x,p)
    basefb=forward_backward(logpi,loga,elli)
    v0=init_logits_from_fb(basefb)
    base,basegrad=objective_and_grad(v0,logpi,loga,elli,y,q)
    def fg(v):
        f,g=objective_and_grad(v,logpi,loga,elli,y,q)
        return -f,-g
    solution=minimize(fg,v0,method='L-BFGS-B',jac=True,
                      options={'maxiter':max_iter,'ftol':1e-10,'gtol':gtol,'maxls':15})
    candidate,_=objective_and_grad(solution.x,logpi,loga,elli,y,q)
    take=bool(np.isfinite(candidate) and candidate>=base+1e-10)
    v=solution.x if take else v0
    q0,Q=unpack_logits(v,len(x),len(logpi))
    fb=markov_marginals(q0,Q)
    m,H=moments_occupancy(fb)
    return {'fb':fb,'m':m,'H':H,'tilt':np.zeros(len(logpi)),
            'local_objective':float(candidate if take else base),
            'full_local_gain':float(max(0.,candidate-base)) if np.isfinite(candidate) else 0.,
            'full_local_grad_norm':float(np.max(np.abs(basegrad))),
            'full_local_iterations':int(solution.nit),
            'full_local_optimizer_success':bool(solution.success),
            'full_local_accepted':take}
