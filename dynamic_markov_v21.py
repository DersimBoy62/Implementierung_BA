"""V2.1 sensitivity study: switch rates per 100 observation intervals, with exact Markov moments.

New, independent research module. Does not load ring data or read any labels.
Features: occupancies, optionally early-vs-late occupancy differences, and
optionally early/late number of off-diagonal transitions per 100 intervals. Time-dependent q is
q0 * prod_t Q[t,z[t],z[t+1]]. No Viterbi/pseudo-label features.

The forward-backward Markov marginal calculations are classical (Rabiner 1989).
The deterministic x100 scaling is a NEW feature/regularization assumption.
With the SAME coefficient-prior variance 0.15 this changes the induced prior on
the rate in original units; it is not an innocuous re-expression.
The additive-path second-moment and reverse-gradient recursions below are our
own derivation; they were checked against brute path enumeration and finite
differences. They are NOT presented as equations from a cited paper.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(func):return func
        return wrap

@dataclass(frozen=True)
class Spec:
    K: int
    variant: str='M3'
    def __post_init__(self):
        if self.K<2 or self.variant not in ('M0','M1','M2','M3'):
            raise ValueError('Expected K>=2 and M0/M1/M2/M3')
    @property
    def temporal(self): return self.variant in ('M2','M3')
    @property
    def switches(self): return self.variant in ('M1','M3')
    @property
    def D(self): return self.K+(self.K if self.temporal else 0)+(2 if self.switches else 0)
    @property
    def feature_names(self):
        out=[f'occupancy_Z{k+1}' for k in range(self.K)]
        if self.temporal:out.extend(f'late_minus_early_Z{k+1}' for k in range(self.K))
        if self.switches:out.extend(('switches_per_100_intervals_early','switches_per_100_intervals_late'))
        return tuple(out)
    @property
    def prior_variances(self):
        return np.r_[np.full(self.K,2.),
                     np.full(self.K,.15) if self.temporal else np.empty(0),
                     np.full(2,.15) if self.switches else np.empty(0)]


def increments(T:int,spec:Spec):
    if T<4:raise ValueError('Need >=4 contiguous time points for split transition features')
    K=spec.K;D=spec.D
    node=np.zeros((T,K,D),dtype=np.float64)
    edge=np.zeros((T-1,K,K,D),dtype=np.float64)
    for t in range(T):
        node[t,np.arange(K),np.arange(K)]=1./T
    o=K
    if spec.temporal:
        ne=T//2;nl=T-ne
        for t in range(T):
            node[t,np.arange(K),o+np.arange(K)]=(-1./ne if t<ne else 1./nl)
        o+=K
    if spec.switches:
        ne=(T-1)//2;nl=(T-1)-ne
        for t in range(T-1):
            for j in range(K):
                for k in range(K):
                    if j!=k:edge[t,j,k,o+(0 if t<ne else 1)]=(100./ne if t<ne else 100./nl)
    return node,edge


@njit(cache=True)
def _moments_grad(q0,Q,node,edge,b,G,logpi,logA,ell):
    """Unnormalized Markov reward moments; exact gradient wrt q0/Q probabilities."""
    T,K,D=node.shape
    P=np.zeros((T,K));M=np.zeros((T,K,D));S=np.zeros((T,K,D,D))
    obj=0.
    for k in range(K):
        pk=q0[k];P[0,k]=pk
        obj+=pk*(logpi[k]+ell[0,k]-np.log(pk))
        for a in range(D):
            va=node[0,k,a];M[0,k,a]=pk*va
            for c in range(D):S[0,k,a,c]=pk*va*node[0,k,c]
    for t in range(1,T):
        for j in range(K):
            pj=P[t-1,j]
            for k in range(K):
                qjk=Q[t-1,j,k]
                obj+=pj*qjk*(logA[j,k]+ell[t,k]-np.log(qjk))
                P[t,k]+=pj*qjk
                for a in range(D):
                    da=node[t,k,a]+edge[t-1,j,k,a]
                    M[t,k,a]+=qjk*(M[t-1,j,a]+pj*da)
                    for c in range(D):
                        dc=node[t,k,c]+edge[t-1,j,k,c]
                        S[t,k,a,c]+=qjk*(S[t-1,j,a,c]+M[t-1,j,a]*dc+da*M[t-1,j,c]+pj*da*dc)
    m=np.zeros(D);H=np.zeros((D,D))
    for k in range(K):
        for a in range(D):
            m[a]+=M[T-1,k,a]
            for c in range(D):H[a,c]+=S[T-1,k,a,c]
    for a in range(D):
        obj+=b[a]*m[a]
        for c in range(D):obj+=G[a,c]*H[a,c]

    # Reverse differentiation through additive Markov reward recursions.
    dp=np.zeros((T,K));dM=np.zeros((T,K,D));dS=np.zeros((T,K,D,D))
    for k in range(K):
        for a in range(D):
            dM[T-1,k,a]=b[a]
            for c in range(D):dS[T-1,k,a,c]=G[a,c]
    dQ=np.zeros((T-1,K,K))
    for t in range(T-1,0,-1):
        for j in range(K):
            pj=P[t-1,j]
            for k in range(K):
                qjk=Q[t-1,j,k]
                reward=logA[j,k]+ell[t,k]-np.log(qjk)
                # Contribution of process terms to gradients.
                dp[t-1,j]+=qjk*reward
                dQ[t-1,j,k]+=pj*(reward-1.)
                # Contributions of current moment arrays to previous arrays.
                dp[t-1,j]+=qjk*dp[t,k]
                dQ[t-1,j,k]+=pj*dp[t,k]
                for a in range(D):
                    da=node[t,k,a]+edge[t-1,j,k,a]
                    dQ[t-1,j,k]+=dM[t,k,a]*(M[t-1,j,a]+pj*da)
                    dp[t-1,j]+=qjk*dM[t,k,a]*da
                    dM[t-1,j,a]+=qjk*dM[t,k,a]
                    for c in range(D):
                        dc=node[t,k,c]+edge[t-1,j,k,c]
                        ds=dS[t,k,a,c]
                        dQ[t-1,j,k]+=ds*(S[t-1,j,a,c]+M[t-1,j,a]*dc+da*M[t-1,j,c]+pj*da*dc)
                        dp[t-1,j]+=qjk*ds*da*dc
                        dM[t-1,j,a]+=qjk*ds*dc
                        dM[t-1,j,c]+=qjk*ds*da
                        dS[t-1,j,a,c]+=qjk*ds
    d0=np.zeros(K)
    for k in range(K):
        d0[k]=logpi[k]+ell[0,k]-np.log(q0[k])-1.+dp[0,k]
        for a in range(D):
            da=node[0,k,a]
            d0[k]+=dM[0,k,a]*da
            for c in range(D):d0[k]+=dS[0,k,a,c]*da*node[0,k,c]
    return obj,d0,dQ,m,H


def moments(q0,Q,spec:Spec):
    q0=np.asarray(q0,float);Q=np.asarray(Q,float)
    if q0.shape!=(spec.K,) or Q.ndim!=3 or Q.shape[1:]!=(spec.K,spec.K):
        raise ValueError('Invalid local Markov dimensions')
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(Q)) or np.min(q0)<=0 or np.min(Q)<=0 or not np.isclose(q0.sum(),1) or not np.allclose(Q.sum(axis=2),1.):
        raise ValueError('Expected normalized strictly positive Markov probabilities')
    node,edge=increments(len(Q)+1,spec)
    _,_,_,m,H=_moments_grad(q0,Q,node,edge,np.zeros(spec.D),np.zeros((spec.D,spec.D)),np.zeros(spec.K),np.zeros((spec.K,spec.K)),np.zeros((len(Q)+1,spec.K)))
    return m,(H+H.T)/2


def objective_logits(v,logpi,logA,ell,b,G,spec:Spec):
    import full_markov_b1 as fm
    T,K=ell.shape
    if K!=spec.K or len(b)!=spec.D or G.shape!=(spec.D,spec.D):raise ValueError('Invalid objective shape')
    q0,Q=fm.unpack_logits(v,T,K)
    node,edge=increments(T,spec)
    f,d0,dQ,m,H=_moments_grad(q0,Q,node,edge,b,G,logpi,logA,ell)
    d0=q0*(d0-q0@d0)
    dQ=Q*(dQ-np.sum(dQ*Q,axis=2,keepdims=True))
    return float(f),np.r_[d0,dQ.ravel()],m,H
