"""Exact moments of additive state rewards, including temporal covariance.

M2 uses h for all K states and late-minus-early for the first K-1 states.
All K h entries remain because their affine sum encodes the intercept.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from .reduced_markov import MarkovChain


def m2_increments(T: int, K: int, reduced: bool = True):
    if int(T)!=T or int(K)!=K or T<4 or K<2:
        raise ValueError("M2 requires integer T>=4 and K>=2")
    D = 2*K-int(reduced)
    v = np.zeros((T,K,D))
    states = np.arange(K)
    v[:,states,states] = 1/T
    ne = T//2
    for k in range(K-int(reduced)):
        v[:ne,k,K+k] = -1/ne
        v[ne:,k,K+k] = 1/(T-ne)
    return v


def validate_rewards(chain, node):
    v = m2_increments(chain.T,chain.K) if node is None else np.asarray(node,dtype=np.float64)
    if v.ndim!=3 or v.shape[:2]!=(chain.T,chain.K) or v.shape[2]<1 or not np.isfinite(v).all():
        raise ValueError("Expected finite additive node rewards[T,K,D]")
    return np.ascontiguousarray(v)


@njit(cache=True)
def _joint_moments(initial, transitions, node):
    T,K,D = node.shape
    p = initial.copy()
    first = initial[:,None]*node[0]
    second = np.zeros((K,D,D))
    for k in range(K):
        for a in range(D):
            for b in range(D):
                second[k,a,b] = initial[k]*node[0,k,a]*node[0,k,b]
    for t in range(T-1):
        pnew = np.zeros(K)
        mnew = np.zeros((K,D))
        hnew = np.zeros((K,D,D))
        for i in range(K):
            for j in range(K):
                q = transitions[t,i,j]
                pnew[j] += q*p[i]
                for a in range(D):
                    mnew[j,a] += q*(first[i,a]+p[i]*node[t+1,j,a])
                    for b in range(D):
                        hnew[j,a,b] += q*(second[i,a,b]
                            + first[i,a]*node[t+1,j,b]
                            + first[i,b]*node[t+1,j,a]
                            + p[i]*node[t+1,j,a]*node[t+1,j,b])
        p,first,second = pnew,mnew,hnew
    return first.sum(axis=0), second.sum(axis=0)


def moments(chain: MarkovChain, node=None):
    rewards = validate_rewards(chain,node)
    m,H = _joint_moments(chain.initial,chain.transitions,rewards)
    return m,(H+H.T)*.5


def augment_moments(m,H,deterministic):
    s = np.asarray(deterministic,dtype=np.float64)
    m = np.asarray(m,dtype=np.float64)
    H = np.asarray(H,dtype=np.float64)
    if s.ndim!=1 or m.ndim!=1 or H.shape!=(len(m),len(m)) or not all(np.isfinite(x).all() for x in (m,H,s)):
        raise ValueError("Invalid moment augmentation")
    return np.r_[m,s],np.block([[H,np.outer(m,s)],[np.outer(s,m),np.outer(s,s)]])
