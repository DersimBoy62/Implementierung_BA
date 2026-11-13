"""Log-space sum-product for a finite HMM with time-varying node evidence."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .reduced_markov import MarkovChain


@dataclass(frozen=True)
class ForwardBackwardResult:
    chain: MarkovChain
    gamma: np.ndarray
    xi: np.ndarray
    log_normalizer: float

    @property
    def logZ(self):
        return self.log_normalizer


def validate_potentials(log_initial, log_transition, emission):
    lp = np.asarray(log_initial, dtype=np.float64)
    la = np.asarray(log_transition, dtype=np.float64)
    ell = np.asarray(emission, dtype=np.float64)
    if (lp.ndim != 1 or len(lp) < 2 or la.shape != (len(lp),len(lp))
            or ell.ndim != 2 or ell.shape[1] != len(lp) or len(ell) < 1):
        raise ValueError("Expected log_initial[K], log_transition[K,K], emission[T,K]")
    if any(not np.isfinite(a).all() for a in (lp,la,ell)):
        raise ValueError("HMM log potentials must be finite")
    return lp,la,ell


def forward_backward(log_initial, log_transition, emission):
    lp, la, ell = validate_potentials(log_initial, log_transition, emission)
    T, K = ell.shape
    alpha = np.empty((T,K))
    beta = np.zeros((T,K))
    # Normalize each forward slice; accumulated normalizers retain log Z.
    scale = np.empty(T)
    alpha[0] = lp+ell[0]
    scale[0] = logsumexp(alpha[0])
    alpha[0] -= scale[0]
    for t in range(1,T):
        alpha[t] = ell[t]+logsumexp(alpha[t-1,:,None]+la,axis=0)
        scale[t] = logsumexp(alpha[t])
        alpha[t] -= scale[t]
    for t in range(T-2,-1,-1):
        beta[t] = logsumexp(la+ell[t+1,None,:]+beta[t+1,None,:],axis=1)-scale[t+1]
    log_gamma = alpha+beta
    log_gamma -= logsumexp(log_gamma,axis=1,keepdims=True)
    gamma = np.exp(log_gamma)
    # Direct conditional transitions avoid xi/gamma divisions at tiny mass.
    log_q = la[None,:,:]+ell[1:,None,:]+beta[1:,None,:]
    log_q -= logsumexp(log_q,axis=2,keepdims=True)
    chain = MarkovChain(gamma[0],np.exp(log_q))
    # Marginals of the returned chain give exact mutual consistency even at
    # extreme potentials. Independently calculated gamma agrees to roundoff.
    return ForwardBackwardResult(chain,chain.gamma,chain.xi,float(scale.sum()))


FB = forward_backward
