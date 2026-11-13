"""Dirichlet process factors and prior-preserving M2 feature map."""
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma, gammaln
from .linalg import finite
from .normal_wishart import NormalWishart


def elog_dirichlet(alpha):
    alpha = finite(alpha, "Dirichlet alpha")
    if alpha.ndim not in (1,2) or np.any(alpha <= 0):
        raise ValueError("Dirichlet concentration must be positive")
    return digamma(alpha)-digamma(alpha.sum(axis=-1,keepdims=True))


def kl_dirichlet(alpha, prior):
    alpha, prior = finite(alpha,"alpha"), finite(prior,"prior")
    if alpha.shape != prior.shape or alpha.ndim != 1 or np.any(alpha<=0) or np.any(prior<=0):
        raise ValueError("Invalid Dirichlet KL parameters")
    return float(gammaln(alpha.sum())-gammaln(alpha).sum()-gammaln(prior.sum())
                 +gammaln(prior).sum()+(alpha-prior)@elog_dirichlet(alpha))


def m2_lift(K, nphys=0):
    if K < 2 or nphys < 0:
        raise ValueError("Invalid M2 dimensions")
    A = np.zeros((2*K+nphys, 2*K-1+nphys))
    A[:K,:K] = np.eye(K)
    A[K:2*K-1,K:2*K-1] = np.eye(K-1)
    A[2*K-1,K:2*K-1] = -1
    A[2*K:,2*K-1:] = np.eye(nphys)
    return A


@dataclass(frozen=True)
class ProcessPosterior:
    pi: np.ndarray
    A: np.ndarray
    emissions: tuple

    def __post_init__(self):
        pi, A = finite(self.pi,"pi"), finite(self.A,"A")
        K = len(self.emissions)
        if pi.shape != (K,) or A.shape != (K,K) or np.any(pi<=0) or np.any(A<=0):
            raise ValueError("Invalid process concentrations")
        if len({len(x.mean) for x in self.emissions}) != 1:
            raise ValueError("Inconsistent emission dimensions")
        object.__setattr__(self,"pi",pi.copy())
        object.__setattr__(self,"A",A.copy())

    def potentials(self, X):
        return elog_dirichlet(self.pi), elog_dirichlet(self.A), np.column_stack([e.expected_log_emission(X) for e in self.emissions])

    def kl_parts(self, prior):
        return dict(kl_initial=kl_dirichlet(self.pi,prior.pi),
                    kl_transition=sum(kl_dirichlet(a,b) for a,b in zip(self.A,prior.A)),
                    kl_nw=sum(a.kl(b) for a,b in zip(self.emissions,prior.emissions)))

    def to_dict(self):
        return dict(pi=self.pi.tolist(),A=self.A.tolist(),emissions=[x.to_dict() for x in self.emissions])

    @classmethod
    def from_dict(cls, d):
        return cls(d["pi"],d["A"],tuple(NormalWishart.from_dict(x) for x in d["emissions"]))


def process_prior(K, D=6):
    return ProcessPosterior(np.ones(K),.5*np.ones((K,K))+4*np.eye(K),
                            tuple(NormalWishart(np.zeros(D),.2,float(D+3),np.eye(D)/(D+3)) for _ in range(K)))


def update_process(prior, rings, chains, weights):
    K, D = len(prior.pi), len(prior.emissions[0].mean)
    n, sx, sxx = np.zeros(K), np.zeros((K,D)), np.zeros((K,D,D))
    pi, A = prior.pi.copy(), prior.A.copy()
    for rid, chain in chains.items():
        w = weights[rid]
        if w == 0:
            continue
        X = rings[rid]
        g, xi = chain.gamma, chain.xi
        pi += w*g[0]
        A += w*xi.sum(axis=0)
        n += w*g.sum(axis=0)
        sx += w*g.T@X
        sxx += w*np.einsum("tk,ti,tj->kij",g,X,X)
    out = ProcessPosterior(pi,A,tuple(prior.emissions[k].update(n[k],sx[k],sxx[k]) for k in range(K)))
    return out, dict(state_N=n.tolist(),sum_X=sx.tolist(),sum_XX=sxx.tolist())

