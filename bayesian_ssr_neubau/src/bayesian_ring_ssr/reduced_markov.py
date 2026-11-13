"""Finite, time-dependent Markov distributions in a fixed softmax gauge.

The last logit in each probability row is zero. This removes a coordinate
redundancy, without restricting the strictly positive Markov family.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, xlogy


@dataclass(frozen=True)
class MarkovChain:
    initial: np.ndarray
    transitions: np.ndarray

    def __post_init__(self):
        p = np.array(self.initial, dtype=np.float64, copy=True)
        q = np.array(self.transitions, dtype=np.float64, copy=True)
        if p.ndim != 1 or len(p) < 2 or q.ndim != 3 or q.shape[1:] != (len(p), len(p)):
            raise ValueError("Expected initial[K] and transitions[T-1,K,K], K>=2")
        if (not np.isfinite(p).all() or not np.isfinite(q).all()
                or (p < 0).any() or (q < 0).any()):
            raise ValueError("Markov probabilities must be finite and nonnegative")
        if not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-12):
            raise ValueError("Initial probabilities are not normalized")
        if not np.allclose(q.sum(axis=-1), 1., atol=1e-12, rtol=1e-12):
            raise ValueError("Transition rows are not normalized")
        p.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "initial", p)
        object.__setattr__(self, "transitions", q)

    @property
    def T(self):
        return len(self.transitions) + 1

    @property
    def K(self):
        return len(self.initial)

    @property
    def q0(self):
        return self.initial

    @property
    def Q(self):
        return self.transitions

    @property
    def gamma(self):
        g = np.empty((self.T, self.K))
        g[0] = self.initial
        for t, transition in enumerate(self.transitions):
            g[t+1] = g[t] @ transition
        return g

    @property
    def xi(self):
        return self.gamma[:-1, :, None] * self.transitions

    def entropy(self):
        return float(-xlogy(self.initial, self.initial).sum()
                     - (self.gamma[:-1, :, None]
                        * xlogy(self.transitions, self.transitions)).sum())


def dimensions(T: int, K: int):
    if isinstance(T, bool) or isinstance(K, bool) or int(T) != T or int(K) != K or T < 1 or K < 2:
        raise ValueError("Require integers T>=1 and K>=2")
    return K + (T-1)*K*K, (K-1)*(1+(T-1)*K)


def unpack(log_odds: np.ndarray, T: int, K: int) -> MarkovChain:
    _, n = dimensions(T, K)
    v = np.asarray(log_odds, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError("Invalid reduced log-odds dimensions or values")
    rows = np.zeros((1+(T-1)*K, K))
    rows[:, :-1] = v.reshape(-1, K-1)
    p = np.exp(rows - logsumexp(rows, axis=1, keepdims=True))
    return MarkovChain(p[0], p[1:].reshape(T-1, K, K))


def pack(chain: MarkovChain, floor: float = 1e-12) -> np.ndarray:
    """Represent a chain by log-odds after explicit numerical flooring.

    The default matches the historical FB start conversion. Flooring is
    applied only when converting a supplied chain into optimizer coordinates;
    forward-backward itself remains unfloored. A floor of zero requires
    strictly positive probabilities and preserves them up to roundoff.
    """
    if not np.isfinite(floor) or floor < 0 or floor >= 1/chain.K:
        raise ValueError("Probability floor must lie in [0,1/K)")
    rows = np.vstack((chain.initial[None], chain.transitions.reshape(-1, chain.K)))
    if floor == 0 and (rows <= 0).any():
        raise ValueError("Zero probabilities cannot have finite log-odds")
    rows = np.maximum(rows, floor)
    rows /= rows.sum(axis=1, keepdims=True)
    return (np.log(rows[:, :-1])-np.log(rows[:, -1:])).ravel()


def full_to_reduced(full_logits: np.ndarray, T: int, K: int) -> np.ndarray:
    n, _ = dimensions(T, K)
    v = np.asarray(full_logits, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError("Invalid full logits")
    rows = v.reshape(-1, K)
    return (rows[:, :-1]-rows[:, -1:]).ravel()


def reduced_to_full(reduced_logits: np.ndarray, T: int, K: int) -> np.ndarray:
    _, n = dimensions(T, K)
    v = np.asarray(reduced_logits, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError("Invalid reduced logits")
    rows = np.zeros((1+(T-1)*K, K))
    rows[:, :-1] = v.reshape(-1, K-1)
    return rows.ravel()
