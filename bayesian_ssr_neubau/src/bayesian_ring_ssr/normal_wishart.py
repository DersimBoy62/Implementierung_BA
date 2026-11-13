"""Normal-Wishart factors. W is the Wishart SCALE of precision Lambda."""
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma, multigammaln
from .linalg import finite, spd, inverse_spd, logdet


@dataclass(frozen=True)
class NormalWishart:
    mean: np.ndarray
    kappa: float
    nu: float
    W: np.ndarray

    def __post_init__(self):
        m = finite(self.mean, "NW mean").copy()
        W = spd(self.W, "Wishart scale W").copy()
        if m.ndim != 1 or W.shape != (len(m), len(m)):
            raise ValueError("Normal-Wishart dimension mismatch")
        if not np.isfinite([self.kappa, self.nu]).all() or self.kappa <= 0 or self.nu <= len(m)-1:
            raise ValueError("Need kappa>0, nu>D-1")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "W", W)

    @property
    def elog_precision(self):
        d = len(self.mean)
        return float(digamma((self.nu-np.arange(d))/2).sum() + d*np.log(2) + logdet(self.W))

    def expected_log_emission(self, X):
        X = finite(X, "emission X")
        d = len(self.mean)
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError("Emission dimensions do not match")
        delta = X-self.mean
        quadratic = np.einsum("ti,ij,tj->t", delta, self.W, delta)
        return .5*(self.elog_precision - d*np.log(2*np.pi) - d/self.kappa - self.nu*quadratic)

    def update(self, n, sx, sxx):
        sx, sxx = finite(sx, "sum X"), finite(sxx, "sum XX")
        d = len(self.mean)
        if not np.isfinite(n) or n < 0 or sx.shape != (d,) or sxx.shape != (d,d):
            raise ValueError("Invalid weighted emission statistics")
        kappa = self.kappa+n
        mean = (self.kappa*self.mean+sx)/kappa
        # The sufficient-statistic form is valid for fractional U counts.
        rate = inverse_spd(self.W)+sxx+self.kappa*np.outer(self.mean,self.mean)-kappa*np.outer(mean,mean)
        return NormalWishart(mean, float(kappa), float(self.nu+n), inverse_spd((rate+rate.T)/2))

    def kl(self, prior):
        d = len(self.mean)
        if len(prior.mean) != d:
            raise ValueError("NW KL dimension mismatch")
        # E_q[log Wishart(q)-log Wishart(p)] with explicit scale normalizers.
        def log_normalizer(nu, W):
            return nu*d/2*np.log(2)+nu/2*logdet(W)+multigammaln(nu/2,d)
        wish = ((self.nu-prior.nu)/2*self.elog_precision
                +.5*self.nu*(np.trace(np.linalg.solve(prior.W,self.W))-d)
                +log_normalizer(prior.nu,prior.W)-log_normalizer(self.nu,self.W))
        delta = self.mean-prior.mean
        normal = .5*(d*(prior.kappa/self.kappa-1+np.log(self.kappa/prior.kappa))
                     +prior.kappa*self.nu*(delta@self.W@delta))
        return float(wish+normal)

    def to_dict(self):
        return dict(mean=self.mean.tolist(), kappa=self.kappa, nu=self.nu, W=self.W.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

