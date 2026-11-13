"""Normal-Inverse-Gamma heads; p(s)=b^a/Gamma(a) s^(-a-1) exp(-b/s)."""
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma, gammaln
from .linalg import finite, spd, inverse_spd, logdet
from .priors import m2_lift


@dataclass(frozen=True)
class NIGHead:
    mean: np.ndarray
    V: np.ndarray
    a: float
    b: float

    def __post_init__(self):
        m = finite(self.mean,"head mean")
        V = spd(self.V,"head V")
        if m.ndim!=1 or V.shape!=(len(m),len(m)):
            raise ValueError("Head dimension mismatch")
        if not np.isfinite([self.a,self.b]).all() or self.a<=0 or self.b<=0:
            raise ValueError("NIG a,b must be positive")
        object.__setattr__(self,"mean",m.copy())
        object.__setattr__(self,"V",V.copy())

    def coefficients(self):
        precision = self.a/self.b
        G = -.5*(precision*np.outer(self.mean,self.mean)+self.V)
        return precision,G

    def expected_log_likelihood(self, y, m, H):
        precision,G = self.coefficients()
        return float(-.5*np.log(2*np.pi)+.5*(digamma(self.a)-np.log(self.b))
                     -.5*precision*y*y+precision*y*self.mean@m+np.sum(G*H))

    def compressed(self, y, s):
        s = finite(s,"physical descriptor")
        d = len(self.mean)-len(s)
        if d < 1 or s.ndim != 1 or not np.isfinite(y):
            raise ValueError("Invalid compressed head data")
        precision,G = self.coefficients()
        b = precision*y*self.mean
        constant = -.5*np.log(2*np.pi)+.5*(digamma(self.a)-np.log(self.b))-.5*precision*y*y
        return b[:d]+2*G[:d,d:]@s, G[:d,:d], float(constant+b[d:]@s+s@G[d:,d:]@s)

    def update(self, ys, moments):
        ys = finite(ys,"head labels")
        if ys.ndim!=1 or len(ys)!=len(moments):
            raise ValueError("Labels/moments differ")
        P0 = inverse_spd(self.V)
        P, rhs = P0.copy(), P0@self.mean
        for y,(m,H) in zip(ys,moments):
            m,H = finite(m,"head m"),finite(H,"head H")
            if m.shape!=self.mean.shape or H.shape!=self.V.shape:
                raise ValueError("Head moments have wrong dimension")
            if not np.allclose(H,H.T,atol=1e-12) or np.linalg.eigvalsh(H-np.outer(m,m)).min() < -1e-9:
                raise ValueError("Inconsistent second moments")
            P += H
            rhs += y*m
        V = inverse_spd(P)
        mean = V@rhs
        b = self.b+.5*(ys@ys+self.mean@P0@self.mean-mean@P@mean)
        return NIGHead(mean,V,float(self.a+len(ys)/2),float(b))

    def kl(self, prior):
        delta = self.mean-prior.mean
        P0 = inverse_spd(prior.V)
        ig = (prior.a*np.log(self.b/prior.b)+gammaln(prior.a)-gammaln(self.a)
              +(self.a-prior.a)*digamma(self.a)+self.a*(prior.b/self.b-1))
        normal = .5*(np.trace(P0@self.V)-len(delta)+logdet(prior.V)-logdet(self.V)
                     +self.a/self.b*(delta@P0@delta))
        return float(ig+normal)

    def predictive_moments(self, m, H):
        if self.a <= 1:
            raise ValueError("Predictive variance requires a>1")
        mean = float(self.mean@m)
        state_variance = float(self.mean@(H-np.outer(m,m))@self.mean)
        if state_variance < -1e-9:
            raise ArithmeticError("Negative Markov predictive variance")
        variance = self.b/(self.a-1)*(1+np.sum(self.V*H))+max(0,state_variance)
        if not np.isfinite(variance) or variance<=0:
            raise ArithmeticError("Invalid predictive variance")
        return mean,float(variance)

    def to_dict(self):
        return dict(mean=self.mean.tolist(),V=self.V.tolist(),a=self.a,b=self.b)

    @classmethod
    def from_dict(cls,d):
        return cls(**d)


def head_prior(K, nphys, physical_var, reduced=True):
    full_mean = np.zeros(2*K+nphys)
    full_V = np.diag(np.r_[np.full(K,2.),np.full(K,.15),np.full(nphys,physical_var)])
    if reduced:
        A = m2_lift(K,nphys)
        return NIGHead(A.T@full_mean,A.T@full_V@A,2.5,1.5)
    return NIGHead(full_mean,full_V,2.5,1.5)


def augment(m, H, s):
    m,H,s = finite(m,"m"),finite(H,"H"),finite(s,"s")
    if m.ndim!=1 or s.ndim!=1 or H.shape!=(len(m),len(m)):
        raise ValueError("Invalid augmented moments")
    out = np.concatenate((m,s))
    HH = np.outer(out,out)
    HH[:len(m),:len(m)] = H
    return out,HH
