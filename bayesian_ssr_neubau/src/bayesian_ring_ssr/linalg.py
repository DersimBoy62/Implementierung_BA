"""Small strictly checked float64 linear algebra helpers."""
import numpy as np
from scipy.linalg import cho_solve


def finite(value, name):
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return out


def spd(value, name):
    out = finite(value, name)
    if out.ndim != 2 or out.shape[0] != out.shape[1] or not len(out):
        raise ValueError(f"{name} must be a nonempty square SPD matrix")
    if not np.allclose(out, out.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    out = (out + out.T) / 2
    np.linalg.cholesky(out)
    return out


def inverse_spd(value):
    value = spd(value, "SPD matrix")
    L = np.linalg.cholesky(value)
    result = cho_solve((L, True), np.eye(len(value)))
    return (result + result.T) / 2


def logdet(value):
    value = spd(value, "logdet matrix")
    return float(2 * np.log(np.diag(np.linalg.cholesky(value))).sum())

