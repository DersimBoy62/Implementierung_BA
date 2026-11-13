"""The same fully weighted variational objective for every coordinate block."""
import numpy as np
from .hmmmoments import moments
from .nig_heads import augment


def ring_terms(X, chain, process, heads=None, y=None, descriptors=None):
    lp,la,ell = process.potentials(X)
    gamma,xi = chain.gamma,chain.xi
    out = dict(initial=float(gamma[0]@lp),transition=float(np.sum(xi*la)),
               emission=float(np.sum(gamma*ell)),entropy=chain.entropy())
    if y is not None:
        m,H = moments(chain)
        for j in range(2):
            a,HH = augment(m,H,descriptors[j])
            out[f"quality_{j}"] = heads[j].expected_log_likelihood(float(y[j]),a,HH)
    return out


def evaluate(rings, chains, process, heads, process_prior, head_priors, labels, descriptors, C):
    result = {f"{group}_{name}":0. for group in ("L","weighted_U")
              for name in ("initial","transition","emission","entropy")}
    result.update(quality_0=0.,quality_1=0.)
    per_ring = {}
    for rid,chain in chains.items():
        labeled = rid in labels
        terms = ring_terms(rings[rid],chain,process,heads,labels.get(rid),descriptors[rid])
        weight,group = (1.,"L") if labeled else (C,"weighted_U")
        for name,value in terms.items():
            key = name if name.startswith("quality_") else f"{group}_{name}"
            result[key] += weight*value
        per_ring[rid] = dict(weight=weight,unweighted=terms)
    result.update(process.kl_parts(process_prior))
    result.update({f"kl_head_{j}":heads[j].kl(head_priors[j]) for j in range(2)})
    result["total"] = float(sum(value if not name.startswith("kl_") else -value for name,value in result.items()))
    result["emission_NW"] = result["L_emission"]+result["weighted_U_emission"]-result["kl_nw"]
    result["transition_Dirichlet"] = result["L_transition"]+result["weighted_U_transition"]-result["kl_transition"]
    result["initial_Dirichlet"] = result["L_initial"]+result["weighted_U_initial"]-result["kl_initial"]
    if not all(np.isfinite(value) for value in result.values()):
        raise ArithmeticError("Nonfinite variational objective")
    return result,per_ring


def require_nondecrease(before, after, atol=1e-6, stage="update"):
    delta = float(after-before)
    if not np.isfinite(delta) or delta < -atol:
        raise ArithmeticError(f"ELBO decreased at {stage}: {delta:.12g}; tolerance={atol}")
    return delta

