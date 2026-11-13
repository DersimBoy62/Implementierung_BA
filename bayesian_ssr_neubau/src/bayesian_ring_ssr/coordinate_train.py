"""Fresh coordinate-ascent training; no historical model import or state loading."""
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .config import Config
from .schema import LabeledRing, UnlabeledRing
from .split_guard import guard_training_ids
from .preprocessing import fit_preprocessing
from .physics_descriptors import PHYSICAL_DESCRIPTOR_INDICES
from .priors import process_prior, ProcessPosterior, update_process
from .normal_wishart import NormalWishart
from .nig_heads import head_prior, augment
from .forward_backward import FB
from .hmmmoments import moments
from .local_vi import optimize_local
from .elbo import evaluate,require_nondecrease


@dataclass
class TrainedModel:
    config: Config
    preprocessing: object
    process: ProcessPosterior
    heads: tuple
    process_prior: ProcessPosterior
    head_priors: tuple
    split_manifest: dict
    dataset_sha256: str
    history: list
    update_history: list
    local_chains: dict = field(default_factory=dict,repr=False)
    math_test_status: str = "NOT_RECORDED"


def head_descriptors(preprocessing, ring):
    desc = preprocessing.descriptors(ring)
    return tuple(desc[list(PHYSICAL_DESCRIPTOR_INDICES[name])] for name in ("outside","height"))


def fit(labeled, unlabeled=(), config=None, dataset_sha256="SYNTHETIC", callback=None):
    config = config or Config()
    labeled = list(labeled.values()) if hasattr(labeled,"values") else list(labeled)
    unlabeled = list(unlabeled.values()) if hasattr(unlabeled,"values") else list(unlabeled)
    if not all(isinstance(r,LabeledRing) for r in labeled):
        raise TypeError("L training requires two known ring targets")
    if not all(isinstance(r,UnlabeledRing) for r in unlabeled):
        raise TypeError("U training requires UnlabeledRing process-only objects")
    split = guard_training_ids([r.ring_id for r in labeled],[r.ring_id for r in unlabeled])
    prep = fit_preprocessing(labeled,unlabeled,config.include_unlabeled_preprocessing)
    raw = {r.ring_id:r for r in sorted(labeled+unlabeled,key=lambda r:r.ring_id)}
    rings = {rid:prep.transform_process(r).X for rid,r in raw.items()}
    labels = {r.ring_id:prep.y_scaler.transform(r.targets) for r in labeled}
    descriptors = {rid:head_descriptors(prep,r) for rid,r in raw.items()}
    prior = process_prior(config.K,6)
    hprior = tuple(head_prior(config.K,n,config.physical_var) for n in (6,5))
    # Historical sampling policy, freshly fitted on this fold's permitted X.
    init_ids = tuple(rid for rid in rings if rid in labels or config.include_unlabeled_preprocessing or config.C>0)
    samples = np.vstack([rings[r][::max(1,len(rings[r])//120)] for r in init_ids])
    with threadpool_limits(limits=1):
        centers = KMeans(n_clusters=config.K,n_init=5,random_state=config.seed).fit(samples).cluster_centers_
    p = ProcessPosterior(prior.pi,prior.A,tuple(NormalWishart(centers[k],4.,prior.emissions[k].nu,prior.emissions[k].W) for k in range(config.K)))
    heads = hprior
    weights = {rid:(1. if rid in labels else config.C) for rid in rings}
    chains = {rid:FB(*p.potentials(X)).chain for rid,X in rings.items() if weights[rid]>0}
    history,events = [],[]
    current,ring_parts = evaluate(rings,chains,p,heads,prior,hprior,labels,descriptors,config.C)

    def record(epoch,stage,before,after,**extra):
        delta = require_nondecrease(before["total"],after["total"],config.monotonic_atol,stage)
        row = dict(epoch=epoch,stage=stage,ELBO_before=before["total"],ELBO_after=after["total"],
                   delta=delta,components_before=before,components_after=after,**extra)
        events.append(row)
        if callback:
            callback(row)
        return delta

    for epoch in range(1,config.train_epochs+1):
        epoch_start = current["total"]
        local_diagnostics = {}
        for rid in chains:
            before = current
            if rid in labels:
                d = 2*config.K-1
                linear,quadratic,constant = np.zeros(d),np.zeros((d,d)),0.
                for j in range(2):
                    b,G,c = heads[j].compressed(labels[rid][j],descriptors[rid][j])
                    linear += b
                    quadratic += G
                    constant += c
                result = optimize_local(*p.potentials(rings[rid]),linear,quadratic,constant,
                                        previous=chains[rid],max_iter=config.local_maxiter)
                chains[rid] = result.chain
                diag = result.diagnostics
            else:
                chains[rid] = FB(*p.potentials(rings[rid])).chain
                diag = dict(message="exact process FB; no target interface",weight=config.C)
            current,ring_parts = evaluate(rings,chains,p,heads,prior,hprior,labels,descriptors,config.C)
            actual_gain = record(epoch,"local",before,current,ring_id=rid,optimizer=diag)
            local_diagnostics[rid] = dict(full_objective_gain=actual_gain,**diag)
        before = current
        new_p,stats = update_process(prior,rings,chains,weights)
        drift = dict(emission_mean_change_l2=float(np.linalg.norm(np.stack([e.mean for e in new_p.emissions])-np.stack([e.mean for e in p.emissions]))),
                     emission_scale_change_l2=float(np.linalg.norm(np.stack([e.W for e in new_p.emissions])-np.stack([e.W for e in p.emissions]))),
                     dirichlet_initial_mean=(new_p.pi/new_p.pi.sum()).tolist(),
                     dirichlet_transition_mean=(new_p.A/new_p.A.sum(axis=1,keepdims=True)).tolist())
        p = new_p
        current,ring_parts = evaluate(rings,chains,p,heads,prior,hprior,labels,descriptors,config.C)
        process_gain = record(epoch,"process",before,current,sufficient_statistics=stats,parameter_changes=drift)
        all_moments = {rid:moments(chains[rid]) for rid in labels}
        for j in range(2):
            before = current
            head_moments = [augment(*all_moments[rid],descriptors[rid][j]) for rid in labels]
            updated = hprior[j].update([labels[rid][j] for rid in labels],head_moments)
            old_head = heads[j]
            heads = tuple(updated if jj==j else heads[jj] for jj in range(2))
            current,ring_parts = evaluate(rings,chains,p,heads,prior,hprior,labels,descriptors,config.C)
            record(epoch,f"head_{j}",before,current,mean_change_l2=float(np.linalg.norm(updated.mean-old_head.mean)),
                   covariance_change_l2=float(np.linalg.norm(updated.V-old_head.V)))
        phase_moments = {}
        for rid,chain in chains.items():
            gamma = chain.gamma
            if raw[rid].phases is not None:
                phase_moments[rid] = {str(phase):gamma[raw[rid].phases==phase].mean(axis=0).tolist() for phase in (1,2) if np.any(raw[rid].phases==phase)}
        row = dict(epoch=epoch,elbo=current["total"],epoch_gain=current["total"]-epoch_start,
                   delta_process=process_gain,components=current,state_N=stats["state_N"],
                   local=local_diagnostics,phase_state_means=phase_moments,
                   dirichlet_transition_mean=drift["dirichlet_transition_mean"])
        history.append(row)
        if callback:
            callback(dict(stage="epoch_complete",**row,checkpoint=dict(process=p.to_dict(),heads=[h.to_dict() for h in heads],
                local={rid:dict(initial=c.initial.tolist(),transitions=c.transitions.tolist()) for rid,c in chains.items()})))
    return TrainedModel(config,prep,p,heads,prior,hprior,split,dataset_sha256,history,events,chains)

