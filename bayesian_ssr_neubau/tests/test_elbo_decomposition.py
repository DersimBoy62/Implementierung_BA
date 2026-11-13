import numpy as np
import pytest
from dataclasses import replace
from bayesian_ring_ssr.coordinate_train import fit,head_descriptors
from bayesian_ring_ssr.elbo import evaluate
from bayesian_ring_ssr.reduced_markov import pack,unpack
from bayesian_ring_ssr.local_vi import objective_and_grad
from bayesian_ring_ssr.priors import update_process


def core_inputs(model,l,u):
    all_r = {**l,**u}
    X = {rid:model.preprocessing.transform_process(r).X for rid,r in all_r.items()}
    y = {rid:model.preprocessing.y_scaler.transform(r.targets) for rid,r in l.items()}
    s = {rid:head_descriptors(model.preprocessing,r) for rid,r in all_r.items()}
    return X,y,s


def test_T6_components_weight_entropy_and_exact_local_delta(synthetic_model,synthetic_rings):
    model = synthetic_model
    l,u,_ = synthetic_rings
    X,y,s = core_inputs(model,l,u)
    def ev(chains,C):
        return evaluate(X,chains,model.process,model.heads,model.process_prior,model.head_priors,y,s,C)[0]
    base = ev(model.local_chains,.2)
    summed = sum(v if not k.startswith('kl_') else -v for k,v in base.items() if k.startswith(('L_','weighted_U_','quality_','kl_')))
    assert base['total'] == pytest.approx(summed,abs=2e-10)
    no_u = ev(model.local_chains,0.)
    full_u = ev(model.local_chains,1.)
    assert base['total']-no_u['total'] == pytest.approx(.2*(full_u['total']-no_u['total']),abs=2e-10)
    assert base['weighted_U_entropy'] == pytest.approx(.2*sum(model.local_chains[r].entropy() for r in u),abs=1e-11)
    rid = next(iter(l))
    chain = model.local_chains[rid]
    v = pack(chain,floor=0)
    alt = v+np.random.default_rng(4).normal(scale=.03,size=len(v))
    d = 2*model.config.K-1
    b,G,c = np.zeros(d),np.zeros((d,d)),0.
    for j in range(2):
        bj,Gj,cj = model.heads[j].compressed(y[rid][j],s[rid][j])
        b,G,c = b+bj,G+Gj,c+cj
    f0 = objective_and_grad(v,*model.process.potentials(X[rid]),b,G,c)[0]
    f1 = objective_and_grad(alt,*model.process.potentials(X[rid]),b,G,c)[0]
    changed = dict(model.local_chains)
    changed[rid] = unpack(alt,chain.T,chain.K)
    assert ev(changed,.2)['total']-base['total'] == pytest.approx(f1-f0,abs=5e-10)


def test_T6_weighted_sufficient_statistics_independent(synthetic_model,synthetic_rings):
    model = synthetic_model
    l,u,_ = synthetic_rings
    X,_,_ = core_inputs(model,l,u)
    w = {rid:1. if rid in l else .2 for rid in X}
    p,stats = update_process(model.process_prior,X,model.local_chains,w)
    K = model.config.K
    n = np.zeros(K)
    sx = np.zeros((K,6))
    sxx = np.zeros((K,6,6))
    for rid,values in X.items():
        for t,value in enumerate(values):
            for k in range(K):
                weight = w[rid]*model.local_chains[rid].gamma[t,k]
                n[k] += weight
                sx[k] += weight*value
                sxx[k] += weight*np.outer(value,value)
    np.testing.assert_allclose(stats['state_N'],n,atol=1e-12)
    np.testing.assert_allclose(stats['sum_X'],sx,atol=1e-12)
    np.testing.assert_allclose(stats['sum_XX'],sxx,atol=1e-12)
    for k in range(K):
        assert p.emissions[k].kappa == pytest.approx(.2+n[k])
        assert p.emissions[k].nu == pytest.approx(9+n[k])
    assert np.sum(n) == pytest.approx(sum(len(r.X) for r in l.values())+.2*sum(len(r.X) for r in u.values()))


def test_T6_C0_has_no_U_posterior_update(synthetic_rings,small_config):
    l,u,_ = synthetic_rings
    cfg = replace(small_config,C=0.,include_unlabeled_preprocessing=False)
    model = fit(l,u,cfg)
    assert set(model.local_chains)==set(l)
    assert set(model.preprocessing.x_scaler.fit_ids)==set(l)
    assert set(model.preprocessing.y_scaler.fit_ids)==set(l)
    assert all(event.get('ring_id') not in u for event in model.update_history)
    empty_u = fit(l,(),cfg)
    np.testing.assert_allclose(model.process.A,empty_u.process.A,atol=1e-12)
    np.testing.assert_allclose(model.heads[0].mean,empty_u.heads[0].mean,atol=1e-12)

