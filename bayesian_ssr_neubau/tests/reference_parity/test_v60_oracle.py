"""Read-only Parität mit dem tatsächlich SHA-geprüften V6.0-Oracle.

Keine alten Modellzustände, Datenlabels oder Ergebnisdateien werden geladen.
Nur dieser Test importiert die historische Implementierung.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from bayesian_ring_ssr.config import Config
from bayesian_ring_ssr.coordinate_train import fit, head_descriptors
from bayesian_ring_ssr.hmmmoments import moments
from bayesian_ring_ssr.local_vi import objective_and_grad
from bayesian_ring_ssr.nig_heads import NIGHead, augment, head_prior
from bayesian_ring_ssr.normal_wishart import NormalWishart
from bayesian_ring_ssr.priors import m2_lift, process_prior, update_process
from bayesian_ring_ssr.reduced_markov import dimensions, unpack
from bayesian_ring_ssr.schema import LabeledRing, UnlabeledRing

ROOT = Path(__file__).resolve().parents[3]
PHYSICAL_VAR = 0.38526108747593474


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    """Originalcode-Hierarchie prüfen; Numba-Caches in TEST-TEMP schreiben."""
    from numba import config as numba_config
    saved_path, saved_cache = list(sys.path), numba_config.CACHE_DIR
    saved_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    numba_config.CACHE_DIR = str(tmp_path_factory.mktemp("oracle_numba_cache"))
    source = ROOT / "research_math_v60/m2_prior_reduction_v60.py"
    handover = ROOT / "handover/04_REFERENZCODE_READ_ONLY/V60/research_math_v60/m2_prior_reduction_v60.py"
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(handover.read_bytes()).digest()
    module_spec = importlib.util.spec_from_file_location("readonly_v60_parity_oracle", source)
    v60 = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(v60)
    v59, v58, mods = v60.dependencies()  # SHA-gated V58/V59 und alle zehn V52-Kerne
    yield v60, v59, v58, mods
    sys.path[:] = saved_path
    numba_config.CACHE_DIR = saved_cache
    sys.dont_write_bytecode = saved_bytecode


@pytest.mark.parametrize("K", [2, 3, 4])
def test_v60_markov_moments_objective_and_analytic_gradient(oracle, K):
    v60, v59, _, mods = oracle
    _, _, fm, _, _, Spec, old_moments, old_objective = mods
    rng, T = np.random.default_rng(5000 + K), K + 5
    logits = rng.normal(0, .6, dimensions(T, K)[1])
    chain = unpack(logits, T, K)
    q0, transitions = fm.unpack_logits(v59.reduced_to_full(logits, T, K), T, K)
    np.testing.assert_allclose(chain.initial, q0, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(chain.transitions, transitions, atol=2e-15, rtol=2e-15)
    old_m, old_H = old_moments(q0, transitions, Spec(K, "M2"))
    selector, _ = v60.state_selector(K)
    new_m, new_H = moments(chain)
    np.testing.assert_allclose(new_m, selector @ old_m, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(new_H, selector @ old_H @ selector.T, atol=2e-14, rtol=2e-14)
    lp, la, ell = rng.normal(size=K), rng.normal(size=(K, K)), rng.normal(size=(T, K))
    linear = rng.normal(size=2 * K - 1)
    raw = rng.normal(size=(2 * K - 1, 2 * K - 1))
    quadratic = (raw + raw.T) / 2
    full_linear, full_quadratic = selector.T @ linear, selector.T @ quadratic @ selector
    old = v59.reduced_objective_and_grad(logits, lp, la, ell, full_linear,
                                          full_quadratic, Spec(K, "M2"), old_objective)
    new = objective_and_grad(logits, lp, la, ell, linear, quadratic)
    np.testing.assert_allclose(new[0], old[0], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(new[1], old[1], atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_v60_normal_wishart_fractional_updates_kl_and_emission(oracle, K):
    _, _, _, mods = oracle
    _, full, fm, _, base, _, _, _ = mods
    rng, D = np.random.default_rng(1000 + K), 6
    prior = process_prior(K, D)
    old_prior = full.make_priors_full(K, D)
    rings, chains, old_local, weights = {}, {}, {}, {}
    for n in range(3):
        rid, T = f"S{n}", n + 7
        rings[rid] = rng.normal(size=(T, D))
        chains[rid] = unpack(rng.normal(size=dimensions(T, K)[1]), T, K)
        old_local[rid] = {"fb": fm.markov_marginals(chains[rid].initial, chains[rid].transitions)}
        weights[rid] = .2 if n == 2 else 1.
    old = full.update_process_full(rings, old_local, old_prior, weights)
    new, stats = update_process(prior, rings, chains, weights)
    np.testing.assert_allclose(new.pi, old.pi, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(new.A, old.A, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(new.kl_parts(prior)["kl_initial"], base.kl_dir(old.pi, old_prior.pi), atol=2e-12)
    old_ell = full.elog_emission_full(rings["S0"], old)
    for k, emission in enumerate(new.emissions):
        np.testing.assert_allclose(emission.mean, old.mu[k], atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(emission.W, old.W[k], atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose([emission.kappa, emission.nu], [old.kappa[k], old.nu[k]], atol=3e-13)
        np.testing.assert_allclose(emission.elog_precision, full.expected_log_precision(old.nu[k], old.W[k]), atol=3e-12)
        np.testing.assert_allclose(emission.expected_log_emission(rings["S0"]), old_ell[:, k], atol=3e-12, rtol=3e-13)
        old_kl = full.kl_normal_wishart(old.mu[k], old.kappa[k], old.nu[k], old.W[k],
                                       old_prior.mu[k], old_prior.kappa[k], old_prior.nu[k], old_prior.W[k])
        np.testing.assert_allclose(emission.kl(prior.emissions[k]), old_kl, atol=3e-12, rtol=3e-13)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_v60_induced_nig_posterior_likelihood_kl_prediction(oracle, K):
    v60, _, _, mods = oracle
    dyn, _, _, th, base, Spec, _, _ = mods
    rng, nphys = np.random.default_rng(700 + K), 3
    full_prior = dyn.prior_head(Spec(K, "M2"), nphys, PHYSICAL_VAR)
    old_prior = v60.reduce_head_prior(full_prior, K, nphys, th)
    new_prior = head_prior(K, nphys, PHYSICAL_VAR)
    np.testing.assert_array_equal(new_prior.V, old_prior.V)
    np.testing.assert_array_equal(m2_lift(K, nphys), v60.full_feature_map(K, nphys)[1])
    ydata, local, labels, all_moments = {}, {}, [], []
    for n in range(4):
        T = n + 6
        chain = unpack(rng.normal(size=dimensions(T, K)[1]), T, K)
        m, H = moments(chain)
        m, H = augment(m, H, rng.normal(size=nphys))
        yy = float(rng.normal())
        ydata[f"S{n}"] = [yy]
        labels.append(yy)
        all_moments.append((m, H))
        local[f"S{n}"] = {"head_moments": [(m, H)]}
    old = th.fit_head(ydata, local, 0, old_prior)
    new = new_prior.update(labels, all_moments)
    np.testing.assert_allclose(new.mean, old.mu, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(new.V, old.V, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose([new.a, new.b], [old.a, old.b], atol=3e-13, rtol=3e-13)
    old_kl = base.kl_reg(old.mu, old.V, old.a, old.b, old_prior.mu, old_prior.V, old_prior.a, old_prior.b)
    np.testing.assert_allclose(new.kl(new_prior), old_kl, atol=3e-12, rtol=3e-13)
    for y, (m, H) in zip(labels, all_moments):
        np.testing.assert_allclose(new.expected_log_likelihood(y, m, H), th.quality_logterm(y, m, H, old), atol=3e-13, rtol=3e-13)
        old_mean = float(old.mu @ m)
        old_variance = float(old.b / (old.a - 1) * (1 + np.trace(old.V @ H))
                             + max(0., old.mu @ H @ old.mu - old_mean ** 2))
        np.testing.assert_allclose(new.predictive_moments(m, H), [old_mean, old_variance], atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_v60_three_epoch_training_from_same_fresh_initialization(oracle, K):
    v60, _, _, _ = oracle
    rng = np.random.default_rng(600 + K)
    labeled, unlabeled = [], []
    for n in range(4):
        T, rid = 8 + n, f"S{n}"
        t = np.arange(T) * .1
        x = rng.normal(size=(T, 6)) + n * .3
        if n < 3:
            labeled.append(LabeledRing(rid, t, x, targets=rng.normal(size=2)))
        else:
            unlabeled.append(UnlabeledRing(rid, t, x))
    cfg = Config(K=K, C=.2, train_epochs=3, local_maxiter=8)
    new = fit(labeled, unlabeled, cfg)
    raw = {r.ring_id: r for r in sorted(labeled + unlabeled, key=lambda r: r.ring_id)}
    xdata = {rid: new.preprocessing.transform_process(r).X for rid, r in raw.items()}
    ydata = {r.ring_id: new.preprocessing.y_scaler.transform(r.targets) for r in labeled}
    sdata = {rid: head_descriptors(new.preprocessing, r) for rid, r in raw.items()}
    with threadpool_limits(limits=1):
        old = v60.train_m2_reduced(xdata, ydata, sdata, K, .2, 3, cfg.seed,
                                   cfg.local_maxiter, PHYSICAL_VAR)
    # Independent exact gradients share the objective; finite optimization can
    # differ slightly. Absolute tolerance fixed before executing this test.
    np.testing.assert_allclose([h["elbo"] for h in new.history],
                               [h["elbo"] for h in old["history"]], atol=2e-6, rtol=0)
    for new_head, old_head in zip(new.heads, old["head_posteriors"]):
        np.testing.assert_allclose(new_head.mean, old_head.mu, atol=2e-6, rtol=0)
        np.testing.assert_allclose(new_head.b, old_head.b, atol=2e-6, rtol=0)
