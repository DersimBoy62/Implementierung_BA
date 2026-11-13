"""T7/T9: semantische Modellverträge trotz absichtlich neu versiegelter Hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from bayesian_ring_ssr.config import Config
from bayesian_ring_ssr.coordinate_train import fit
from bayesian_ring_ssr.persistence import load_model, save_model
from bayesian_ring_ssr.posterior_predictive import predict_new_ring
from bayesian_ring_ssr.schema import LabeledRing, UnlabeledRing


@pytest.fixture(scope="module")
def valid_contract_bundle(tmp_path_factory):
    rng = np.random.default_rng(450)
    labeled, unlabeled = [], []
    for n in range(4):
        t = np.arange(9 + n) * .1
        x = rng.normal(size=(len(t), 6)) + .15 * n
        if n < 3:
            labeled.append(LabeledRing(f"PL{n}", t, x, targets=rng.normal(size=2)))
        else:
            unlabeled.append(UnlabeledRing(f"PU{n}", t, x))
    model = fit(labeled, unlabeled, Config(train_epochs=2, local_maxiter=4))
    bundle = tmp_path_factory.mktemp("persist_contract") / "original"
    save_model(model, bundle)
    return bundle, model, unlabeled[0]


def _rewrite_and_reseal(bundle, model_transform=None, contract_transform=None):
    if model_transform is not None:
        path = bundle / "model.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        model_transform(data)
        path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        # Keep the external split document mutually consistent. Invalid TRAIN
        # semantics must still be rejected, independent of crossfile checks.
        (bundle / "split_manifest.json").write_text(json.dumps(data["split_manifest"]), encoding="utf-8")
    if contract_transform is not None:
        path = bundle / "prediction_contract.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        contract_transform(data)
        path.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = bundle / "SHA256_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Assert the fixture really has valid byte integrity after adversarial edit.
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected


def _shrink_scaler(data, key):
    scaler = data["preprocessing"][key]
    scaler["mean"] = scaler["mean"][:-1]
    scaler["scale"] = scaler["scale"][:-1]


def _shrink_head(data):
    head = data["heads"][0]
    head["mean"] = head["mean"][:-1]
    head["V"] = [row[:-1] for row in head["V"][:-1]]


def _wrong_prior_states(data):
    p = data["process_prior"]
    p["pi"] = p["pi"][:-1]
    p["A"] = [row[:-1] for row in p["A"][:-1]]
    p["emissions"] = p["emissions"][:-1]


MUTATIONS = [
    ("K_vs_process", lambda d: d["config"].update(K=3)),
    ("one_head", lambda d: d.update(heads=d["heads"][:1])),
    ("one_head_prior", lambda d: d.update(head_priors=d["head_priors"][:1])),
    ("wrong_head_dimension", _shrink_head),
    ("wrong_x_dimension", lambda d: _shrink_scaler(d, "x_scaler")),
    ("wrong_y_dimension", lambda d: _shrink_scaler(d, "y_scaler")),
    ("wrong_descriptor_dimension", lambda d: _shrink_scaler(d, "descriptor_scaler")),
    ("validation_in_x_fit", lambda d: d["preprocessing"]["x_scaler"]["fit_ids"].append("VAL99")),
    ("unlabeled_in_y_fit", lambda d: d["preprocessing"]["y_scaler"]["fit_ids"].append("PU3")),
    ("validation_in_descriptor_fit", lambda d: d["preprocessing"]["descriptor_scaler"]["fit_ids"].append("VAL99")),
    ("train_validation_overlap", lambda d: d["split_manifest"]["VAL"].append("PL0")),
    ("locked_ring_training", lambda d: d["split_manifest"]["U"].append("DM07")),
    ("wrong_prior_states", _wrong_prior_states),
    ("feature_order", lambda d: d.update(feature_names=d["feature_names"][::-1])),
    ("target_order", lambda d: d.update(targets=d["targets"][::-1])),
    ("phase6_in_model_window", lambda d: d["f2"].update(phase804=[1, 2, 6])),
]


@pytest.mark.parametrize("name,mutation", MUTATIONS, ids=[x[0] for x in MUTATIONS])
def test_signed_but_semantically_invalid_model_is_rejected(valid_contract_bundle, tmp_path, name, mutation):
    source, _, _ = valid_contract_bundle
    bundle = tmp_path / name
    shutil.copytree(source, bundle)
    _rewrite_and_reseal(bundle, model_transform=mutation)
    # Passing the cryptographic gate is necessary, but insufficient: no
    # malformed model may escape load and fail only during prediction.
    with pytest.raises((ValueError, TypeError), match=".+"):
        load_model(bundle)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(requires_targets=True),
    lambda d: d.update(unit="m"),
    lambda d: d.update(head_dimensions=[10, 9]),
    lambda d: d.update(features=d["features"][::-1]),
], ids=["requires_unknown_quality", "wrong_output_unit", "old_full_head_dimensions", "feature_order"])
def test_signed_prediction_side_contract_cannot_contradict_model(valid_contract_bundle, tmp_path, mutation):
    source, _, _ = valid_contract_bundle
    bundle = tmp_path / "side_contract"
    shutil.copytree(source, bundle)
    _rewrite_and_reseal(bundle, contract_transform=mutation)
    with pytest.raises((ValueError, TypeError), match=".+"):
        load_model(bundle)


def test_valid_contract_still_predicts_exactly_after_load(valid_contract_bundle):
    bundle, model, process = valid_contract_bundle
    assert predict_new_ring(load_model(bundle), process) == predict_new_ring(model, process)


def test_manifest_path_traversal_rejected_before_read(valid_contract_bundle, tmp_path):
    source, _, _ = valid_contract_bundle
    bundle = tmp_path / "traversal"
    shutil.copytree(source, bundle)
    manifest_path = bundle / "SHA256_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../outside-secret.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity|manifest|path|Artifact"):
        load_model(bundle)
