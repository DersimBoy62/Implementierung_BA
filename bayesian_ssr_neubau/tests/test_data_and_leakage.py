"""T7/T9 plus reale F2- und Merkmalsparität gegen unabhängige Alt-Oracles."""
from __future__ import annotations
import ast
import json
import numpy as np
import pytest

from bayesian_ring_ssr.data_io import (DATA_SHA256, LOGGER_COLUMNS, audit_data,
    export_development_logger, export_process_npz, f2_indices, load_development,
    load_development_processes, read_process_csv, read_process_file, sha256)
from bayesian_ring_ssr.schema import FEATURES, TARGETS, LabeledRing, RingProcessSequence, UnlabeledRing
from bayesian_ring_ssr.preprocessing import Preprocessing, fit_preprocessing
from bayesian_ring_ssr.physics_descriptors import extract_descriptors
from bayesian_ring_ssr.split_guard import (DEVELOPMENT_IDS, LABELED_IDS, UNLABELED_IDS,
                                          LOCKED_IDS, development_folds, guard_training_ids)

@pytest.fixture(scope="module")
def development(private_data_path):
    return load_development(private_data_path)


def _synthetic(rid, offset=0, labeled=False):
    t = np.arange(10) * 0.1
    x = np.arange(60).reshape(10, 6) + offset
    return LabeledRing(rid, t, x, targets=np.array([1 + offset, 2 + offset])) if labeled else UnlabeledRing(rid, t, x)


def _old_function(legacy_root, filename, function):
    """Read-only Oracle: nur konkrete alte Funktion, keine Architekturimportabhängigkeit."""
    tree = ast.parse((legacy_root / "final_candidate_v52/final_core" / filename).read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    module = ast.Module(body=[definition], type_ignores=[])
    env = {"np": np}
    exec(compile(module, str(filename), "exec"), env)
    return env[function]


def test_trusted_data_sha_and_ids(development, private_data_path):
    assert sha256(private_data_path) == DATA_SHA256
    labeled, unlabeled = development
    assert tuple(labeled) == LABELED_IDS
    assert tuple(unlabeled) == UNLABELED_IDS
    assert not (set(labeled) | set(unlabeled)) & set(LOCKED_IDS)
    assert len(set(labeled) | set(unlabeled)) == 12


def test_sha_failure_refuses_deserialization(tmp_path):
    f = tmp_path / "untrusted.pkl"
    f.write_bytes(b"not-a-pickle")
    with pytest.raises(ValueError, match="SHA256"):
        load_development(f)


def test_unlabeled_interface_contains_no_quality(development):
    labeled, unlabeled = development
    for ring in unlabeled.values():
        assert isinstance(ring, UnlabeledRing)
        assert not hasattr(ring, "targets") and not hasattr(ring, "y") and not hasattr(ring, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            ring.targets = np.array([10, 20])
    for ring in labeled.values():
        process = ring.process_only()
        assert type(process) is RingProcessSequence
        assert not hasattr(process, "targets")
        assert ring.targets.shape == (2,)


def test_exact_f2_times_rows_and_features_vs_historical_oracle(development, private_data_path, legacy_root):
    # Hash checked before the explicitly authorized local deserialization.
    import pickle
    assert sha256(private_data_path) == DATA_SHA256
    raw = pickle.loads(private_data_path.read_bytes())
    old_view = _old_function(legacy_root, "dm_data.py", "view")
    class OldRing:
        def __init__(self, rid, t, x, y, names):
            self.ring_id, self.t, self.x, self.y, self.names = rid, t, x, y, names
    old_view.__globals__.update(SETS={"V1_Kern_6": FEATURES}, Ring=OldRing, TARGETS=TARGETS)
    old = old_view({rid: raw[rid] for rid in DEVELOPMENT_IDS})
    new = dict(development[0], **development[1])
    for rid in DEVELOPMENT_IDS:
        matrix, columns = raw[rid]["logger"]["matrix"], raw[rid]["logger"]["matrix_columns"]
        phase, laser, mess = (matrix[:, columns.index(k)] for k in ("phase804", "LaserOK", "MessEin"))
        direct_indices = np.flatnonzero(np.isin(phase, (1, 2)) & (laser == 1) & (mess == 1))
        assert np.array_equal(new[rid].row_indices, direct_indices)
        assert np.array_equal(new[rid].times, old[rid].t)
        assert np.array_equal(new[rid].X, old[rid].x)
        assert new[rid].feature_names == old[rid].names == FEATURES
        if rid in development[0]:
            assert np.array_equal(new[rid].targets, old[rid].y)


def test_descriptor_values_bitwise_equal_to_readonly_reference(development, legacy_root):
    old = _old_function(legacy_root, "quality_link_challenge.py", "extract_descriptors")
    pre = fit_preprocessing(*development)
    for ring in dict(development[0], **development[1]).values():
        x = pre.x_scaler.transform(ring.X)
        assert np.array_equal(extract_descriptors(x, ring.times), old(x, ring.times, "physical"))


def test_train_scalers_exclude_validation_labels_and_processes(development):
    l, u = development
    fold = next(development_folds())
    train = {rid: l[rid] for rid in fold["L"]}
    pre = fit_preprocessing(train, u)
    assert pre.x_scaler.fit_ids == tuple(sorted((*fold["L"], *fold["U"])))
    assert pre.y_scaler.fit_ids == tuple(fold["L"])
    assert pre.descriptor_scaler.fit_ids == pre.x_scaler.fit_ids
    assert not set(fold["VAL"]) & set(pre.x_scaler.fit_ids)
    expected_x = np.vstack([r.X for r in sorted([*train.values(), *u.values()], key=lambda r: r.ring_id)])
    expected_y = np.stack([train[rid].targets for rid in sorted(train)])
    np.testing.assert_array_equal(pre.x_scaler.mean, expected_x.mean(0))
    np.testing.assert_array_equal(pre.y_scaler.mean, expected_y.mean(0))
    strict = fit_preprocessing(train, u, include_unlabeled=False)
    assert strict.x_scaler.fit_ids == tuple(fold["L"])
    assert strict.descriptor_scaler.fit_ids == tuple(fold["L"])
    # Counterfactual U corruption can affect the allowed SSL X fit, never y fit.
    modified = {rid: UnlabeledRing(rid, r.times, r.X + 1000, phases=r.phases) for rid, r in u.items()}
    changed = fit_preprocessing(train, modified)
    np.testing.assert_array_equal(pre.y_scaler.mean, changed.y_scaler.mean)
    strict_changed = fit_preprocessing(train, modified, include_unlabeled=False)
    np.testing.assert_array_equal(strict.x_scaler.mean, strict_changed.x_scaler.mean)


def test_scaler_roundtrip_and_process_only_transform(development):
    pre = fit_preprocessing(*development)
    clone = Preprocessing.from_dict(json.loads(json.dumps(pre.to_dict())))
    ring = development[0]["DM04"]
    assert not hasattr(pre.transform_process(ring), "targets")
    np.testing.assert_array_equal(pre.descriptors(ring), clone.descriptors(ring))
    np.testing.assert_allclose(pre.y_scaler.inverse_transform(pre.y_scaler.transform(ring.targets)), ring.targets,
                               rtol=1e-15, atol=1e-15)


def test_complete_ring_csv_process_only_roundtrip(tmp_path, development, private_data_path):
    f = tmp_path / "DM04.csv"
    export_development_logger(private_data_path, "DM04", f)
    assert tuple(f.read_text().splitlines()[0].split(",")) == LOGGER_COLUMNS
    assert "ovality" not in f.read_text() and "planarity" not in f.read_text()
    ring = read_process_csv(f)
    np.testing.assert_array_equal(ring.X, development[0]["DM04"].X)
    np.testing.assert_array_equal(ring.times, development[0]["DM04"].times)
    assert not hasattr(ring, "targets")
    with pytest.raises(FileExistsError):
        export_development_logger(private_data_path, "DM04", f)
    with pytest.raises(ValueError):
        export_development_logger(private_data_path, "DM07", tmp_path / "locked.csv")


def test_process_npz_never_exports_targets(tmp_path, development):
    f = tmp_path / "process.npz"
    export_process_npz(development[0]["DM04"], f)
    with np.load(f, allow_pickle=False) as a:
        assert "targets" not in a.files and "y" not in a.files
    ring = read_process_file(f)
    np.testing.assert_array_equal(ring.X, development[0]["DM04"].X)
    with pytest.raises(FileExistsError):
        export_process_npz(ring, f)


def test_development_folds_separate_ring_ids():
    seen = []
    for fold in development_folds():
        assert len(fold["L"]) == 6 and len(fold["U"]) == 4 and len(fold["VAL"]) == 2
        assert not (set(fold["L"]) | set(fold["U"])) & set(fold["VAL"])
        seen += fold["VAL"]
    assert sorted(seen) == sorted(LABELED_IDS)
    with pytest.raises(ValueError, match="Doppelte"):
        guard_training_ids(["AA", "AA"], [])
    with pytest.raises(ValueError, match="Leakage"):
        guard_training_ids(["AA"], [], ["AA"])
    with pytest.raises(ValueError, match="Außenring"):
        guard_training_ids(["DM07"], [])


@pytest.mark.parametrize("kind", ["short", "nan", "time", "order", "phase"])
def test_invalid_ring_contracts(kind):
    t = np.arange(8) * 0.1
    x = np.ones((8, 6))
    kwargs = {}
    if kind == "short": t, x = t[:3], x[:3]
    if kind == "nan": x[0, 0] = np.nan
    if kind == "time": t[1] = t[0]
    if kind == "order": kwargs["feature_names"] = FEATURES[::-1]
    if kind == "phase": kwargs["phases"] = np.zeros(len(t))
    with pytest.raises(ValueError):
        RingProcessSequence("AA", t, x, **kwargs)


def test_missing_quality_and_wrong_u_type():
    with pytest.raises(ValueError, match="zwei"):
        LabeledRing("AA", np.arange(8), np.ones((8, 6)), targets=[1])
    with pytest.raises(ValueError, match="NaN"):
        LabeledRing("AA", np.arange(8), np.ones((8, 6)), targets=[1, np.nan])
    with pytest.raises(TypeError, match="U muss"):
        fit_preprocessing([_synthetic("AA", labeled=True)], [_synthetic("BB", labeled=True)])


def test_f2_missing_boundary_islands_channels_and_clock_fail():
    n = 20
    a = np.zeros((n, len(LOGGER_COLUMNS)))
    a[:, 0] = np.arange(n) * .1
    a[:, 1:7] = 1
    a[:, 7] = 6
    a[3:10, 7] = 2
    a[10:17, 7] = 1
    a[3:17, 8:] = 1
    assert len(f2_indices(a, LOGGER_COLUMNS)) == 14
    b = a.copy(); b[3:17, 7] = 2
    with pytest.raises(ValueError, match="Phasengrenze"): f2_indices(b, LOGGER_COLUMNS)
    b = a.copy(); b[8, 8] = 0
    with pytest.raises(ValueError, match="zusammenhängend"): f2_indices(b, LOGGER_COLUMNS)
    with pytest.raises(ValueError, match="Loggerkanäle"): f2_indices(a[:, :-1], LOGGER_COLUMNS[:-1])
    b = a.copy(); b[8, 0] += .03
    with pytest.raises(ValueError, match="Raster"): f2_indices(b, LOGGER_COLUMNS)


def test_data_audit_accesses_no_quality_or_locked_processes(private_data_path):
    qa = audit_data(private_data_path)
    assert qa["active_ring_count"] == 16 and qa["development_ring_count"] == 12
    assert qa["locked_processes_accessed"] is False
    assert all(not row["quality_values_accessed"] for row in qa["rings"])
    assert sum(r["f2_rows"] for r in qa["rings"]) == 7440
