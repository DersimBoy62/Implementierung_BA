import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from bayesian_ring_ssr.schema import LabeledRing,UnlabeledRing,RingProcessSequence
from bayesian_ring_ssr.config import Config


@pytest.fixture(scope="session")
def private_data_path():
    """Only real-data tests need the separately supplied private input file."""
    workspace = Path(__file__).resolve().parents[2]
    configured = os.environ.get("BAYESIAN_SSR_DATA")
    path = Path(configured).expanduser() if configured else workspace / "input/DM_16_aktive_Ringe.pkl"
    path = path.resolve()
    if not path.is_file():
        pytest.skip(
            f"Private Rohdaten fehlen: {path}. Datei lokal unter input/ ablegen "
            "oder BAYESIAN_SSR_DATA auf DM_16_aktive_Ringe.pkl setzen."
        )
    return path


@pytest.fixture(scope="session")
def legacy_root():
    """Check every immutable reference source before executing historical code."""
    fixtures = Path(__file__).resolve().parent / "reference_parity/fixtures"
    manifest = json.loads((fixtures / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "bayesian-ring-ssr-reference-fixtures/1"
    root = fixtures / "legacy"
    expected_files = manifest["files"]
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    assert actual_files == set(expected_files), "Historical reference source inventory changed"
    for relative, expected_hash in expected_files.items():
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, (
            f"Historical reference source changed: {relative}"
        )
    return root


@pytest.fixture
def synthetic_rings():
    rng = np.random.default_rng(318)
    labeled,unlabeled = {},{}
    for r in range(7):
        T = 16+2*r
        z = np.empty(T,dtype=int)
        z[0] = rng.integers(2)
        for t in range(1,T):
            z[t] = z[t-1] if rng.uniform()<.8 else 1-z[t-1]
        X = rng.normal(scale=.25,size=(T,6))+z[:,None]*np.array([1.,-.5,.7,-.2,.4,.3])+r*.07
        X[:,2] += np.linspace(0,.3+r*.02,T)
        y = np.array([.6+.4*z.mean()+.2*X[:,0].mean(),2.+2*z.mean()+.5*X[:,1].mean()])+rng.normal(scale=.05,size=2)
        rid = f"SY{r:02}"
        args = (rid,np.arange(T)*.1,X)
        if r<4:
            labeled[rid] = LabeledRing(*args,targets=y)
        elif r<6:
            unlabeled[rid] = UnlabeledRing(*args)
        else:
            test = RingProcessSequence(*args)
    return labeled,unlabeled,test


@pytest.fixture
def small_config():
    return Config(train_epochs=3,local_maxiter=5)


@pytest.fixture
def synthetic_model(synthetic_rings,small_config):
    from bayesian_ring_ssr.coordinate_train import fit
    l,u,_ = synthetic_rings
    return fit(l,u,small_config)

