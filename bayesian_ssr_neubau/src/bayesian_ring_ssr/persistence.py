"""Inspectable JSON persistence and verified write-once artifact directories."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from . import __version__
from .config import Config
from .schema import FEATURES,TARGETS
from .data_io import F2_CRITERIA
from .preprocessing import Preprocessing
from .priors import ProcessPosterior
from .nig_heads import NIGHead
from .coordinate_train import TrainedModel
from .physics_descriptors import DESCRIPTOR_NAMES,PHYSICAL_DESCRIPTOR_INDICES
from .split_guard import guard_training_ids

SCHEMA_VERSION = "bayesian-ring-ssr/1"


def write_json(path,data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")


def environment():
    result = {"python":platform.python_version(),"platform":platform.platform()}
    for name in ("numpy","scipy","scikit-learn","pandas","numba","pytest","pypdf","setuptools","wheel"):
        result[name] = importlib.metadata.version(name)
    return result


def model_dict(model):
    return dict(schema=SCHEMA_VERSION,package_version=__version__,config=model.config.to_dict(),
        feature_names=list(FEATURES),targets=list(TARGETS),f2=F2_CRITERIA,
        descriptor_names=list(DESCRIPTOR_NAMES),physical_descriptor_indices={k:list(v) for k,v in PHYSICAL_DESCRIPTOR_INDICES.items()},
        preprocessing=model.preprocessing.to_dict(),process=model.process.to_dict(),heads=[h.to_dict() for h in model.heads],
        process_prior=model.process_prior.to_dict(),head_priors=[h.to_dict() for h in model.head_priors],
        split_manifest=model.split_manifest,dataset_sha256=model.dataset_sha256,history=model.history,
        update_history=model.update_history,math_test_status=model.math_test_status,environment=environment(),
        uncertainty_contract="marginal first/second moments under fitted structured VI; no calibrated quantiles")


def seal_directory(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*")) if p.is_file() and p.name!="SHA256_MANIFEST.json"}
    write_json(directory/"SHA256_MANIFEST.json",dict(schema=SCHEMA_VERSION,files=files))


def save_model(model, directory, create=True):
    validate_model_contract(model)
    directory = Path(directory)
    if create:
        directory.mkdir(parents=True,exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError(directory)
    write_json(directory/"model.json",model_dict(model))
    write_json(directory/"split_manifest.json",model.split_manifest)
    write_json(directory/"prediction_contract.json",dict(features=list(FEATURES),targets=list(TARGETS),
        f2=F2_CRITERIA,requires_targets=False,unit="mm",head_dimensions=[len(h.mean) for h in model.heads]))
    seal_directory(directory)


def load_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_VERSION or "model.json" not in manifest["files"]:
        raise ValueError("Invalid integrity manifest")
    for name,expected in manifest["files"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    d = json.loads((directory/"model.json").read_text(encoding="utf-8"))
    if (d.get("schema") != SCHEMA_VERSION or d["feature_names"]!=list(FEATURES) or d["targets"]!=list(TARGETS)
            or d["f2"]!=F2_CRITERIA or d["descriptor_names"]!=list(DESCRIPTOR_NAMES)
            or d["physical_descriptor_indices"]!={k:list(v) for k,v in PHYSICAL_DESCRIPTOR_INDICES.items()}):
        raise ValueError("Incompatible prediction contract")
    model = TrainedModel(Config(**d["config"]),Preprocessing.from_dict(d["preprocessing"]),
        ProcessPosterior.from_dict(d["process"]),tuple(NIGHead.from_dict(h) for h in d["heads"]),
        ProcessPosterior.from_dict(d["process_prior"]),tuple(NIGHead.from_dict(h) for h in d["head_priors"]),
        d["split_manifest"],d["dataset_sha256"],d["history"],d["update_history"],math_test_status=d["math_test_status"])
    validate_model_contract(model)
    side_split = json.loads((directory/"split_manifest.json").read_text(encoding="utf-8"))
    side_contract = json.loads((directory/"prediction_contract.json").read_text(encoding="utf-8"))
    if side_split!=model.split_manifest:
        raise ValueError("Sidefile split manifest differs from model")
    expected_contract = dict(features=list(FEATURES),targets=list(TARGETS),f2=F2_CRITERIA,
                             requires_targets=False,unit="mm",head_dimensions=[len(h.mean) for h in model.heads])
    if side_contract!=expected_contract:
        raise ValueError("Sidefile prediction contract differs from model")
    return model


def validate_model_contract(model):
    """Cross-check independently serialized factors, dimensions and fit provenance."""
    K = model.config.K
    if len(model.process.pi)!=K or len(model.process_prior.pi)!=K:
        raise ValueError("Process K differs from configuration")
    if any(len(e.mean)!=6 for p in (model.process,model.process_prior) for e in p.emissions):
        raise ValueError("Process emission feature dimension mismatch")
    dimensions = [2*K-1+6,2*K-1+5]
    if len(model.heads)!=2 or len(model.head_priors)!=2 or [len(h.mean) for h in model.heads]!=dimensions or [len(h.mean) for h in model.head_priors]!=dimensions:
        raise ValueError("Exactly two heads with the configured M2 dimensions required")
    if any(h.a<=1 for h in model.heads):
        raise ValueError("Predictive variance undefined for head a<=1")
    split = model.split_manifest
    expected = guard_training_ids(split["L"],split["U"],split.get("VAL",()))
    if split!=expected:
        raise ValueError("Inconsistent split manifest")
    prep = model.preprocessing
    expected_X = set(split["L"]) | (set(split["U"]) if model.config.include_unlabeled_preprocessing else set())
    if (prep.include_unlabeled!=model.config.include_unlabeled_preprocessing
            or set(prep.x_scaler.fit_ids)!=expected_X or set(prep.descriptor_scaler.fit_ids)!=expected_X
            or set(prep.y_scaler.fit_ids)!=set(split["L"])):
        raise ValueError("Scaler fit provenance differs from TRAIN split")
    if [len(s.mean) for s in (prep.x_scaler,prep.y_scaler,prep.descriptor_scaler)]!=[6,2,12]:
        raise ValueError("Scaler feature dimensions differ from prediction contract")
