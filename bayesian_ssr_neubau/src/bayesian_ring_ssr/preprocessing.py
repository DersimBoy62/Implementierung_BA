"""Ausschließlich TRAIN-angepasste Normalisierer mit Herkunft pro Fit."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .schema import RingProcessSequence, LabeledRing
from .split_guard import guard_training_ids, unique_ids
from .physics_descriptors import extract_descriptors


@dataclass(frozen=True, slots=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_ids: tuple[str, ...]

    def __post_init__(self):
        mean, scale = np.array(self.mean, float, copy=True), np.array(self.scale, float, copy=True)
        if mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Ungültige Scalerparameter")
        mean.flags.writeable = scale.flags.writeable = False
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "fit_ids", tuple(sorted(unique_ids(self.fit_ids))))

    @property
    def fitted_ids(self):
        return self.fit_ids

    def _check(self, values):
        a = np.asarray(values, float)
        if a.ndim == 0 or a.shape[-1] != len(self.mean) or not np.isfinite(a).all():
            raise ValueError("Shape-/Finite-Fehler bei Scalertransform")
        return a

    def transform(self, values):
        return (self._check(values) - self.mean) / self.scale

    def inverse_transform(self, values):
        return self._check(values) * self.scale + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit_ids": list(self.fit_ids)}

    @classmethod
    def from_dict(cls, values):
        if set(values) != {"mean", "scale", "fit_ids"}:
            raise ValueError("Unerwarteter Scalervertrag")
        return cls(values["mean"], values["scale"], tuple(values["fit_ids"]))


def fit_scaler(values, fit_ids):
    a = np.asarray(values, float)
    ids = unique_ids(fit_ids)
    if a.ndim != 2 or not len(a) or not ids or not np.isfinite(a).all():
        raise ValueError("Scalerfit benötigt endliche TRAIN-Matrix und Ring-IDs")
    sd = a.std(axis=0, ddof=0)
    return Scaler(a.mean(axis=0), np.where(sd > 1e-10, sd, 1.0), ids)


@dataclass(frozen=True, slots=True)
class Preprocessing:
    x_scaler: Scaler
    y_scaler: Scaler
    descriptor_scaler: Scaler
    include_unlabeled: bool

    def transform_process(self, ring: RingProcessSequence):
        if not isinstance(ring, RingProcessSequence):
            raise TypeError("Prozesstransformation erwartet RingProcessSequence")
        return RingProcessSequence(ring.ring_id, ring.times, self.x_scaler.transform(ring.X),
                                   ring.feature_names, ring.phases, ring.row_indices)

    def descriptors(self, ring: RingProcessSequence):
        scaled = self.transform_process(ring)
        return self.descriptor_scaler.transform(extract_descriptors(scaled.X, scaled.times))

    def to_dict(self):
        return {"x_scaler": self.x_scaler.to_dict(), "y_scaler": self.y_scaler.to_dict(),
                "descriptor_scaler": self.descriptor_scaler.to_dict(),
                "include_unlabeled": self.include_unlabeled,
                "process_policy": "P0_pooled_train_timepoints", "descriptor_policy": "train_rings_equal_weight"}

    @classmethod
    def from_dict(cls, data):
        if data.get("process_policy") != "P0_pooled_train_timepoints" or data.get("descriptor_policy") != "train_rings_equal_weight":
            raise ValueError("Unbekannte Preprocessing-Policy")
        if type(data["include_unlabeled"]) is not bool:
            raise ValueError("include_unlabeled muss bool sein")
        return cls(*(Scaler.from_dict(data[n]) for n in ("x_scaler", "y_scaler", "descriptor_scaler")), data["include_unlabeled"])


def _rings(values):
    return list(values.values()) if hasattr(values, "values") else list(values)


def fit_preprocessing(labeled, unlabeled=(), include_unlabeled=True):
    labeled, unlabeled = _rings(labeled), _rings(unlabeled)
    if not all(isinstance(r, LabeledRing) for r in labeled):
        raise TypeError("Zielscaler darf ausschließlich bekannte L-TRAIN-Labels erhalten")
    if not all(type(r) is not LabeledRing and isinstance(r, RingProcessSequence) for r in unlabeled):
        raise TypeError("U muss eine Prozess-only-Schnittstelle besitzen")
    guard_training_ids([r.ring_id for r in labeled], [r.ring_id for r in unlabeled])
    train = sorted(labeled + (unlabeled if include_unlabeled else []), key=lambda r: r.ring_id)
    labeled = sorted(labeled, key=lambda r: r.ring_id)
    sx = fit_scaler(np.vstack([r.X for r in train]), [r.ring_id for r in train])
    sy = fit_scaler(np.vstack([r.targets for r in labeled]), [r.ring_id for r in labeled])
    features = np.stack([extract_descriptors(sx.transform(r.X), r.times) for r in train])
    sd = fit_scaler(features, [r.ring_id for r in train])
    return Preprocessing(sx, sy, sd, bool(include_unlabeled))
