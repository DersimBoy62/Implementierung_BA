"""Explizite, unveränderliche Ringverträge; U besitzt keine Qualitätsfelder."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

FEATURES = ("F_rad63", "F_ax64", "Ist_D150", "Ist_s151", "Ist_h152", "RWG55")
TARGETS = ("ovality.ovality_radial_mm", "planarity.height_deviation_mm")


def _array(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("NaN/Inf im Ringvertrag")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class RingProcessSequence:
    ring_id: str
    times: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...] = FEATURES
    phases: np.ndarray | None = None
    row_indices: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.ring_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", self.ring_id):
            raise ValueError("Ungültige Ring-ID")
        t, x = _array(self.times), _array(self.X)
        if t.ndim != 1 or len(t) < 4 or x.shape != (len(t), 6):
            raise ValueError("Erwartet T>=4 und X.shape=(T,6)")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Zeitachse muss streng aufsteigen")
        if tuple(self.feature_names) != FEATURES:
            raise ValueError("Feature-Reihenfolge weicht vom sechs-Kanal-Vertrag ab")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "feature_names", FEATURES)
        if self.phases is not None:
            p = _array(self.phases)
            if p.shape != t.shape or not np.isin(p, (1, 2)).all():
                raise ValueError("F2-Phasen müssen 1 oder 2 sein")
            object.__setattr__(self, "phases", p)
        if self.row_indices is not None:
            indices = _array(self.row_indices, dtype=np.int64)
            original = np.asarray(self.row_indices)
            if indices.shape != t.shape or not np.array_equal(indices, original) or np.any(np.diff(indices) != 1):
                raise ValueError("F2-Zeilenindizes müssen ganzzahlig und zusammenhängend sein")
            object.__setattr__(self, "row_indices", indices)

    @property
    def t(self):
        return self.times

    @property
    def x(self):
        return self.X

    def process_only(self):
        return RingProcessSequence(self.ring_id, self.times, self.X, self.feature_names,
                                   self.phases, self.row_indices)


@dataclass(frozen=True, slots=True)
class LabeledRing(RingProcessSequence):
    targets: np.ndarray = field(kw_only=True)

    def __post_init__(self):
        super(LabeledRing, self).__post_init__()
        y = _array(self.targets)
        if y.shape != (2,):
            raise ValueError("Exakt zwei endliche Qualitätslabels erforderlich")
        object.__setattr__(self, "targets", y)


@dataclass(frozen=True, slots=True)
class UnlabeledRing(RingProcessSequence):
    """Kein y-/targets-Attribut und kein __dict__ für versteckte Labels."""

