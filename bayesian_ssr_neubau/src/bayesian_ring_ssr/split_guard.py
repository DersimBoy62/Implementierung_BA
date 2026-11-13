"""Ringweise Entwicklungs-Splits und harte Sperre der historischen Außenringe."""
from __future__ import annotations

LABELED_IDS = ("DM04", "DM05", "DM06", "DM09", "DM12", "DM16", "DM17", "DM18")
UNLABELED_IDS = ("DM01", "DM02", "DM10", "DM14")
LOCKED_IDS = ("DM07", "DM08", "DM11", "DM15")
DEVELOPMENT_IDS = tuple(sorted((*LABELED_IDS, *UNLABELED_IDS)))
ACTIVE_IDS = tuple(sorted((*DEVELOPMENT_IDS, *LOCKED_IDS)))


def unique_ids(ids):
    values = tuple(ids)
    if len(set(values)) != len(values):
        raise ValueError("Doppelte Ring-ID")
    return values


def guard_training_ids(labeled_ids, unlabeled_ids, validation_ids=()):
    l, u, v = map(unique_ids, (labeled_ids, unlabeled_ids, validation_ids))
    if not l:
        raise ValueError("Training benötigt gelabelte Ringe")
    if set(l) & set(u) or (set(l) | set(u)) & set(v):
        raise ValueError("Ring-Leakage zwischen TRAIN/L/U/VAL")
    if (set(l) | set(u) | set(v)) & set(LOCKED_IDS):
        raise ValueError("Historischer Außenring ist für Entwicklung/Training gesperrt")
    return {"L": sorted(l), "U": sorted(u), "VAL": sorted(v), "LOCKED": list(LOCKED_IDS)}


def development_folds():
    """Vier vorab festgelegte Folds; jedes L-Ringlabel einmal VAL."""
    for fold in range(4):
        validation = LABELED_IDS[2 * fold:2 * fold + 2]
        training = tuple(r for r in LABELED_IDS if r not in validation)
        yield {"fold": fold + 1, **guard_training_ids(training, UNLABELED_IDS, validation)}

