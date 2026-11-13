"""SHA-verifizierter lokaler DM-Import und strikt prozessbezogene CSV/NPZ-IO.

Das einmalige Entserialisieren des freigegebenen Sammelpickles materialisiert
auch eingeschlossene Altlabels. Nur L-TRAIN-Zielwerte werden danach gelesen;
an U-Training, Skalierung und Prediction gelangen ausschließlich Prozessobjekte.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import pickle
import numpy as np

from .schema import FEATURES, TARGETS, RingProcessSequence, LabeledRing, UnlabeledRing
from .split_guard import (ACTIVE_IDS, DEVELOPMENT_IDS, LABELED_IDS, UNLABELED_IDS,
                          LOCKED_IDS, guard_training_ids, unique_ids)

DATA_SHA256 = "bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5"
LOGGER_COLUMNS = ("time", *FEATURES, "phase804", "LaserOK", "MessEin")
FEATURE_UNITS = ("%", "%", "m", "m", "m", "m/s")
TIME_STEP_S = 0.1
TIME_ABSOLUTE_TOLERANCE_S = 0.002
F2_CRITERIA = {"phase804": [1, 2], "LaserOK": 1, "MessEin": 1,
               "contiguous_original_rows": True, "time_step_s": TIME_STEP_S,
               "time_absolute_tolerance_s": TIME_ABSOLUTE_TOLERANCE_S,
               "phase_boundary": "exactly_one_2_to_1", "features": list(FEATURES)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trusted_raw(path):
    """Erlaubt ausschließlich die konkret freigegebene, unveränderte Datei."""
    path = Path(path)
    if sha256(path) != DATA_SHA256:
        raise ValueError("Daten-SHA256-Abweichung; Deserialisierung abgebrochen")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or tuple(sorted(raw)) != ACTIVE_IDS:
        raise ValueError("Erwartet genau 16 aktive DM-Ringe ohne DM03/DM13")
    for rid, record in raw.items():
        if record.get("ring_id") != rid or record.get("train_eligible") is not True:
            raise ValueError(f"Inkonsistenter Aktivstatus: {rid}")
    return raw


def _validate_time(times):
    differences = np.diff(times)
    if np.any(differences <= 0) or np.any(np.abs(differences - TIME_STEP_S) > TIME_ABSOLUTE_TOLERANCE_S):
        raise ValueError("Zeitordnung/Raster verletzt: 0,1 s ±0,002 s erforderlich; keine Interpolation")


def f2_indices(matrix, columns, require_boundary=True):
    """Exakt historische Zeilenauswahl; zusätzliche Prüfungen ändern keine Zeile."""
    columns = tuple(columns)
    if len(set(columns)) != len(columns) or not set(LOGGER_COLUMNS).issubset(columns):
        raise ValueError("Fehlende oder doppelte Loggerkanäle")
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns) or not np.isfinite(data).all():
        raise ValueError("Ungültige Logger-Matrix/NaN/Inf")
    positions = {column: columns.index(column) for column in LOGGER_COLUMNS}
    times = data[:, positions["time"]]
    _validate_time(times)
    phase, laser, measurement = (data[:, positions[n]] for n in ("phase804", "LaserOK", "MessEin"))
    if not np.isin(laser, (0, 1)).all() or not np.isin(measurement, (0, 1)).all():
        raise ValueError("LaserOK und MessEin müssen binär sein")
    if not np.isin(phase, (1, 2, 6)).all():
        raise ValueError("Unbekannte Loggerphase; Prozessdefinition prüfen")
    selected = np.flatnonzero(np.isin(phase, (1, 2)) & (laser == 1) & (measurement == 1))
    if len(selected) < 6:
        raise ValueError("F2 zu kurz: mindestens sechs Zeitpunkte für Deskriptoren")
    if np.any(np.diff(selected) != 1):
        raise ValueError("F2-Fenster nicht zusammenhängend; kein Zusammenkleben von Inseln")
    selected_phase = phase[selected]
    changes = np.flatnonzero(np.diff(selected_phase) != 0)
    if require_boundary and (len(changes) != 1 or selected_phase[0] != 2 or selected_phase[-1] != 1):
        raise ValueError("Fehlende oder unzulässige 2→1-Phasengrenze im vollständigen F2")
    return selected


def _process_from_record(rid, record, ring_type=UnlabeledRing):
    logger = record["logger"]
    columns = tuple(logger["matrix_columns"])
    if columns != ("time", *logger["signal_names"]):
        raise ValueError(f"Loggerheader inkonsistent: {rid}")
    matrix = np.asarray(logger["matrix"], float)
    indices = f2_indices(matrix, columns)
    feature_columns = [columns.index(name) for name in FEATURES]
    units = tuple(logger["signal_units"][logger["signal_names"].index(name)] for name in FEATURES)
    if units != FEATURE_UNITS:
        raise ValueError(f"Unverifizierte Sensoreinheiten: {rid}")
    values = dict(ring_id=rid, times=matrix[indices, 0], X=matrix[np.ix_(indices, feature_columns)],
                  phases=matrix[indices, columns.index("phase804")], row_indices=indices)
    if ring_type is LabeledRing:
        try:
            # 'zielwerte' ist der belegte Container der all_targets-Excel-Übernahme.
            values["targets"] = np.array([float(record["zielwerte"][name]) for name in TARGETS])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Fehlendes/ungültiges Qualitätslabel in L: {rid}") from error
    return ring_type(**values)


def load_development(path, labeled_ids=LABELED_IDS, unlabeled_ids=UNLABELED_IDS):
    l, u = unique_ids(labeled_ids), unique_ids(unlabeled_ids)
    guard_training_ids(l, u)
    if not set(l).issubset(LABELED_IDS) or not set(u).issubset(UNLABELED_IDS):
        raise ValueError("Nur registrierte Entwicklungslabels und U-Prozesse zulässig")
    raw = _trusted_raw(path)
    labeled = {rid: _process_from_record(rid, raw[rid], LabeledRing) for rid in sorted(l)}
    unlabeled = {rid: _process_from_record(rid, raw[rid], UnlabeledRing) for rid in sorted(u)}
    return labeled, unlabeled


def load_development_processes(path):
    raw = _trusted_raw(path)
    return {rid: _process_from_record(rid, raw[rid], UnlabeledRing) for rid in DEVELOPMENT_IDS}


def audit_data(path):
    raw = _trusted_raw(path)
    rows = []
    for rid in DEVELOPMENT_IDS:
        ring = _process_from_record(rid, raw[rid], UnlabeledRing)
        phase_changes = np.flatnonzero(np.diff(ring.phases) != 0)
        rows.append({"ring_id": rid, "raw_rows": len(raw[rid]["logger"]["matrix"]),
                     "f2_rows": len(ring.times), "f2_start_index": int(ring.row_indices[0]),
                     "f2_end_index": int(ring.row_indices[-1]),
                     "f2_t0_s": float(ring.times[0]), "f2_tlast_s": float(ring.times[-1]),
                     "median_time_step_s": float(np.median(np.diff(ring.times))),
                     "phase_boundary_count": int(len(phase_changes)),
                     "phase_boundary_time_s": float(ring.times[phase_changes[0] + 1]),
                     "feature_min": ring.X.min(axis=0).tolist(),
                     "feature_max": ring.X.max(axis=0).tolist(),
                     "missing_values": 0, "quality_values_accessed": False})
    return {"dataset_sha256": DATA_SHA256, "active_ring_count": len(raw),
            "active_ring_ids": sorted(raw), "development_ring_count": len(rows),
            "excluded_ids": ["DM03", "DM13"], "locked_ids": list(LOCKED_IDS),
            "locked_processes_accessed": False, "targets": list(TARGETS),
            "target_container_in_verified_pickle": "zielwerte",
            "feature_units": list(FEATURE_UNITS), "f2": F2_CRITERIA, "rings": rows}


def _completion_check(matrix, indices):
    phase, laser, measurement = matrix[:, -3:].T
    idle = (phase == 6) & (laser == 0) & (measurement == 0)
    if len(matrix) < 20 or not np.all(phase[:3] == 6) or not np.all(idle[-3:]):
        raise ValueError("Unvollständiger Logger: Phase-6-Vorlauf und inaktiver Nachlauf erforderlich")
    start, end = int(indices[0]), int(indices[-1])
    if start < 3 or end >= len(matrix) - 3:
        raise ValueError("F2 ohne nachgewiesenen Vor-/Nachlauf")
    following = np.flatnonzero(idle[end + 1:])
    if not len(following) or following[0] > 5 or np.any(~idle[end + 1 + int(following[0]):]):
        raise ValueError("Kein zeitnaher, dauerhafter inaktiver Prozessabschluss")


def read_process_csv(path, ring_id=None):
    path = Path(path)
    rid = ring_id or path.stem
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != LOGGER_COLUMNS:
            raise ValueError("CSV erwartet exakt Prozessheader/-reihenfolge: " + ",".join(LOGGER_COLUMNS))
        try:
            rows = [list(map(float, row)) for row in reader]
            matrix = np.asarray(rows, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("CSV enthält nichtnumerische/unvollständige Prozesszeilen") from error
    indices = f2_indices(matrix, LOGGER_COLUMNS)
    _completion_check(matrix, indices)
    return UnlabeledRing(rid, matrix[indices, 0], matrix[np.ix_(indices, range(1, 7))],
                         phases=matrix[indices, 7], row_indices=indices)


def export_development_logger(path, ring_id, output_path):
    """Kontrolliertes Prozess-only-Demofile; niemals Endmessung exportieren."""
    if ring_id not in DEVELOPMENT_IDS:
        raise ValueError("Nur Entwicklungsprozesse dürfen exportiert werden")
    raw = _trusted_raw(path)[ring_id]["logger"]
    columns = tuple(raw["matrix_columns"])
    selected = np.asarray(raw["matrix"])[:, [columns.index(name) for name in LOGGER_COLUMNS]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOGGER_COLUMNS)
        writer.writerows(selected)
    return sha256(output)


def export_process_npz(ring, output_path):
    """Nur für kontrollierte, bereits F2-validierte Prozessschnittstellen."""
    if not isinstance(ring, RingProcessSequence):
        raise TypeError("Prozessring erwartet")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, schema=np.array("bayesian-ring-f2-process-v1"),
                            ring_id=np.array(ring.ring_id), times=ring.times, X=ring.X,
                            feature_names=np.array(FEATURES),
                            phases=np.array([]) if ring.phases is None else ring.phases)
    return sha256(path)


def read_process_file(path, ring_id=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_process_csv(path, ring_id)
    if path.suffix.lower() != ".npz":
        raise ValueError("Nur Prozess-only CSV oder vertraglich markiertes F2-NPZ erlaubt")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"schema", "ring_id", "times", "X", "feature_names", "phases"} or str(archive["schema"]) != "bayesian-ring-f2-process-v1":
            raise ValueError("Unbekannter/qualitätsbehafteter Prozess-NPZ-Vertrag")
        phases = archive["phases"]
        ring = UnlabeledRing(ring_id or str(archive["ring_id"]), archive["times"], archive["X"],
                             tuple(archive["feature_names"].tolist()),
                             None if not len(phases) else phases)
    _validate_time(ring.times)
    return ring
