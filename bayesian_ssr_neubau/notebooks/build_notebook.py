"""Build the German demonstration notebook; the model implementation is unchanged."""
from pathlib import Path
import argparse
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cell = nbf.v4.new_code_cell(textwrap.dedent(source).strip())
    cell.metadata["name"] = f"schritt_{sum(c.cell_type == 'code' for c in cells)+1:02d}"
    cells.append(cell)


md(r"""
# Bayesian-SSR-HMM: vollständige Ausführung und Funktionsnachweis

Dieses Notebook führt das **bestehende Basismodell ohne Optimierung** aus.
Es prüft die Daten, startet die mathematischen Tests, trainiert das Modell
frisch über 30 Epochen und erstellt neue Vorhersagen sowie eine vollständige
vierfache Entwicklungsevaluation mit fünf Vergleichsarmen.

**Bedienung in VS Code:** Als Kernel die Python-Umgebung `.venv` des Projekts
auswählen und **Restart Kernel → Run All** ausführen. Die gespeicherte Ausgabe
zeigt einen tatsächlich ausgeführten Lauf. Jeder neue Gesamtlauf erzeugt einen
eigenen Ausgabeordner; die vorhandene Referenz wird nicht überschrieben.

Der vollständige Lauf dauert deutlich länger als die Tests, insbesondere durch
die zwölf zusätzlichen HMM-Fits der Cross-Validation. Es werden keine alten
Gewichte oder fertigen CV-Ergebnisse als Ersatz für einen neuen Fit geladen.

Das Notebook unterscheidet:

1. **Rechenkorrektheit:** unabhängige Tests, exakte kurze Pfade, ELBO-Bilanz.
2. **Funktionsfähigkeit:** frisches Training, Prozess-only-Prediction, Persistenz.
3. **Vorhersagefehler:** zurückgehaltene vollständige Ringe in Entwicklungs-CV.

Die acht Labelringe wurden historisch bereits zur Entwicklung verwendet.
Diese Ausführung ist deshalb keine neue unabhängige externe Validierung.
""")

code(r"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

def find_workspace():
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if ((candidate / "input/DM_16_aktive_Ringe.pkl").is_file()
                and (candidate / "bayesian_ssr_neubau/pyproject.toml").is_file()):
            return candidate
    raise FileNotFoundError("Notebook innerhalb des Workspace Implementierung_BA öffnen.")

ROOT = find_workspace()
PROJECT = ROOT / "bayesian_ssr_neubau"
NOTEBOOKS = PROJECT / "notebooks"
DATA = ROOT / "input/DM_16_aktive_Ringe.pkl"
if (ROOT / ".venv").is_dir():
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), (
        "Bitte in VS Code den Python-Kernel der Projekt-.venv auswählen."
    )
sys.path.insert(0, str(PROJECT / "src"))

RUN_ID = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
RUN_DIR = PROJECT / "outputs" / "notebook_demos" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)
FIGURES = RUN_DIR / "figures"
FIGURES.mkdir()
TOTAL_START = time.perf_counter()
os.environ.setdefault("MPLCONFIGDIR", str(NOTEBOOKS / ".runtime" / "matplotlib"))

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import bayesian_ring_ssr
from bayesian_ring_ssr.config import Config
from bayesian_ring_ssr.data_io import (
    DATA_SHA256, FEATURES, TARGETS, LOGGER_COLUMNS, audit_data, load_development,
    export_development_logger, read_process_file, sha256,
)
from bayesian_ring_ssr.coordinate_train import fit
from bayesian_ring_ssr.diagnostics import training_diagnostics
from bayesian_ring_ssr.evaluation import cross_validate, ARMS
from bayesian_ring_ssr.forward_backward import forward_backward
from bayesian_ring_ssr.hmmmoments import moments
from bayesian_ring_ssr.metrics import regression_metrics
from bayesian_ring_ssr.persistence import save_model, load_model, write_json, seal_directory
from bayesian_ring_ssr.posterior_predictive import predict_new_ring
from bayesian_ring_ssr.split_guard import (
    LABELED_IDS, UNLABELED_IDS, LOCKED_IDS, development_folds,
)
from scipy.special import logsumexp

assert Path(bayesian_ring_ssr.__file__).resolve().is_relative_to(PROJECT / "src")
plt.rcParams.update({
    "figure.figsize": (10, 4), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "font.size": 10,
})

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def show_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    display(fig)
    plt.close(fig)

print("Python:", sys.executable)
print("Workspace:", ROOT)
print("Neuer Lauf:", RUN_DIR)
""")

md(r"""
## 1. Umgebung und unveränderte Modellreferenz

Die zusätzlichen Jupyter-Pakete ändern keine Modellgleichungen oder
Hyperparameter. Die Hashprüfung kontrolliert alle Dateien des vorhandenen
Baseline-Releases. Die nachfolgenden Ergebnisse werden separat neu berechnet.
""")

code(r"""
release_manifest = read_json(PROJECT / "RELEASE_SHA256.json")
changed = [name for name, expected in release_manifest["files"].items()
           if not (PROJECT / name).is_file() or sha256(PROJECT / name) != expected]
assert not changed, f"Referenzdateien verändert: {changed}"
source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((PROJECT / "src").rglob("*.py"))}
package_names = ["numpy", "scipy", "scikit-learn", "pandas", "numba", "pytest",
                 "nbformat", "nbclient", "ipykernel", "matplotlib"]
versions = {name: metadata.version(name) for name in package_names}
display(pd.DataFrame({"Paket": versions.keys(), "Version": versions.values()}))
print(f"Unveränderte Referenz: {len(release_manifest['files'])} Dateien geprüft.")
write_json(RUN_DIR / "environment.json", {
    "python": sys.version, "executable": sys.executable, "packages": versions,
    "baseline_manifest_sha256": sha256(PROJECT / "RELEASE_SHA256.json"),
    "source_sha256": source_hashes,
})
installed = {d.metadata["Name"]: d.version for d in metadata.distributions()
             if d.metadata["Name"].lower().replace("_", "-") != "bayesian-ring-ssr"}
(RUN_DIR / "requirements-executed.txt").write_text(
    "\n".join(f"{name}=={version}" for name, version in sorted(installed.items())) + "\n",
    encoding="utf-8",
)
""")

md(r"""
## 2. Datenprüfung und Trennung der Ringgruppen

Die Datei wird nur nach Übereinstimmung mit dem freigegebenen SHA256 gelesen.
Das vertrauenswürdige Sammelpickle enthält physisch auch weitere Zielwerte;
an Training und Prediction werden ausschließlich die erlaubten Datenverträge
weitergegeben. U-Objekte besitzen weder `y` noch `targets`.

Die vier historischen Außenringe DM07/08/11/15 werden nicht trainiert oder neu
gescort. F2 besteht aus den zusammenhängenden aktiven Phase-1/2-Zeitpunkten mit
LaserOK=MessEin=1. Die Reihenfolge der sechs Features ist fest.
""")

code(r"""
assert sha256(DATA) == DATA_SHA256, "Datenhash stimmt nicht; Abbruch."
qa = audit_data(DATA)
labeled, unlabeled = load_development(DATA)
assert set(labeled) == set(LABELED_IDS) and set(unlabeled) == set(UNLABELED_IDS)
assert not (set(labeled) | set(unlabeled)) & set(LOCKED_IDS)
assert all(not hasattr(r, "targets") and not hasattr(r, "y") for r in unlabeled.values())
assert qa["active_ring_count"] == 16 and qa["development_ring_count"] == 12
ring_table = pd.DataFrame([
    {"Ring": row["ring_id"], "Gruppe": "L" if row["ring_id"] in labeled else "U",
     "F2-Zeitpunkte": row["f2_rows"], "Start [s]": row["f2_t0_s"],
     "Ende [s]": row["f2_tlast_s"], "Phasengrenze [s]": row["phase_boundary_time_s"],
     "Fehlende Werte": row["missing_values"]}
    for row in qa["rings"]
])
display(ring_table)
print("Daten-SHA256:", qa["dataset_sha256"])
print("Features:", ", ".join(FEATURES))
print("Ziele:", ", ".join(TARGETS))
print("L-Ringe:", sorted(labeled), "\nU-Ringe:", sorted(unlabeled))
print("Historische Außenringe gesperrt:", LOCKED_IDS)
write_json(RUN_DIR / "data_qa.json", qa)
ring_table.to_csv(RUN_DIR / "ring_overview.csv", index=False, encoding="utf-8")
""")

md(r"""
### Beispiel einer vollständigen F2-Prozesssequenz

DM04 wird hier als Eingangssignal gezeigt. Die Kraftkanäle sind laut
Loggerheader **Prozentwerte**, die Geometriekanäle Meter und RWG55 eine
Ringwachstumsrate in m/s. Die gestrichelte Linie markiert die beobachtete
Loggerphasengrenze; sie ist kein zusätzliches Modellfeature.
""")

code(r"""
example = labeled["DM04"].process_only()
units = ["%", "%", "m", "m", "m", "m/s"]
boundary_times = example.times[np.flatnonzero(np.diff(example.phases) != 0) + 1]
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, layout="constrained")
for j, ax in enumerate(axes.flat):
    ax.plot(example.times, example.X[:, j], color="#276c9e", linewidth=1.3)
    for boundary in boundary_times:
        ax.axvline(boundary, color="#a34b20", linestyle="--", linewidth=1)
    ax.set(title=FEATURES[j], xlabel="Zeit [s]", ylabel=f"[{units[j]}]")
fig.suptitle("DM04: sechs beobachtete Prozesskanäle im F2-Fenster")
show_figure(fig, "01_prozesssequenz_DM04")
""")

md(r"""
## 3. Anschaulicher Mathematikcheck: alle kurzen Zustandspfade

Für eine kleine Kette mit K=2 und T=4 werden alle 16 Pfade unabhängig
aufgezählt. Verglichen werden Forward-Backward-Marginalen und die exakten
ersten/zweiten M2-Momente. Die reduzierte Statistik lautet
`g=(h_1,h_2,Delta_1)`. Insbesondere wird die zweite Momentenmatrix nicht durch
das äußere Produkt des Mittelwerts ersetzt.
""")

code(r"""
K, T = 2, 4
logpi = np.log([0.6, 0.4])
logA = np.log([[0.85, 0.15], [0.2, 0.8]])
ell = np.array([[-0.2, -1.0], [-0.5, -0.6], [-1.2, -0.1], [-0.8, -0.3]])
fb_small = forward_backward(logpi, logA, ell)
paths = list(itertools.product(range(K), repeat=T))
path_logs, path_features = [], []
for z in paths:
    path_logs.append(logpi[z[0]] + sum(ell[t, z[t]] for t in range(T))
                     + sum(logA[z[t-1], z[t]] for t in range(1, T)))
    h = np.bincount(z, minlength=K) / T
    early = np.bincount(z[:T//2], minlength=K) / (T//2)
    late = np.bincount(z[T//2:], minlength=K) / (T-T//2)
    path_features.append(np.r_[h, (late-early)[:-1]])
weights = np.exp(np.asarray(path_logs) - logsumexp(path_logs))
features = np.asarray(path_features)
m_enum = weights @ features
H_enum = np.einsum("n,ni,nj->ij", weights, features, features)
m_exact, H_exact = moments(fb_small.chain)
gamma_enum = np.zeros((T, K))
for z, weight in zip(paths, weights):
    for t, state in enumerate(z):
        gamma_enum[t, state] += weight
np.testing.assert_allclose(fb_small.gamma, gamma_enum, atol=2e-12, rtol=0)
np.testing.assert_allclose(m_exact, m_enum, atol=2e-12, rtol=0)
np.testing.assert_allclose(H_exact, H_enum, atol=2e-12, rtol=0)
display(pd.DataFrame({"Merkmal": ["h1", "h2", "Delta1"],
                      "Rekursion": m_exact, "Pfadenumeration": m_enum}))
display(pd.DataFrame(H_exact, index=["h1", "h2", "Delta1"], columns=["h1", "h2", "Delta1"]))
print("Max. Momentenabweichung:", float(np.max(np.abs(H_exact-H_enum))))
print("||E[ggᵀ] − E[g]E[g]ᵀ||:", float(np.linalg.norm(H_exact-np.outer(m_exact, m_exact))))
print("Kurze Pfadenumeration und Markov-Rekursion stimmen überein.")
""")

md(r"""
## 4. Die vollständige Testsuite tatsächlich ausführen

Diese Zelle startet pytest mit **dem aktuellen Notebook-Python**. Die aktuelle
Testanzahl und der Exitcode stammen aus diesem neuen Aufruf und seiner neuen
JUnit-Datei. Die Suite umfasst T1–T10, synthetisches End-to-End-Training,
Leakage-/Persistenzprüfungen sowie getrennte read-only-V6.0-Oracletests.
Der Produktivcode importiert keine alten Modellkerne.
""")

code(r"""
junit_path = RUN_DIR / "pytest_results.xml"
test_command = [sys.executable, "-m", "pytest", str(PROJECT / "tests"), "-q",
                f"--junitxml={junit_path}"]
test_start = time.perf_counter()
test_run = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
test_output = test_run.stdout + test_run.stderr
(RUN_DIR / "pytest_output.log").write_text(test_output, encoding="utf-8")
print(test_output)
assert test_run.returncode == 0, f"pytest fehlgeschlagen: Exitcode {test_run.returncode}"
suite_root = ET.parse(junit_path).getroot()
suites = list(suite_root.iter("testsuite"))
test_counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
               for name in ("tests", "failures", "errors", "skipped")}
assert test_counts["tests"] > 0
assert test_counts["failures"] == test_counts["errors"] == test_counts["skipped"] == 0
test_report = {"command": test_command, "exitcode": test_run.returncode, **test_counts,
               "elapsed_seconds": time.perf_counter()-test_start,
               "junit_sha256": sha256(junit_path)}
write_json(RUN_DIR / "test_report.json", test_report)
display(pd.DataFrame([test_report]).drop(columns=["command", "junit_sha256"]))
""")

md(r"""
## 5. Festgelegte Baselinekonfiguration

Es werden keine Hyperparameter gesucht oder optimiert. Die bestehende
Konfiguration K2/M2/C0,2 mit 30 Epochen wird unverändert verwendet. Die ersten
K Aufenthaltsmerkmale bleiben erhalten; nur das letzte Delta wird mit
induziertem Prior entfernt. Dadurch haben die Qualitätsköpfe Dimension 9 und 8.

Für C≠1 ist J_C eine **gewichtete/generaliserte Variationszielfunktion**.
Ein größerer Wert belegt keine bessere Vorhersage oder kalibrierte Unsicherheit.
""")

code(r"""
CONFIG_PATH = PROJECT / "configs/baseline_k2_m2.json"
config = Config.load(CONFIG_PATH)
assert (config.K, config.variant, config.C, config.train_epochs, config.local_maxiter) == (2, "M2", 0.2, 30, 8)
display(pd.DataFrame(config.to_dict().items(), columns=["Parameter", "Wert"]))
print("Trainingsringe:", len(labeled), "L +", len(unlabeled), "U")
print("Initialisierung: frischer Prior und frische K-Means-Zentren aus TRAIN-X.")
""")

md(r"""
## 6. Frisches vollständiges Training auf 8L + 4U

Die nächste Zelle berechnet alle 30 Epochen neu. Pro lokalem Update,
Prozessupdate und Kopfupdate wird dieselbe vollständige J_C auf Nichtabnahme
geprüft. Ereignisse und Epochen-Checkpoints werden in einen neuen Modellordner
geschrieben. Erst nach Abschluss aller Dateien wird dieser versiegelt.
""")

code(r"""
MODEL_DIR = RUN_DIR / "model_8L4U"
MODEL_DIR.mkdir(exist_ok=False)
write_json(MODEL_DIR / "config.json", config.to_dict())
write_json(MODEL_DIR / "data_qa.json", qa)
write_json(MODEL_DIR / "test_evidence.json", test_report)

def training_progress(event):
    event = dict(event)
    checkpoint = event.pop("checkpoint", None)
    with (MODEL_DIR / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    if checkpoint is not None:
        write_json(MODEL_DIR / f"checkpoint_epoch_{event['epoch']:03d}.json", checkpoint)
        print(f"Epoche {event['epoch']:02d}/{config.train_epochs}: J_C = {event['elbo']:.9f}", flush=True)

training_start = time.perf_counter()
model = fit(labeled, unlabeled, config=config, dataset_sha256=DATA_SHA256,
            callback=training_progress)
training_seconds = time.perf_counter()-training_start
model.math_test_status = (
    f"PASS: {test_counts['tests']} aktuell ausgeführte pytest-Tests, Exit 0; "
    f"JUnit-SHA256={test_report['junit_sha256']}"
)
diagnostics = training_diagnostics(model, {**labeled, **unlabeled})
write_json(MODEL_DIR / "diagnostics.json", diagnostics)
write_json(MODEL_DIR / "training_provenance.json", {
    "training_seconds": training_seconds, "initialization": "fresh TRAIN-X KMeans and prior",
    "source_sha256": source_hashes, "config_sha256": sha256(CONFIG_PATH),
    "data_sha256": DATA_SHA256, "notebook_run_id": RUN_ID,
})
save_model(model, MODEL_DIR, create=False)
print(f"Training beendet: {training_seconds/60:.2f} Minuten; Modell gespeichert: {MODEL_DIR}")
""")

md(r"""
## 7. Trainingsverlauf und kontrollierte Updateblöcke

Die Diagramme zeigen den Rechenverlauf, nicht die Generalisierung. Kleine
negative Rundungsreste bis zur vorab festgelegten Toleranz 10⁻⁶ sind erlaubt;
eine echte Verletzung hätte das Training abgebrochen.
""")

code(r"""
history = pd.DataFrame(model.history)
updates = pd.DataFrame(model.update_history)
assert len(history) == 30 and len(updates) == 450
assert np.isfinite(history["elbo"]).all()
assert updates["delta"].min() >= -config.monotonic_atol
assert [len(head.mean) for head in model.heads] == [9, 8]
weighted_points = sum(len(r.X) for r in labeled.values()) + config.C*sum(len(r.X) for r in unlabeled.values())
np.testing.assert_allclose(sum(history.iloc[-1]["state_N"]), weighted_points, atol=1e-6)
block_summary = updates.groupby("stage")["delta"].agg(["count", "min", "sum"])
display(block_summary.rename(columns={"count": "Updates", "min": "Kleinster Gewinn", "sum": "Gesamtgewinn"}))
print("Finale J_C:", history.iloc[-1]["elbo"])
print("Kleinste Blockdifferenz:", updates["delta"].min())
print("Gewichtete Prozesszeitpunkte:", weighted_points)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
axes[0].plot(history["epoch"], history["elbo"], marker="o", markersize=3, color="#246b87")
axes[0].set(title="Gewichtete Trainingszielfunktion", xlabel="Epoche", ylabel="J_C")
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
gains = updates.pivot_table(index="epoch", columns="stage", values="delta", aggfunc="sum").fillna(0)
axes[1].plot(gains.index, gains["local"], label="Lokale Faktoren", color="#246b87")
axes[1].plot(gains.index, gains["process"], label="Globaler Prozess", color="#c77620")
axes[1].plot(gains.index, gains["head_0"]+gains["head_1"], label="Qualitätsköpfe", color="#488454")
axes[1].set(title="Gewinn je Updategruppe", xlabel="Epoche", ylabel="ΔJ_C")
axes[1].legend()
show_figure(fig, "02_training_und_blockgewinne")
""")

code(r"""
local_optimizer_rows = [event["optimizer"] for event in model.update_history
                        if event["stage"] == "local" and "optimizer_success" in event["optimizer"]]
optimizer_table = pd.DataFrame(local_optimizer_rows)
optimizer_summary = pd.DataFrame([{
    "Lokale Optimiereraufrufe": len(optimizer_table),
    "Akzeptierte Optimiererschritte": int(optimizer_table["accepted"].sum()),
    "Ohne Erfolgsstatus": int((~optimizer_table["optimizer_success"]).sum()),
    "Max. Abschlussgradient über alle Aufrufe": optimizer_table["grad_inf"].max(),
}])
display(optimizer_summary)
print("Nichtabnahme ist geprüft; lokale oder globale Optimalität wird nicht daraus abgeleitet.")
""")

md(r"""
## 8. Prozess-only-Prediction aus einer echten Loggerdatei

Es wird eine neue CSV ausschließlich mit Loggerkanälen exportiert, wieder
eingelesen und durch das frisch gespeicherte Modell vorhergesagt. Die API
bekommt kein Qualitätslabel. **DM04 war im Vollfit enthalten**: dieses Beispiel
prüft die Ein-/Ausgabeschnittstelle und die Persistenz, nicht die Vorhersagegüte.
Die Qualitätsbewertung folgt ausschließlich mit den VAL-Ringen der CV.

SD bedeutet die marginale Standardabweichung der gewählten Momentenprädiktion.
Es handelt sich nicht um ein kalibriertes 95%-Intervall.
""")

code(r"""
process_file = RUN_DIR / "process_only_DM04.csv"
process_hash = export_development_logger(DATA, "DM04", process_file)
process_columns = tuple(pd.read_csv(process_file, nrows=0).columns)
assert process_columns == LOGGER_COLUMNS
process_ring = read_process_file(process_file, ring_id="DM04")
assert not hasattr(process_ring, "targets") and not hasattr(process_ring, "y")
prediction_memory = predict_new_ring(model, process_ring)
loaded_model = load_model(MODEL_DIR)
prediction_loaded = predict_new_ring(loaded_model, process_ring)
assert prediction_memory == prediction_loaded
assert np.isfinite(prediction_loaded["mean_mm"]).all()
assert np.isfinite(prediction_loaded["sd_mm"]).all() and min(prediction_loaded["sd_mm"]) > 0
display(pd.DataFrame({
    "Ziel": ["Außenunrundheit", "Höhenabweichung"],
    "Vorhersage [mm]": prediction_loaded["mean_mm"],
    "Marginale SD [mm]": prediction_loaded["sd_mm"],
}))
print("CSV-Spalten:", ", ".join(process_columns))
print("Vorhersage im Speicher und nach Laden: exakt gleich.")
try:
    predict_new_ring(loaded_model, labeled["DM04"])
except TypeError as error:
    print("Erwartete Eingabesicherung bei LabeledRing:", error)
else:
    raise AssertionError("Prediction hat ein Objekt mit Qualitätslabels angenommen.")
write_json(RUN_DIR / "process_only_prediction.json", prediction_loaded)
""")

md(r"""
## 9. Was macht die Hidden-Markov-Kette?

Die folgende Inferenz wird **erneut nur aus dem Prozess X** mit den geladenen
globalen Faktoren berechnet. Es ist keine qualitätskonditionierte lokale
Trainingskette. Die Zustandsnummern sind permutierbar und nicht automatisch
mit Loggerphase 1 oder 2 identisch.
""")

code(r"""
scaled_process = loaded_model.preprocessing.transform_process(process_ring).X
fb_process = forward_backward(*loaded_model.process.potentials(scaled_process))
gamma, xi = fb_process.gamma, fb_process.xi
np.testing.assert_allclose(gamma.sum(axis=1), 1, atol=2e-12, rtol=0)
np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=2e-12, rtol=0)
np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=2e-12, rtol=0)
transition_mean = loaded_model.process.A / loaded_model.process.A.sum(axis=1, keepdims=True)
display(pd.DataFrame(transition_mean, index=["Von Z1", "Von Z2"], columns=["Nach Z1", "Nach Z2"]))
expected_switches = float(xi.sum() - np.trace(xi, axis1=1, axis2=2).sum())
print("DM04: erwartete Zustandswechsel unter Prozess-only-q:", expected_switches)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
for k, color in enumerate(("#276c9e", "#bd6829")):
    axes[0].plot(process_ring.times, gamma[:, k], label=f"q(z_t=Z{k+1}|X)", color=color)
axes[0].set(title="Prozess-only-Zustandswahrscheinlichkeiten – DM04", ylabel="Wahrscheinlichkeit", ylim=(-0.03, 1.03))
axes[0].legend()
axes[1].plot(process_ring.times, process_ring.X[:, 0], label="F_rad63", color="#276c9e")
axes[1].plot(process_ring.times, process_ring.X[:, 1], label="F_ax64", color="#bd6829")
axes[1].set(xlabel="Zeit [s]", ylabel="Logger-Kraftkanal [%]")
axes[1].legend()
for ax in axes:
    for boundary in boundary_times:
        ax.axvline(boundary, linestyle="--", color="#555555", linewidth=1)
show_figure(fig, "03_prozess_only_zustandsverlauf")
""")

md(r"""
### Sichtbare Modellgrenze: serielle Residualstruktur

Diese Heatmap zeigt die bereits im Basismodell implementierte deskriptive
Lag1-Diagnose an den Trainingsfaktoren. Die Residuen hängen über gamma selbst
von X und bei L auch von der Qualitätskopplung ab. Sie sind kein formaler
Innovationstest und kein bereits implementiertes AR-Modell.
""")

code(r"""
residual_acf = pd.DataFrame({rid: row["residual_lag1"]
                             for rid, row in diagnostics["rings"].items()}, index=FEATURES).T
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
im = ax.imshow(residual_acf.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(6), FEATURES, rotation=25, ha="right")
ax.set_yticks(range(len(residual_acf)), residual_acf.index)
ax.set(title="Deskriptive Lag1-Korrelation der Trainingsresiduen", ylabel="Entwicklungsring")
fig.colorbar(im, ax=ax, label="Lag1-Korrelation")
show_figure(fig, "04_residualdiagnose")
""")

md(r"""
## 10. Vollständige ringgetrennte Entwicklungsevaluation neu ausführen

In jedem der vier Folds werden sechs L-Ringe zum Fit und zwei andere L-Ringe
ausschließlich zur Bewertung verwendet. U darf gemäß dem jeweiligen Arm nur
die erlaubten Prozessanteile beitragen. Alle Scaler werden foldlokal angepasst.

Die fünf festen Arme sind TRAIN-Mittelwert, physikalische NIG-Regression,
HMM strict-C0, HMM shared-C0 und HMM SSL C=0,2. Die HMM-Arme werden in jedem
Fold frisch über 30 Epochen trainiert. Diese Zelle lädt keine alten CV-Berichte.

Die Baselines sind Vergleichsverfahren. Insbesondere ist die physikalische
NIG-Regression wegen abweichendem Design-/Prior- und teils Scalervertrag keine
vollständig isolierte Ablation des HMM-Nutzens.
""")

code(r"""
CV_DIR = RUN_DIR / "development_cv"
cv_start = time.perf_counter()

def cv_progress(event):
    event = {**event, "elapsed_seconds": time.perf_counter()-cv_start}
    with (RUN_DIR / "cv_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"Fold {event['fold']}/4 · {event['arm']}: {event['status']} "
          f"({event['elapsed_seconds']/60:.1f} min)", flush=True)

cv_report = cross_validate(DATA, config, CV_DIR, arms=ARMS, progress=cv_progress)
cv_seconds = time.perf_counter()-cv_start
seal_directory(CV_DIR)
assert len(cv_report["folds"]) == 20
print(f"Alle vier Folds und fünf Arme abgeschlossen: {cv_seconds/60:.2f} Minuten.")
""")

code(r"""
fold_checks = []
for run in cv_report["folds"]:
    split = run["fold"]
    L, U, V = set(split["L"]), set(split["U"]), set(split["VAL"])
    assert len(L) == 6 and len(U) == 4 and len(V) == 2
    assert not (L | U) & V and not (L | U | V) & set(LOCKED_IDS)
    if "scalers" in run["model"]:
        scalers = run["model"]["scalers"]
        allowed_X = L | U if run["arm"] in ("shared_c0", "ssl") else L
        assert set(scalers["y_scaler"]["fit_ids"]) == L
        assert set(scalers["x_scaler"]["fit_ids"]) == allowed_X
        assert set(scalers["descriptor_scaler"]["fit_ids"]) == allowed_X
    fold_checks.append({"Fold": split["fold"], "Arm": run["arm"],
                        "L-TRAIN": ", ".join(sorted(L)), "VAL": ", ".join(sorted(V)),
                        "Split und Scaler": "geprüft"})
for arm in ARMS:
    rows = [r for r in cv_report["per_ring"] if r["arm"] == arm]
    assert len(rows) == 8 and sorted(r["ring_id"] for r in rows) == list(LABELED_IDS)
    recalculated = regression_metrics([r["truth_mm"] for r in rows], [r["pred_mm"] for r in rows])
    np.testing.assert_allclose(recalculated["rmse_mm"], cv_report["metrics"][arm]["rmse_mm"], atol=1e-12, rtol=0)
display(pd.DataFrame(fold_checks))
print("Je Arm: acht echte Out-of-Fold-Vorhersagen; kein VAL-Ring im Scalerfit.")
""")

md(r"""
## 11. Qualitätsfehler auf zurückgehaltenen Ringen

RMSE und MAE werden gleichgewichtet über die acht vollständigen
Out-of-Fold-Ringe berechnet und für beide Ziele in mm angegeben. Jeder Punkt
stammt aus einem Fit, der das Label und die Prozesssequenz dieses VAL-Rings
nicht zum Training oder Scalerfit verwendet hat.

Die Ergebnisse sind explorativ: Die acht Ringe und die Ausgangskonfiguration
haben eine historische Entwicklungsgeschichte. Es wird keine unabhängige
Überlegenheit oder validierte 95%-Deckung behauptet.
""")

code(r"""
arm_labels = {"train_mean": "TRAIN-Mittelwert", "physical_nig": "Physikalisches NIG",
              "strict_c0": "HMM strict-C0", "shared_c0": "HMM shared-C0", "ssl": "HMM SSL"}
score_rows = []
for arm in ARMS:
    metrics = cv_report["metrics"][arm]
    score_rows.append({"Verfahren": arm_labels[arm], "Ringe": metrics["n_rings"],
                      "RMSE außen [mm]": metrics["rmse_mm"][0],
                      "RMSE Höhe [mm]": metrics["rmse_mm"][1],
                      "MAE außen [mm]": metrics["mae_mm"][0],
                      "MAE Höhe [mm]": metrics["mae_mm"][1]})
scores = pd.DataFrame(score_rows).set_index("Verfahren")
display(scores.round(6))
scores.to_csv(RUN_DIR / "cv_metrics.csv", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
colors = ["#9099a2", "#5a8bb0", "#d58b40", "#8b76b0", "#388568"]
for j, ax in enumerate(axes):
    values = np.array([cv_report["metrics"][arm]["rmse_mm"][j] for arm in ARMS])
    bars = ax.bar(range(len(ARMS)), values, color=colors)
    ax.set_xticks(range(len(ARMS)), [arm_labels[a] for a in ARMS], rotation=25, ha="right")
    ax.set(title=["Außenunrundheit", "Höhenabweichung"][j], ylabel="Out-of-Fold-RMSE [mm]", ylim=(0, values.max()*1.22))
    ax.bar_label(bars, fmt="%.3f", padding=3)
show_figure(fig, "05_ring_cv_rmse_vergleich")
""")

code(r"""
ssl_rows = sorted([r for r in cv_report["per_ring"] if r["arm"] == "ssl"], key=lambda r: r["ring_id"])
truth = np.array([r["truth_mm"] for r in ssl_rows])
predicted = np.array([r["pred_mm"] for r in ssl_rows])
oof_table = pd.DataFrame({"Ring": [r["ring_id"] for r in ssl_rows],
    "Messwert außen [mm]": truth[:, 0], "OOF außen [mm]": predicted[:, 0],
    "Messwert Höhe [mm]": truth[:, 1], "OOF Höhe [mm]": predicted[:, 1]})
display(oof_table.round(4))
oof_table.to_csv(RUN_DIR / "ssl_out_of_fold_predictions.csv", index=False, encoding="utf-8")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
for j, ax in enumerate(axes):
    lo = min(truth[:, j].min(), predicted[:, j].min())
    hi = max(truth[:, j].max(), predicted[:, j].max())
    margin = max((hi-lo)*0.12, 0.05)
    bounds = (lo-margin, hi+margin)
    ax.plot(bounds, bounds, linestyle="--", color="#666666", label="Ideale Übereinstimmung")
    ax.scatter(truth[:, j], predicted[:, j], color="#388568", s=45, zorder=3)
    for r, x, y in zip(ssl_rows, truth[:, j], predicted[:, j]):
        ax.annotate(r["ring_id"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(title=["Außenunrundheit – SSL", "Höhenabweichung – SSL"][j],
           xlabel="Gemessener Endwert [mm]", ylabel="Out-of-Fold-Vorhersage [mm]",
           xlim=bounds, ylim=bounds, aspect="equal")
    ax.legend(fontsize=8)
show_figure(fig, "06_ssl_out_of_fold_messwert_vorhersage")
""")

md(r"""
## 12. Zusammenfassung dieses tatsächlich berechneten Laufs

Die folgende Übersicht wird aus den aktuellen Objekten und neu geschriebenen
Artefakten erzeugt. Tests, Training und Prediction belegen die Funktionsfähigkeit
im spezifizierten Basismodellvertrag. Die Ring-CV beschreibt die beobachteten
Entwicklungsfehler. Sie beweist weder ein globales VI-Optimum noch allgemeine
Überlegenheit oder Unsicherheitskalibrierung.

Offen bleiben insbesondere serielle Emissionsfehlstruktur, echte Multistart-
Robustheit, vollständig kontrollierte HMM-/Bayes-Ablationen und externe
Generalisierung. Der Optimierungsfahrplan wird hier nicht umgesetzt.
""")

code(r"""
final_changed = [name for name, expected in release_manifest["files"].items()
                 if sha256(PROJECT / name) != expected]
assert not final_changed
status_rows = [
    ("Originalreferenz", f"{len(release_manifest['files'])} Hashes unverändert"),
    ("Daten", "SHA256 und 8L/4U-Vertrag geprüft"),
    ("Kurze Zustandskette", "Pfadenumeration stimmt mit Rekursion überein"),
    ("Mathematische Tests", f"{test_counts['tests']} bestanden, Exitcode 0"),
    ("Frisches Haupttraining", f"{len(history)} Epochen, {len(updates)} kontrollierte Blöcke"),
    ("Blockbilanz", f"kleinste ΔJ_C={updates['delta'].min():.3g}; Toleranz={config.monotonic_atol}"),
    ("Prozess-only-Prediction", "endlich, zwei Ziele in mm, keine Label-Eingabe"),
    ("Speichern und Laden", "Prediction exakt gleich"),
    ("Neue Entwicklungs-CV", "4 Folds × 5 Arme, 8 OOF-Ringe je Arm"),
]
display(pd.DataFrame(status_rows, columns=["Prüfung", "Tatsächliches Ergebnis"]))
summary = {
    "run_id": RUN_ID, "status": "COMPLETE", "baseline_unchanged": True,
    "data_sha256": DATA_SHA256, "source_sha256": source_hashes,
    "test_report": test_report, "epochs": len(history), "controlled_updates": len(updates),
    "first_epoch_J": float(history.iloc[0]["elbo"]), "last_epoch_J": float(history.iloc[-1]["elbo"]),
    "minimum_block_delta": float(updates["delta"].min()), "monotonic_atol": config.monotonic_atol,
    "prediction_load_equal": prediction_memory == prediction_loaded,
    "process_file_sha256": process_hash, "cv_metrics": cv_report["metrics"],
    "training_seconds": training_seconds, "cv_seconds": cv_seconds,
    "total_seconds": time.perf_counter()-TOTAL_START,
    "model_directory": str(MODEL_DIR.relative_to(ROOT)),
    "limitations": ["historically reused development rings", "uncalibrated predictive moments",
                    "conditional temporal emission independence", "no independent external evaluation"],
}
write_json(RUN_DIR / "notebook_summary.json", summary)
seal_directory(RUN_DIR)
print("Alle Schritte abgeschlossen. Neue Artefakte:", RUN_DIR)
print("Diagramme als PNG und SVG:", FIGURES)
print("Modell:", MODEL_DIR)
print(f"Gesamtdauer: {summary['total_seconds']/60:.2f} Minuten")
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "Bayesian_SSR_HMM_Demonstration.ipynb")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (.venv Implementierung_BA)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "description": "Fresh baseline training, actual tests, process-only prediction and full ring-wise CV.",
    })
    nbf.validate(notebook)
    with args.out.open("x", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"Created {args.out}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")


if __name__ == "__main__":
    main()
