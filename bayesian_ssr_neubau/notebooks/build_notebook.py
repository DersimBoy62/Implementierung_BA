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
# Unser Bayesian-SSR-HMM Schritt für Schritt verstehen und ausführen

Dieses Notebook ist eine geführte Demonstration unseres Modells. Du kannst es
von oben nach unten lesen und ausführen, ohne jeden Python-Befehl verstehen zu
müssen. Die Texte erklären jeweils das Ziel, die nächste Berechnung und die
Bedeutung der Ausgabe.

**Die Aufgabe des Modells:** Aus dem zeitlichen Verlauf eines Ringwalzprozesses
sollen zwei gemessene Endgrößen geschätzt werden: Außenunrundheit und
Höhenabweichung, jeweils in Millimetern. Ein Ring ist dabei **ein vollständiger
Versuch** mit vielen Zeitpunkten. Viele Messzeitpunkte desselben Rings sind
keine vielen unabhängigen Beispiele für die Endqualität.

Das Modell verbindet drei Ideen:

- **Hidden Markov Model, kurz HMM:** Es beschreibt den Prozess mit verborgenen
  Zuständen, zwischen denen sich die Zugehörigkeit im Zeitverlauf ändern kann.
- **Semi-supervised Regression, kurz SSR:** Es nutzt acht Ringe mit bekannten
  Endqualitäten und zusätzlich vier Ringe, von denen nur die Prozessdaten
  verwendet werden.
- **Bayesian:** Unbekannte Größen werden durch Wahrscheinlichkeitsverteilungen
  beschrieben. Die Rechnung verwendet dafür eine berechenbare Näherung,
  die Variationsinferenz, kurz VI.

Wir prüfen hier drei verschiedene Aussagen:

| Aussage | Nachweis im Notebook |
|---|---|
| Die untersuchten Rechnungen liefern die erwarteten Ergebnisse. | Kleines vollständig nachrechenbares Beispiel und automatisierte Tests, Abschnitte 3–4. |
| Der ganze Modellablauf funktioniert. | Frisches Training, Vorhersage und identisches Ergebnis nach Speichern/Laden, Abschnitte 5–9. |
| Die Vorhersagen haben einen bestimmten Fehler. | Vergleich mit gemessenen Werten zurückgehaltener Ringe, Abschnitte 10–11. |

Ein bestandener Rechentest bedeutet noch keinen kleinen Vorhersagefehler.
Deshalb sind alle drei Teile enthalten.

## 0. Starten und den Ablauf verfolgen

1. Wähle in VS Code oben rechts den Kernel **`.venv` des Projekts**. Der Kernel
   ist der Python-Prozess, der die Codezellen ausführt.
2. Wähle **Kernel neu starten / Restart Kernel**, danach **Alle ausführen / Run All**.
3. Lies die Erklärungen und Ausgaben der Reihe nach. Spätere Zellen benötigen
   die Ergebnisse früherer Zellen.

Textzellen wie diese erklären den Ablauf. Codezellen führen die Berechnungen
aus; darunter erscheinen Tabellen, Meldungen oder Diagramme. Ein `[*]` an
einer Codezelle zeigt, dass sie noch läuft. Während des Trainings erscheinen
Epochenmeldungen, während der Vergleichsläufe Meldungen zu Fold und Verfahren.
Ohne Fehlermeldung kann eine Zelle trotzdem längere Zeit rechnen.

Der gesamte Lauf erstellt ein neues Hauptmodell mit 30 Trainingsrunden und
zwölf weitere HMMs für die Vergleichsauswertung. Es gibt keine Suche nach neuen
Modelleinstellungen; innerhalb jedes Trainings werden die Modellgrößen aber
wie vorgesehen optimiert. Alte Modelle oder fertige Vergleichsergebnisse
ersetzen keine dieser Rechnungen. Die Dauer hängt von deinem Rechner ab.

Du kannst zunächst bis Abschnitt 4 mitlesen und die Daten- und Rechenprüfungen
ansehen. Für die abschließenden Fehlertabellen musst du auch die späteren
Trainingszellen ausführen. Nach einem Neustart des Kernels beginnt der Ablauf
wieder oben. Die leeren Ausgabefelder der Vorlage sind normal: Sie werden erst
bei deinem eigenen Lauf gefüllt.

### Begriffe, die in den Ausgaben wiederkehren

| Begriff | Bedeutung in diesem Notebook |
|---|---|
| `X`, Features, Prozessdaten | Die sechs Eingangskanäle über die Zeit. |
| `y`, Targets, Labels | Die beiden gemessenen Endqualitäten eines Rings. |
| `L` / `U` | Ringe mit genutztem Qualitätslabel / Ringe, deren Qualität nicht verwendet wird. |
| Fit oder Training | Anpassen der Modellgrößen an die erlaubten Trainingsdaten. |
| Epoche | Eine vollständige Runde der vorgesehenen Modellupdates. |
| Scaler | Eine aus Trainingsdaten berechnete Umrechnung der Größen auf vergleichbare Skalen. |
| Fold, `VAL`, Out-of-Fold oder `OOF` | Eine Vergleichsrunde, die zurückgehaltenen Ringe und deren Vorhersagen durch ein ohne sie trainiertes Modell. |

Die erste Codezelle richtet die Umgebung ein, sucht die lokale Rohdatei und
legt einen neuen Ausgabeordner an. Danach zeigt sie den verwendeten Python,
den Projektordner und den Ordner **dieses** Laufs. Fehlen die Rohdaten, stoppt
sie mit einem Hinweis auf `input/DM_16_aktive_Ringe.pkl` beziehungsweise
`BAYESIAN_SSR_DATA`. Es wird dann kein Training gestartet.
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
        if (candidate / "bayesian_ssr_neubau/pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Notebook innerhalb des Workspace Implementierung_BA öffnen.")

ROOT = find_workspace()
PROJECT = ROOT / "bayesian_ssr_neubau"
NOTEBOOKS = PROJECT / "notebooks"
DATA = Path(os.environ.get("BAYESIAN_SSR_DATA", ROOT / "input/DM_16_aktive_Ringe.pkl")).expanduser().resolve()
if not DATA.is_file():
    raise FileNotFoundError(
        f"Private Rohdaten fehlen: {DATA}. Die freigegebene Datei lokal unter "
        "input/DM_16_aktive_Ringe.pkl bereitstellen oder BAYESIAN_SSR_DATA setzen."
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
## 1. Umgebung und geprüfte Modellquellen

Damit Ergebnisse nachvollziehbar sind, halten wir fest, **mit welchen Dateien
und Paketversionen** gerechnet wird. Zwei verschiedene Programmstände könnten
sonst ähnlich aussehen und trotzdem andere Ergebnisse liefern.

Die nächste Zelle vergleicht den Modellcode und seine Einstellungen mit
`SOURCE_SHA256.json`. Eine SHA256-Prüfsumme ist ein digitaler Fingerabdruck
einer Datei. Auch eine kleine Änderung führt normalerweise zu einer anderen
Prüfsumme. Zusätzlich wird geprüft, dass keine Quelldatei hinzugekommen ist
oder fehlt.

**Die Ausgabe lesen:** Die Tabelle nennt die installierten Pakete und ihre
Versionen. Die Meldung `Modellquellen: ... Dateien geprüft` bestätigt den
Vergleich mit dem festgehaltenen Quellenstand. Das ist eine Prüfung der
Dateiidentität; ob die Rechnungen richtig sind, untersuchen wir erst danach.

Bei `Quellenmanifest weicht ab` den angezeigten Dateinamen prüfen und den
gewünschten Projektstand klären. Das Manifest sollte nicht nur deshalb neu
geschrieben werden, damit die Meldung verschwindet. Gespeicherte Modelle oder
alte Trainingsausgaben werden für diesen Startschritt nicht benötigt.
""")

code(r"""
def validate_source_manifest():
    manifest = read_json(PROJECT / "SOURCE_SHA256.json")
    if manifest.get("schema") != "bayesian-ring-ssr-source-sha256/1":
        raise ValueError("Unbekanntes Schema des Quellenmanifests.")
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Quellenmanifest enthält keine gültige Dateiliste.")
    if not (PROJECT / "src/bayesian_ring_ssr/__init__.py").is_file():
        raise FileNotFoundError("Modellquellen fehlen.")
    paths = [PROJECT / "pyproject.toml"]
    paths.extend(sorted((PROJECT / "src").rglob("*.py")))
    paths.extend(sorted((PROJECT / "configs").rglob("*.json")))
    actual = {path.relative_to(PROJECT).as_posix(): sha256(path) for path in sorted(paths)}
    differences = [name for name in sorted(set(recorded) | set(actual))
                   if recorded.get(name) != actual.get(name)]
    if differences:
        raise ValueError("Quellenmanifest weicht ab: " + ", ".join(differences))
    return manifest

source_manifest = validate_source_manifest()
source_manifest_sha256 = sha256(PROJECT / "SOURCE_SHA256.json")
source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((PROJECT / "src").rglob("*.py"))}
package_names = ["numpy", "scipy", "scikit-learn", "pandas", "numba", "pytest",
                 "nbformat", "nbclient", "ipykernel", "matplotlib"]
versions = {name: metadata.version(name) for name in package_names}
display(pd.DataFrame({"Paket": versions.keys(), "Version": versions.values()}))
print(f"Modellquellen: {len(source_manifest['files'])} Dateien geprüft.")
write_json(RUN_DIR / "environment.json", {
    "python": sys.version, "executable": sys.executable, "packages": versions,
    "source_manifest_sha256": source_manifest_sha256,
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

Hier kontrollieren wir, welche Ringe das Modell bekommt und welche
Informationen es von ihnen verwenden darf. Die lokale Datei enthält 16 aktive
Ringe. Für diesen Lauf verteilen sie sich so:

| Gruppe | Anzahl und Ringe | Verwendete Information |
|---|---|---|
| L: gelabelte Entwicklungsringe | 8: DM04, DM05, DM06, DM09, DM12, DM16, DM17, DM18 | Prozessverlauf und zwei Endqualitäten. |
| U: ungelabelte Entwicklungsringe | 4: DM01, DM02, DM10, DM14 | Nur Prozessverlauf; keine Qualitätslabels. |
| Historische Außenringe | 4: DM07, DM08, DM11, DM15 | Für diesen Lauf gesperrt; kein Training und keine neue Bewertung. |

Die Datei wird erst nach passender SHA256-Prüfsumme eingelesen. Technisch
enthält das Sammelpickle auch weitere Endwerte, die beim Laden materialisiert
werden. Anschließend werden nur die erlaubten Informationen weitergereicht:
Ein U-Trainingsobjekt besitzt weder `y` noch `targets`. Die Trennung in der
Trainingsschnittstelle ist damit ausdrücklich geprüft.

**F2** ist der hier festgelegte aktive Ausschnitt aus dem Loggerverlauf:
Phase 1 oder 2, `LaserOK=1` und `MessEin=1`. Die ausgewählten Zeitpunkte müssen
zusammenhängen und die weiteren gespeicherten Zeit-/Phasenregeln erfüllen.
Es wird nicht beliebig geglättet oder ein fehlender Zeitpunkt ergänzt.

**Die Tabelle lesen:** Jede Zeile gehört zu einem Entwicklungsring. Die Zahl
der F2-Zeitpunkte zeigt seine Sequenzlänge; Start und Ende sind Loggerzeiten in
Sekunden. Die Phasengrenze ist eine beobachtete Loggergrenze. `Fehlende Werte`
soll für die hier freigegebenen Sequenzen null sein. Die sechs anschließend
genannten Features sind Eingaben, die zwei Targets sind die Endgrößen.

Diese Prüfung verhindert unter anderem, dass ein gesperrter Ring oder ein
U-Qualitätslabel versehentlich im Training landet. Sie beweist noch keine
gute Vorhersage.
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

Jetzt sehen wir die **Eingabedaten** eines Rings, bevor das Modell daraus eine
Endqualität schätzt. Das Beispiel DM04 besitzt sechs Diagramme, eines für
jeden verwendeten Kanal.

| Kanal | Bedeutung laut Loggerheader | Einheit der dargestellten Rohwerte |
|---|---|---|
| `F_rad63`, `F_ax64` | Radialer und axialer Kraftkanal | Prozent; keine hier verifizierten kN. |
| `Ist_D150`, `Ist_s151`, `Ist_h152` | Außendurchmesser, Wanddicke, Höhe | Meter. |
| `RWG55` | Ringwachstumsrate | Meter pro Sekunde. |

Auf der waagerechten Achse steht die Zeit, auf der senkrechten der jeweilige
Messwert. Die sechs Kurven haben verschiedene Einheiten und Skalen; ihre
Zahlenhöhen sind deshalb nicht unmittelbar vergleichbar. Intern werden sie
später mit auf Trainingsdaten bestimmten Scalern umgerechnet.

Die gestrichelte Linie markiert die beobachtete Grenze der Loggerphasen. Sie
ist eine Orientierung im Diagramm, kein zusätzliches Modellfeature und keine
vorgegebene Grenze zwischen den versteckten HMM-Zuständen. Auch eine plausible
Kurve liefert für sich allein keinen Nachweis guter Qualitätsvorhersagen.
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

Bevor wir lange echte Sequenzen berechnen, prüfen wir einen winzigen Fall,
den man vollständig durchgehen kann: **zwei mögliche Zustände und vier
Zeitpunkte**. Ein möglicher Zustandspfad lautet beispielsweise `1 → 1 → 2 → 2`.
Insgesamt gibt es `2⁴ = 16` mögliche Pfade.

Die nächste Zelle berechnet dasselbe Ergebnis auf zwei Wegen:

1. Sie zählt alle 16 Pfade auf und gewichtet sie mit ihren Wahrscheinlichkeiten.
2. Sie verwendet die effizienten Rekursionen, die auch im Modell eingesetzt werden.

Beide Wege sollen übereinstimmen. Das Beispiel arbeitet mit künstlichen
Zahlen; es lernt dabei nichts über die Qualität eines echten Rings.

Die drei zusammengefassten Zustandsmerkmale sind:

| Merkmal | Einfache Bedeutung |
|---|---|
| `h1` | Anteil der Zeit in Zustand 1. |
| `h2` | Anteil der Zeit in Zustand 2; zusammen ergeben `h1` und `h2` immer 1. |
| `Delta1` | Anteil von Zustand 1 in der späteren Hälfte minus seinem Anteil in der frühen Hälfte. |

Beim Beispielpfad `1 → 1 → 2 → 2` wären das `h1=0,5`, `h2=0,5` und
`Delta1=−1`. In der Tabelle stehen allerdings die **über alle Pfade gemittelten**
Werte, nicht die Merkmale nur dieses einen Pfads.

**Die Ausgabe lesen:** `Rekursion` und `Pfadenumeration` sollen bis auf sehr
kleine Rundungsfehler gleich sein. `Max. Momentenabweichung` soll nahe null
liegen; eine Zahl wie `1e-16` ist etwa 0,0000000000000001.

Die zusätzliche Matrix beschreibt `E[ggᵀ]`, also gemittelte Produkte der
Merkmale. Sie enthält Informationen, die im Mittelwert allein fehlen. Der
danach ausgegebene Unterschied `||E[ggᵀ] − E[g]E[g]ᵀ||` **darf positiv sein**:
Er zeigt gerade die Merkmalsunsicherheit und ist kein Rechenfehler. Für
unsichere Zustände darf man diese beiden Ausdrücke nicht einfach gleichsetzen.
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

Ein automatisierter Test gibt einer Funktion kontrollierte Eingaben und
vergleicht ihre Ausgabe mit einer erwarteten Antwort oder einer unabhängig
formulierten Rechnung. Viele unterschiedliche Tests untersuchen verschiedene
Fehlermöglichkeiten.

Die nächste Zelle startet das Testprogramm **pytest neu mit dem Python dieses
Notebooks**. Sie liest keinen alten Erfolgsbericht ein. Geprüft werden unter
anderem Zustandswahrscheinlichkeiten und Gradienten, die bayesschen Updates,
Trainingsdaten-Trennung sowie Speichern und Laden. Manche Tests benutzen
synthetische, also künstlich erzeugte Ringe; andere prüfen die echten Daten
oder vergleichen mit den bytegleich gesicherten historischen Testreferenzen.

**Die Ausgabe lesen:** Im aktuellen Stand erwarten wir `107 passed`. Ein Punkt
steht normalerweise für einen bestandenen Test. Die genaue Zahl stammt aus
deinem tatsächlichen Aufruf und kann sich nach bewussten Testerweiterungen
ändern. In der Tabelle sollen `exitcode`, `failures`, `errors` und `skipped`
jeweils null sein. Dieses vollständige Notebook hat die Rohdatei bereits
geprüft und verlangt deshalb auch die tatsächliche Ausführung der Datentests.

Bei einem Fehler stoppt die Ausführung. Die letzte Meldung und
`pytest_output.log` im neuen Laufordner helfen beim Einordnen. Das Notebook
setzt den Ablauf nicht mit einem behaupteten Testerfolg fort.

Bestandene Tests sind konkrete Belege für die untersuchten Eigenschaften.
Sie garantieren weder absolute Fehlerfreiheit noch kleine Vorhersagefehler
bei neuen Ringen. Letzteres untersuchen wir ab Abschnitt 10.
""")

code(r"""
junit_path = RUN_DIR / "pytest_results.xml"
test_command = [sys.executable, "-m", "pytest", str(PROJECT / "tests"), "-q",
                f"--junitxml={junit_path}"]
test_start = time.perf_counter()
test_run = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                               "BAYESIAN_SSR_DATA": str(DATA)})
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

Vor dem Training sehen wir nach, welche Einstellungen tatsächlich verwendet
werden. Das ist unser festgelegtes Ausgangsmodell, hier **Baseline** genannt.
Das Notebook sucht nicht automatisch nach einer besseren Einstellung.

| Einstellung | Bedeutung |
|---|---|
| `K=2` | Zwei verborgene Prozesszustände. Das sind keine zwei Zielgrößen oder zwei vorher festgelegten Loggerphasen. |
| `variant=M2` | Die Qualitätsrechnung verwendet Aufenthaltsanteile und den Unterschied zwischen später und früher Zustandsbelegung. |
| `C=0,2` | Der gesamte Prozessbeitrag eines U-Rings zur Trainingszielfunktion wird mit 0,2 gewichtet. Das ist weder eine Erfolgswahrscheinlichkeit noch ein Anteil künstlicher Labels. |
| `train_epochs=30` | 30 vollständige Runden der Modellupdates. |
| `local_maxiter=8` | Höchstens acht interne Optimiereriterationen pro lokalem L-Update; keine Vorgabe von acht Gesamtepochen. |
| `seed=20261111` | Festgelegter Startwert für die verwendeten Zufallsverfahren. |

Die M2-Darstellung entfernt nur das letzte überflüssige Delta-Merkmal.
Die Aufenthaltsmerkmale bleiben erhalten; ihre Summe trägt den konstanten
Regressionsanteil. Der Prior, also die vor dem Fit festgelegte Verteilung,
wird passend in die reduzierte Darstellung übertragen. Die beiden
Qualitätsköpfe haben dadurch interne Vektorlängen 9 und 8. Es werden weiterhin
genau **zwei** Qualitätswerte vorhergesagt.

Die Trainingszielfunktion heißt `J_C`. Für `C≠1` ist sie eine gewichtete,
generalisierte Variationszielfunktion. Merke dir für die nächsten Diagramme:
**Größer ist für dieses Trainingsziel besser; daraus folgt noch kein kleinerer
Vorhersagefehler.**
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

Hier beginnt das eigentliche Lernen aus allen acht L- und vier U-Ringen.
Die Modellgrößen werden frisch initialisiert; ein alter trainierter Zustand
wird nicht als Startpunkt geladen. Die L-Ringe liefern Prozess- und
Qualitätsinformation, die U-Ringe nur den erlaubten Prozessbeitrag.

Eine **Epoche** durchläuft nacheinander diese Updates:

1. Für jeden der zwölf Ringe wird seine Zustandsverteilung aktualisiert.
2. Das gemeinsame Prozessmodell wird mit den erlaubten Informationen aktualisiert.
3. Der Qualitätskopf für Außenunrundheit und der Qualitätskopf für
   Höhenabweichung werden jeweils aktualisiert.

Das ergibt `12 + 1 + 2 = 15` kontrollierte Updates pro Epoche und bei 30 Epochen
insgesamt **450 Updates**. Die lokalen L-Rechnungen nutzen einen numerischen
Optimierer; U-Zustände werden mit der dafür vorgesehenen Forward-Backward-
Rechnung aktualisiert. Innerhalb einer Epoche baut jeder Schritt auf dem
aktuellen Stand der anderen Modellteile auf.

**Den Fortschritt lesen:** Nach jeder vollständigen Epoche erscheint etwa
`Epoche 01/30: J_C = ...`. Negative Zahlen sind dabei normal. Beispielsweise
ist `−900` größer als `−1000`; die Bewegung von `−1000` nach `−900` wäre also
eine Verbesserung dieses Trainingsziels. Diese Beispielzahlen sind keine
vorgegebenen Ergebnisse für deinen Lauf.

Die Gesamtzielfunktion wird vor und nach den einzelnen Updates kontrolliert.
Ein Schritt darf sie nicht stärker als die festgelegte numerische Toleranz
verschlechtern. Pro Epoche wird außerdem ein Zwischenstand gespeichert.
Die abschließende Meldung nennt Trainingsdauer und Modellordner. Dort liegen
das neue Modell, seine Einstellungen und Prüfsummen zum späteren Laden.
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

Nach dem Training prüfen wir die gespeicherten Rechenschritte. Die nächste
Zelle verlangt 30 Epochen und 450 Updateblöcke und kontrolliert, dass die Werte
endlich sind und die erlaubte Verschlechterung nicht überschritten wurde.

**Die Tabelle lesen:** `Updates` zählt die Schritte einer Gruppe.
`Kleinster Gewinn` ist die kleinste Änderung von `J_C` in dieser Gruppe;
`Gesamtgewinn` addiert ihre Änderungen. Ein Gewinn von null kann bedeuten,
dass ein Update bereits nichts mehr verbessert oder ein schlechterer
Optimierervorschlag nicht übernommen wurde.

**Linkes Diagramm:** Die waagerechte Achse zählt Epochen, die senkrechte zeigt
`J_C`. Die Kurve sollte im Rahmen der Rundungstoleranz steigen oder flach
verlaufen. Eine flachere Kurve bedeutet kleinere Fortschritte dieses Fits;
sie beweist noch kein global bestmögliches Modell.

**Rechtes Diagramm:** Die Linien zerlegen die Verbesserung je Epoche nach
Zustandsupdates, gemeinsamem Prozessmodell und Qualitätsköpfen. So sieht man,
welcher Teil wie viel zum Trainingsziel beigetragen hat. Die Höhen sind keine
prozentualen Wichtigkeiten der Sensoren.

Eine Differenz bis `−1e-6`, also minus 0,000001, wird als numerische Toleranz
akzeptiert. Größere Verletzungen würden zum Fehler führen. Auch ein sauber
steigender Verlauf sagt noch nicht, wie gut unbekannte Ringe vorhergesagt
werden; dafür folgt die getrennte Vergleichsauswertung.
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

md(r"""
### Die zusätzliche Optimierertabelle richtig einordnen

Die nächste Tabelle betrachtet die internen Optimiereraufrufe für die acht
L-Ringe über 30 Epochen. Bei unveränderter Konfiguration sind das
`8 × 30 = 240` Aufrufe. Die vier U-Ringe haben eigene direkte Zustandsupdates
und zählen hier nicht als solche Optimiereraufrufe.

`Akzeptierte Optimiererschritte` zählt Aufrufe, deren Vorschlag tatsächlich
übernommen wurde. `Ohne Erfolgsstatus` bedeutet, dass der Optimierer sein
internes Abschlusskriterium nicht als erfüllt meldete; beispielsweise kann
bereits die Grenze von acht Iterationen erreicht worden sein. Dieser Zähler
allein bedeutet deshalb keinen Programmabbruch. Die Annahmekontrolle prüft
zusätzlich, ob der vorgeschlagene Schritt die Zielfunktion verbessert.

Der Abschlussgradient beschreibt, wie stark sich das lokale Ziel in der
untersuchten Richtung noch ändern kann. Ein kleiner Gradient ist ein Hinweis
auf lokale Ruhe, kein Beweis eines globalen Optimums. Die Tabelle hilft beim
Einordnen des Rechenverlaufs; sie ist kein Vorhersagefehler in Millimetern.
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

Jetzt testen wir den Weg, den eine Vorhersage im späteren Gebrauch nimmt:

1. Die Loggerdaten von DM04 werden als neue CSV ohne Endqualitätsfelder geschrieben.
2. Die CSV wird als reine Prozesseingabe wieder eingelesen.
3. Das aktuelle Modell erzeugt zwei Qualitätsvorhersagen.
4. Das gespeicherte Modell wird neu geladen und muss genau dieselben Werte liefern.

Die Vorhersagefunktion bekommt dabei keine gemessene Endqualität. Das verhindert,
dass sie die gesuchte Antwort direkt aus ihrer Eingabe abliest.
**DM04 war allerdings im Training des Hauptmodells enthalten.** Dieses Beispiel
zeigt daher, dass Datenübergabe, Vorhersage und Speichern/Laden funktionieren.
Die echte Fehlermessung auf zurückgehaltenen Ringen folgt in Abschnitt 11.

**Die Tabelle lesen:** Jede Zeile gehört zu einer Zielgröße. `Vorhersage [mm]`
ist ihr geschätzter Mittelwert. `Marginale SD [mm]` ist die vom Modell
berechnete Standardabweichung dieser einzelnen Zielgröße innerhalb seiner
Näherung. Eine größere SD beschreibt eine breitere vorhergesagte Verteilung.
Sie ist weder der bereits gemessene Fehler dieses Rings noch eine Garantie,
dass der wahre Wert in einem bestimmten Abstand liegt. Insbesondere ist
`Mittelwert ± 2 × SD` hier kein nachgewiesenes 95%-Intervall.

Die Meldung `Vorhersage im Speicher und nach Laden: exakt gleich` bestätigt
den Speicher-/Ladetest. Die anschließende `Erwartete Eingabesicherung bei
LabeledRing` ist **beabsichtigt**: Das Notebook probiert ausdrücklich eine
unzulässige Eingabe mit Labels aus, fängt deren Ablehnung ab und läuft weiter.
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

Das HMM beschreibt nicht direkt beobachtete Prozesszustände. Für jeden
Zeitpunkt schätzt es, wie wahrscheinlich Zustand 1 beziehungsweise Zustand 2
ist. Es muss dabei nicht zu jedem Zeitpunkt einen Zustand mit absoluter
Sicherheit auswählen.

Hier wird diese Zustandsrechnung für DM04 erneut durchgeführt, **nur aus den
Prozessdaten und den geladenen Modellgrößen**. Während des Trainings können
bei L-Ringen auch deren Qualitätslabels die lokale Zustandsverteilung
beeinflussen; die hier gezeigte Vorhersagerechnung bekommt diese Labels nicht.

**Die Übergangstabelle lesen:** Eine Zeile beginnt bei `Von Z1` oder `Von Z2`,
die Spalten zeigen den nächsten Zustand. Ein großer Diagonaleintrag steht für
eine hohe modellierte Wahrscheinlichkeit, beim nächsten Zeitpunkt im selben
Zustand zu bleiben. Die Einträge einer Zeile summieren sich zu eins.

**Oberes Diagramm:** Für jeden Zeitpunkt werden zwei Wahrscheinlichkeiten
zwischen 0 und 1 gezeigt, die zusammen eins ergeben. Liegen sie etwa bei
0,5 und 0,5, ist die Zuordnung unsicher. Liegt eine nahe eins, bevorzugt das
Modell dort diesen Zustand. Der angegebene erwartete Wechselzähler darf eine
Kommazahl sein, weil er über unsichere Zustandsfolgen mittelt.

**Unteres Diagramm:** Die beiden Kraftkanäle zeigen den beobachteten Prozess
zum selben Zeitverlauf. So kannst du Zustandsänderungen und Signalverlauf
nebeneinander betrachten.

`Z1` und `Z2` bedeuten nicht automatisch „gut/schlecht“ oder Loggerphase 1/2.
Die Zustandsnummern sind grundsätzlich vertauschbar. Die gestrichelte
Loggergrenze ist Orientierung; eine Übereinstimmung mit Zustandsänderungen
müsste inhaltlich untersucht werden.
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

Ein **Residuum** ist hier die Differenz zwischen einem beobachteten
Prozesskanal und seinem vom Zustandsmodell beschriebenen Mittelwert. Gemeint
sind somit die sechs Prozesssignale, nicht die Fehler der zwei Endqualitäten.

Die farbige Tabelle, eine Heatmap, betrachtet pro Ring und Kanal, wie stark
benachbarte Residuen zeitlich zusammenhängen. `Lag1` bedeutet einen Abstand
von genau einem Messschritt.

| Farbe beziehungsweise Wert | Bedeutung dieser Beschreibung |
|---|---|
| Rot, nahe +1 | Benachbarte Residuen hängen stark in gleicher Richtung zusammen. |
| Blau, nahe −1 | Benachbarte Residuen hängen stark in entgegengesetzter Richtung zusammen. |
| Nahe 0 | Geringer linearer Zusammenhang bei genau diesem Zeitabstand. |

Unser Basismodell behandelt die Emissionen bei gegebenem Zustand und gegebenen
globalen Parametern als zeitlich unabhängig. Deutliche Restmuster sind deshalb
ein Hinweis darauf, dass zeitliche Struktur übrigbleibt. Kanalzusammenhänge
am selben Zeitpunkt und Zusammenhänge zwischen Zeitpunkten sind verschiedene
Dinge.

Die Darstellung ist eine beschreibende Trainingsdiagnose. Die verwendeten
Zustandsgewichte hängen selbst von den Prozessdaten und bei L auch von der
Qualitätskopplung ab. Die Heatmap ist daher weder ein formaler statistischer
Test noch eine externe Qualitätsbewertung. Sie ergänzt auch kein neues
autoregressives Modell, sondern macht eine Grenze des Basismodells sichtbar.
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

Hier beginnt der Teil zur **Vorhersagegüte auf zurückgehaltenen Ringen**.
Die Qualität eines bereits zum Fit verwendeten Rings wiederzugeben wäre
einfacher und könnte einen zu guten Eindruck vermitteln.

Wir teilen die acht L-Ringe deshalb in vier feste Vergleichsrunden, die
**Folds**, auf. In jeder Runde werden sechs L-Ringe zum Training verwendet.
Die zwei übrigen heißen `VAL`: Ihre Prozesse und Endqualitäten dürfen weder
in den Fit noch in die Berechnung der Scaler dieses Modells gelangen. Erst
nach dem jeweiligen Fit werden ihre Prozessverläufe dem Modell zur Vorhersage
übergeben und die Ergebnisse mit ihren gemessenen Endwerten verglichen.

Nach vier Runden war jeder der acht L-Ringe genau einmal zurückgehalten.
So entstehen acht **Out-of-Fold-Vorhersagen pro Verfahren**: Jede stammt aus
einem Modell, das diesen Ring nicht zum Lernen benutzt hat. Die vier U-Ringe
können je nach Verfahren ihre erlaubten Prozessinformationen beitragen.

Wir vergleichen fünf festgelegte Verfahren. Ein „Arm“ in den Ausgaben ist
einfach eines dieser Verfahren:

| Name in den Ausgaben | Vorgehen |
|---|---|
| `TRAIN-Mittelwert` / `train_mean` | Sagt immer den Durchschnitt der sechs L-TRAIN-Qualitäten des jeweiligen Folds vorher. Das ist der einfachste Vergleich. |
| `Physikalisches NIG` / `physical_nig` | Bayessche Regression mit zusammengefassten Prozessmerkmalen, etwa ausgewählten Mittelwerten und Änderungen; ohne HMM-Zustandsmodell. |
| `HMM strict-C0` / `strict_c0` | HMM mit Qualitätsköpfen, aber ohne U-Einfluss auf Fit, Skalierung oder Initialisierung. |
| `HMM shared-C0` / `shared_c0` | U-Prozesse helfen bei Vorverarbeitung und Initialisierung; ihr Beitrag zur eigentlichen Trainingszielfunktion bleibt null. |
| `HMM SSL` / `ssl` | Unser Basismodell: U-Prozesse werden zusätzlich mit `C=0,2` in der Trainingszielfunktion verwendet. |

Für jeden Fold beginnt jeder Fit neu. Drei Verfahren sind HMMs, also entstehen
**4 × 3 = 12 neue HMM-Trainingsläufe** über jeweils 30 Epochen. Zusammen mit
den einfacheren Vergleichen ergeben sich 4 × 5 = 20 Fold-/Verfahrensberichte.
Diese Zelle ist daher oft der längste Teil des Notebooks. Die Meldungen
`started` und `complete` zeigen, welches Verfahren in welchem Fold gerade
berechnet wird. Alte CV-Berichte werden nicht anstelle neuer Fits geladen.

Die NIG-Regression ist ein nützlicher Vergleich, unterscheidet sich jedoch
auch bei Design und Priors. Ein Unterschied zu ihr misst deshalb nicht
isoliert nur den Nutzen des HMM. Ebenso können die acht historisch schon zur
Entwicklung verwendeten Ringe keine neue unabhängige externe Studie ersetzen.
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

md(r"""
### Kontrollieren, dass die Trennung wirklich eingehalten wurde

Nach den Fits kontrolliert die nächste Zelle für jeden Fold und jedes
Verfahren die Ringzuordnung. In der Tabelle sollen pro Zeile sechs
`L-TRAIN`-Ringe und zwei andere `VAL`-Ringe stehen. `geprüft` wird erst nach
den entsprechenden Programmprüfungen eingetragen.

Ein **Scaler** lernt ebenfalls aus Daten, auch wenn er nur Mittelwerte und
Streuungen zur Umrechnung bestimmt. Deshalb wird kontrolliert, dass auch er
keinen VAL-Ring gesehen hat. Die erlaubte Einbeziehung von U bei den beiden
gemeinsamen Vorverarbeitungsarmen wird dabei gesondert berücksichtigt.

Zusätzlich wird geprüft, dass pro Verfahren genau die acht erwarteten
zurückgehaltenen Ringe vorliegen und die gespeicherten Fehler mit den neu aus
den Einzelvorhersagen berechneten Fehlern übereinstimmen.
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

Jetzt stehen für jeden zurückgehaltenen Ring Vorhersage und Messwert fest.
Ihr Unterschied ist der Vorhersagefehler. Die Tabelle fasst diese Fehler
getrennt für Außenunrundheit und Höhenabweichung zusammen.

| Spalte | Bedeutung |
|---|---|
| `Ringe` | Zahl der ausgewerteten Ringe pro Verfahren; hier acht. |
| `MAE ... [mm]` | Durchschnitt der absoluten Fehler. Vorzeichen heben sich dabei nicht auf. |
| `RMSE ... [mm]` | Quadratwurzel aus dem Durchschnitt der quadrierten Fehler; große Einzelfehler fallen stärker ins Gewicht. |

**Für beide Fehlermaße gilt: kleiner ist besser, null wäre perfekte
Übereinstimmung.** Ein MAE von beispielsweise 0,5 mm hieße, dass der absolute
Fehler im Mittel 0,5 mm beträgt. Das Beispiel ist keine Vorgabe für dein
Ergebnis; einzelne Ringe können deutlich stärker abweichen. Jeder Ring zählt
gleich viel, unabhängig von der Länge seiner Loggersequenz.

Beginne beim Vergleich mit der Zeile `TRAIN-Mittelwert`. Hat das komplexere
Modell kleinere Fehler als dieser einfache Vergleich? Untersuche dann auch
die anderen Verfahren und beide Zielgrößen getrennt. Ein Vorteil bei
Außenunrundheit muss kein Vorteil bei Höhe sein. Ist der Fehler größer, muss
man das als solches festhalten; ein komplizierteres Modell ist nicht
automatisch besser.

**Die Balkendiagramme lesen:** Links steht der RMSE der Außenunrundheit,
rechts der Höhenabweichung. Die Höhe eines Balkens entspricht dem Fehler in
Millimetern, also ist innerhalb eines Diagramms ein niedrigerer Balken besser.
Die Achsenskalierungen der beiden Diagramme können verschieden sein.

Diese Zahlen beschreiben die beobachteten Entwicklungsringe. Wegen der kleinen,
historisch wiederverwendeten Datenbasis sind sie keine allgemeine
Überlegenheitsgarantie für künftig unbekannte Ringe. Für eine solche Aussage
wäre eine zusätzliche unabhängig geplante Bewertung nötig.
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

md(r"""
### Die einzelnen Ringe und das Streudiagramm lesen

Ein Mittelwert kann verbergen, dass manche Ringe gut und andere schlecht
vorhergesagt werden. Deshalb zeigt die nächste Tabelle jeden der acht Ringe
einzeln für unser SSL-Modell. `Messwert` ist die bekannte Endqualität,
`OOF` die Vorhersage aus dem Fold, der diesen Ring zurückgehalten hat.

Im anschließenden Streudiagramm steht der **Messwert auf der waagerechten**
und die **Vorhersage auf der senkrechten Achse**. Jeder beschriftete Punkt ist
ein Ring. Die gestrichelte Diagonale bedeutet perfekte Übereinstimmung:

- Nahe der Linie: kleiner Fehler dieses Rings.
- Oberhalb der Linie: Die Vorhersage ist größer als der Messwert.
- Unterhalb der Linie: Die Vorhersage ist kleiner als der Messwert.

Die zwei Diagramme gehören zu den zwei Zielgrößen. Sie zeigen keine zusätzliche
neue Testmenge und keine garantierten Unsicherheitsintervalle, sondern dieselben
acht OOF-Vorhersagen, aus denen die SSL-Fehler in der vorherigen Tabelle stammen.
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

Die letzte Zelle fasst zusammen, **was in deinem aktuellen Lauf wirklich
berechnet und geprüft wurde**. Ihre Ergebnisse stammen aus den gerade
erzeugten Objekten und Dateien. Zu Beginn und am Ende müssen die geprüften
Modellquellen übereinstimmen.

Wenn die Zelle ohne Fehler endet, wurden die vorherigen Programmprüfungen
erfüllt. Die neue Übersicht nennt unter anderem die tatsächliche Testanzahl,
die Epochen und Updates sowie die identische Vorhersage vor und nach dem
Laden. Die Datei `notebook_summary.json` erhält dann den Status `COMPLETE`.
Beim Terminal-Hilfsskript steht der Abschlussstatus außerdem im gesonderten
Ausführungsbericht.

Für deine Beurteilung bleiben diese drei Aussagen getrennt:

1. **Rechenprüfung:** Die kontrollierten mathematischen Beispiele und Tests bestehen.
2. **Ablaufprüfung:** Training, prozessbasierte Vorhersage und Speichern/Laden
   funktionieren für die hier festgelegten Verträge.
3. **Beobachteter Vorhersagefehler:** Die Fehlertabellen zeigen, wie die fünf
   Verfahren auf den zurückgehaltenen Entwicklungsringen abschneiden. Ob das
   SSL-Modell dort besser ist, liest du aus diesen Zahlen, nicht aus `COMPLETE`.

Die Abschlussmeldungen nennen den Ordner dieses Laufs. Darin findest du
Diagramme als PNG und SVG unter `figures/`, das trainierte Modell unter
`model_8L4U/`, Vergleichsergebnisse unter `development_cv/` und Tabellen als CSV.
Die Dateien bleiben lokal unter `outputs/notebook_demos/<lauf_id>/`; ein neuer
Gesamtlauf erzeugt einen neuen Ordner.

Offen bleiben unter anderem zeitliche Restmuster, Empfindlichkeit gegenüber
anderen Initialisierungen und eine unabhängige externe Bewertung. Die
berechneten Standardabweichungen haben keine nachgewiesene Intervallabdeckung.
Das Notebook belegt damit einen ausgeführten, geprüften Basismodellablauf und
seine beobachteten Entwicklungsfehler; weitere Optimierungsversuche sind ein
eigener Arbeitsschritt.
""")

code(r"""
final_manifest = validate_source_manifest()
assert final_manifest == source_manifest, "Quellenmanifest wurde während des Laufs verändert."
assert sha256(PROJECT / "SOURCE_SHA256.json") == source_manifest_sha256
status_rows = [
    ("Modellquellen", f"{len(source_manifest['files'])} Hashes unverändert"),
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
