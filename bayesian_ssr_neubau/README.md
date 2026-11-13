# Bayesian Ring SSR

Eigenständiges variationales Bayesian-SSR-HMM für vollständige DM-Ringsequenzen:
volle Normal-Wishart-Emissionen, gemeinsame Markov-Zustände und zwei ringweite
Bayesian-NIG-Qualitätsköpfe. Produktivcode unter `src/` importiert keine alten
Modellkerne. Die bytegleich übernommenen historischen Referenzen liegen
ausschließlich bei den Tests.

Die Basiskonfiguration bleibt K=2, M2, C=0,2, Seed 20261111, 30 Epochen und acht
lokale L-BFGS-B-Iterationen. C gewichtet den gesamten U-Prozessbeitrag
einschließlich Entropie; für C≠1 ist dies eine generalisierte Variationszielfunktion.
Alle Fits starten frisch. Die induzierten M2-Priors behalten alle Aufenthaltsmerkmale
und entfernen nur die letzte Delta-Komponente; die Kopfgrößen sind 9 und 8.

## Installation

Python 3.11 oder 3.12 verwenden. Die folgenden Befehle laufen vom
Repository-Root `Implementierung_BA` aus:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\bayesian_ssr_neubau[audit,notebook]"
.\.venv\Scripts\python.exe -m pytest -q
```

Für die festgehaltene Windows-/Python-3.11-Umgebung zuerst
`requirements-lock-windows-py311.txt` installieren und danach das lokale Paket
mit `--no-deps -e .\bayesian_ssr_neubau`. Das optionale Extra `audit` enthält
pytest; `notebook` ergänzt die Pakete für die Demonstration. Für reine
Modellnutzung genügt `pip install -e .\bayesian_ssr_neubau`.

## Lokale Daten

Die freigegebenen Rohdaten gehören nach `input/DM_16_aktive_Ringe.pkl` im
Repository-Root und werden nicht versioniert. Alternativ:

```powershell
$env:BAYESIAN_SSR_DATA = 'C:\Daten\DM_16_aktive_Ringe.pkl'
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli audit-data
```

Ein explizites `--input PFAD` hat Vorrang vor der Umgebungsvariable. Vor jeder
Deserialisierung muss die Datei SHA256
`bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`
haben. Ziele sind `ovality.ovality_radial_mm` und
`planarity.height_deviation_mm` aus `zielwerte`. U-Trainingsobjekte enthalten
keine Qualitätslabels. Die historischen Außenringe DM07/08/11/15 bleiben gesperrt.
Die technische Deserialisierung des Sammelpickles materialisiert dessen Inhalt;
erst danach werden ausschließlich die freigegebenen L-Ziele weitergereicht.

## Training und Evaluation

Für jeden Lauf einen neuen Ausgabeordner wählen:

```powershell
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli train --config bayesian_ssr_neubau\configs\baseline_k2_m2.json --out bayesian_ssr_neubau\outputs\baseline_8L4U_002
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli evaluate --config bayesian_ssr_neubau\configs\baseline_k2_m2.json --out bayesian_ssr_neubau\outputs\development_cv_002
```

Die deterministische vierfache Ring-CV vergleicht TRAIN-Mean, physikalische
NIG-Regression, strict-C0, shared-C0 und SSL mit C=0,2. Jeder Fold hält zwei
der acht L-Ringe zurück und fittet alle Normalisierer erneut ausschließlich auf
dem jeweils erlaubten TRAIN-Anteil. Es gibt keine automatische Hyperparameterwahl
und keinen unabhängigen äußeren Testscore.

Ausgaben enthalten Modellfaktoren, Split-/Featureverträge, TRAIN-Scaler,
Blockbilanzen, Checkpoints, Diagnostik und Integritätsmanifest. Laden prüft
Integrität und semantische Verträge. Checkpoints dokumentieren den frischen Fit;
sie sind kein automatischer Resume-Pfad. Vorhandene lokale Läufe bleiben
verwendbar und werden nicht nach GitHub übertragen.

## Vorhersage

Eine vollständige Rohlogger-CSV muss exakt diese Spalten in dieser Reihenfolge
enthalten:

```text
time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin
```

Vollständiger Vor-/Nachlauf und die gespeicherten F2-Regeln sind erforderlich.
Keine Endqualitätsfelder oder beliebigen Pickles als neue Predict-Eingaben.
Für kontrollierte bereits-F2-Daten ist zusätzlich NPZ möglich; die Grenzen
stehen in [DATENPROVENIENZ.md](docs/DATENPROVENIENZ.md).

```powershell
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli predict --model bayesian_ssr_neubau\outputs\baseline_8L4U_002 --ring-file C:\Daten\neuer_ring.csv --ring-id NEU01 --out bayesian_ssr_neubau\outputs\prediction_NEU01.json
```

Die API `predict_new_ring(model, process_ring)` akzeptiert keine
`LabeledRing`-Objekte. Ein explizites `.process_only()` entfernt vorhandene
Labels. Ergebnisse sind marginale Mittelwerte und Standardabweichungen in mm.

## Tests, Quellprüfung und Notebook

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\check_source_manifest.py
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\audit_importgraph.py
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\run_verification.py
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\execute_notebook.py
```

Die synthetischen Mathematik-/Persistenztests und historischen Paritätstests
funktionieren ohne private Rohdaten oder alte Workspace-Ordner. Tests mit
echten Ringen werden ohne Daten ausdrücklich übersprungen; mit verfügbarer
freigegebener Datei müssen sie vollständig laufen. Das Notebook benötigt die
Rohdaten und erzeugt frische Trainings-/CV-Ergebnisse. Seine Vorlage enthält
keine früheren Ausgaben. Details: [Testanleitung](docs/TESTPROTOKOLL.md) und
[Notebook-Anleitung](notebooks/README.md).

`SOURCE_SHA256.json` bindet aktuelle Produktivquellen, Konfiguration und
Paketdefinition; frühere Ausgabedateien sind keine Startvoraussetzung. Nach
bewusst geprüften Quelländerungen aktualisiert
`scripts/check_source_manifest.py --write` dieses Manifest. Die historischen
Testreferenzen haben ein getrenntes unveränderliches Fixture-Manifest.

## Fachliche Grenzen und Archiv

Die volle Emissionskovarianz erfasst keine zeitliche Residualabhängigkeit.
Die Prediction integriert Markov-Mischungsmomente innerhalb der gewählten
VI-Approximation, keine vollständige globale Posteriorverteilung. Die
Unsicherheiten sind nicht kalibriert. Acht wiederholt verwendete Entwicklungslabels
erlauben keinen neuen unabhängigen Generalisierungs- oder Überlegenheitsnachweis.
Die metrologische Gleichsetzung der Planaritätshöhenabweichung mit normativem
Höhenschlag bleibt offen. Das Paket ist ein Basismodell, kein finaler Freeze.

Die mathematische Herleitung, Datenverträge und Forschungsgrenzen stehen in
`docs/`. Der vollständige frühere Projektstand mit Originalinventuren,
historischen Befunden und Optimierungsplanung liegt im Git-Branch
`archive/vollstand-2026-11-13`. Die [Projektstruktur](docs/PROJEKTSTRUKTUR.md)
erklärt die Bereinigung und die zusätzliche lokale Sicherung.
