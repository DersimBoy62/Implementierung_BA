# Bayesian Ring SSR – eigenständiger Basismodell-Neubau

Neuer Python-Code für vollständige DM-Loggersequenzen, mit zwei ringweiten
Bayesian-NIG-Qualitätsköpfen und einem gemeinsamen variationalen Bayesian-HMM.
Die alten Modellkerne dienen ausschließlich als separat geprüfte Test-Oracles.
Der Produktivcode unter `src/` importiert keinen davon.

Die Basis verwendet K=2, M2, C=0,2, Seed20261111, 30 Epochen und acht lokale
L-BFGS-B-Iterationen. C gewichtet den gesamten U-Prozessbeitrag einschließlich
Entropie: für C≠1 ist dies eine generalisierte Variationszielfunktion. Alle Fits
starten frisch. Die induzierten M2-Priors erhalten beide Aufenthaltsmerkmale und
entfernen nur die letzte Delta-Komponente; die Kopfgrößen sind 9 und 8.

Der tatsächlich ausgeführte 8L+4U-Lauf liegt unter
`outputs/baseline_8L4U_001`. Nach 30 Epochen beträgt J_C
−22028,914444073278; keine Blockabnahme überschreitet die festgelegte
Toleranz1e-6. Das ist ein Rechen-/Trainingsbefund. Die Entwicklungsevaluation wird
gesondert in `outputs/development_cv_001/development_cv.json` dokumentiert.

## Installation und echte Befehle

Alle Befehle vom bisherigen Workspace-Root `Implementierung_BA` aus ausführen.
Python3.11.9 und die vorhandene `.venv` wurden verwendet. Fehlende Pakete und
Buildwerkzeuge wurden ergänzt; alte Daten, Kernmodule und Ergebnisdateien wurden
nicht überschrieben. Ein plattformspezifischer vollständiger Versionslock wird
mit `scripts/run_verification.py` als `requirements-lock-windows-py311.txt` erfasst.
`pyproject.toml` enthält die unterstützten Versionsbereiche. Für eine frische
Umgebung zuerst den Lock installieren, danach das lokale Paket:

```powershell
.\.venv\Scripts\python.exe -m pip install -r bayesian_ssr_neubau\requirements-lock-windows-py311.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .\bayesian_ssr_neubau
.\.venv\Scripts\python.exe -m pytest bayesian_ssr_neubau\tests -q
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli audit-data --input input\DM_16_aktive_Ringe.pkl
```

Der folgende Trainingsbefehl wurde mit `baseline_8L4U_001` ausgeführt. Für eine
Wiederholung **einen neuen Namen** verwenden, beispielsweise `_002`:

```powershell
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli train --config bayesian_ssr_neubau\configs\baseline_k2_m2.json --out bayesian_ssr_neubau\outputs\baseline_8L4U_002
```

Für die ebenfalls ausgeführte vierfache Entwicklungs-CV:

```powershell
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli evaluate --config bayesian_ssr_neubau\configs\baseline_k2_m2.json --out bayesian_ssr_neubau\outputs\development_cv_002
```

Arme: TRAIN-Mean; physikalische NIG-Regression mit Intercept und denselben
Deskriptorgruppen; strict-C0 mit ausschließlich L-Preprocessing/Initialisierung;
shared-C0 mit U-X-Preprocessing/Initialisierung; SSL mit C=0,2. Jeder Fold hält
zwei der acht L-Ringe zurück, alle Trainingsnormalisierer werden neu gefittet.
Es gibt keine automatische Hyperparameterwahl und keinen äußeren Testscore.

## Vorhersage ohne Endqualitätszugriff

Die Rohlogger-CSV enthält exakt diese Spalten, in dieser Reihenfolge:

```text
time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin
```

Sie enthält vollständigen Vor-/Nachlauf; F2 wird mit den gespeicherten Regeln
ausgewählt. Keine Qualitätsfelder, Walzprotokoll-Endmessungen oder Pickles als
neue Predict-Eingaben. Ein kontrolliertes bereits-F2-NPZ ist zusätzlich möglich;
dessen Vertragsgrenzen sind in `docs/DATENPROVENIENZ.md` erklärt.

Diese beiden Befehle wurden mit `_001` ausgeführt; neue Ausgabenamen verwenden:

```powershell
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli export-process --ring-id DM04 --out bayesian_ssr_neubau\outputs\prediction_process_DM04_002.csv
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli predict --model bayesian_ssr_neubau\outputs\baseline_8L4U_001 --ring-file bayesian_ssr_neubau\outputs\prediction_process_DM04_002.csv --ring-id DM04 --out bayesian_ssr_neubau\outputs\prediction_DM04_002.json
```

Dieses Beispiel ist ein **Schnittstellen-/Persistenztest an einem Trainingsring**,
keine Qualitätsvalidierung. Echte Train/Prediction-Trennung liefern die
synthetischen zurückgehaltenen Ringe und die vollständigen VAL-Ringe der CV.
Die Funktion `predict_new_ring(model, process_ring)` akzeptiert kein
`LabeledRing`-Objekt. Bei vorhandenen Labels ist `.process_only()` explizit nötig.
Vorhersage liefert die beiden marginalen Mittelwerte und SD in mm.

## Daten- und Artefaktvertrag

- Genau die SHA-verifizierte Datei `input/DM_16_aktive_Ringe.pkl` wird importiert.
  SHA256: `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`.
- Ziele: `ovality.ovality_radial_mm`, `planarity.height_deviation_mm`; der reale
  Pickle-Container heißt `zielwerte`. Die Bedeutung einer normativen
  Höhenschlagmessung ist damit noch nicht metrologisch belegt.
- Die vier historisch geöffneten Außenringe DM07/08/11/15 sind gesperrt.
  U-Trainingsobjekte besitzen weder `y` noch `targets`. Das SHA-freigegebene
  Sammelpickle materialisiert beim Deserialisieren physisch alle enthaltenen
  Werte; ausschließlich die freigegebenen L-Ziele werden danach ausgelesen und
  weitergereicht. Dieser Importgrenze wird keine stärkere Eigenschaft unterstellt.
- Jedes Trainingsverzeichnis ist neu, mit JSON-Faktoren, Split, Feature-/Gate-
  Vertrag, TRAIN-Scalern, Daten-QA, Update-JSONL, 30 Epochen-Checkpoints,
  Diagnostik und SHA256-Manifest. Laden prüft Integrität und semantische Verträge.
- Checkpoints dokumentieren den laufenden frischen Fit. Automatisches Resume
  aus alten oder neuen Checkpoints ist bewusst kein Basismodell-Trainingspfad.
- Vollständige Exportartefakte sind lesbare JSON/CSV/NPZ; Modelle werden nicht
  als beliebige Python-Pickles gespeichert oder geladen.

## Mathematik, Tests und Forschungsgrenzen

Die Dokumente unter `docs/` enthalten das mathematische Modell, den A–D/V6.0-
Abgleich, Herkunftsinventur und Originalquellenstatus, Datenprovenienz, reale
Testprotokolle und Abschlussbefunde. `tests/reference_parity/` darf den alten
Oracle read-only importieren; unabhängige Enumeration, Dichteintegrale und
Finite-Differenzen stehen zusätzlich in den übrigen Tests.

Die Emissionen haben volle Kanalkovarianz, sind zeitlich bedingt unabhängig.
Starke serielle Residualkorrelation bleibt eine wesentliche Modellgrenze. Kein
AR-HMM wurde ergänzt. Die Prediction verwendet erwartete Logpotenziale und
exakte Markov-Mischungsmomente innerhalb der gewählten VI-Approximation. Sie
liefert keine volle globale Posteriorintegration und keine kalibrierten
95%-Mischungsquantile. Acht wiederholt benutzte Entwicklungslabels erlauben
keine neue unabhängige Generalisierungs- oder Überlegenheitsbehauptung.

Der Optimierungsfahrplan unter `handover/08_OPTIMIERUNG_NACH_NEUBAU` bleibt ein
separater Folgeauftrag. Dieses Paket ist ein Basismodell, kein finaler Freeze.
