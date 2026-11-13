# Bayesian Bachelor — V5.4 technische Entwicklungsintegration (V5.2-Release)

**Status 12.11.2026: QA-Overlay, kein Modell-Freeze, keine neue Validierung.**

Dieses Overlay ergänzt ausschließlich den Ordner `qa_v54_real_replay`. Es ändert weder
`final_candidate_v52` noch die V5.1-Eingangsprüfung, die zehn SHA-gesicherten mathematischen
Kernmodule, die festen Entwicklungs-/Test-IDs oder alte V5.3/V5.4-Ergebnisse.

## Ziel und Grenzen

Die ursprüngliche V5.2-CLI `main_bayesian_ssr_v52.py train` erzeugt **einen**
ausdrücklich vorläufigen **8L+4U-Entwicklungsfit** mit der bereits fixierten
Konfiguration (K2/M2/P0/C0.2/physical_var0.38526108747593474, 10 Epochen,
8 lokale VI-Schritte). Das originale V5.2-Release speichert atomar
`TRAINED_MODEL.pkl` + `TRAIN_MANIFEST.json` mitsamt X-/Y-/Deskriptorskalierern,
Eingangsrichtlinie, Umgebungsdaten und SHA256-Prüfsummen.

Das zusätzliche QA-Overlay **trainiert nie selbst**. Es lädt das von V5.2
lokal erzeugte und anhand aller ursprünglichen SHA256-Werte verifizierte Paket
zweimal unabhängig im gleichen Python-Prozess und wiederholt numerisch die
Vorhersagen ausschließlich für **DM01 (U-Prozess-Trainingsring)** und
**DM04 (gelabelter Trainingsring)**. Außerdem vergleicht es alle vorhandenen
Skalierungsparameter mit den ausschließlich aus den zwölf zulässigen
Entwicklungsringen neu berechneten Referenzen. Das `verify`-Kommando wiederholt
die Prüfung in einem **frisch gestarteten Python-Prozess** und vergleicht die
Gesamtprüfung bytekanonisch mit einem Write-once-Bericht.

**Es werden keine Qualitätslabels für eine Vorhersagebewertung verwendet.**
Das heißt: keine RMSE, MAE, R², Konfidenzintervallabdeckung, weitere Modellauswahl
oder Aussagen zur unabhängigen Generalisierung. DM07/DM08/DM11/DM15 bleiben
untersagt. Das zugrunde liegende DM-Pickle wird vom Original-Entwicklungsloader
als Gesamtobjekt deserialisiert; keine der historischen Außenring-Einträge wird
anschließend ausgelesen oder featurisiert. Nur vertrauenswürdige eigene
Pickles verwenden.

Eine wichtige **nicht geprüfte Schnittstelle** bleibt die reale Prognose eines
vollständig neuen Rings mittels `predict` (korrekte Prozess-CSV und unabhängig
bestätigter Ringabschluss). Historische DM-IDs dürfen nicht durch Umbenennen
als „neuer Ring“ missbraucht werden. Diese Integration testet das
Train/Save/Load/Replay und ist kein prospektiver Feldtest.

## Installation

Das ZIP direkt in `Implementierung_BA` entpacken. Es entsteht allein
`qa_v54_real_replay` und es wird keine bestehende Datei überschrieben. Die
vorhandene originale und Windows-korrigierte `final_candidate_v52` muss daneben
liegen. Alle folgenden Befehle im Projektordner ausführen.

### A. Tests ohne Training

```powershell
.\.venv\Scripts\python.exe .\qa_v54_real_replay\check_v52_development_replay.py self-test
.\.venv\Scripts\python.exe .\qa_v54_real_replay\check_v52_development_replay.py preflight --data input\DM_16_aktive_Ringe.pkl
```

`preflight` prüft erst den SHA256-Wert und deserialisiert dann das ausschließlich
vertrauenswürdige Originalpickledatenset. `self-test` öffnet keinerlei Datensatz.

### B. **Einmaliger** vorläufiger Entwicklungsfit mit unveränderter V5.2-CLI

```powershell
New-Item -ItemType Directory -Force results\v54_development_replay | Out-Null
.\.venv\Scripts\python.exe .\final_candidate_v52\main_bayesian_ssr_v52.py train --data input\DM_16_aktive_Ringe.pkl --out results\v54_development_replay\MODEL_8L4U_RUN01
```

**Nicht mehrfach ausführen:** V5.2 verweigert vorhandene Modelldestinationen.
Bei Fehlern zunächst Meldungen prüfen, das Original nicht löschen oder
überschreiben. Keine weiteren Hyperparameter ändern. Dieser Befehl ist
explizit ein *Entwicklungstraining*, **kein** offizielles finales BA-Training.

### C. Read-only Modellprüfung und unabhängige Replay-Prüfung

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v52\main_bayesian_ssr_v52.py verify-model --model-dir results\v54_development_replay\MODEL_8L4U_RUN01
.\.venv\Scripts\python.exe .\qa_v54_real_replay\check_v52_development_replay.py audit --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --out results\v54_development_replay\V54_REAL_DEV_REPLAY_AUDIT_01.json
.\.venv\Scripts\python.exe .\qa_v54_real_replay\check_v52_development_replay.py verify --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --report results\v54_development_replay\V54_REAL_DEV_REPLAY_AUDIT_01.json
```

`audit` schreibt seinen JSON-Prüfbericht **einmalig**. `verify` ist read-only.
Wenn eine Datei schon existiert, nicht überschreiben; einen neuen Namen wählen.

## Erwartete Prüfungen

- V5.2-Konfigurationshash, zehn Mathematik-SHA256 und V5.1-Codehash bleiben unverändert.
- Das V5.2-Manifest, trainiertes Modell und Python-/Paketversionen stimmen bei
  jedem Laden exakt überein; ungültige/unerwartete Pakete werden abgewiesen.
- X-Scaler: 8 L + 4 U, über gepoolte Zeitpunkte, genau dieselben Mittel/Skalen.
- y-Scaler: nur die 8 gelabelten Ringe (keine U-Labels).
- Physikalische Deskriptorskalierer: über 8 L + 4 U, je ein Ring als Zeile.
- Sicherheitsrichtlinie: ausschließlich über erlaubte Entwicklungs-Prozessdaten.
- Zwei Trainingsprozess-Vorhersagen nach zwei geprüften Neuladungen numerisch
  exakt gleich; separate erneute Ausführung `verify` vergleicht Bericht erneut.
- Die zehn ELBO-Werte und ihr letzter Zuwachs sind nachvollziehbar archiviert.
  **Konvergenz bleibt ungeprüft.** Eine nichtmonotone ELBO oder ein letzter
  signifikanter Zuwachs darf nicht durch Neutuning auf den mehrfach verwendeten
  Entwicklungslabels „behoben“ werden.

## Provenienz / Vertrauensgrenze

Das Manifest verknüpft Modell und lokale Softwareversion, ist aber **nicht
extern digital signiert**; SHA256 ist kein Herkunftsnachweis bei gleichzeitig
manipulierbarem Modell und Manifest. Nie fremde oder untrusted Pickles laden.
Ein tatsächlich neues Ringexperiment mit unabhängigen Qualitätszielen bleibt
für echte externe Generalisierungsprüfung erforderlich. `RWG55` bleibt in seiner
physikalischen Semantik anhand der Maschinenunterlagen gesondert zu bestätigen.
