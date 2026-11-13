# Bayesian Bachelor — V5.4 / technischer PRE-FREEZE-Prüfzweig

**Stand:** 12.11.2026. **Status:** `PRE-FREEZE / NO MODEL APPROVAL`. Dieses Paket ist ein **reines QA-Overlay**. Es ist **keine** neue Modellversion, trainiert keine echten DM-Ringe und optimiert keine Hyperparameter. Die existierende `final_candidate_v52`-Version und ihre zehn SHA-gesicherten mathematischen Kernmodule werden **nicht geändert**. Die V5.3-Originalergebnisse bleiben unverändert.

## Ziel

1. V5.3-CV4-Originalergebnisse (acht historische Entwicklungsetiketten) anhand der **vier vollständigen Fold-JSONs und der Originalzusammenfassung** unabhängig nachrechnen und provenance-/split-/leakage-kontrollieren.
2. ELBO-Verläufe aller acht Fit-Arme ohne neue Experimente dokumentieren. Die Höhe des ELBO-Gewinns in Epoche 10 wird **nicht** als Beleg für Konvergenz oder neue Epochenwahl interpretiert.
3. Negative Mutationstests gegen verfälschte Fold-/Scaler-/Summendaten ausführen, ohne Originaldateien zu verändern.
4. Ein echtes **synthetisch trainiertes** Bayesian-HMM zweimal mit gleicher Konfiguration fitten, prädizieren, als **synthetisches lokales Pickle (nicht als V5.2-DM-Modellrelease!)** serialisieren und Vorhersagen nach Laden vergleichen.

## Installation

ZIP **direkt** in `Implementierung_BA` entpacken; neuer Ordner `qa_v54_freeze_gate`. Erwartete Nachbarordner:

- `final_candidate_v52` — die unter Windows bereits erfolgreich geprüfte, **nicht überschriebene** korrigierte V5.2-Version.
- `research_v53_feedback` — unberührt.
- `results/v53_feedback_devonly` — die **fünf bereits vorhandenen** Original-JSONs (Fold 1–4 + Summary).

**Nicht in einen anderen Projektordner kopieren** und nicht die V5.2-Core-Dateien verschieben. Ein neues Ergebnisverzeichnis wird ausdrücklich vorher erstellt; die Audit-JSON wird nur einmal geschrieben.

## PowerShell (zunächst nur QA, kein Training mit DM-Daten)

```powershell
.\.venv\Scripts\python.exe .\qa_v54_freeze_gate\pre_freeze_gate_v54.py self-test
.\.venv\Scripts\python.exe .\qa_v54_freeze_gate\pre_freeze_gate_v54.py test-audit-negative --results-dir results\v53_feedback_devonly
.\.venv\Scripts\python.exe .\qa_v54_freeze_gate\pre_freeze_gate_v54.py synthetic-replay
New-Item -ItemType Directory -Force results\v54_pre_freeze | Out-Null
.\.venv\Scripts\python.exe .\qa_v54_freeze_gate\pre_freeze_gate_v54.py audit-v53 --results-dir results\v53_feedback_devonly --out results\v54_pre_freeze\V54_ORIGINAL_CV4_AUDIT.json
```

**Erwartung auf dem Entwickler-System:** 7/7 synthetische Selbsttests, 7/7 negative Ergebnismutationstests, synthetische Doppeltraining-/Pickle-Wiederholung, unabhängig bestätigte CV4-Gesamtmetriken. Unter Windows mit Python 3.11.9 ist das neue Overlay **noch nicht geprüft**; auch kleine numerische Abweichungen zwischen Systemen sind möglich.

Die letzte Audit-Ausgabe verweigert ein vorhandenes Ziel (`open('x')`). Falls du den Befehl mehrfach ausführst, verwende einen **anderen** neuen Dateinamen; vorhandene Auswertungen niemals überschreiben.

## Wissenschaftliche Interpretation

- CV4 ist Explorationsforschung auf acht bereits mehrfach verwendeten gelabelten Entwicklungsringen, **keine unabhängige Validierung**.
- Die beiden HMM-/SSR-Arme unterscheiden sich ausschließlich in der lokalen VI-Rückkopplung (`steps=0` versus `steps=8`); **beide** nutzen gelabelte Qualitätsköpfe und die vier U-Prozessringe. Das ist **nicht** Strict Supervised versus Semi-Supervised.
- Keine automatische Änderung von `K=2`, `C=0.2`, `P0`, `M2`, `physical_var=0.38526108747593474`, `epochs=10`, `steps=8` für den eingefrorenen Kandidaten.
- Beide Arms zeigen in Fold 3 noch einen besonders großen ELBO-Zuwachs in Epoche 10; daher ist **Konvergenz nicht nachgewiesen**. Das ist ein Befund, kein Anlass zur nachträglichen Verbesserung von Entwicklungsmetriken.
- Das Original-Feld `final_train_local_gain_sum` ist semantisch irreführend: es enthält den **letzten Epochenwert**, nicht die Summe. Das Overlay dokumentiert getrennt beide Größen und verändert die Originalresultate **nicht**.
- Ein synthetisches Pickle mit vertrauenswürdiger Herkunft beweist nur technische Serialisierung in der Testumgebung. Es ist **kein** signierter Release und **kein** Beweis der Echtheit fremder Pickles oder plattformübergreifender bitweiser Reproduzierbarkeit. **Nie fremde Pickles laden.**
- Während dieser QA wird die originale DM-Pickledatei **nicht** geöffnet; es findet weder realer DM-Fit noch eine Qualitätsbewertung historischer Testringe DM07/DM08/DM11/DM15 statt.

## Verbleibende echte Freigaben

1. Die vier Befehle **unter Windows/Python 3.11.9** laufen lassen, Ausgaben und neue Audit-JSON archivieren.
2. Reproduzierbaren **entwicklungsbasierten** End-to-End-V5.2-Modellrelease-Test planen, bei dem nach explizitem Start ein realer 8L+4U-Entwicklungsfit einschließlich X-/Y- und Deskriptor-Skalierer gespeichert, geladen und ohne neue Qualitätsmetrik wiedergegeben wird. **Nicht Teil dieses QA-Overlays.**
3. Modellkonvergenz als Limitierung diskutieren; keine post-hoc Epochensuche im wiederverwendeten Entwicklungsdatensatz.
4. Erst nach dokumentierter wissenschaftlicher und technischer Prüfung eine Freigabeentscheidung treffen. **V5.4 hier bedeutet nur Vorprüfung, nicht finalen Model-Freeze.**

## Datei-/Hashmanifest

`V54_QA_SHA256_MANIFEST.json` enthält SHA256 des Python-Prüfskripts und dieser README. Die V5.2-Config und alle zehn Kernmodule werden zur Laufzeit direkt gegen den ursprünglichen V5.2-Lock geprüft. Änderungen am Overlay dürfen nicht stillschweigend als bestandener ursprünglicher Code ausgegeben werden.
