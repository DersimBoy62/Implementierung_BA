# V5.4 – Prognoseschnittstelle, synthetische Negativtests

Diese eigenständige Ergänzung prüft ausschließlich bereits implementierte Eingabe-/Freigabesperren der originalen `final_candidate_v52`-Version. **Keine Änderungen am Hauptcode oder an den zehn mathematischen HMM/VI-Modulen.** Kein Fit, keine DM-Datensatzdeserialisierung, kein historischer Test, keine Prognose und keine Quality-Scores.

## Ausführen im Ordner `Implementierung_BA` (PowerShell)

1. Dieses ZIP dort entpacken. Es erstellt nur `qa_v54_prediction_boundary`.
2. Ohne Modellzugriff: ` .\.venv\Scripts\python.exe .\qa_v54_prediction_boundary\test_prediction_boundary_v54.py self-test `
3. Optional auf dem **selbst auf deinem Rechner erzeugten, vertrauenswürdigen** Entwicklungsmodell (ohne Pickle zu laden):

```powershell
.\.venv\Scripts\python.exe .\qa_v54_prediction_boundary\test_prediction_boundary_v54.py audit-model --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --out results\v54_development_replay\V54_PREDICTION_BOUNDARY_AUDIT_01.json
```

Der `audit-model`-Befehl erfordert das bestehende `results\v54_development_replay`-Verzeichnis und eine **noch nicht existierende** Ergebnisdatei; das ist eine absichtliche Write-once-Sperre. `audit-model` kontrolliert SHA256 und Manifest sowie Konfiguration/Umgebung, `load_pickle=False`. Es werden **keine** Vorhersagen erzeugt.

## Was danach weiterhin ausdrücklich fehlt

- Ein **neuer echter Ringversuch** mit vollständigem RAW-Logger-CSV (`time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin`).
- Eine **genuine, unabhängig überprüfbare** End-of-Ring-Bestätigung entsprechend `final_candidate_v52/CLOSEOUT_TEMPLATE.json`. Ein korrektes JSON-Format und übereinstimmende SHA256 sind **kein Beweis**, dass der Ring wirklich fertiggestellt wurde. Das reine Bestehen der V5.1-Datengates bestätigt ebenfalls keinen Fertigring.
- Eine erfolgreiche Positivprüfung der vollständigen `predict`-Schnittstelle an so einem Ring und anschließend **wirklich unabhängige** Qualitätsmesswerte, wenn die Generalisierungsleistung überprüft werden soll.
- Ein eigenständiger Konvergenznachweis. Bei 10 festgelegten Epochen war die zuletzt beobachtete ELBO-Zunahme +28.977708607875684.

**Bitte keine historischen IDs (DM01–DM18) als neue IDs ausgeben. Keine falschen Betreiberbestätigungen erstellen. Keine neue Hyperparametersuche auf den bereits mehrfach verwendeten Entwicklungsringen.**
