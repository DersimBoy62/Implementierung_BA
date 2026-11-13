# V5.4 — Read-only ELBO-Verlaufsdiagnostik des realen Entwicklungsmodells

**Zweck:** Das bereits unter Windows trainierte Entwicklungsmodell `MODEL_8L4U_RUN01` anhand des bekannten SHA256-Fingerabdrucks nur lesend analysieren. Keine Modellparameter ändern, kein weiteres Training, keine neuen Qualitätsmetriken, kein Lesen des DM-Datenpickles, kein Zugriff auf den alten Test.

**Vertrauensgrenze:** Das Skript nutzt die bereits implementierte V5.2-Prüfung `verify_bundle(...,load_pickle=True)` und darf daher nur auf das eigene, lokal erzeugte und vertrauenswürdige Modellverzeichnis angewandt werden. SHA256/Manifest sind keine extern authentifizierten Signaturen.

Das ZIP im Projektverzeichnis `Implementierung_BA` entpacken, sodass `qa_v54_elbo_audit` parallel zu `final_candidate_v52` liegt. Danach in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\qa_v54_elbo_audit\audit_model_history_v54.py self-test

.\.venv\Scripts\python.exe .\qa_v54_elbo_audit\audit_model_history_v54.py audit --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --replay-report results\v54_development_replay\V54_REAL_DEV_REPLAY_AUDIT_01.json --out results\v54_development_replay\V54_ELBO_HISTORY_AUDIT_01.json

.\.venv\Scripts\python.exe .\qa_v54_elbo_audit\audit_model_history_v54.py verify --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --replay-report results\v54_development_replay\V54_REAL_DEV_REPLAY_AUDIT_01.json --report results\v54_development_replay\V54_ELBO_HISTORY_AUDIT_01.json
```

`audit` schreibt einmalig; bei bestehendem Ziel nicht überschreiben. `verify` ist nur lesend. Es wird die **vollständige Zehnerfolge der ELBO**, der Veränderung pro Epoche, der Zustandsmassen und des lokalen ELBO-Zugewinns ausgewiesen. Ferner die letzte Änderung relativ zur absoluten finalen ELBO sowie die korrekte Summe aller zehn lokalen Gewinne (das ursprüngliche Feld `final_train_local_gain_sum` war missverständlich benannt).

**Wissenschaftlich:** Das Skript beweist keine Konvergenz, legt keinen nachträglichen Schwellenwert fest und startet kein weiteres Training. Die Epochenzahl von zehn bleibt gesperrt. Eine verbesserte oder verschlechterte Güte auf dem mehrfach verwendeten Entwicklungssatz lässt sich damit weder nachweisen noch selektieren. Das am 12.11.2026 trainierte Modell bleibt vorläufig.
