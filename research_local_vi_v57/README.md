# V5.7 | Lokale L-BFGS-B-Variationsinferenz untersuchen

**Forschungsphase, kein Model-Freeze und kein Generalisierungstest.**

Wir haben nach V5.5 und V5.6 gelernt, dass bei gleicher Initialisierung das Verdoppeln des maximalen lokalen Optimierer-Budgets (8 auf 16) zu derselben End-ELBO führt. Eine Plateauphase ist jedoch kein Beleg für lokale Stationarität.

Dieses Diagnose-Overlay lässt den vollständig unveränderten V5.2-Trainingskern erneut für 30 Epochen laufen und beobachtet *in-memory* die Aufrufe von `scipy.optimize.minimize` sowie die vom Kern tatsächlich akzeptierten lokalen Updates. Während des Aufrufs werden die Originalfunktionen nur zeitweilig durch protokollierende Wrapper ersetzt und danach zurückgesetzt. Die Kernmodule auf dem Datenträger bleiben unverändert und werden erneut SHA-geprüft. Die Wiederholung muss die bereits archivierte Original-ELBO bis maximal 1e-5 reproduzieren. Es werden keine Modellgewichte gespeichert, keine historischen Outer-Tests und keine Vorhersage-Qualitätsmetriken berechnet.

Gemessen wird bei jedem der 8 gelabelten Ringe in jeder der 30 Epochen: Anzahl Iterationen/Funktionsauswertungen, `OptimizeResult.success/status/message`, Erreichen von `maxiter`, erste und letzte maximale absolute Gradientenkomponente, Verbesserung der lokalen Zielfunktion und Übernahme des Kandidaten durch den ursprünglichen Kern. Die Logitparametrisierung ist redundanter Natur; kleine Ableitungen alleine sind kein globaler Konvergenzbeweis.

## Entpacken
Den Ordner `research_local_vi_v57` direkt unter `Implementierung_BA` ablegen. `final_candidate_v52` und `research_convergence_v55` müssen wie bisher vorhanden sein. Python-Umgebung unverändert lassen.

## Selbsttest
```
.\.venv\Scripts\python.exe .\research_local_vi_v57\audit_local_optimizer_v57.py self-test
```

## Zunächst 8 lokale Iterationen instrumentieren
```
New-Item -ItemType Directory -Force results\v57_local_vi | Out-Null
.\.venv\Scripts\python.exe .\research_local_vi_v57\audit_local_optimizer_v57.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --local-steps 8 --reference results\v55_convergence_trainonly\CONVERGENCE_30E_TRAINONLY_01.json --out results\v57_local_vi\LOCAL_VI_TRACE_L8_E30_01.json
```

Bei erfolgreicher Prüfung können wir 16 Iterationen getrennt instrumentieren, indem `--local-steps 16` sowie die Referenz `results\v56_train_stability\SEED_20261111_L16_E30_01.json` verwendet werden und ein neuer Outputname `LOCAL_VI_TRACE_L16_E30_01.json` gewählt wird.

Alle Outputs sind write-once. Vor Wiederholung vorhandene JSON prüfen, nicht löschen. Reine Analyse der lokalen Optimierer-Ursachen, **kein Beleg für vorhergesagte Qualität oder neue Test-Unabhängigkeit**.
