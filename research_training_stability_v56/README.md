# V5.6 – Initialisierung und lokale VI als kontrollierte Trainingsforschung

**Nur Forschung auf 8 gelabelten und 4 maskierten Prozessringen.** Kein neu unabhängiger Test, kein Hyperparameter-Tuning nach Qualitätsmetrik, keine Auswertung von DM07/08/11/15. V5.2-Kernmodule und gespeicherter Originalmodellstand bleiben unverändert. Dies ist ein separater Forschungsversuch, kein Freeze und keine Qualitätsfreigabe.

## Warum?

V5.5 belegt bei K=2/C=0,2/P0/M2, Seed 20261111 und maximal 8 lokalen L-BFGS-B-Iterationen eine Plateauphase nach einer **erheblichen ELBO-Verbesserung erst ab Epoche 11**. Um Aussagen zur Trainingskonvergenz robuster zu machen, untersuchen wir unabhängig von Testmetriken (1) Seed-abhängige K-Means-Initialisierung, (2) das Budget der lokalen VI. Wir behalten 30 Epochen, dieselben 8L+4U-Ringe, identische Skalierung und unveränderte Kernmodule bei.

**Voraussetzung:** Ordner `research_convergence_v55` und `final_candidate_v52` im selben Projektordner wie `research_training_stability_v56`.

## Schritt 1 – Synthetik

```powershell
.\.venv\Scripts\python.exe .\research_training_stability_v56\training_stability_v56.py self-test
```

## Schritt 2 – Ein einziger ergänzender Initialisierungslauf (Seed 20261112, acht lokale Schritte)

```powershell
New-Item -ItemType Directory -Force results\v56_train_stability | Out-Null
.\.venv\Scripts\python.exe .\research_training_stability_v56\training_stability_v56.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --seed 20261112 --local-steps 8 --out results\v56_train_stability\SEED_20261112_L8_E30_01.json
```

Erst nach Auswertung dieses Laufs folgen gegebenenfalls Seed 20261113 und der kontrollierte Vergleich 8 gegen 16 lokale L-BFGS-B-Maximaliterationen bei gleichen Seeds. Keine automatischen Mehrfachtrainings.

## Interpretation und Grenzen

Die ELBO kann lokale Optima oder unterschiedliche Zustandszuordnungen zeigen. Sortierte Zustandsanteile sind eine grobe, permutationstolerante Kennzahl, kein Beweis identischer latenter Zustände. Die fünf letzten relativen ELBO-Änderungen unter 1e-4 sind lediglich ein beschreibendes Kriterium. **Insbesondere bleiben Optimierer-Success, lokale Gradienten und die ELBO-Beiträge bisher ungeloggt**; das erfordert ein separates Instrumentierungsprojekt.

## Optional: Vergleich vorhandener Forschungsberichte

```powershell
.\.venv\Scripts\python.exe .\research_training_stability_v56\training_stability_v56.py compare --reports results\v55_convergence_trainonly\CONVERGENCE_30E_TRAINONLY_01.json results\v56_train_stability\SEED_20261112_L8_E30_01.json
```

Vergleiche sind rein trainingsbezogen und **keine Auswahl des prädiktiv besten Modells**.
