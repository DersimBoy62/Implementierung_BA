# V6.3 – Wissenschaftlicher Diagnosezweig: Phasengrenzen und Emissionszeitstruktur

**Kein neues Modell, kein Hyperparameter-Tuning, keine Testauswertung.** V6.3 reproduziert den unveränderten 30-Epochen-Trainingslauf und instrumentiert die ringweisen Markov-Marginalen der *letzten* Epoche. Die zehn ursprünglichen mathematischen V5.2-Kernmodule und V5.8–V6.2 werden **nicht verändert**. Der SHA von V6.2 und die vollständige Abhängigkeitskette werden im Selbsttest geprüft.

## Installation

Nur `research_math_v63` in `Implementierung_BA` kopieren. Die bisherigen Verzeichnisse `final_candidate_v52`, `research_convergence_v55`, `research_math_v58` bis `research_math_v62` bleiben an ihrem Ort. Die Originalmodell-Dateien werden weder überschrieben noch als Initialgewichte übernommen; das Training wird von den registrierten Startparametern reproduziert.

## Befehle in PowerShell im Projektordner

```powershell
.\.venv\Scripts\python.exe .\research_math_v63\phase_emission_audit_v63.py self-test
New-Item -ItemType Directory -Force results\v63_phase_emissions | Out-Null
.\.venv\Scripts\python.exe .\research_math_v63\phase_emission_audit_v63.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --reference-v61 results\v61_process_audit\V61_PROCESS_POSTERIOR_30E_01.json --reference-v62 results\v62_hmm_dynamics\V62_HMM_ZEITDYNAMIK_30E_01.json --out results\v63_phase_emissions\V63_PHASE_TIME_RESIDUAL_30E_01.json
```

Das Skript verlangt einen **neuen** JSON-Ausgabepfad; alte Ergebnisse bleiben erhalten. Das bekannte vertrauenswürdige, per SHA verifizierte DM-Pickle wird gelesen, um die Logger-Zeitachse und `phase804` zu entnehmen. Die Qualitätswerte der historischen äußeren Testringe werden **nicht für Schätzung, Tuning oder Auswertung verwendet**. Die zugelassene Entwicklungsansicht enthält weiterhin acht gelabelte und vier maskierte ungelabelte Ringe. Es wird kein Modell gespeichert und kein Qualitäts-RMSE berechnet.

## Definition der Kennzahlen

- Die echte Zeitachse wird aus dem **identischen F2-Fenster** wie beim Training rekonstruiert und mit der Trainingsansicht exakt abgeglichen. Die 12 Entwicklungsringe besitzen jeweils eine zusammenhängende 0,1-s-Zeitachse und **eine** beobachtete `phase804`-Änderung 2→1.
- Posterior-erwartete Wechsel pro Zeitintervall: `sum_{i != j} xi[t,i,j]` der **unveränderten** Markov-Variationsverteilung; summiert über Zeit ergeben sie erwartete Zustandswechsel, nicht harte Pfadwechsel.
- Nähe zu einer *beobachteten Phasengrenze*: ±2,0 s (±20 Zeitkanten) um die jeweilige 2→1-Kante. Der Anteil der erwarteten Wechselmasse dort wird mit einer rein **deskriptiven** Referenz verglichen: dieselbe Wechselmasse wäre entlang der einzelnen Zeitkanten eines Rings gleichmäßig verteilt. Unterschiedliche Ringdauern und randnahe Phasengrenzen werden berücksichtigt.
- Harte Wechsel = Wechsel von `argmax(gamma_t)`, **nicht Viterbi**. Distanz zur Phasengrenze in Sekunden.
- Beobachtetes Phasenlabel und HMM-Latentzustand sind nicht dieselbe Definition. Ein hoher Zusammenhang beweist keine kausale/physikalische Zustandsinterpretation. Der Vergleich ist **keine Signifikanzprüfung**.
- Zeitliche Abhängigkeit: lag-1-Autokorrelation der posterior-gemittelten Emissionsresiduen `r_t = x_t - sum_k gamma_tk mu_k`, danach getrennt innerhalb der zwei beobachteten Phasen nach Abzug je eines linearen Trends. Zusätzlich ACF der **ersten Differenzen**. Die Residuen hängen über `gamma` selbst von `x` ab; **kein formaler HMM-Goodness-of-fit-Test**.
- Prüfkriterien: ELBO-Verlauf muss V6.1 bis `1e-5` entsprechen und die erwarteten ringweisen Wechsel müssen V6.2 bis `1e-5` reproduzieren. Ausgabedatei wird nur nach bestandenem Test neu geschrieben.

## Lokale, vorläufige Entwicklungsdiagnostik (noch nicht Windows-verifiziert)

Aus dem verfügbaren, identischen 12-Ring-Entwicklungsbestand und dem bisherigen V6.1-Prozesskern wurde lokal gemessen:

- Alle 12 Ringe haben im F2-Fenster genau eine 2→1-Loggerphasengrenze.
- Posterior-erwartete Wechsel ungewichtet: 28,5451. In ±2 s um die Phasengrenzen: 3,3296 (11,66 %).
- Ringweise gleichmäßige Referenz: 1,6805 erwartete Wechsel im selben Fenster, also beobachtete Wechselmasse etwa **1,98-fach**. Diese Referenz kontrolliert **nicht** zeitliche Trends, ringweise Heterogenität oder multiple Hypothesentests.
- Nur 3 der 12 Ringe besitzen einen harten argmax-gamma-Wechsel innerhalb von ±2 s der Phasengrenze.
- Ringmedian der kanalweisen mittleren absoluten Lag-1-Residualkorrelation: 0,9961; nach phasenweiser linearer Trendbereinigung: 0,9866; auf den Residualdifferenzen: 0,2543.
- Lokal maximaler V6.1-ELBO-Unterschied ~`2.8e-8`; erwartete Ringwechsel stimmen mit V6.2 bis `1e-11` überein.

## Weiterer Forschungsbedarf

1. Prüfen, wie die HMM-Wechsel zu den tatsächlichen Walzgrößen und Ereignissen (Kraft-/Geometrieänderungen) passen – nicht bloß zu `phase804`.
2. Die Emissionszeitabhängigkeit modellgerecht untersuchen, z. B. mit phasen-/zustandsbezogenen Vorhersageresiduen und **hold-out Prozesslikelihood** auf ausschließlich Entwicklungsringen.
3. Erst dann eine kontrollierte state-abhängige AR-Emissionsalternative herleiten (`p(x_t|x_{t-1},z_t)`), inklusive Priors, Anfangsschritt und mathematisch vergleichbarer Zielfunktion. Keine automatische Übernahme einer komplexeren Architektur ohne kontrollierte Ablation und ringweise Validierung.
4. Die eigenständigen Vorteile von HMM, Bayes und SSL und die Generalisierungsgrenzen bleiben offen.
