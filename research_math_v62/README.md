# V6.2 – HMM-Zeitdynamik und bedingte Emissionsannahmen

**Forschungsdiagnose, kein neues finales Modell.** Dieser Zweig instrumentiert die Originalrechnung von V6.1 *im Arbeitsspeicher* und verändert keine Dateien in `final_candidate_v52`, `research_math_v58`, `research_math_v59`, `research_math_v60` oder `research_math_v61`. Es werden keine Qualitaetsvorhersagen bewertet, keine alten aeusseren Testringe verwendet und keine Modellgewichte gespeichert. Erforderliche Modul-SHA256-Pruefungen bleiben bestehen.

## Installation

Nur `research_math_v62` aus diesem ZIP direkt in den Ordner `Implementierung_BA` kopieren. Die Ordner `final_candidate_v52`, `research_convergence_v55`, `research_math_v58`, `research_math_v59`, `research_math_v60`, `research_math_v61` muessen daneben liegen. Alte Ordner nicht ueberschreiben.

## Befehle (PowerShell im Projektverzeichnis)

```powershell
.\.venv\Scripts\python.exe .\research_math_v62\hmm_dynamics_audit_v62.py self-test
New-Item -ItemType Directory -Force results\v62_hmm_dynamics | Out-Null
.\.venv\Scripts\python.exe .\research_math_v62\hmm_dynamics_audit_v62.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --reference results\v61_process_audit\V61_PROCESS_POSTERIOR_30E_01.json --epochs 30 --out results\v62_hmm_dynamics\V62_HMM_ZEITDYNAMIK_30E_01.json
```

`--out` muss **neu** und der Zielordner bereits vorhanden sein; bestehende Dateien werden nicht ueberschrieben. Der 30-Epochen-Lauf erfolgt erneut von denselben initialen Annahmen und dient der reinen Instrumentierung. Die gesamten 30 ELBO-Werte werden mit der V6.1-Referenz (max. erlaubte Abweichung 1e-5) verglichen.

## Was wird direkt aus q(z) gemessen?

- Posterior-erwartete Wechselanzahl: Summe aller Offdiagonal-Elemente aus \(\xi_t(i,j)\). Diese Zahl ist nicht identisch mit der Anzahl harter Wechsel.
- Harte *argmax-gamma*-Wechsel; **keine Viterbi-Dekodierung**.
- Ringweise mittlere Zustandsbesetzungen, Unsicherheit, ungefaehre Verteilung am fruehen/mittleren/spaeten Ringzeitpunkt.
- U-Ring-Beitraege werden nur in der *gewichteten* Gesamtsumme mit \(C=0.2\) gewichtet. Ungewichtete ringweise Wechsel bleiben separat erhalten.
- Interne Konsistenztests \(\sum_j \xi_t(i,j)=\gamma_t(i)\), \(\sum_i \xi_t(i,j)=\gamma_{t+1}(j)\); Offdiagonal-Summen gegen die Dirichlet-Suffizienzstatistik.
- Posterior-Mittelwerte der Uebergangsmatrix, theoretische mittlere Verweildauer in **Logger-Zeitschritten** unter geometrischer Verweildauerannahme, nicht in Sekunden.
- Explorative Lag-1-Autokorrelation von Loggerkanaelen und Residuen \(x_t-\sum_k\gamma_{t,k}\mu_k\). Da \(\gamma\) von \(x\) abhaengt, ist dies **kein formaler Test** bedingter Emissionsunabhaengigkeit.

## Bereits lokal geprüfte Ausgangslage

Die Datei `V62_VORLAEUFIGE_LOKALE_ENTWICKLUNGSDIAGNOSTIK.json` ist eine **lokale, nicht auf dem Nutzerrechner verifizierte** 30-Epochen-Probe, berechnet mit den ausschliesslich freigegebenen Entwicklungsringen und denselben mathematischen Kernmodulen. Sie soll die Windows-Pruefung nicht ersetzen.

In der lokalen Probe nach Epoche 30: 4/12 Ringe ohne argmax-gamma-Wechsel, 8/12 mit Wechseln, zusammen 28.545 erwartete Wechsel (ungewichtet), 27.745 mit C-gewichteten U-Ringen; 4/12 Ringe mit >95% Besetzung eines Zustands. Modell-Zustandswechsel sind **nicht automatisch** physikalische Walzphasen. Die Lag-1-Residualautokorrelation ist sehr hoch; das erfordert weitergehende Diagnose von Trends, Schrittweite und expliziter zeitlicher Fehlerstruktur.

## Forschungsgrenzen und nächste mathematische Schritte

1. Beobachtete Zustandswechsel gegen bekannte Walzphasen, Prozessmerkmale und Wechselpunkte abgleichen; zeitliche Aufloesung pruefen.
2. Bedingte Emissionsunabhaengigkeit und logarithmische Evidenz kritisch untersuchen. Sehr eng abgetastete, glatte Sensorverlaeufe koennen die angenommene effektive Unabhaengigkeit stark verletzen.
3. Fairer mathematisch gekoppelter Vergleich gegen HMM-freie Regression, IID-Mischmodell und ggf. autoregressives HMM unter denselben Entwicklungs-Splits und Priors. Vorhersageleistung separat bewerten.
4. Keine Schlussfolgerung zu einem unabhaengigen externen Test; historische aeussere Qualitaetslabels wurden frueher bereits gesehen.

**Hinweis:** Kein Model-Freeze und keine Hyperparameteroptimierung durch diesen Diagnosezweig.
