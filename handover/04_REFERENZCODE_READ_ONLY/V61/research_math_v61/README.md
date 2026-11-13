# V6.1 — Mathematisches Audit des globalen Bayesian-Prozess-Posterior-Updates

**Forschungszweig, keine finale Modellfreigabe.** Nutzt V6.0, V5.9, V5.8 und die **SHA-gesperrten unveränderten** zehn V5.2-Mathematik-Kernmodule. Es werden nur die zwölf festgelegten Entwicklungsringe (acht gelabelte, vier Qualitätsziele maskiert) verwendet; DM07/08/11/15 werden nicht ausgewertet. Vorliegende Ergebnisse werden nicht überschrieben.

## Fragestellung

Woher kommt der Prozess-ELBO-Sprung in Epoche 11? Für **unverändert gehaltene** lokale Markov-Faktoren und Qualitätsköpfe berechnet die Diagnostik die Änderung nach dem geschlossenen globalen Prozess-Posterior-Update, einschließlich einer unabhängigen additiven Bilanz:

- `emission_NW = Δ(l_x + C*u_x - KL_NormalWishart)`
- `transition_Dirichlet = Δ(l_transition + C*u_transition - KL_transition)`
- `initial_Dirichlet = Δ(l_initial + C*u_initial - KL_initial)`

Deren Summe muss dem gemessenen gesamten Prozess-ELBO-Gewinn entsprechen. Die unveränderten Qualitäts-/Entropiebeiträge müssen einzeln exakt konstant bleiben. Die gewichteten Zustands-/Übergangs-/Startzustandszahlen werden unabhängig berechnet und mit der Normal-Wishart- bzw. Dirichlet-Posteriorformel verglichen.

**Zusätzliche Diagnostik:** gewichtete Besetzungszahlen beider Zustände, Zustandsanteile je Ring, Änderung der globalen Mittelwerte (nur Normen), prädiktive Emissionskovarianz-Eigenwerte, erwartete Übergangswahrscheinlichkeiten und Emissionsbeiträge nach Ring. Kein Qualitäts-RMSE, keine neue Vorhersage, keine Speicherausgabe eines vollständig einsetzbaren Modells.

## Installation

Nur `research_math_v61/` direkt nach `Implementierung_BA/` entpacken. Bereits vorhandene `final_candidate_v52/`, `research_convergence_v55/`, `research_math_v58/`, `research_math_v59/`, `research_math_v60/` unverändert lassen.

```powershell
.\.venv\Scripts\python.exe .\research_math_v61\process_posterior_audit_v61.py self-test
New-Item -ItemType Directory -Force results\v61_process_audit | Out-Null
.\.venv\Scripts\python.exe .\research_math_v61\process_posterior_audit_v61.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --reference results\v60_math_m2\V60_INDUCED_PRIOR_M2_30E_01.json --epochs 30 --out results\v61_process_audit\V61_PROCESS_POSTERIOR_30E_01.json
```

**Wichtig:** Die Referenzdatei `V60_INDUCED_PRIOR_M2_30E_01.json` wurde bereits vom Nutzer auf Windows erstellt; für diesen Vergleich wird sie nur gelesen. Die Ausgabe-Zieldatei muss **neu** sein. Keine existierende Datei löschen oder überschreiben.

## Lokal geprüfte Ergebniszahlen (kein unabhängiger äußerer Test)

Das lokale Originaldaten-Replay mit dem registrierten `physical_var=0.38526108747593474`, Seed 20261111, K2, C0.2 und 30 Epochen reproduzierte V6.0 bis auf max `8.948e-8` ELBO-Differenz. In Epoche 11: globaler Prozess-Update-Gewinn `1146.3054`, aufgeteilt in `1143.3443` Emission+Normal-Wishart, `2.9612` Übergangs-Dirichlet, `~0` Startzustands-Dirichlet. Diese Zahlen sind **Gewinne während des globalen Updates bei festgehaltenem q(z)**, nicht die gesamte Epochendifferenz (`1322.4469`). Die anderen `~176.1346` stammen aus dem lokalen Koordinatenschritt; Qualitätsköpfe `~0.0060`.

Von Epoche 10 bis 11 verschieben sich die gewichteten Zustandsbesetzungen etwa von `[4542.055,1800.345]` auf `[4456.056,1886.344]` (Gesamtsumme 6342.4). Die Norm der Änderung der Emissions-Posterior-Mittelwerte ist im Update von Epoche 11 `0.1596` (gegenüber `0.02846` in Epoche 10). Die Spur ist ein Hinweis auf eine starke Neuordnung des Emissionsmodells, kein Beweis ihrer physikalischen Plausibilität oder Generalisierung.

Die Gamma-Zustandswahrscheinlichkeiten und die globalen Prozessparameter sind gegenseitig gekoppelt; aus der isolierten Posterior-Update-Bilanz folgt **keine einzelne kausale Auslösung des Epochenwechsels**. Darüber hinaus ist `C=0.2` eine gewichtete VI-Zielfunktion, keine gewöhnliche ungewichtete Log-Evidenz. Nicht als unabhängige Testperformance ausgeben.

## Weiterführender Mathematikcode

Nach dem Audit: (1) prüfen, ob die hohe Persistenz der Übergänge (~0.995) und die ringweise fast konstanten Zustandszuweisungen die intendierte Prozessphasenmodellierung widerspiegeln; (2) Emissionsresiduen/autokorrelationsbedingte Unabhängigkeit prüfen; (3) getrennte HMM-/Bayes-/SSL-Ablationen und ringweise CV. Noch **kein** finaler Model-Freeze.
