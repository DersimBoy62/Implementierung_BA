# Bayesian Bachelor V6.0: M2-Redundanz **mit induziertem Bayesian-Prior**

**Forschungszweig vom 12.11.2026 – NICHT freigegebenes Finalmodell.** Dieses Paket verändert weder die zehn mathematischen Originaldateien von V5.2 noch die Forschungszweige V5.8/V5.9. Es liefert einen eigenständigen ausführbaren Code-Audit mit einer exakt äquivalenten Reparametrisierung der M2-Regressionsköpfe. Ziel: Mathematik im Code prüfen, nicht Hyperparameter tunen. Keine historische Test-Ringauswertung und keine gespeicherten Modellgewichte.

## Wichtige mathematische Präzisierung

Die originalen M2-Merkmale sind `g=(h_1,...,h_K, Delta_1,...,Delta_K)`. Dabei gilt für jede Zustandsfolge

`sum_k h_k = 1` und `sum_k Delta_k = 0`.

**Nur die Differenzmerkmale weisen ohne separaten Intercept eine entbehrliche lineare Spalte auf.** Die `K` Belegungsmerkmale dürfen nicht ohne Ersatz eines Intercepts auf `K-1` verkürzt werden: Ihre Summe repräsentiert den konstanten Term. Die exakte reduzierte Darstellung lautet daher

`g_red=(h_1,...,h_K, Delta_1,...,Delta_(K-1))`.

Für `K=2`: Vier originale Zustandsmerkmale werden **drei** (nicht zwei). Bei den konkreten Regressionsköpfen verringert sich die Dimension von **10 -> 9** (Außenunrundheit) und von **9 -> 8** (Höhenschlag).

Sei `r` der reduzierte komplette Merkmalsvektor einschließlich unveränderter physikalischer Kovariaten und `f = A @ r` der alte Merkmalsvektor. In `A` wird die letzte Differenz durch `Delta_K=-sum(Delta_1:Delta_(K-1))` rekonstruiert. Bei originalem NIG-Prior

`beta_full | sigma^2 ~ N(mu_full, sigma^2 V_full)`, `sigma^2 ~ IG(a,b)`

lautet der **induzierte** Prior für die wirksamen Regressionskoeffizienten

`beta_red=A.T @ beta_full`,

`mu_red=A.T @ mu_full`, `V_red=A.T @ V_full @ A`, `a_red=a`, `b_red=b`.

Für den tatsächlich verwendeten Originalprior mit unabhängigen temporalen Koeffizienten und Varianz `0.15` ist die Varianz jeder reduzierten Differenzkoeffizienten **0.30**. Bei `K>2` entsteht darüber hinaus eine **nichtdiagonale** Kovarianz von `0.15` zwischen den reduzierten Differenzkoeffizienten. Einfach eine Spalte zu löschen und den Prior unverändert zu lassen wäre **keine** exakt äquivalente bayesianische Modelländerung.

## Ausführliche Code-Prüfungen

`m2_prior_reduction_v60.py self-test` prüft:

1. Vollständige SHA256-Identität aller zehn V5.2-Mathematikmodule und des V5.8-/V5.9-Quellcodes (dynamische Verifikation über `checked_v58`).
2. Für `K=2,3,4` und verschiedene physikalische Feature-Dimensionen die exakte algebraische und stochastische Identität der ersten und zweiten Zustandsmerkmalsmomente einschließlich `Delta_K`.
3. Induzierten NIG-Prior `A.T V A` und positiven Definitheitsnachweis.
4. Exakte Übereinstimmung der voll- und reduziert-parametrisierten *konjugierten* Regressions-Posteriorparameter (`mu`, `V`, `a`, `b`).
5. Gleichheit der KL-Divergenz **nach** dem konjugierten Posteriorupdate, der erwarteten Qualitätsloglikelihood und der prädiktiven Mittelwerte/Varianzen.
6. Gleichheit der vollständigen lokalen VI-Zielfunktion und des analytischen vektoriellen Markov-Gradienten nach Pullback des Qualitätsbeitrags.
7. Drei synthetische koordinatenweise Update-Epochen ohne berichtete ELBO-Blockverschlechterung.

Die Überprüfung ist mathematisch wie numerisch bewusst **nicht** die Behauptung, dass für beliebige nicht-konjugate oder frei gewählte `q(beta_full)` der volle und reduzierte KL-Term immer gleich seien. Die KL-Identität gilt hier für den tatsächlich verwendeten konjugierten Posterior, dessen bedingte Verteilung über redundante Koeffizienten die Priorbedingte bleibt.

## Voraussetzungen und Einbau unter Windows

Entpacke **nur** den Ordner `research_math_v60` direkt neben den bereits vorhandenen Ordnern:

- `final_candidate_v52/` mit **unveränderten** V5.2-Kernen;
- `research_math_v58/math_train_audit_v58.py`;
- `research_math_v59/reduced_markov_logits_v59.py`;
- `research_convergence_v55/convergence_probe_v55.py`;
- `.venv/` sowie `input/DM_16_aktive_Ringe.pkl`.

### Erst synthetische Mathematik prüfen (keine DM-Daten)

```powershell
.\.venv\Scripts\python.exe .\research_math_v60\m2_prior_reduction_v60.py self-test
```

Erwartete Schlusszeile: `V60 SELF-TEST PASS: ...; NO DM/TEST DATA`

### Dann reale Entwicklungsdaten (8 L + 4 U), 30 Epochen, kein äußerer Score

```powershell
New-Item -ItemType Directory -Force results\v60_math_m2 | Out-Null
.\.venv\Scripts\python.exe .\research_math_v60\m2_prior_reduction_v60.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --reference results\v59_math_reduced\V59_REDUCED_M2_WARMSTART_30E_01.json --epochs 30 --out results\v60_math_m2\V60_INDUCED_PRIOR_M2_30E_01.json
```

Die Ausgabe ist **write-once**. Existiert die Ziel-JSON schon, wähle einen **neuen Namen**, statt sie zu löschen oder zu überschreiben. `--reference` ist optional, aber für den numerischen Vergleich zu V5.9 empfehlenswert. `--model-dir` ist optional für den lokalen Forschungslauf, sollte auf dem Nutzer-PC aber gesetzt bleiben, um die originale V5.2-Bundle-Identität zu überprüfen. Alle Trainings- und physikalischen Skalierer werden wie in V5.5 aus den **Entwicklungsdaten** neu erstellt; die äußeren Test-Ring-IDs werden nicht als Trainingsdaten verwendet.

## Bereits in der Arbeitsumgebung unabhängig durchgeführte Tests

- Synthetischer Selbsttest erfolgreich inklusive K2/K3/K4, NIG-Posterior und KL, prädiktiven Momenten sowie lokalem Gradient.
- Echte 3-Epochen-Probe (8 L + 4 U): alle Blockbilanzen nichtnegativ, ELBO zur V5.9-Basis numerisch gleich.
- Echte 30-Epochen-Probe (8 L + 4 U): letzte ELBO `-22028.914444055594`; 0 Blockverletzungen mit Toleranz `-1e-6`; finale ELBO-Abweichung von lokalem V5.9-Vergleich `-4.53e-9`; größte Abweichung über 30 Epochen `8.54e-8`. (Vergleich nur der Trainingszielfunktion, keine Vorhersage.)

Die Reparametrisierung zeigt damit numerische Gleichwertigkeit. **Das ist kein Gewinn an Vorhersagequalität und kein Beweis für ein globales Optimum.** Unsere bisherige Modellarchitektur bleibt weiter ein Entwicklungsmodell.

## Als Nächstes im Forschungsvorhaben

Keine weitere reine Hyperparameteroptimierung. Stattdessen mathematischer Audit des überraschenden globalen Prozess-Posterior-Sprungs nach Epoche 10: Zustandsposterior, Normal-Wishart-Parameter, Emissionsbeiträge und Nachvollziehbarkeit der Updategleichungen. Danach die bedingte Unabhängigkeitsannahme der Emissionen, HMM-Abhängigkeit und die Bayes-/SSL-Argumentation untersuchen.
