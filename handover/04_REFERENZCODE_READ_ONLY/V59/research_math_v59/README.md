# Bayesian Bachelor V5.9 – mathematisch äquivalente Markov-Parametrisierung

**Status: Forschungszweig, KEIN endgültiges Modell, 12.11.2026.** Keine historische äußere Testbewertung, kein Speichern neuer Modellgewichte. Die zehn ursprünglichen V5.2-Mathematikmodule sowie das V5.8-Trainingsmodul bleiben unverändert und sind per SHA256 geprüft.

## Was ist mathematisch verändert?

Die Variationsfamilie ist weiterhin die frei zeitabhängige Markov-Verteilung

`q(z_0:T-1) = q0[z0] * product_t Q[t-1,z[t-1],z[t]]`.

Bisher: Für jede Verteilung mit K Kategorien wurden K Softmax-Logits optimiert, obwohl die Addition derselben Konstanten zu allen Logits die Wahrscheinlichkeiten unverändert lässt. Daher enthält der ursprüngliche Raum je Zeile eine flache, nicht-identifizierbare Richtung.

Neu: Der **letzte Logit jeder Zeile wird auf null fixiert**. Die anderen K-1 Werte sind freie Log-Odds `r_j = log(p_j/p_K)`; daraus folgen `p_j = exp(r_j)/(1+sum_i exp(r_i))` und `p_K = 1/(1+sum_i exp(r_i))`. Die Implementierung arbeitet mit stabiler Softmax im eingefrorenen V5.2-Kern. Die Zahl der freien Parameter beträgt

- vorher `K + (T-1)*K*K`;
- jetzt `(K-1)*(1+(T-1)*K)`.

Für `K=2` ist das exakt halb so viel, z. B. bei DM12 von 5.170 auf 2.585.

Der unveränderte V5.2-Kern berechnet dieselbe Zielfunktion und denselben vollständigen analytischen Gradient. Die Ableitung nach den reduzierten Koordinaten ist durch die Kettenregel exakt die erste `K-1`-Komponente jeder Logit-Zeile (nicht eine Differenz der Gradienteneinträge). `full_to_reduced` dient nur zur Wahl einer eindeutigen Repräsentation derselben `q`-Verteilung.

**Wichtig:** Der *mathematische Funktionsraum* bleibt gleich, aber L-BFGS-B ist unter einer Koordinatentransformation nicht invariant. Bei endlicher Iterationszahl entstehen kleine Unterschiede der optimierten Trajektorien. Deshalb ist nicht jede Epochen-ELBO identisch.

## Voraussetzungen

Den Ordner `research_math_v59` direkt in `Implementierung_BA` entpacken. Bereits vorhanden sein müssen:

- `final_candidate_v52/` mit unverändertem V5.2-Kern
- `research_math_v58/math_train_audit_v58.py` **unverändert**
- `research_convergence_v55/convergence_probe_v55.py`
- `.venv`, `input/DM_16_aktive_Ringe.pkl`, `results/v54_development_replay/MODEL_8L4U_RUN01`

Das V5.9-Programm verweigert den Lauf, falls der per SHA256 fixierte V5.8-Audit oder eine der zehn originalen mathematischen Kern-Dateien verändert wurden. Modell/Datensatz/Config werden im Real-Lauf durch das V5.5-Preflight geprüft. Die Ausgabe ist *write-once*; einen vorhandenen Ergebnisdateinamen nie überschreiben.

## Selbsttest ohne echte DM-Daten

```powershell
.\.venv\Scripts\python.exe .\research_math_v59\reduced_markov_logits_v59.py self-test
```

Enthalten: K=2/3 und M0/M2/M3; Gleichheit der Verteilungen für beliebige Softmax-Gauge-Verschiebungen; exakte Gleichheit von Zielwert und ersten/zweiten Zustandsmerkmalmomenten; analytische Kettenregel vs. eingefrorene vollständige Gradienten; unabhängige zentrale finite Differenzen; L-BFGS-B-Machbarkeit; drei synthetische koordinatenweise Trainings-Epochen mit Block-ELBO-Bilanz. Keine DM-Daten, kein Testscore.

## Reallauf: Vergleich V5.9 zu V5.8 ohne Testauswertung

```powershell
New-Item -ItemType Directory -Force results\v59_math_reduced | Out-Null
.\.venv\Scripts\python.exe .\research_math_v59\reduced_markov_logits_v59.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --epochs 30 --reference results\v58_math_audit\V58_M2_WARMSTART_30E_01.json --out results\v59_math_reduced\V59_REDUCED_M2_WARMSTART_30E_01.json
```

*Die Referenz-JSON ist der V5.8-Warmstart-30-Epochen-Lauf. Der Start überprüft Datumshash und Epochenzahl. Wenn diese Datei anders heißt, gib den tatsächlichen Pfad an. Die Referenz kann ganz weggelassen werden; dann wird kein Kurvenabgleich erzeugt.*

## Bereits unabhängig im aktuellen Arbeitscontainer durchgeführter Realvergleich

12 Entwicklungsringe, davon 8 L und 4 U (historische äußere Test-Ring-IDs DM07/08/11/15 ausgeschlossen), K=2, C=0.2, M2, Seed 20261111, max. 8 lokale Iterationen, 30 Epochen, physical_var=0.38526108747593474:

- Original V5.8 abgesicherter Warmstart: finale Trainings-ELBO `-22028.91444410429`.
- V5.9 reduzierte Logits: finale Trainings-ELBO `-22028.914444051064`.
- Finale Differenz (V5.9 minus V5.8): `+5.3227e-8` (praktisch null).
- Über alle 30 Epochen betrug die maximale absolute ELBO-Abweichung `0.0074799065` (beachte lokale Optimierungsgeometrie).
- Keine Block-ELBO-Verschlechterung unter `-1e-6` im 30-Epochen-Lauf.
- Summe lokal frei optimierter Logits aller 8 L-Ringe je Epoche: `24256 -> 12128`.

**Keine Interpretation der kleinen positiven End-ELBO-Differenz als Vorhersageverbesserung oder Beweis für schnellere Optimierung.** Vor Freigabe einer finalen Modellvariante sollten Laufzeit, konvergierte lokale Gradienten und Validierungsmetriken separat geprüft werden. Das Paket bietet Mathematik und Forschungs-Code, keine automatisch übernommene Release-Änderung.

## Nächste Forschung

Prüfen der **M2-Merkmalsredundanz und ihrer Prior-Transformation** (bei zwei Zuständen summiert die zeitliche Besetzungsdifferenz auf null), außerdem eingehende Emissions-Posterior-/ELBO-Änderungen beim Sprung in Epoche 11. Eine Reduktion der M2-Merkmale ohne induzierte Prior-Transformation wäre eine Änderung des statistischen Modells und daher keine reine Vereinfachung.
