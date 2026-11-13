# Bayesian Bachelor – V5.8: mathematische Trainingsprüfung, Priorität 1

**Stand: 12.11.2026. Dies ist ein separater Forschungszweig, kein neues freigegebenes Endmodell.**

## Warum V5.8?

Das bestehende `final_candidate_v52/final_core/dynamic_quality_v2.py` beginnt die lokale Variationsoptimierung der gelabelten Ringe **bei jeder Epoche** mit der prozessbasierten Forward-Backward-Lösung. Das ist nicht an sich falsch. Für einen formalen *koordinatenweisen* Verbesserungsnachweis darf aber eine Lösung `q_{e-1}(z)` bei neuen globalen Faktoren nicht ignoriert werden, ohne zu prüfen, ob der neue E-Schritt sie mindestens erreicht.

V5.8 führt daher **ohne Änderung der zehn originalen mathematischen Core-Dateien** einen Forschungsalgorithmus ein:

1. Für jeden L-Ring die prozessbasierte Markov-Verteilung `q_FB` bestimmen.
2. Den bisher gespeicherten Markov-Faktor `q_alt` ebenfalls unter **denselben aktuellen** globalen Parametern `q(theta)` und Qualitätsköpfen bewerten.
3. Die lokale Zielfunktion am besseren der beiden Startpunkte optimieren; die Lösung nur bei tatsächlicher Verbesserung übernehmen.
4. Die vorige lokale Verteilung *pro Ring* für die nächste Epoche speichern. Für U-Ringe bleibt die unveränderte exakte prozessbasierte Forward-Backward-Inferenz.
5. Die **vollständige gewichtete ELBO** in drei festgehaltenen Zuständen berechnen: nach dem lokalen Schritt (`after_local`), nach dem globalen Prozess-Posterior-Update (`after_process`) und nach den beiden Qualitätskopf-Updates (`elbo`). Jede Blockdifferenz muss innerhalb numerischer Rundung nichtnegativ sein.
6. Die Bestandteile `L_pi`, `L_A`, `L_X`, `L_entropy`, `C*U_*`, `Y1`, `Y2`, `KL_pi`, `KL_A`, `KL_NW`, `KL_heads` separat ausgeben. Die globale KL-Zerlegung wird unabhängig gegen `two_head_b1.kl_all` geprüft.

**Die zusätzlichen Checks testen eine Epochenfolge in einem festgelegten numerischen Modell; sie sind kein allgemeiner mathematischer Beweis einer global optimalen Lösung.**

## Voraussetzung

Entpacke `research_math_v58` direkt nach `Implementierung_BA`. Dort müssen weiterhin vorhanden sein:

- `final_candidate_v52/final_core/` samt unverändertem ursprünglichem V5.2-Core und zugehöriger Release-Konfiguration
- `research_convergence_v55/convergence_probe_v55.py`
- `input/DM_16_aktive_Ringe.pkl` (SHA256-Prüfung im Lauf)
- `results/v54_development_replay/MODEL_8L4U_RUN01/` (nur zur Integritätsbestätigung des bekannten Referenzstands)
- die gewohnte Windows-`.venv`-Umgebung

Die originale Modellkonfiguration und alle zehn mathematischen Kernmodule werden mittels SHA256 gegen die geprüften V5.2-Dateien verifiziert; sonst wird der Vorgang verweigert.

## 1. Synthetischer Test ohne Ringdaten

```powershell
.\.venv\Scripts\python.exe .\research_math_v58\math_train_audit_v58.py self-test
```

Dieser Test enthält: exakte Kern-Hashes, alternativen Markov-Start, M2-Momente, eine **Identität der lokalen Zielfunktionsdifferenz mit der vollständigen ELBO-Differenz bei festgehaltenen globalen Faktoren**, vier synthetische Epochen mit drei Ringsequenzen und getrennte ELBO-Kontrollen nach jedem Block.

## 2. Realer mathematischer Forschungsversuch (keine Testbewertung)

```powershell
New-Item -ItemType Directory -Force results\v58_math_audit | Out-Null
.\.venv\Scripts\python.exe .\research_math_v58\math_train_audit_v58.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --epochs 30 --mode warmstart --out results\v58_math_audit\V58_M2_WARMSTART_30E_01.json
```

Der erste `run`-Aufruf mit dem Zielnamen darf ihn nur **neu** erzeugen; falls die Datei schon existiert, einen anderen Namen verwenden. Ergebnisdateien werden nie überschrieben. Die bestehende V5.2-Version oder das gespeicherte Modell werden nicht verändert. Das Script speichert **keine Gewichte**, keine Qualitätsvorhersagen, keinen Test-RMSE.

Optional kann `--mode original_restart` dieselbe ursprüngliche E-Schritt-Startregel mit genau derselben neuen ELBO-Bilanz instrumentieren; Originaldateien bleiben unverändert. Ergebnisse dieser beiden Modi dürfen nicht als unabhängige Generalisierung interpretiert werden.

## Bereits lokal unabhängig reproduzierte Realergebnisse (Entwicklungsdaten, nur Training)

Auf denselben 12 Entwicklungsringen (8 L + 4 U) wurden beide Algorithmen auf dem hier vorhandenen Datensatz geprüft (30 Epochen, K2/C0.2/M2, Seed 20261111, `steps=8`, `physical_var=0.38526108747593474`).

- `original_restart`: ELBO nach Epoche 30 `-22028.914445293016`; max. Abweichung von gespeicherter V5.5-ELBO-Zeitreihe `8.4114e-8`; **keine Block-Abnahmen größer als `1e-6`**.
- `warmstart`: ELBO nach Epoche 30 `-22028.91444410429`; unter 240 L-Ring-Epochen wurde 108-mal der vorherige Faktor als besserer Start gewählt; **keine Block-Abnahmen größer als `1e-6`**.
- Maximale ELBO-Differenz der gesamten 30-Epochen-Kurven zwischen beiden Varianten `1.2612e-6`; weder deutliche Verbesserung noch Verschlechterung.
- Im alten `original_restart`-Ablauf steigt die ELBO während **Epoche 11** um `1322.44690974537`; davon ungefähr `176.1358774` im lokalen Schritt, `1146.3050167` im globalen Prozess-Update und nur `0.0060156` in den Qualitätsköpfen.
- Die ELBO-Komponente `L_X` wächst von Epoche 10 auf 11 um `+1362.752616`; die übrigen Komponenten gleichen sich zum Gesamtgewinn aus. Damit liegt der Hauptbeitrag im **Prozessmodell**, nicht in einer plötzlichen zusätzlichen Optimierung der Qualitätsköpfe.
- Diese Ergebnisse charakterisieren den gegebenen Trainingslauf; sie beweisen **keine** Überlegenheit des Warmstarts, keine globale Optimalität, keine bessere Vorhersagequalität und keine unabhängige Generalisierung.

Die vollständigen Epochenbilanzen befinden sich in `V58_Beide_Methoden_30E_Entwicklungsdaten.json`.

## Als Nächstes

Vor einer Implementationsentscheidung müssen wir (a) die tatsächliche Quelle des Prozess-ELBO-Sprungs tiefer prüfen: Zustandsmassen, globale Normal-Wishart-Parameter und Emissions-/Übergangsterme; (b) erst dann die redundante Softmax-Parametrisierung als **mathematisch äquivalente** Alternative implementieren und durch Gradienten/ELBO-Invarianztests bestätigen; (c) die M2-Redundanz und passende Prior-Transformation untersuchen. Das alte V5.2-Modell ist unverändert, eine Modellfreigabe wurde nicht vorgenommen.
