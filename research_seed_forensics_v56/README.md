# V5.6 Seed-Forensik (read-only) – Bayesian Bachelor

## Wissenschaftliche Frage

Seed 20261112 ergab bei 30 Epochen und acht lokalen VI-Schritten praktisch dieselbe ELBO wie Seed 20261111. Um zwischen **tatsächlich verschiedenen Starts, die zum selben lokalen Optimum laufen**, und **identischen K-Means-Initialisierungen** zu unterscheiden, wird die exakt in `dynamic_quality_v2.train` verwendete Initialisierungsanweisung auf den unverändert vorbereiteten 8L+4U-Entwicklungsprozessen repliziert. Kein Modelltraining und keine Qualitätsmetrik.

K-Means verwendet `KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled)`. Weil Zustandsnummern beliebig permutierbar sind, werden Zentren per minimaler Zuordnung (Hungarian Assignment) verglichen und die Clusterbelegung entsprechend ausgerichtet.

## Verwendung aus `Implementierung_BA`

Nur den Ordner `research_seed_forensics_v56` ergänzen. Erforderliche vorhandene Ordner: `research_convergence_v55`, `final_candidate_v52`. Kein sonstiger Code wird ersetzt.

```powershell
.\.venv\Scripts\python.exe .\research_seed_forensics_v56\audit_kmeans_seeds_v56.py self-test
.\.venv\Scripts\python.exe .\research_seed_forensics_v56\audit_kmeans_seeds_v56.py audit --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --out results\v56_train_stability\KMEANS_INIT_AUDIT_01.json
```

Der zweite Befehl liest die SHA-geprüfte Entwicklungssicht der Original-Pickle (Python Pickle nur aus eigener vertrauenswürdiger Quelle), die IDs und X-Skalierung. Er führt K-Means mit drei Seeds durch, aber **keine Bayesian-Modelltrainingsläufe**, keine Fehlerbewertung, keine Nutzung historischer Testringe und keine Modelländerung. Der Bericht wird einmalig erzeugt.

## Interpretation

- Gleiche Zentren nach Permutation und gleiche Clusterzuordnung: die Seed-Studie hat praktisch dieselbe Initialisierung erzeugt; **kein belastbarer Test weit auseinanderliegender Startpunkte**.
- Verschiedene Zentren und gleicher End-ELBO-Wert: konsistent mit einem gemeinsamen lokalen Optimum, aber ohne Parameter-/Zustandsabgleich kein Beweis.
- Unterschiede in K-Means-Starts können auch unterschiedliche **Label-Permutationen**, nicht andere physikalische Prozesszustände sein.
- Nächster Forschungszweig: bei Bedarf eine gezielte, vorab festgelegte Startwertvariation und separate 8-vs-16-L-BFGS-B-Analyse, jeweils ohne historischen Testscore.
