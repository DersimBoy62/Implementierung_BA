# Unabhängiger numerischer Vergleich mit dem echten V6.0-Oracle

Am 13.11.2026 ausgeführt:

```powershell
.\.venv\Scripts\python.exe -m pytest bayesian_ssr_neubau\tests\reference_parity\test_v60_oracle.py -q
```

Historisches Ergebnis vor der Projektbereinigung: **12 passed in 34.27 s**,
Exitcode 0. Das Oracle ist die bytegleiche Kopie von
`research_math_v60/m2_prior_reduction_v60.py` unter
`tests/reference_parity/fixtures/legacy/`. Vor der Übernahme wurde sie gegen die
Übergabekopie geprüft; im aktiven Projekt sichert das Fixture-Manifest ihre
Integrität ohne eine zweite Übergabekopie. Die originale Funktion
`dependencies()` prüft ihrerseits SHA256 der V5.9-/V5.8-Dateien und aller zehn
V5.2-Kernmodule. Nur der Test importiert diesen alten Code. Alte Modellzustände,
Posterioren, Zentren, Skalierer oder Ergebnisdateien werden nicht gelesen.
Numba-Caches des Oracles werden in das pytest-Temp-Verzeichnis umgeleitet;
Python-Bytecode-Schreiben wird während des Oracleimports deaktiviert.

Für jedes K=2,3,4 werden vier getrennte Vergleiche durchgeführt:

1. Gleiche zeitabhängige Markov-Verteilungen, reduzierte M2-Momente und
   vollständige lokale Zielfunktion einschließlich analytischem Gradient.
   Toleranzen: Verteilungen `atol=rtol=2e-15`, Momente `2e-14`, lokale Funktion
   und Gradient `3e-13`.
2. Normal-Wishart-Posterior mit echten gewichteten U-Statistiken, erwartete
   Log-Präzision, Emissionslikelihood und KL gegen alte kontrollierte Funktionen.
   Toleranzen: Posterioren `3e-13`, Log-/KL-Werte `atol=3e-12`.
3. Induzierte NIG-Priors, Posterioren, Qualitätslikelihood, KL und erste/zweite
   Posterior-Predictive-Momente. Priormatrix und Lift sind bitgleich;
   weitere Toleranzen `3e-13`, KL `atol=3e-12`.
4. Drei vollständige neue Trainings-Epochen gegen drei frische
   `train_m2_reduced`-Epochen. Vier synthetische sechs-Kanal-Ringe,
   drei L mit je zwei Qualitätslabels und ein U; gleiche frisch gefittete
   TRAIN-Skalierung, frische K-Means-Initialisierung, Seed20261111, C0,2,
   acht lokale Iterationen. Gesamte ELBO-Kurve sowie Head-Mittelwerte und b
   werden mit **vor Ausführung festgelegtem** `atol=2e-6, rtol=0` verglichen.

Die zweite Implementierung verwendet einen eigenständig formulierten
Conditional-Reward-Gradienten statt des historischen Rückwärts-Differenzierens.
Die Tests belegen numerische Gleichheit der mathematischen Funktionen in
diesen kontrollierten Beispielen. Endliche Optimierertrajektorien oder ein
beliebiger freier voller q(beta)-Faktor müssen deshalb nicht identisch sein.
Die Tests sind Korrektheits-/Paritätsbelege und keine Prüfung der Vorhersagegüte
oder eine externe Generalisierungsstudie.
