# Ausgeführte Tests und numerische Toleranzen

Stand13.11.2026. Der maschinenlesbare abschließende Lauf wird in
`verification_001.json`, `verification_001.xml` und `verification_001.log`
gespeichert. Diese Dateien enthalten tatsächlichen Befehl, Exitcode, Testanzahl,
Laufzeit und Umgebung. `scripts/run_verification.py` führt pytest aus; es erzeugt
keine Erfolgsangabe aus einem vorgegebenen Erwartungswert.

Der abschließende gespeicherte Lauf ist ausgeführt: **105 passed in18.25s**, Exit0,
0failures,0errors,0skipped (Gesamtdauer des subprocess-Aufrufs19.93s).

## Tatsächlich ausgeführte Zwischenprüfungen

| Befehl (jeweils mit `.\.venv\Scripts\python.exe`) | Ergebnis | Exitcode |
|---|---|---|
| `-m pytest bayesian_ssr_neubau\tests\test_nw_dirichlet_nig.py bayesian_ssr_neubau\tests\test_m2_induced_prior.py -q` | 16 passed in12.34s | 0 |
| `-m pytest bayesian_ssr_neubau\tests\test_data_and_leakage.py -q` | 18 passed in0.50s | 0 |
| `-m pytest bayesian_ssr_neubau\tests\reference_parity\test_v60_oracle.py -q` | 12 passed in34.27s | 0 |
| `-m pytest bayesian_ssr_neubau\tests -q` vor den zusätzlichen Persistenzangriffen | 83 passed in15.85s | 0 |
| Persistenz-/Datenprüfung nach Vertragshärtung | 40 passed in4.54s | 0 |
| Volle Suite nach Vertragshärtung | 105 passed in16.41s | 0 |

Die Teilmengen werden nicht addiert; die abschließende Suite enthält sie.
Ein früher Datenparitätstest zeigte Rundungsdifferenzen bei achsenweiser statt
kanalweiser Mittelwertberechnung. Die neue Funktion wurde auf die historische
Summationsfolge korrigiert und erneut geprüft. Der erste editable-Installversuch
scheiterte am fehlenden `bdist_wheel`; nach Ergänzung der Buildwerkzeuge war die
lokale Paketinstallation erfolgreich. Diese Zwischenfehler werden nicht als
bestandene Prüfungen gezählt.

## Abdeckung des beauftragten Katalogs

| Katalog | Tatsächliche unabhängige Kontrolle |
|---|---|
| T1 | Forward-/Backward gegen vollständige kurze Pfade, T1/T2-Randfälle, Marginalidentitäten, Subnormalisierung der Elog-Potenziale, langes T2800 und Zustandspermutation. |
| T2 | M2-Momente gegen Pfadenumeration für K2/3/4; Kovarianz einschließlich Zeitkreuztermen; analytische lokale Gradienten gegen zentrale Differenzen. |
| T3 | Softmax-Gauge, reduzierte Logits und volle Koordinaten-Finite-Differenzen; erste K−1 Gradienteneinträge werden ausgewählt. |
| T4 | D1-Wishart als Gamma mit Skala2W; NW-KL als unabhängiges Gamma-/Normalintegral; Dirichlet als Betaintegral; NIG-KL als unabhängiges IG-/Normalintegral; handgerechnete Posteriorupdates und SPD-Fehler. |
| T5 | K2/3/4, unterschiedliche physikalische Dimensionen; induzierte Prioroffdiagonalen; volle/reduzierte konjugierte Posterioren, KL, Qualitätslikelihood und prädiktive Momente. |
| T6 | Gleiche Gesamtkomponentensumme; U-Entropie vollständig C-gewichtet; lokaler analytischer Gain gleich vollständiger J_C-Differenz; unabhängig summierte gewichtete Statistiken; C0 hat keine U-Prozessupdates; Warmstart-/Blocknichtabnahme. |
| T7 | Ringweise Folds, gesperrte Außenringe, U ohne y/targets, L-only-Y-Fit, TRAIN-X-/Deskriptorfit, ausschließlich Prozessfelder im Export und kein VAL-Fit. |
| T8 | Gleicher Seed reproduziert die Rechnung; JSON load gegenüber memory identisch; write-once; SHA-Änderung und intern neu versiegelte semantische Modellverletzungen werden abgelehnt. |
| T9 | Falsche Shapes/Featurefolge, fehlende Labels/Kanäle, NaN/Inf, zu kurze Sequenzen, SPD-/Zeit-/Gatefehler, doppelte IDs, SHA-Fehler, Außenringzugriff, inkonsistente Head-/Scaler-/Split-/Sidefile-Verträge. |
| T10 | Synthetische echte HMM-Pfade erzeugen sechs-Kanal-Ringe und zwei ringweite Qualitäten; L und U werden gemeinsam frisch trainiert; J_C verbessert sich mit nichtabnehmenden Blöcken, Vorhersage an einem prozess-only Holdout. |

## Vorab festgelegte Toleranzen und Reichweite

- Pfadenumeration/Marginalen/Momente: absolute Toleranzen etwa1e-14 bis3e-13,
  je nach getesteter Summation; genaue Werte stehen im jeweiligen Test.
- Lokale zentrale Differenzen: Schritt2e-6, `atol=3e-9`, `rtol=3e-6`.
- Hand-/Dichteintegrale: etwa1e-9; volle/reduzierte NIG-Identitäten etwa3e-13
  bis3e-12.
- Alter V6.0-Oracle: Momente2e-14, Gradient3e-13; unabhängig optimierte
  Drei-Epochen-Verläufe und Headparameter `atol=2e-6, rtol=0`.
- Trainingsblockbilanz: absolute1e-6; echte Verletzung führt zum Abbruch.
- Load-/Memory-Vorhersage: im gleichen Prozess und dokumentierten Environment
  exakt gleiche serialisierte Zahlen; Seed-Reprotests bis1e-12.

Die mathematischen Tests prüfen die festgelegten Funktionen. Sie beweisen keine
globale VI-Optimalität, unabhängige Generalisierung oder Unsicherheitskalibrierung.
Ein zweiter Seed kann die gleichen K-Means-Zentren ergeben und ist deshalb
allein kein Nachweis verschiedener Initialisierungen.

## Reale Pipelinebefehle

`-m bayesian_ring_ssr.cli audit-data --input input\DM_16_aktive_Ringe.pkl
--out bayesian_ssr_neubau\docs\CLI_DATEN_AUDIT.json` wurde mit Exit0 ausgeführt.

`-m bayesian_ring_ssr.cli train --config bayesian_ssr_neubau\configs\baseline_k2_m2.json
--out bayesian_ssr_neubau\outputs\baseline_8L4U_001` wurde mit Exit0 ausgeführt:
30 Epochen, 450 kontrollierte Blöcke, keine Verletzung, J_C am Ende
−22028,914444073278. Kleinste gemessene Blockdifferenz−7,28e-12.
240 lokale Optimiereraufrufe, davon6 ohne Erfolgsstatus; maximale abschließende
lokale Gradientennorm-infinity2,096e-4. Der Lauf ist damit nicht als Beweis
lokaler Stationarität bei gtol1e-5 zu lesen. Komplette Statusmeldungen stehen
im Update-JSONL. Einige nahezu flache Kandidaten werden abgelehnt; daher sind
Annahmezahlen und endliche Optimiererwege nicht mit alten Berichten identisch.

`export-process` für DM04 und `predict` mit der erzeugten Rohlogger-CSV wurden
jeweils mit Exit0 ausgeführt. Die Prediction erhält keine Endqualitäten.
Das Beispiel ist ein Trainingsring und dient ausschließlich dem IO-Vertrag.
Unabhängige Train/Pred-Trennung innerhalb der Entwicklung wird durch synthetische
Holdouts und die zurückgehaltenen VAL-Ringe der CV geprüft.

Der vierfache `evaluate`-Lauf speichert je Fold/Arm seine eigenen Scaler,
Konfigurationen, Verlaufswerte und zwei Ringvorhersagen. Abschlussstatus und
Qualitätsmetriken sind in `ABSCHLUSSBERICHT.md` dokumentiert.
