# V5.3 — Reale Qualitätsrückkopplung: kontrolliertes Entwicklungsexperiment

**Stand: 12.11.2026 — zusätzliche Forschungsdiagnostik, kein neues freigegebenes Hauptmodell.**

## Zweck

Die numerischen 14/14-Synthetikprüfungen der lokalen Variationsrechnung sind erfolgreich; das beweist nicht, dass die Kopplung im Ringwalzdatensatz nennenswert wirkt. Diese **vorab fixierte** Studie untersucht für die gleiche M2-Modellarchitektur und identische Prozessaufteilung:

1. **Innerhalb desselben trainierten `steps=8`-Modells:** Wie stark ändern sich die lokalen q(z)-Marginalen und die M2-Momente, wenn unter exakt denselben globalen HMM- und NIG-Parametern 0 statt 8 lokale Optimierungsschritte ausgeführt werden? Dieser Vergleich eliminiert Zustandspermutationen zwischen zwei unabhängig trainierten Modellen. Pro L-Ring werden `local_gain`, Akzeptanz, Anzahl Iterationen sowie Gamma- und Momentenverschiebung protokolliert.
2. **Zwischen zwei getrennten Trainingseinheiten (`steps=0` und `steps=8`):** Welche Abweichungen zeigen die **bereits wiederverwendeten** acht Entwicklungslabels bei denselben vier festen Zweier-Validierungsgruppen? Erfasst werden RMSE, MAE, ringweise Fehler und ELBO-Verläufe. Ein Effekt kann durch Trainingspfad und finale Parameter entstehen und darf nicht als isolierter rein lokaler Mechanismus interpretiert werden.

## Verbindliche Nichtänderungen

- **Unverändertes V5.2-Paket** im Ordner `final_candidate_v52`; keine Kernmodule überschreiben.
- Festgelegtes Modell `K=2`, `M2`, `P0`, `C=0.2`, `physical_var=0.38526108747593474`, `epochs=10`, `seed=20261111`, `F2`, sechs Originalsensoren, zwei NIG-Qualitätsköpfe `SEPARATE_AXIAL`.
- **L:** DM04, DM05, DM06, DM09, DM12, DM16, DM17, DM18.
- **U (Qualität maskiert):** DM01, DM02, DM10, DM14.
- **Alte historische Tests DM07, DM08, DM11, DM15 sind strikt ausgeschlossen.** Das ursprüngliche Gesamt-Pickle wird beim Laden trotzdem als vertrauenswürdiges Gesamtobjekt deserialisiert; es werden daraus keine Test-/U-Qualitätswerte gelesen und keine äußeren Testsignale zum Fit benutzt. Das ist keine physische Dateitrennung.
- Prozess- und Deskriptorskalierer werden **pro Fold** nur auf sechs trainierenden L- und vier U-Ringen angepasst; Zielskaler nur auf den sechs Trainings-L-Ringen. Validierungslabels werden erst nach beiden Modellvorhersagen zur Fehlerberechnung genutzt.
- Keine Nachoptimierung von Schritten, K, C, Priorvarianz, Epochen, Features oder Datensplit anhand dieses Experiments; keine externe Generalisierungsbehauptung.

## Windows-Ausführung aus `Implementierung_BA`

Dieses ZIP fügt ausschließlich den Ordner `research_v53_feedback/` hinzu. `qa_v53/` und `final_candidate_v52/` bleiben unverändert.

Zunächst ohne DM-Daten und ohne Training:

```powershell
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py self-test
```

Danach ausschließlich eine Datenvorprüfung — sie liest das bekannte vertrauenswürdige Pickle nach SHA256-Prüfung, startet jedoch kein Training und keine Metrikberechnung:

```powershell
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py preflight --data input\DM_16_aktive_Ringe.pkl
```

**Erst wenn die beiden Checks erfolgreich sind** und du die explorative Auswertung beginnen möchtest, führe **jeweils genau einen festen Fold** aus:

```powershell
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py run-fold --fold 1 --data input\DM_16_aktive_Ringe.pkl --out-dir results\v53_feedback_devonly
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py run-fold --fold 2 --data input\DM_16_aktive_Ringe.pkl --out-dir results\v53_feedback_devonly
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py run-fold --fold 3 --data input\DM_16_aktive_Ringe.pkl --out-dir results\v53_feedback_devonly
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py run-fold --fold 4 --data input\DM_16_aktive_Ringe.pkl --out-dir results\v53_feedback_devonly
```

Die festen Validierungsringe sind: Fold 1 DM04/DM05, Fold 2 DM06/DM09, Fold 3 DM12/DM16, Fold 4 DM17/DM18. Ein Fold trainiert **zweimal** mit denselben hyperparametrischen Einstellungen (nur lokale Schritte 0 gegenüber 8). Die vier Folds dürfen nacheinander ausgeführt werden. Ergebnisdateien werden **nicht überschrieben**, so dass ältere Versuche unverändert bleiben.

Nach Abschluss aller vier Folds aggregieren (kein Training):

```powershell
.\.venv\Scripts\python.exe .\research_v53_feedback\paired_feedback_dev_v53.py summarize --out-dir results\v53_feedback_devonly
```

## Dateien und Einordnung

- `FOLD_1_PAIRED_STEPS_0_8.json` bis `FOLD_4_PAIRED_STEPS_0_8.json`: feste Ring-IDs, Skalierer-Fit-Gruppen, ELBO je Epoche, Val-Vorhersagen, realer `steps=8`-Rückkopplungseffekt bei fixierten globalen Parametern.
- `V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json`: jeweils **acht** paarige Entwicklungs-Val-Vorhersagen, zwei dimensionsspezifische RMSE-/MAE-Werte, ringweise Residuen und signierte Differenzen `steps8 minus steps0`.
- Das Vorzeichen der Differenz ist wichtig: **negative Fehlerdifferenz = `steps=8` schneidet auf diesen Entwicklungsringen besser ab**; es ist keine Signifikanz- oder Generalisierungsaussage.

## Technische Evidenz vor Veröffentlichung

- 5/5 synthetische Kontrolltests (gesperrter Split, feste Fold-Aufteilung, Zielmapping, Paaraggregation, veränderte Ringzuordnung zurückweisen).
- Ein separater durchgängiger synthetischer Probe-Fold mit je 10 echten Codeepochen (`steps=0` und `steps=8`) bestätigt die Schnittstellen und Ergebnisstruktur; **keine DM-Daten in dieser Probe**.
- Echtes DM-Daten-SHA256 `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5` in der **Vorprüfung** verifiziert, 12 erlaubte F2-Signale erkannt; **kein reales DM-Training in dieser Releasevorbereitung**.
- Das reproduzierte Windows-Protokoll V5.2 hat 19/19 bestanden, wobei die native Symlink-Erstellung aufgrund von Windows-Berechtigungen übersprungen und der Prüfpfeil simuliert wurde. V5.3-Synthetik hat auf dem Nutzer-PC 14/14 bestanden.

## Methodische Grenzen

Dieses Experiment ist eine weitere **explorative Analyse derselben acht mehrfach verwendeten Entwicklungslabels**. Auch eine positive RMSE-Differenz darf nicht als externe Bestätigung, optimaler Hyperparameterwert, validiertes Unsicherheitsintervall oder reales Prozessverbesserungsergebnis ausgegeben werden. Der Freeze bleibt vorläufig und wird durch diesen Vergleich nicht automatisch geändert. Die Ergebnisse der Schritte 0 und 8 dürfen nur unter identischen eingefrorenen Parametern diskutiert werden.

Das Skript prüft beim Start die reale V5.2-Konfigurationsbindung und alle zehn Kernmodule über SHA256. Es benutzt keine neuen trainierbaren Modelltypen. **Der Benutzer entscheidet explizit durch `run-fold`, ob ein reales Entwicklungstraining ausgeführt wird.**
