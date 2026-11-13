# Update für den bestehenden Ordner `Implementierung_BA` (11.11.2026)

## Ergebnis von OFF und TILT_JOINT

Die gerundeten Terminalausgaben von `OFF` und `TILT_JOINT` sind gleich. In den
Referenzberichten liegt bei Epoche 3 die ELBO bei OFF bei -45017.541004468,
bei TILT_JOINT bei -45017.540845862; der Unterschied ist +0.000158605.
Das belegt einen sehr kleinen, aber nicht identisch null Rückkopplungseffekt.
Daraus folgt **keine** Aussage über die Vorhersageleistung auf neuen Ringen.

## Installation: Nur diese fünf Dateien übernehmen

Aus dem ZIP in den **bereits vorhandenen** Projektordner `Implementierung_BA` kopieren:

1. `vb_b1.py` **ersetzen** (vorher sichern)
2. `run_dm_smoke.py` **ersetzen** (vorher sichern)
3. `full_markov_b1.py` **hinzufügen**
4. `test_full_markov.py` **hinzufügen**
5. `compare_smoke_runs.py` **hinzufügen**

**Nicht überschreiben oder löschen:** `input/`, `results/`, `dm_data.py`, `.venv/`,
`test_math.py`, `test_data.py` sowie alle bisherigen Auswertungen.

Eine mögliche Sicherung der beiden Dateien in der PowerShell (im Projektordner):

```powershell
Copy-Item .\vb_b1.py .\vb_b1_V0_1_backup.py
Copy-Item .\run_dm_smoke.py .\run_dm_smoke_V0_1_backup.py
```

## Ausführung

```powershell
.\.venv\Scripts\python.exe test_math.py
.\.venv\Scripts\python.exe test_data.py
.\.venv\Scripts\python.exe test_full_markov.py
```

**Danach** (gleicher Split und dieselben Konfigurationen wie die vorhandenen Läufe):

```powershell
.\.venv\Scripts\python.exe run_dm_smoke.py --epochs 3 --mode FULL_JOINT --local-steps 8 --out results/smoke_v1_f2_k3_full_joint_c1
```

Anschließend die drei bereits gespeicherten Läufe vergleichen:

```powershell
.\.venv\Scripts\python.exe compare_smoke_runs.py --full results/smoke_v1_f2_k3_full_joint_c1/smoke_report.json
```

Falls deine OFF- bzw. TILT-Berichte an anderer Stelle liegen, ihre Pfade
mit `--off` und `--joint` explizit angeben.

## Mathematische Besonderheit

Der neue `FULL_JOINT`-Modus parametrisiert die gesamte lokale
Markov-Verteilung eines Ringes frei durch eine Anfangsverteilung und
**separat optimierbare Übergangsmatrizen für jeden Zeitpunkt**.
Die ringweiten Zustandsaufenthalte `h` werden in der lokalen ELBO
**mit dem ersten und zweiten Moment** integriert. Die Gradienten
werden analytisch durch Rückwärtsableitung der exakten
Markov-Momentenrekursion berechnet (nicht über Pseudolabels).

Eine zusätzliche mathematische Validierung enthält:
- Exakte Enumeration aller Zustandsfolgen eines kleinen Testfalls,
- Abgleich analytischer Gradienten mit zentraler numerischer Prüfung
  via `scipy.optimize.check_grad`,
- Test eines deutlichen Rückkopplungseffekts in einem konstruierten Fall.

## Status und Grenzen

- Alle Original-DM-Daten unverändert; keine Altdaten (SFA) verwendet.
- Genau zwei Targets: Außenunrundheit, Höhenschlag. Keine anderen Zielwertfelder als Eingaben.
- Logger-only V1 F2; keine Walzprotokollmerkmale, keine Sollwertduplikate.
- Vier Testringe bleiben strikt gesperrt, auch nicht als ungelabeltes Training.
- Der Aufruf mit `--local-steps 8` ist ein **begrenzter technischer Test**,
  **kein** nachgewiesenes Konvergenzergebnis der freien lokalen Optimierung.
- Kleine ELBO-Veränderungen bestätigen weder bessere Qualitätsvorhersage noch
  valide Unsicherheitsintervalle.
- Die Auswertung der diagonal-gaußschen Emissionsannahme **innerhalb der
  geschätzten Zustände**, der numerischen Konvergenz und die verschachtelte
  ringweise Validierung stehen weiterhin aus.
- Optuna wird erst nach stabiler, validierter Trainings- und
  Evaluationsschnittstelle gestartet.

## Bereits unabhängig geprüft

Am 11.11.2026 wurden in der Entwicklungsumgebung durchgeführt:
- 5/5 bisherige Mathematiktests PASS,
- 8/8 bisherige Datentests PASS,
- 3/3 neue freie-Markov-Tests PASS,
- Drei-Epochen-Lauf `FULL_JOINT`: ELBO (gerundet) -49585.83, -45985.53, -45017.12.

Dies sind technische Prüfungen auf der vorhandenen Referenzversion; bitte den
Lauf unter deiner Windows-/Python-3.11-Umgebung wiederholen.
