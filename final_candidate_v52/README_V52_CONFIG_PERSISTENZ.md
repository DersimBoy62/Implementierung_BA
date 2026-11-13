# BAYESIAN BACHELOR — V5.2

## Konfiguration, Modellpersistenz, Nachvollziehbarkeit

**Datum:** 12.11.2026. **Forschungsstatus:** `PRE-FREEZE`, Entwicklungsmodell, **kein** unabhängiger Test. Dieses Paket ersetzt weder den älteren V5-/V5.1-Code noch bestätigt es eine finale Modellfreigabe. **In dieser Veröffentlichung ist kein echtes Modell neu trainiert worden.**

### Installation

Das ZIP im Projektordner `Implementierung_BA` entpacken. Es erzeugt einen eigenständigen Unterordner `final_candidate_v52`; die Ordner `final_candidate_v5` und `final_candidate_v51` werden nicht überschrieben. Empfohlen: Python 3.11.9 mit projektinternem `.venv`. Auf dem Erzeugungssystem wurden die technischen Tests unter Python 3.13.5/Linux ausgeführt; die eigentliche Windows-Umgebung muss separat geprüft werden.

### Vorläufige Testbefehle (kein Training)

Aus dem Verzeichnis `Implementierung_BA` (PowerShell):

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v52\main_bayesian_ssr_v52.py verify-config
.\.venv\Scripts\python.exe .\final_candidate_v52\test_release_integrity_v52.py
.\.venv\Scripts\python.exe .\final_candidate_v52\test_input_safety_v51.py
.\.venv\Scripts\python.exe .\final_candidate_v52\test_development_logger_equivalence_v51.py --data input\DM_16_aktive_Ringe.pkl
```

- `verify-config` kontrolliert **alle** 26 registrierten JSON-Konfigurationseinträge über Schema, inhaltliche Konsistenz und SHA256 der Originaldatei, zehn unveränderte HMM-/VI-Kernmodule und die alte V5.1-Sicherheits-/Wrapper-Datei. Keine Deserialisierung, keine Modellbewertung.
- `test_release_integrity_v52.py`: 19 isolierte, synthetische Tests ohne Daten-Pickle. Transaktional gespeicherte **Testdummies** sind kein trainiertes Modell und für Vorhersagen ungeeignet.
- Die V5.1-Tests sind unverändert und reproduzieren 18 synthetische Gates bzw. bei vorhandenen vertrauenswürdigen DM-Rohdaten zwölf F2-Prozesse. Technisch wird beim zweiten Test das ursprüngliche monolithische Pickle deserialisiert; nur L-/U-Prozesskanäle werden anschließend inspiziert, keine vier alten Test-Qualitätswerte bewertet.

### Geänderte Dateien gegenüber V5.1

- `main_bayesian_ssr_v52.py` — neuer Einstieg für `verify-config`, `verify-model`, `prepare-policy`, `verify-input`, explizites `train`, `predict`.
- `release_integrity_v52.py` — vollständiger Lock der registrierten Konfiguration, atomare (write-once) Modell-/Scaler-/Manifest-Transaktion, Paketversionen und Integritätsprüfungen vor jedem vertrauenswürdigen `pickle.loads`.
- `test_release_integrity_v52.py` — 19 synthetische Testfälle.
- `V52_QA_RESULTS.json`, `V52_RELEASE_MANIFEST.json`, dieser README und der V5.2-Prüfbericht.

**Unverändert:** Originales `PROPOSED_MODEL_FREEZE.json` (SHA256 `88f9da6aa51be71db7968a8222cb36e5e4838b8a8431988bd7c54c09bace51ab`), `input_safety_v51.py`, `main_bayesian_ssr_v51.py`, alle zehn `final_core/*.py` und alle ursprünglichen V5.1-Dokumente. Speziell `ovality.ovality_radial_mm` bleibt die Außenunrundheit und `planarity.height_deviation_mm` der Höhenschlag.

### Technische Strategie

1. Ein Modellfit darf nur ausdrücklich über `train` gestartet werden und verweigert bereits existierende Ausgabeordner. Der Datensatz wird vor dem Deserialisieren anhand seines exakten SHA256 geprüft; zugelassen sind acht L- und vier U-Ringe. Die vier historischen Test-Ringe sind nie Trainingsinput.
2. Modell, X-/Y-Scaler, die beiden Deskriptor-Scaler, Modellinput-Policy und die wissenschaftliche Konfigurationskennung werden **gemeinsam** in ein einzelnes Modell-Pickle paketiert. Da das Datenmodell ursprünglich Pickle verwendet, wird dieses Format nicht ohne wissenschaftliche Notwendigkeit umgestellt.
3. Ein Manifest hält Modell-SHA256, SHA256 der Eingangs-Policy, zehn Kernhashes, Release-Codehashes, Daten-/Konfigurationshash, Ringaufteilung, endgültige ELBO, Epochenzahl sowie Python-, Betriebssystem- und Paketversionen fest.
4. Beide Dateien werden in einem verborgenen temporären Verzeichnis auf dem **gleichen Dateisystem** erzeugt, synchronisiert, geprüft und der Ordner erst danach mit `os.replace` atomar auf den endgültigen Namen gesetzt. Eine reguläre Ausnahme bereinigt den Staging-Ordner. Bei einem harten Systemabbruch kann ein isolierter, verborgener Restordner verbleiben, **nicht jedoch ein regulär benanntes halbfertiges Modellverzeichnis**.
5. `verify-model` prüft Manifest/Modellhash/Quellcode-/Konfigurations- und Environment-Gleichheit, **ohne** Pickle zu laden. Die prädiktive Funktion prüft dieselben Voraussetzungen und lädt nur danach die **zuvor überprüften Bytes** des lokal als vertrauenswürdig eingestuften Pickles; es gibt keine ungesicherte Prüfung mit anschließend erneut geöffnetem Dateipfad.
6. Bei Vorhersagen steht im Bericht der Modell-SHA256, der Manifest-SHA256 und der Environment-SHA256. Die V5.1-Sicherheitsgates und Abschlussbestätigung bleiben Pflicht.

### Grenzen und Sicherheitswarnungen

- **Kein echter V5.2-Trainingslauf in diesem Paket**, keine neuen RMSE-Ergebnisse und keine neu überprüfte Konvergenz.
- **Lokale Hashes sind keine Signaturen.** Wer Modell und Manifest kontrolliert, kann beides manipulieren. Python-Pickle kann beim Laden Code ausführen; NIEMALS fremde oder nicht vertrauenswürdig erzeugte Modelle laden. Für weitergehenden Herkunftsnachweis wären getrennt gesicherte Signaturen oder ein sichereres Austauschformat notwendig.
- Der Environment-Abgleich sperrt absichtlich das Laden bei anderer Python-/NumPy-/SciPy-/sklearn-/Numba-/threadpoolctl-Version oder anderem Betriebssystem/CPU-Typ. Die Versionen garantieren keine bitgenaue, hardware- und threadunabhängige Wiederholung. Ein verbindlicher Windows-Lock muss auf dem Windows-Rechner erstellt werden.
- Das ursprüngliche Gesamt-Pickle kann technisch alle 16 Ringe enthalten. Die Entwicklungsansicht maskiert U-Labels, schließt Testdaten aber erst nach Deserialisierung aus. Keine echte physische Datenisolierung behaupten.
- Die in V5.1 abgeleiteten Sensordomänen sind Daten-Gates, keine universellen physikalischen Zulässigkeitsgrenzen. Ein Betreiberabschlussnachweis kann technisch gebunden, aber nicht automatisch als wahr authentifiziert werden.
- Forschungsentscheidungen sind wiederholte Entwicklungsdaten-Selektionen. Die vier früher geöffneten Test-Ringe sind nicht mehr unabhängig; für Generalisierungsansprüche braucht es neue Ringe.

### Weitere Arbeit und spätere Kommandos

V5.3: mathematische/numerische Robustheit und lokale VI-Rückkopplung `steps=0` vs `steps=8` ausschließlich im registrierten Entwicklungsdesign. V5.4: eindeutige fachliche Freigabe, dann explizites neues Fit-Artefakt mit dokumentierter Windows-Umgebung. Ein späterer regulärer Trainingsaufruf könnte so aussehen (NICHT jetzt ausführen):

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v52\main_bayesian_ssr_v52.py train --data input\DM_16_aktive_Ringe.pkl --out results\v52\RUN_01
```

Voraussetzung ist, dass `results\v52` **bereits als Elternverzeichnis existiert**, `RUN_01` dagegen nicht. Bei einer späteren echten V5.2-Laufversion wird die gespeicherte Umgebung verbindlich und ist beim Laden identisch wiederherzustellen. Bei V5.4 würden wir sinnvollerweise eine **neue** Version und einen anderen, zuvor unbenutzten Zielordner verwenden, statt das V5.2-Dummy-/QA-Paket umzudeuten.
