# Tests und tatsächliche Prüfergebnisse

Die Tests sind aus dem Repository-Root mit
`.\.venv\Scripts\python.exe -m pytest -q` ausführbar. Sie hängen nicht von den
historischen Workspace-Verzeichnissen ab. `scripts/run_verification.py`
erzeugt zusätzlich ein neues Verzeichnis unter `outputs/verification/` mit
tatsächlichem pytest-Log, JUnit-Bericht, Exitcode, Testzahlen, Laufzeit und Umgebung.
Es überschreibt keine historischen Prüfberichte und erzeugt keinen Erfolg aus
einer vorgegebenen erwarteten Testanzahl.

## Umfang

- Forward-/Backward- und Markov-Momente gegen unabhängige Enumeration.
- Analytische Gradienten, induzierte M2-Priors, konjugierte NIG-/Normal-Wishart-
  und Dirichlet-Rechnungen sowie die gewichtete Blockbilanz.
- Frische synthetische Trainingsläufe, prozessbasierte Prediction, Speichern/
  Laden und semantische Ablehnung beschädigter Modellverträge.
- V60-Parität für K=2,3,4 gegen die SHA256-geprüften lokalen Testreferenzen.
- Reale Daten-/Featureparität, F2-Gates und TRAIN-only-Skalierung mit der
  freigegebenen lokalen Rohdatei.
- Datenpfad-Vorgabe per `BAYESIAN_SSR_DATA` und Vorrang eines expliziten `--input`.

Die ursprünglichen 105 Prüfungen bleiben erhalten. Zwei Datenpfad-Prüfungen
ergänzen das neue Verhalten auf insgesamt 107 Tests. Die tatsächlichen Zahlen
sind dem jeweils ausgeführten Bericht zu entnehmen.

## Ohne private Rohdaten

Mathematik, Referenzvergleiche und synthetische End-to-End-Prüfungen laufen
weiter. Ausschließlich neun Tests mit echten Ringen werden mit einer sichtbaren
Begründung übersprungen. Mit `-rs` zeigt pytest die Gründe:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Für einen abweichenden lokalen Speicherort:

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\run_verification.py --data C:\Daten\DM_16_aktive_Ringe.pkl
```

Eine vorhandene Datei mit falschem SHA256 darf nicht als fehlende Datei
übersprungen werden; der verbindliche Datenvertrag führt dann zum Fehler.

## Unabhängigkeit und Quellenintegrität

`scripts/audit_importgraph.py` prüft die Produktivimporte und legt einen
neuen Bericht unter `outputs/verification/` ab. Alte Modellimporte,
unzulässige externe Module und dynamische Imports werden abgelehnt.
`scripts/check_source_manifest.py` kontrolliert zusätzlich die Menge und
SHA256-Werte der Quellen und Konfigurationen gegen `SOURCE_SHA256.json`.
Die 13 bytegleichen historischen Testmodule haben ein eigenes Fixture-Manifest.

Die früheren Läufe und Berichte vor der Bereinigung liegen unverändert auf
`archive/vollstand-2026-11-13`. Sie belegen ihren jeweiligen historischen
Stand. Ein bestandener Test nach der Umstrukturierung ist kein neu ausgeführter
30-Epochen- oder vierfacher CV-Lauf; solche Modellläufe erzeugt das Notebook
gesondert. Numerische Parität ist kein unabhängiger Vorhersagegütenachweis.

## Ausgeführt für die Bereinigung am 13.11.2026

| Prüfung | Tatsächliches Ergebnis |
|---|---|
| Vollständige Suite mit vorhandener freigegebener Rohdatei | 107 bestanden, 0 übersprungen, 0 Fehler; 19,30 s |
| Sauberer Checkout des Git-Index ohne Rohdaten und historische Ordner | 98 bestanden, genau 9 datenabhängige Tests übersprungen, 0 Fehler; 36,55 s |
| Quellenmanifest und Importprüfung im aktiven und im sauberen Checkout | 25 Quelldateien stimmen überein; 23 Produktivmodule ohne unzulässige Imports |
| Erste fünf Notebook-Codezellen im echten Kernel | 0 Fehler; 13,37 s; Vorlage blieb unverändert |
| Vorhandenes trainiertes Modell und Prozess-CSV | Laden und prozessbasierte Vorhersage erfolgreich |
| Vergleich mit archiviertem Produktivcode und Basiskonfiguration | Bytegleich, ausgenommen die ergänzte CLI-Datenpfadwahl |

Der saubere Checkout wurde ausschließlich aus dem vorgemerkten Git-Inhalt
erzeugt; er enthielt weder Rohdaten noch die alten Workspace-Verzeichnisse.
Die Tests importierten nachweislich sein eigenes `src/`, nicht den lokalen
Arbeitsbaum über die vorhandene Editable-Installation.
Die realen pytest-/JUnit-Berichte liegen lokal unter
`outputs/verification/main_cleanup_001/` beziehungsweise im lokalen sauberen
Checkout unter `outputs/verification/clean_checkout_001/`.
Der Notebooknachweis liegt unter `outputs/notebook_executions/eb74f41a53/`.
Die Prüfungen dieser Umstrukturierung wiederholen keinen vollständigen
30-Epochen- oder CV-Lauf.
