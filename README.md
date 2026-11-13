# Bayesian Ring SSR

Dieses Repository enthält das aktuelle Bayesian-SSR-HMM unter
[`bayesian_ssr_neubau/`](bayesian_ssr_neubau/README.md). Modellcode,
Konfigurationen, Tests, mathematische Dokumentation und das Demonstrationsnotebook
bilden das aktive Projekt.

## Branches

- **`main`**: gekürztes, eigenständig prüfbares Projekt.
- **[`archive/vollstand-2026-11-13`](https://github.com/DersimBoy62/Implementierung_BA/tree/archive/vollstand-2026-11-13)**:
  unveränderter Git-Stand vor der Bereinigung, einschließlich alter Experimente,
  Forschungsberichte und Übergabeunterlagen.

Die zuvor nicht versionierten lokalen Dateien sind weiterhin lokal vorhanden:
Rohdaten unter `input/`, aktuelle Modellartefakte unter
`bayesian_ssr_neubau/outputs/` und der übrige Altbestand unter
`_local_archive/vollstand-2026-11-13/`. Diese Dateien werden nicht zu GitHub
übertragen. Der Archivbranch bewahrt den zuvor versionierten Stand.

## Start unter Windows

Python 3.11 oder 3.12 verwenden. Alle Befehle vom Repository-Root aus ausführen:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\bayesian_ssr_neubau[audit,notebook]"
.\.venv\Scripts\python.exe -m pytest -q
```

Eine vorhandene Projektumgebung kann weiterverwendet werden. Die Datei
`bayesian_ssr_neubau/requirements-lock-windows-py311.txt` hält die konkret
geprüfte Windows-/Python-3.11-Umgebung fest.

Die mathematischen, synthetischen und Referenztests funktionieren ohne private
Daten. Nur die Tests mit echten DM-Ringen werden bei fehlenden Rohdaten mit
ausdrücklicher Begründung übersprungen. Für Training, Evaluation und die ganze
Notebook-Demonstration die freigegebene Datei
`input/DM_16_aktive_Ringe.pkl` lokal bereitstellen oder einen anderen Speicherort
angeben:

```powershell
$env:BAYESIAN_SSR_DATA = 'C:\Daten\DM_16_aktive_Ringe.pkl'
```

Der vorhandene SHA256-Datenvertrag bleibt verbindlich. Details, Trainings- und
Vorhersagebefehle stehen in der [Projektanleitung](bayesian_ssr_neubau/README.md).

## Prüfung nach Änderungen

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\check_source_manifest.py
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\audit_importgraph.py
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\run_verification.py
```

Neue Prüfergebnisse werden unter `bayesian_ssr_neubau/outputs/verification/`
gespeichert. Die Umstrukturierung ist in
[`docs/PROJEKTSTRUKTUR.md`](bayesian_ssr_neubau/docs/PROJEKTSTRUKTUR.md) beschrieben.
