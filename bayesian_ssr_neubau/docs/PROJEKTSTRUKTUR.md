# Aktives Projekt und archivierter Vollstand

Stand: 13.11.2026. Der Branch `archive/vollstand-2026-11-13` bewahrt Commit
`e1a15ca6f808cc16bf863472a552e3c1dea33394` unverändert. `main` enthält das
gekürzte Projekt. Kein Force-Push und keine Umschreibung der bisherigen
Commit-Historie sind dafür nötig.

## Verzeichnisaufteilung

```text
Implementierung_BA/
  README.md
  pytest.ini
  bayesian_ssr_neubau/
    src/bayesian_ring_ssr/       Modell und CLI
    configs/                    Basiskonfiguration
    tests/                      Synthetische und reale Datenprüfungen
      reference_parity/fixtures/legacy/
                                13 SHA256-geprüfte Testreferenzen
    scripts/                    Aktuelle Quellen- und Testprüfungen
    notebooks/                  Demonstration aus den aktuellen Quellen
    docs/                       Mathematik, Datenverträge und Testanleitung
    SOURCE_SHA256.json          Prüfsummen des aktuellen Produktivcodes
    outputs/                    Lokal erzeugte Modelle und Prüfergebnisse
  input/                        Lokale Rohdaten; nur Anleitung/Manifest in Git
  .venv/                        Lokale Python-Umgebung
  _local_archive/                Lokale Sicherung des übrigen Altbestands
```

## Unveränderte Mathematik, lokale Testreferenzen

Die Modellgleichungen, Basishyperparameter, Daten-SHA, Ringaufteilung und
Trainingsverfahren wurden durch die Bereinigung nicht geändert. Die CLI erlaubt
zusätzlich `BAYESIAN_SSR_DATA` als Vorgabe für den Rohdatenpfad; ein explizites
`--input` hat weiterhin Vorrang.

Die Vergleichstests behalten den historischen V60-/V59-/V58-Code und die zehn
benötigten V52-Kernmodule bytegleich unter `tests/reference_parity/fixtures/legacy/`.
Der ursprüngliche V60-Code wurde vor der Übernahme mit der Übergabekopie
verglichen. Das Fixture-Manifest prüft jetzt die lokale, minimale Kopie;
zusätzlich bleiben die SHA-Prüfungen der ursprünglichen Referenzfunktionen aktiv.
Die `.gitattributes` verhindert eine Veränderung der geprüften Bytes durch
plattformabhängige Git-Zeilenenden. Produktivcode importiert keine Testreferenzen.

## Historische Berichte und neue Prüfergebnisse

Historische Inventuren, Quellenauslese-, Trainings- und Releaseberichte sowie die
auf den alten Gesamtworkspace zugeschnittenen Hilfsskripte bleiben auf dem
Archivbranch. Sie sind keine Voraussetzung mehr für einen neuen Testlauf.
Die fachlichen Ableitungen, Grenzen und Quellenhinweise bleiben im aktiven
`docs/`-Ordner; darin bezeichnete historische Befunde sind keine neu ausgeführten
Experimente.

Das frühere `RELEASE_SHA256.json` band auch lokale Trainingsergebnisse ein.
`SOURCE_SHA256.json` prüft ausschließlich aktuelle Produktivquellen,
Konfiguration und Paketdefinition. Das Notebook benötigt deshalb keine
vorherigen Trainingsausgaben und erzeugt eigene neue Laufartefakte.
Neue Test-/Importberichte liegen unter `outputs/verification/`. Der Datenvertrag
prüft weiterhin vor jeder Deserialisierung die freigegebene Rohdatei.

## Lokale Sicherung und Branchwechsel

`_local_archive/vollstand-2026-11-13/` enthält die aus dem aktiven Workspace
entfernten Dateien einschließlich zuvor nicht versionierter Archive, Literatur
und alter Ergebnisse. Rohdaten und aktuelle `outputs/` wurden an ihrem bisherigen
Ort belassen. Diese lokalen Dateien sind weiterhin von Git ausgeschlossen;
der Archivbranch ist keine Sicherung zuvor ignorierter Dateien.

Mit `git switch archive/vollstand-2026-11-13` lässt sich der alte versionierte
Gesamtstand wiederherstellen; `git switch main` kehrt zum aktiven Projekt zurück.
Vor einem Branchwechsel laufende Trainings beenden und eigene Änderungen committen.
