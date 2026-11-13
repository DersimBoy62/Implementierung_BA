# Unveränderte Testreferenzen

Diese 13 Python-Dateien bilden die kleinste benötigte Referenzkette der
mathematischen Vergleichstests: V60 → V59 → V58 → zehn V52-Kernmodule.
Die historischen Dateiinhalte wurden unverändert aus Commit
`e1a15ca6f808cc16bf863472a552e3c1dea33394` übernommen. Der vollständige
Projektstand bleibt auf Branch `archive/vollstand-2026-11-13` erhalten.

Vor der Übernahme wurde die V60-Datei gegen die ursprüngliche Handover-Kopie
geprüft. `SHA256_MANIFEST.json` hält Herkunft und Prüfsummen aller übernommenen
Dateien fest. Die gemeinsame Fixture `legacy_root` kontrolliert den vollständigen
Katalog, bevor ein Test Referenzcode ausführt; V59 und V58 prüfen zusätzlich
ihre historischen Abhängigkeiten selbst.

Diese Dateien dienen ausschließlich als unabhängige Testreferenzen. Der
Produktivcode unter `src/` importiert keine davon. Historische Kommandozeilen-
und Trainingsfunktionen innerhalb der kopierten Dateien gehören nicht zum
unterstützten neuen Projekt und benötigen gegebenenfalls weitere Altdateien.
Keine Rohdaten, alten Modellzustände oder Ergebnisdateien sind enthalten.
