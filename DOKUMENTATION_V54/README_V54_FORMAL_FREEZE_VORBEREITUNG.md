# Bayesian Bachelor – V5.4, Archivierung und vorbereitete Freigabe

**Status: TECHNISCHER ENTWICKLUNGSSTAND GEPRÜFT; FORMALER MODEL-FREEZE NOCH NICHT GENEHMIGT.**

Die beiden PowerShell-Skripte werden im Wurzelordner `Implementierung_BA` abgelegt. Sie starten **kein Training**, laden **kein Pickle**, erstellen **keine Prognosen** und lesen **keine Qualitätslabels**. Die Quelldaten-Pickle wird ausschließlich per SHA256 gehasht und **nicht** in das ZIP kopiert. Die vier historisch benutzten Testringe werden nicht ausgewertet.

## Archiv erstellen

In PowerShell im Ordner `Implementierung_BA`:

```powershell
.\BA_ARCHIVIEREN_V54_DEVONLY.ps1
```

Falls PowerShell die Skriptausführung verweigert, zunächst die Datei-Eigenschaften und Vertrauenswürdigkeit prüfen. Die globale Execution Policy nicht unnötig ändern. Alternativ den Skriptinhalt direkt in PowerShell ausführen, wenn die Datei geprüft wurde.

Erwartet: `PASS: V5.4 DEVELOPMENT ARCHIVE WRITTEN ONCE; NOT A FORMAL MODEL FREEZE`. Das ZIP wird in `results\V54_ENTWICKLUNGSSTAND_ARCHIV_2026-11-12.zip` angelegt. Ein vorhandenes Archiv wird **nicht** überschrieben; für eine neue Version ausdrücklich `-ArchiveName V54_ENTWICKLUNGSSTAND_ARCHIV_2026-11-12_v2` angeben. Vor Archivierung werden die registrierte V5.2-Konfiguration, alle zehn mathematischen Kerne, Daten-SHA und die gespeicherte Modell-Pickle durch SHA256 geprüft.

## Archiv unabhängig prüfen

```powershell
.\BA_ARCHIV_PRUEFEN_V54.ps1 -ZipPath results\V54_ENTWICKLUNGSSTAND_ARCHIV_2026-11-12.zip
```

Erwartet: `PASS: ARCHIVE VERIFIED` samt Datei-Anzahl und ZIP-SHA256.

## Was das ZIP enthält

V5.2-Codezweig, V5.3-/V5.4-Prüf-Overlays, 4 Original-CV-JSONs und CV-Zusammenfassung, V5.4-Auditberichte und das auf Windows erzeugte Entwicklungsmodell `MODEL_8L4U_RUN01` mit Manifest, sowie SHA256-Metadaten für **jede einzelne Datei**. Keine Eingangs-Pickle, kein historisches Testlabel, kein neues Training.

## Methodische und technische Grenzen

* Das Modell wurde schon vor diesem Schritt auf Entwicklung L8+U4 trainiert. Das Archiv bedeutet **keine** unabhängige Modellvalidierung.
* `PROPOSED_MODEL_FREEZE.json` darf nicht geändert werden: Ihre SHA256 ist mit V5.2-Release-Checks hart verknüpft. Eine spätere formale Genehmigung erfolgt separat und signiert/freigegeben, nicht durch Überschreiben der Konfiguration.
* Windows-Symlink-Negativtest lief nur simuliert (`WinError 1314`), nicht nativ.
* SHA256 schützt Datei-Integrität, ist aber kein Vertrauensnachweis gegen einen vollständig kontrollierten Datensatz/Host.
* Zusätzliche Konvergenz, externe Generalisierung, Kalibrierung prädiktiver Unsicherheit und ein positiver Praxistest mit neuem Ring sind offen.
* ZIP-Kopien mit Modell-Pickle nur vertrauenswürdig verwahren und **nicht** aus fremden Quellen deserialisieren.
* Das Archivieren erzeugt keine „final train“-Version; der technische Modell-Freeze bedarf einer ausdrücklichen Freigabeentscheidung.

## Herkunft

Die unten dokumentierten Fingerabdrücke der Windows-Modell-Pickle und des Vorhersage-Replays stammen aus dem vom Nutzer übermittelten PowerShell-Protokoll vom 12.11.2026. Die Binärdateien liegen hier nicht vor und wurden in diesem Chat nicht erneut ausgeführt. Das Archivierungsskript wird erst auf dem Windows-Rechner die vollständigen Hashes real nachprüfen.
