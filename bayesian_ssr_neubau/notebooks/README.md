# Das Demonstrationsnotebook selbst ausführen und verstehen

Öffne **[Bayesian_SSR_HMM_Demonstration.ipynb](Bayesian_SSR_HMM_Demonstration.ipynb)**.
Es führt das vorhandene Bayesian-SSR-HMM Schritt für Schritt aus: Daten ansehen,
Rechnungen prüfen, ein Modell trainieren, Vorhersagen erzeugen und deren Fehler
an zurückgehaltenen Ringen messen. Die Berechnungen stehen direkt zwischen den
Erklärungen. Du musst die Python-Befehle zum Ausführen nicht verändern.

Das Modell schätzt zwei Qualitätswerte eines Rings: die Außenunrundheit und die
gespeicherte Planaritätshöhenabweichung. Beide werden in Millimetern ausgegeben.
Als Eingabe verwendet es den zeitlichen Verlauf der Loggerkanäle während des
ausgewählten Prozessfensters. Die Bedeutung und Grenzen der Zielgrößen stehen
in [DATENPROVENIENZ.md](../docs/DATENPROVENIENZ.md).

## 1. Start in VS Code

1. Öffne den gesamten Projektordner `Implementierung_BA` in VS Code und darin
   das oben verlinkte Notebook.
2. Klicke oben rechts auf **Kernel auswählen / Select Kernel**. Wähle den
   Python-Interpreter aus der Projektumgebung `.venv`, in diesem Projekt
   Python 3.11. Der Kernel ist der Python-Prozess, der die Notebook-Befehle ausführt.
3. Wähle **Kernel neu starten / Restart Kernel**. Damit beginnt Python ohne
   Variablen und Modelle aus einem vorherigen Versuch.
4. Wähle **Alle ausführen / Run All**. Die Codezellen werden von oben nach unten
   abgearbeitet. Lass das Notebook währenddessen geöffnet.
5. Lies die Ausgaben unter den jeweiligen Zellen. Der vollständige Lauf endet
   mit der Übersicht in Abschnitt 12 und der Meldung **Alle Schritte abgeschlossen**.

Die versionierte Notebookvorlage enthält absichtlich keine alten Ergebnisse.
Leere Ausgabefelder vor dem Start bedeuten daher nicht, dass etwas fehlt. Sobald
du eine Codezelle ausführst, erscheinen dort die für deinen Lauf berechneten Werte.

### Was sind Zellen und Ausführungsnummern?

Ein Notebook besteht aus einzelnen Zellen. **Textzellen** erklären den nächsten
Schritt; sie rechnen nichts aus. **Codezellen** enthalten die Python-Befehle und
haben eine Schaltfläche zum Ausführen. Tabellen, Meldungen und Diagramme unter
einer Codezelle sind deren Ausgaben.

Während Python rechnet, zeigt VS Code einen laufenden Zustand an, häufig einen
Stern `[*]` oder eine Fortschrittsanzeige. Nach dem Ende erscheint eine
Ausführungsnummer wie `[1]`. Diese Nummer zählt die Ausführungen im aktuellen
Kernel und ist keine Testbewertung. Ein leerer Codeblock ist noch nicht gelaufen.

Die Reihenfolge ist wichtig: Das Modell aus Abschnitt 6 wird in späteren
Abschnitten weiterverwendet. Wenn du direkt eine spätere Zelle startest, fehlen
möglicherweise vorher erzeugte Variablen. Bei einem `NameError` deshalb zuerst
prüfen, ob alle vorherigen Codezellen im selben Kernel ausgeführt wurden.

## 2. Was vorher vorhanden sein muss

Die Projektumgebung braucht das installierte Paket und die Notebook-Pakete.
Falls die Umgebung noch nicht eingerichtet ist, führe vom Repository-Root aus
im PowerShell-Terminal aus:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\bayesian_ssr_neubau[audit,notebook]"
```

Die vollständige Demonstration benötigt außerdem die private Rohdatei
`input/DM_16_aktive_Ringe.pkl`. Sie bleibt lokal und wird nicht von GitHub
heruntergeladen. Liegt sie woanders, kann vor dem Start von VS Code bzw. des
Notebook-Kernels der Datenpfad gesetzt werden:

```powershell
$env:BAYESIAN_SSR_DATA = 'C:\Daten\DM_16_aktive_Ringe.pkl'
```

Die Variable muss im gestarteten Kernel ankommen. Eine nachträglich in einem
anderen Terminal gesetzte Variable ändert einen bereits laufenden Kernel nicht.
Am einfachsten ist der vorgesehene lokale `input/`-Ordner. Bei Verwendung des
Terminals kann die Variable unmittelbar vor dem weiter unten gezeigten
Ausführungsbefehl gesetzt werden.

Das Notebook prüft den SHA256 der Rohdatei. Ein SHA256 ist ein aus den Dateibytes
berechneter Fingerabdruck: Eine andere oder veränderte Datei wird vor dem
Einlesen abgelehnt. Diese Prüfung bestätigt die erwartete Datei, nicht die
wissenschaftliche Qualität ihrer Messungen.

## 3. Der Ablauf in einfachen Worten

Die Abschnittsnummern beziehen sich auf die Überschriften im Notebook.

| Abschnitt | Was passiert? | Was kannst du daran erkennen? |
|---|---|---|
| Vorbereitung und 1 | Python, Datenpfad, Paketversionen und aktuelle Modellquellen werden geprüft. | Der Lauf verwendet die vorgesehene Umgebung und die zum Quellenmanifest passenden Dateien. |
| 2 | Die Loggerdaten werden gelesen und die erlaubten Ringgruppen angezeigt. Ein Prozessverlauf wird gezeichnet. | Welche Ringe eingehen, wie viele Zeitpunkte sie haben und welche sechs Kanäle verwendet werden. |
| 3 | Bei einer sehr kurzen künstlichen Zustandskette werden alle möglichen Zustandsfolgen durchgerechnet. | Die schnelle Momentenrechnung stimmt in diesem überschaubaren Beispiel mit einer direkten Rechnung überein. |
| 4 | Die automatisierten Tests laufen tatsächlich erneut. | Welche Prüfungen in deinem aktuellen Lauf bestehen oder fehlschlagen. |
| 5 | Die festgelegte Modellkonfiguration wird gezeigt. | Mit welchen Einstellungen das folgende Training ausgeführt wird. |
| 6 | Ein Modell wird frisch mit 8 Ringen mit Qualitätswerten und 4 Ringen ohne verwendete Qualitätswerte trainiert. | Dass ein vollständiger Trainingslauf ausgeführt und ein Modell gespeichert wird. |
| 7 | Trainingskurve und einzelne Rechenschritte werden zusammengefasst. | Wie sich die Trainingszielfunktion während der 30 Epochen entwickelt. |
| 8 | Eine Logger-CSV wird vorhergesagt; dieselbe Vorhersage wird nach Speichern und Laden wiederholt. | Dass die Vorhersageschnittstelle und die Speicherung zusammen funktionieren. |
| 9 | Zustandswahrscheinlichkeiten und die zeitliche Reststruktur der Trainingsdaten werden dargestellt. | Welche Zustandsverläufe das Modell annimmt und wo seine Annahmen möglicherweise nicht gut passen. |
| 10 | Die Ringe werden viermal in Trainings- und Bewertungsgruppen aufgeteilt. Fünf feste Verfahren werden verglichen. | Dass bewertete Ringe im jeweiligen Fit zurückgehalten werden. |
| 11 | Vorhersagefehler, Messwerte und Vorhersagen der zurückgehaltenen Ringe werden gezeigt. | Wie gut die Verfahren auf diesen Entwicklungsringen vorhersagen. |
| 12 | Die Ergebnisse und Prüfungen des aktuellen Laufs werden zusammengefasst. | Ob der vollständige Ablauf beendet wurde und wo seine Dateien liegen. |

**L** bedeutet hier: Ein Ring besitzt Qualitätswerte, die im jeweiligen Training
verwendet werden dürfen. **U** bedeutet: Es werden seine Prozessdaten verwendet,
aber keine Qualitätslabels an das Training weitergegeben. Die vier zusätzlich
vorhandenen historischen Außenringe bleiben gesperrt und werden nicht neu bewertet.

## 4. Die wichtigsten Ausgaben richtig lesen

### Tests und Fehlermeldungen

Eine pytest-Ausgabe wie `107 passed` heißt, dass 107 automatisierte Prüfungen
bestanden haben. Entscheidend ist die tatsächlich angezeigte Ausgabe deines
Laufs, nicht eine vorab erwartete Zahl. Im vollständigen Notebook müssen die
Tests ohne Fehler und ohne übersprungene Tests enden, weil die echten Rohdaten
hier bereits vorausgesetzt werden. Ein roter Python-Fehler beendet den Ablauf;
die folgenden Zellen liefern dann noch kein vollständiges Ergebnis.

Bestandene Tests prüfen konkrete Verträge und bekannte Vergleichsfälle. Sie
belegen nicht automatisch, dass jede zukünftige Vorhersage genau sein wird.
Die Bewertung an zurückgehaltenen Ringen erfolgt deshalb zusätzlich.

### Trainingskurve und Epochen

Eine **Epoche** ist ein vollständiger Trainingsdurchgang über die vorgesehenen
Aktualisierungen der Modellfaktoren. Es werden 30 Epochen ausgeführt. Die Kurve
`J_C` ist die gewichtete Trainingszielfunktion, an der sich die Optimierung
orientiert. Sie darf negativ sein; ein negativer Wert ist kein Programmfehler.
Bei unverändertem Daten- und Modellvertrag ist ein größerer Wert für dieses
Trainingsziel günstiger. Daraus folgt noch keine kleinere Vorhersageabweichung.

Kleine negative Rundungsreste einzelner Aktualisierungen bis zur festgelegten
Toleranz sind erlaubt. Eine relevante Verletzung dieser Rechenbedingung führt
zum Abbruch. Auch eine saubere Trainingskurve beweist kein global bestes Modell.

### Vorhersage und Standardabweichung

Die Tabelle in Abschnitt 8 zeigt einen geschätzten Qualitätswert und eine
**marginale Standardabweichung (SD)** für jedes Ziel in mm. Die SD beschreibt
die vom angepassten Modell berechnete Streuung. Sie ist weder der gemessene
Vorhersagefehler dieses Rings noch ein garantiertes 95-%-Intervall.

Der dort verwendete Ring DM04 war beim vollständigen Training dabei. Dieses
Beispiel zeigt daher, ob Einlesen, Vorhersagen und Speichern funktionieren.
Für die Vorhersagebewertung sind die zurückgehaltenen Ringe in Abschnitt 11
maßgeblich.

### Zustände und Farben

Ein HMM nimmt verborgene Zustände an, um den Prozessverlauf zu beschreiben.
Die Kurven in Abschnitt 9 zeigen für jeden Zeitpunkt Wahrscheinlichkeiten dieser
Zustände. Eine Wahrscheinlichkeit nahe 1 bedeutet eine starke Zuordnung im
verwendeten Modell. Die Bezeichnungen Z1 und Z2 sind keine automatisch bewiesenen
physikalischen Prozessphasen und nicht automatisch identisch mit den Loggerphasen.

### Vergleichsverfahren und Vorhersagefehler

Die **Cross-Validation (CV)** in Abschnitt 10 hält in jedem Durchlauf zwei der
acht L-Ringe vollständig zurück und trainiert mit den anderen sechs L-Ringen.
Jeder der acht Ringe wird so einmal pro Verfahren vorhergesagt, ohne in diesem
Fit zum Training oder zur Skalierung verwendet worden zu sein. Diese Vorhersagen
heißen **Out-of-Fold (OOF)**. Die vier U-Ringe dürfen je nach Verfahren die
vorgesehenen Prozessanteile beitragen.

Der TRAIN-Mittelwert ist ein einfacher Vergleich: für neue Ringe stets den
mittleren Qualitätswert der jeweiligen Trainingsgruppe vorhersagen. Die
physikalische NIG-Regression und die drei HMM-Arme sind weitere festgelegte
Vergleichsverfahren. Das Notebook sucht keine neuen Hyperparameter.

**MAE** ist der mittlere absolute Abstand zwischen Vorhersage und Messwert.
**RMSE** gewichtet große Abweichungen stärker. Beide stehen in mm; kleinere
Werte bedeuten bei demselben Ziel und denselben bewerteten Ringen kleinere
beobachtete Fehler. Die beiden Qualitätsziele sollten getrennt beurteilt werden.
In den Balkendiagrammen sind kleinere Balken entsprechend günstiger.

Im Messwert-Vorhersage-Diagramm wäre eine perfekte Vorhersage auf der gestrichelten
Diagonalen. Je weiter ein Punkt davon entfernt liegt, desto größer ist seine
Abweichung. Die Ringbeschriftungen helfen, einzelne auffällige Fälle wiederzufinden.

Die acht Ringe wurden historisch bereits zur Entwicklung verwendet. Die CV ist
deshalb ein Vergleich innerhalb dieses Entwicklungsbestands und keine neue
unabhängige Prüfung an unbekannten Produktionsdaten. Gute CV-Werte sind ein
interessanter Befund, aber kein allgemeines Güteversprechen.

## 5. Laufzeit, Unterbrechen und erneut starten

Für einen ersten Überblick kannst du nach einem Kernel-Neustart die Codezellen
von oben bis einschließlich Abschnitt 4 einzeln ausführen. Damit siehst du Daten,
eine Prozessgrafik, den kurzen Mathematikcheck und die Tests. Training und
Vorhersagebewertung sind damit noch nicht ausgeführt.

Der vollständige Lauf enthält danach ein Haupttraining mit 30 Epochen. In der
CV kommen zwölf weitere HMM-Trainings mit jeweils 30 Epochen hinzu: drei
HMM-Verfahren in vier Durchläufen. Deshalb dauert die CV deutlich länger als
der reine Testlauf. Die tatsächliche Dauer hängt vom Rechner ab. Der Fortschritt
wird unter anderem mit Epochen sowie Fold-/Verfahrensmeldungen ausgegeben.

Mit **Ausführung unterbrechen / Interrupt** kannst du die laufende Berechnung
anhalten. Ein abgebrochener Lauf ist kein vollständiger Funktionsnachweis.
Für einen sauberen neuen Gesamtlauf den Kernel neu starten und wieder alle
Zellen ausführen. Das Notebook erzeugt dann eine neue Lauf-ID. Vorhandene
Laufordner werden nicht überschrieben; ein automatisches Fortsetzen aus
abgebrochenen Checkpoints ist hier nicht vorgesehen.

## 6. Wo liegen die Ergebnisse?

Das Notebook gibt am Anfang und am Ende seinen aktuellen Laufordner aus:

```text
bayesian_ssr_neubau/outputs/notebook_demos/<lauf_id>/
```

In diesem Ordner liegen unter anderem:

- `figures/`: die Diagramme zusätzlich als PNG und SVG.
- `model_8L4U/`: das frisch trainierte Modell mit Konfiguration, Prüfungen und
  Trainingsprotokollen.
- `development_cv/`: die Ergebnisse der vierfachen Entwicklungsevaluation.
- `cv_metrics.csv` und `ssl_out_of_fold_predictions.csv`: Fehlertabelle und
  einzelne OOF-Vorhersagen zum späteren Nachlesen.
- `notebook_summary.json`: Zusammenfassung, die erst nach dem vollständigen
  erfolgreichen Durchlauf geschrieben wird.

Neue Rechenergebnisse und Rohdaten bleiben lokal. Sie werden nicht mit Git
versioniert. Die Quelle `SOURCE_SHA256.json` prüft die vorgesehenen Modellquellen
und Konfigurationen; für die Referenztests sind die benötigten alten Module
bereits unter `tests/reference_parity/fixtures/legacy/` enthalten.

## 7. Häufige Probleme einordnen

| Beobachtung | Was du prüfen solltest |
|---|---|
| `Private Rohdaten fehlen` | Liegt die freigegebene Datei in `input/`, oder kennt der Kernel `BAYESIAN_SSR_DATA`? Ein Git-Klon enthält die private Datei nicht. |
| `ModuleNotFoundError` | Ist der Kernel wirklich die Projekt-`.venv`, und wurden das Paket sowie die Notebook-Pakete dort installiert? |
| `NameError` bei einer späteren Zelle | Wurden alle vorherigen Codezellen seit dem letzten Kernel-Neustart ausgeführt? |
| Abweichender Datenhash | Prüfe, ob du die richtige unveränderte Rohdatei ausgewählt hast. Die Prüfung nicht entfernen, um eine andere Datei einzulesen. |
| Quellenmanifest weicht ab | Modellquellen oder Konfigurationen passen nicht zum Manifest. Änderungen zuerst vergleichen und bewusst prüfen; die Kontrolle nicht einfach überspringen. |
| Eine Zelle rechnet lange ohne neue Tabelle | In Abschnitt 6 oder 10 kann ein Training laufen. Fortschrittsmeldungen und die laufende Zellanzeige beachten. |
| Nach einem Abbruch fehlt die Abschlussübersicht | Der Gesamtlauf ist unvollständig. Für einen neuen Nachweis Kernel neu starten und von oben ausführen. |

## 8. Alternativ vollständig aus dem Terminal ausführen

Du kannst statt der manuellen Bedienung auch den vorgesehenen Helfer verwenden:

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\execute_notebook.py
```

Er verwendet genau den aufrufenden Python-Interpreter und eine vorübergehende
lokale Kernelspezifikation. Die ausgeführte Notebookkopie und ihr Erfolgs- oder
Fehlerbericht liegen unter `outputs/notebook_executions/<id>/`; die versionierte
Vorlage bleibt dabei unverändert. Modelle und Diagramme werden weiterhin im
zugehörigen `outputs/notebook_demos/<lauf_id>/` gespeichert.

Bei manueller Ausführung in VS Code kannst du Ausgaben im geöffneten Notebook
speichern. Vor einem Commit die Ausgaben wieder leeren, damit private Laufdaten
nicht Bestandteil der versionierten Vorlage werden.

`build_notebook.py` ist nur für die Pflege der Notebookvorlage nötig: Es erzeugt
sie aus den Text- und Codezellen im Generator und überschreibt absichtlich keine
vorhandene Datei. Für die normale Benutzung öffnest du direkt die `.ipynb`-Datei.
