# Jupyter-Demonstration des Basismodells

Öffne `Bayesian_SSR_HMM_Demonstration.ipynb` in VS Code, wähle den Python-Kernel
der Projektumgebung und führe **Restart Kernel → Run All** aus. Die versionierte
Vorlage enthält keine alten Ausgaben. Jeder Gesamtlauf berechnet die Ergebnisse
neu und schreibt sie in einen eigenen lokalen Ausgabeordner.

Voraussetzungen sind das installierte Paket, die Notebook-Abhängigkeiten und die
freigegebenen privaten Rohdaten. Lege `DM_16_aktive_Ringe.pkl` im `input/`-Ordner
des Repositorys ab. Alternativ kann `BAYESIAN_SSR_DATA` auf die Datei zeigen.
Ihr SHA256 wird vor dem Einlesen geprüft. Die Rohdaten werden nicht mit Git
versioniert; ohne sie kann die vollständige Demonstration nicht ausgeführt werden.

Alle Befehle vom Repository-Root aus:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\bayesian_ssr_neubau[audit,notebook]"
# Optional, falls die Datei an einem anderen Ort liegt:
# $env:BAYESIAN_SSR_DATA = 'C:\Daten\DM_16_aktive_Ringe.pkl'
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\execute_notebook.py
```

Das Notebook führt aus:

1. Umgebungs-, Quellenhash- und Datenaudit einschließlich L/U-Trennung.
2. Visualisierung einer vollständigen F2-Prozesssequenz und einen unabhängigen
   Pfadenumerationscheck der Markov-Mathematik.
3. Die vollständige pytest-Suite mit neuem Exitcode, Log und JUnit-Bericht.
4. Ein frisches 30-Epochen-Training auf 8L+4U mit Blockbilanzen und Checkpoints.
5. Speichern, Laden und identische Prozess-only-Prediction aus einer Logger-CSV.
6. J_C-, Zustands- und Residualdiagramme mit bezeichnetem Diagnoseumfang.
7. Eine vollständige vierfache Ring-CV mit fünf festgelegten Armen,
   Fehlertabelle und Out-of-Fold-Diagrammen für beide Qualitätsziele.
8. Ein Laufprotokoll und Hashmanifest aller neuen Artefakte.

`SOURCE_SHA256.json` prüft die versionierten Modellquellen und Konfigurationen.
Für die mathematischen Vergleichstests liegen die benötigten Referenzquellen
unter `tests/reference_parity/fixtures/legacy/`. Historische Projektordner,
Übergaben und gespeicherte Trainingsläufe werden nicht benötigt.

Der CV-Teil trainiert zwölf zusätzliche HMMs über je 30 Epochen und dauert
entsprechend länger. Er ist standardmäßig aktiv und lädt keine alten Ergebnisse
als Ersatz. Die vier historisch geöffneten Außenringe bleiben gesperrt.

Die CSV-Vorhersage an DM04 ist ein Schnittstellenbeispiel an einem Trainingsring.
Für Vorhersagequalität werden ausschließlich die neuen VAL-/Out-of-Fold-
Ergebnisse betrachtet. Prädiktive SD sind unkalibrierte marginale
Momentenunsicherheit, keine zugesicherten 95%-Intervalle. Die acht
Entwicklungslabels erlauben keine unabhängige externe Validierungsbehauptung.

`execute_notebook.py` verwendet seinen aufrufenden Python-Interpreter und eine
vorübergehende lokale Kernelspezifikation. Die ausgeführte Notebookkopie und ihr
Erfolgs-/Fehlerbericht liegen unter
`outputs/notebook_executions/<id>/`; die versionierte Vorlage bleibt unverändert.
Modell, CV und Diagramme liegen unter `outputs/notebook_demos/<lauf_id>/`.
Diagramme werden zusätzlich als PNG und SVG gespeichert. Vorhandene Laufdateien
werden nicht überschrieben. Beide Ausgabeverzeichnisse bleiben lokal.

Bei manueller Ausführung in VS Code können Ausgaben im geöffneten Notebook
gespeichert werden. Vor einem Commit die Notebook-Ausgaben wieder leeren, damit
die Vorlage weiter ohne private Laufdaten versioniert bleibt.

`build_notebook.py` erzeugt das Notebook aus lesbaren Zellquellen und verweigert
das Überschreiben einer vorhandenen Datei. Es ist für die normale Bedienung
nicht erforderlich. Beispiel für eine neue Notebookkopie:

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\build_notebook.py --out bayesian_ssr_neubau\notebooks\Neue_Demonstration.ipynb
```
