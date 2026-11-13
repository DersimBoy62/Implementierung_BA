# Jupyter-Demonstration des unveränderten Basismodells

Öffne `Bayesian_SSR_HMM_Demonstration.ipynb` in VS Code und wähle oben rechts
den Python-Kernel der Projektumgebung `.venv`. Das Notebook wird mit tatsächlichen
Ausgaben geliefert. Mit **Restart Kernel → Run All** läuft die komplette
Demonstration erneut in einem neuen Ausgabeordner.

Das Notebook führt aus:

1. Umgebungs-, Referenzhash- und Datenaudit einschließlich L/U-Trennung.
2. Visualisierung einer vollständigen F2-Prozesssequenz und einen kurzen
   unabhängigen Pfadenumerationscheck der Markov-Mathematik.
3. Die vollständige pytest-Suite, mit neuem Exitcode, Log und JUnit-Bericht.
4. Ein frisches 30-Epochen-Training auf 8L+4U mit Blockbilanzen und Checkpoints.
5. Speichern, Laden und identische Prozess-only-Prediction aus einer Logger-CSV.
6. J_C-/Zustands-/Residualdiagramme mit ausdrücklich bezeichnetem Diagnoseumfang.
7. Eine neue vollständige vierfache Ring-CV mit fünf festgelegten Armen,
   Fehlertabelle und Out-of-Fold-Diagrammen für beide Qualitätsziele.
8. Ein Laufprotokoll und Hashmanifest aller neuen Artefakte.

Der vollständige CV-Teil trainiert zwölf zusätzliche HMMs über je 30 Epochen
und dauert entsprechend länger. Er ist standardmäßig aktiv und lädt keine alten
Ergebnisse als Ersatz. Das Modell wird dabei nicht optimiert oder verändert.
Die vier historisch geöffneten Außenringe bleiben gesperrt.

Die CSV-Vorhersage an DM04 ist ein Schnittstellenbeispiel an einem Trainingsring.
Für Vorhersagequalität werden ausschließlich die neuen VAL-/Out-of-Fold-
Ergebnisse betrachtet. Prädiktive SD sind unkalibrierte marginale
Momentenunsicherheit, keine zugesicherten 95%-Intervalle.

Alle Befehle vom Workspace-Root `Implementierung_BA` aus:

```powershell
.\.venv\Scripts\python.exe -m pip install -r bayesian_ssr_neubau\notebooks\requirements.txt
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\execute_notebook.py
```

`execute_notebook.py` verwendet exakt seinen aufrufenden Python-Interpreter und
eine vorübergehende lokale Kernelspezifikation. Es ist keine globale
Kernelregistrierung nötig. Das Notebook wird während der Ausführung regelmäßig
mit den bisher eingegangenen Ausgaben gespeichert. Ein neuer
`*.execution_<id>.json`-Bericht dokumentiert Erfolg oder Fehler und den Hash der
ausgeführten Notebookdatei.

Neue Rechenartefakte liegen in
`bayesian_ssr_neubau/outputs/notebook_demos/<neue_lauf_id>/`.
Diagramme werden zusätzlich als PNG und SVG gespeichert. Modell- und
Referenzquellen sowie vorhandene Laufdateien werden nicht überschrieben.

`build_notebook.py` erzeugt das Notebook aus lesbaren Zellquellen und verweigert
das Überschreiben einer vorhandenen Datei. Es ist für die normale Bedienung
nicht erforderlich. Beispiel für eine neue Notebookkopie:

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\notebooks\build_notebook.py --out bayesian_ssr_neubau\notebooks\Neue_Demonstration.ipynb
```
