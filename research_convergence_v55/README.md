# V5.5 – Getrennte Konvergenzdiagnose (Forschung, keine Modelländerung)

**Ziel**: Die noch offene Frage prüfen, ob zehn Epochen für den fest registrierten K=2/C=0,2/P0/M2-Hauptkandidaten ausreichen. Mit genau denselben Trainingsringen und identischer Initialisierung wird das mathematisch unveränderte `dynamic_quality_v2.train` für mehr Epochen aufgerufen. Diese zusätzlichen Fits sind **Entwicklungs-/Optimierungsdiagnostik**, keine finalen Qualitätsmodelle. Es werden weder äußere Tests noch CV-Validierungswerte betrachtet.

Ordner `research_convergence_v55` direkt in `Implementierung_BA` entpacken. Bitte `final_candidate_v52` und `MODEL_8L4U_RUN01` nicht verändern.

```powershell
.\.venv\Scripts\python.exe .\research_convergence_v55\convergence_probe_v55.py self-test
.\.venv\Scripts\python.exe .\research_convergence_v55\convergence_probe_v55.py preflight --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01
New-Item -ItemType Directory -Force results\v55_convergence_trainonly | Out-Null
.\.venv\Scripts\python.exe .\research_convergence_v55\convergence_probe_v55.py run --data input\DM_16_aktive_Ringe.pkl --model-dir results\v54_development_replay\MODEL_8L4U_RUN01 --epochs 30 --out results\v55_convergence_trainonly\CONVERGENCE_30E_TRAINONLY_01.json
```

**Sicherheit**: Das JSON wird nur einmal geschrieben, der ursprüngliche Modell-/Konfigurations-Hash muss stimmen; das V5.2-Pickle darf nur aus eigener vertrauenswürdiger lokaler Erzeugung geladen werden. Das originale DM-Pickle enthält die historischen vier Outer-Ringe, aber die registrierte Development-View verwendet nur 8 L + 4 U; keine alten Testlabels werden ausgewertet. Das Skript verweigert unpassende Modell-Hashes und Quellhashes.

**Auswertung**: Alle ELBO pro Epoche, absolute/relative Verbesserungen, Zustandsanteile und die lokalen Gewinne werden gespeichert. Die ersten zehn ELBO-Werte werden gegen die gespeicherte ursprüngliche 10-Epochen-Historie geprüft (Abweichung maximal 1e-7). Der rein deskriptive Wert „letzte fünf relative Schritte unter 1e-4“ ist keine vorab bewiesene Konvergenzregel und keine Aussage über globale Optima. Ohne unabhängige Daten dürfen diese Läufe nicht zur Optimierung der Vorhersagequalität verwendet werden.

Es werden **keine** Quality-RMSEs, MAEs, Test- oder Validierungsprognosen berechnet. Falls die ersten zehn ELBO-Werte nicht reproduzierbar sind, wird abgebrochen, statt einen längeren Verlauf mit einem anderen Referenzlauf zu vermischen.
