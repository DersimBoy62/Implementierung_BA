# Bayesian Bachelor – vorläufige Hauptmodell-Pipeline V5

**Stand:** 12.11.2026. **Status:** `PROVISIONAL_USER_APPROVAL_PENDING` – technisch getestet, aber **nicht endgültig unabhängig validiert**. Die Ergebnisse aus mehrfach verwendeten acht Entwicklungsringen sind explorativ. Dieser Code ändert den bereits abgeschlossenen Test vom 11.11.2026 **nicht** und darf nicht auf dessen vier früher geöffneten Testlabels optimiert werden.

## 1. Wissenschaftliche Entscheidung

**Vorgeschlagener Hauptpfad:** volle Gaussian-Wishart-Prozess-HMM-Emission, `K=2`, dynamische Qualitätskopplung `M2`, Prozessskalierung `P0`, sechs Kanäle `V1_Kern_6`, zusammenhängendes vollständiges `F2`-Prozessfenster, `C=0.2`, physische NIG-Priorvarianz `0.38526108747593474`, zwei separat parametrisierte Bayesian-NIG-Qualitätsköpfe, 10 Epochen, 8 lokale VI-Schritte, Seed 20261111.

`C=0.2` ist keine als global optimal bestätigte Konstante. Bei K2 mit `C=0.1` waren die Mittelwerte über drei Entwicklungssplits bei **beiden** Qualitäts-RMSE minimal kleiner; bei `C=0.2` wurden hingegen **beide** Ziele gegenüber `C=0` in **allen drei** untersuchten Aufteilungen verbessert. Diese Robustheitspriorisierung ist eine nachvollziehbare, aber post-hoc/entwicklungsspezifische Abwägung. **Alternative:** K2 C0.1. **Wichtiger abweichender Kandidat:** K3 C2 (optimierte Außen-RMSE auf den wiederverwendeten Entwicklungsringen). In der Nested-CV werden K3 und C≈0.05–0.11 häufig bevorzugt; es gibt daher keine universell nachgewiesene optimale Konfiguration.

**Unverändert eingefrorenes Datenschema:**
- Gelabelte Entwicklungsringe: DM04, DM05, DM06, DM09, DM12, DM16, DM17, DM18.
- Ungelabelte Prozessringe, Qualitätslabels maskiert: DM01, DM02, DM10, DM14.
- Historische äußere Testringe – **nicht für dieses Modell**: DM07, DM08, DM11, DM15.
- Verbindliche Pickle-SHA256: `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`.
- Qualitätstargets: `ovality.ovality_radial_mm` (Außenunrundheit) und `planarity.height_deviation_mm` (Höhenschlag).

## 2. Ordner und Code-Abgrenzung

- `PROPOSED_MODEL_FREEZE.json`: vollständige, hashgesicherte Konfiguration mit vorläufigem Status.
- `main_bayesian_ssr_v1.py`: aufgeräumter Trainings-/Vorhersageeintrittspunkt mit maskiertem U, SHA-Prüfungen, Ring-ID-Schutz, getrennten Skalierern und Write-once-Ergebnisdateien.
- `test_main_bayesian_ssr_v1.py`: acht Vertragstests, ohne Zugriff auf die DM-Pickle-Datei.
- `final_core/`: **byteidentische Original-Mathematikmodule** des V3-Forschungsstandes; nicht mit dem neuen Wrapper gleichsetzen und nicht nachträglich verändern. Die Hashes werden beim Start geprüft.
- `sources/`: drei originale, unveränderte C-Dosis-Zusammenfassungen.
- `C_DOSE_VERIFIED_THREE_SPLITS_SUMMARY.json`, `.csv`, `figures/` und `FORSCHUNGSBERICHT_...docx`: wissenschaftliche Auswertung für Kapitel 4 und 5.

Der Code wurde im Container tatsächlich auf dem verifizierten Datenbestand ausgeführt: vollständiges Train-Fit über **ausschließlich 8 L + 4 U** (K2, C0.2, 10 Epochen) und anschließende erfolgreiche Prozessvorhersage nach Laden des persistierten Modells. Die QA-Prognose verwendete **nur Prozessdaten eines bereits im Training enthaltenen Rings** zum technischen Replay, ohne Qualitätsvergleich; sie ist kein unabhängiger Test. Die QA-Artefakte werden **nicht** als Trainingsgüte ausgewiesen und nicht in das veröffentlichte ZIP-Paket aufgenommen.

## 3. Windows/PowerShell – Projektablauf

Entpacke alle Paketdateien in einen **neuen Unterordner**, z. B. `Implementierung_BA\final_candidate_v5\`. Insbesondere `final_core/` muss unverändert neben dem Einstiegsskript liegen. Die bisherige Forschungsstruktur bleibt vollständig unberührt.

In deinem bestehenden Projektordner `Implementierung_BA`:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v5\test_main_bayesian_ssr_v1.py
```

Erwartung: `8 PROVISIONAL PIPELINE CONTRACT TESTS PASSED; NO RING PICKLE LOADED`.

**Kandidatenmodell anhand der 8 L + 4 U-Ringe trainieren** – ohne alte Testringe:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v5\main_bayesian_ssr_v1.py train `
  --data input\DM_16_aktive_Ringe.pkl `
  --out results\final_candidate_v5\TRAIN_DEV_L8_U4_V1
```

Ausgabe: `TRAINED_MODEL.pkl` (nur lokal selbst erzeugte Artefakte laden!) und `TRAIN_MANIFEST.json` mit Modellhash, Datenhash, Quellenhashes und Trainingsmetadaten. Bei vorhandenem Ergebnisordner nicht überschreiben; für neue Fits neue Ausgabepfade verwenden.

**Für eine tatsächlich neue Ringmessung** eine CSV mit folgenden *exakt diesen* Spalten und vollständig durchgeführtem Prozess bereitstellen:

```text
time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin
```

Rohzeitachse in aufsteigenden Sekunden und 6 Prozesskanäle mit denselben Einheiten wie beim Training; `phase804`, `LaserOK` und `MessEin` werden zur Auswahl des F2-Zeitfensters verwendet. In diesem Modell muss das F2-Fenster zusammenhängend sein. **Keine nachgelagerten Qualitätsmessungen hinzufügen**. Dann:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v5\main_bayesian_ssr_v1.py predict `
  --model-dir results\final_candidate_v5\TRAIN_DEV_L8_U4_V1 `
  --ring-id NEU01 `
  --csv input\NEU01_process_logger.csv `
  --out results\final_candidate_v5\NEU01_VORHERSAGE.json
```

Erzeugt zwei erwartete Qualitätskennwerte in mm und deren **prädiktive Standardabweichungen**; diese sind ohne neue Kalibrationsstudie keine garantierten Konfidenzintervalle. Bei numerisch problematischem Eingang wird die Prognose abgebrochen statt ein scheinbar gültiges Resultat ausgegeben.

## 4. Forschungs- und BA-Einordnung

**Kapitel 4:** F2-Zeitreihen, P0-Skalierung, Markov-VI, Modellarchitektur, drei C-Kurven (zwei Zielgrößen, drei Aufteilungen), Nested-CV, Optuna-Vergleiche, kontrollierte U-Ablationen, ringweise Kennzahlen aus den vorhandenen Einzelberichten; Vorhersageunsicherheiten vorsichtig berichten.

**Kapitel 5:** Zielkonflikt Unrundheit/Höhenschlag, wechselnd günstiges C, kleine Datenmenge, potentielle Instabilität durch einzelne Ringe, wiederverwendete Entwicklungslabels, Optimismus aus Modellwahl, K2/C0.2 als Robustheitsentscheidung, Alternativen K2/C0.1 und K3/C2 und der Unterschied zwischen Prognose und tatsächlich **nachgewiesener** Prozessoptimierung.

**Offene wissenschaftliche Freigaben:** Bedeutung des Loggerkanals `RWG55` im Original prüfen. Frühere Deskriptoren nennen Mittelwert und SD dieses Kanals `radial_speed` – semantisch nicht ohne Sensordokumentation gesichert. Überwache die Ringqualität ausschließlich über die beiden zugelassenen Zielgrößen. Externe Validierung benötigt neue, bisher unberührte Ringe. Die früher geöffneten vier äußeren Testlabels dürfen nicht zur Modellwahl genutzt werden.

## 5. Quellen- und Prüfstatus

Primärdaten: die drei beigefügten `C_DOSE_SUMMARY_*.json` wurden vollständig gelesen, inklusive gepaarter Verbesserungszählungen und Leave-one-ring-out-Vorzeichenprüfung. Diese Summary-Dateien enthalten **keine einzelnen C-Dosis-Vorhersagen je Ring**; für eine vollständige 10×3×2 ringweise C-Analyse müssten die ursprünglichen Trial-Berichte zusätzlich hochgeladen werden. Zur Nested-CV sind getrennt die originalen `OUTER_PREDICTIONS.csv`, `INNER_SELECTED_BY_OUTER_FOLD.csv` und `NESTED_SUMMARY.json` geprüft worden.

Zitierte methodische Originale in früheren Forschungsnotizen: Varma & Simon (2006); Cawley & Talbot (2010). Die modellbezogenen Original-PDFs Shao et al. (2022), Mowbray et al. (2022) und Herleitungen A–D wurden **für dieses Dokument nicht neu seitenweise geprüft**. Keine ungeprüften Seitenzahlen übernehmen.

## 6. Optionale Reproduktion der Word-Auswertung

Im neuen Paket-Ordner, falls `python-docx`, `matplotlib`, `numpy` vorhanden:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v5\build_report.py
```

Das reproduziert das Diagramm-/Dokumentenmaterial anhand der drei mitgelieferten unveränderten JSON-Dateien, ohne den Ringdaten-Pickle zu öffnen. Achtung: Der Bericht wird dabei im Paketordner aktualisiert; Forschungs-Originale weiterhin separat archivieren.
