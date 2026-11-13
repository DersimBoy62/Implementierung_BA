# BAYESIAN BACHELOR – konsolidierte Projektübernahme und Datei-Inventar

**Erhebungsstand: 12.11.2026.** Dieses Dokument fasst den fortsetzungsrelevanten historischen Projektstand zusammen und erfasst die in dieser Sitzung tatsächlich materialisierten ZIP- und Word-Dateien. Es ist **kein** wissenschaftlich peer-reviewter Audit, keine neue Generalisierungsstudie und keine Modell-Freigabe. Aussagen aus älteren Handovern werden bewusst als *historisch* gekennzeichnet, wenn sie inzwischen durch DM-Daten/V5.1 überholt sind.

## 1. Verbindliche Identität

- Forschungsfrage: vollständig Bayesian Semi-Supervised Regression aus ringweisen radial-axialen Prozesssequenzen, **ohne Pseudo-Labeling**.
- Prozessmodell: Hidden Markov Model mit Markov-strukturierter Variationsinferenz; Normal-Wishart/Gaussian-Wishart Prozess-Emissionen mit **voller Kovarianz**; gewichtete ELBO für unlabeled Prozessdaten.
- Zwei voneinander getrennte Bayesian-Normal-Inverse-Gamma Qualitätsköpfe für **Außenunrundheit** (`ovality.ovality_radial_mm`) und **Höhenschlag** (`planarity.height_deviation_mm`). Die ursprüngliche dritte Innenunrundheit wurde verworfen. Die grünen markierten Originalwerte in `all_targets` müssen für die endgültige Abgabe gegen die maßgebliche Excel-Version geprüft werden.
- Stand der aktiven Datenbasis: 16 DM-Ringe. DM03 (kein vollständiger Versuch/Logger) und DM13 (prozessual unbrauchbar) ausgeschieden. Historische SFA-Ringe und deren Proxy-Ziele sind **nicht** finale Modelleingaben.
- Ringdaten-SHA256 `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5` wurde in dieser Sitzung unabhängig an `DM_16_aktive_Ringe.pkl` verifiziert.
- Feste zulässige Modelltrainingsdaten: L = DM04, DM05, DM06, DM09, DM12, DM16, DM17, DM18; U = DM01, DM02, DM10, DM14, deren Qualitätsziele maskiert sind. Historischer äußerer Test DM07, DM08, DM11, DM15 wurde am 11.11. bereits geöffnet; nie wieder als unabhängigen Blindtest oder für die Modellwahl verwenden.

## 2. Chronologische Dokumentationskette

1. **September–Oktober 2026:** Recherche/Quellenkatalog, Projektarbeit, Bayesian-HMM/VI-Konzept, original Herleitungen A–D, erste SFA-Verarbeitungsentscheidungen. Der Schwerpunkt wurde nach Betreuervorgabe vom 26.09. auf fully Bayesian SSR ohne Teacher–Student-Hauptpfad gelegt. Ursprüngliche MASTER-HANDOVER-Abschnitte 1–62 und 62–115 sind als aus früheren Gesprächszusammenfassungen bekannte Vorläufer zu verstehen; ihre exakten separaten Ursprungsdateien wurden in dieser Sitzung noch nicht byteweise nachgewiesen.
2. **31.10.–05.11.:** 17 SFA-Zeitreihen, SFA16-Ausschluss, Modellkopplung, Kapitel-2/3-Word-Fassungen und 05.11. `Bayesian_Bachelor_ULTIMATIVE_Masteruebergabe_2026-11-05.docx`.
3. **06.–09.11.:** SFA-Proxymodell, Hilfslabels, Labels fehlen real, Diagnostik und Forschungsvarianten. Master-Nachtrag vom 06.11.; diese Ergebnisse historisch, nicht die heutige Hauptbasis.
4. **10.11.:** Übergang zu qualitativ gemessenen DM01–DM18, Annahme der zwei offiziellen Targets; Archivierung der alten SFA-Abschlussphase in `Bayesian_Bachelor_MASTER_HANDOVER_ABSCHLUSSPHASE_2026-11-10.docx`.
5. **11.11.:** Neuer DM-Modellkern, Merkmals-/Kovarianz-/Wishart-/Emissions-/Markov-/Prior- und Qualitätskopplungsvergleiche, getrennte NIG-Qualitätsköpfe und U-Ablationen. Einmalige, später als vorbelastet zu betrachtende äußere V1-Bewertung. Neuverfasste Herleitungen A–D samt Abgleichmatrix. Verbindliches `MASTER_HANDOVER_BAYESIAN_BACHELOR_DM_2026-11-11.docx`.
6. **11.11. V2:** Dynamische Pfadmerkmale M0–M3, Wechsel- und Late-Early-Momente; P0/P1/P2 Vorverarbeitung; 16 manuelle K-Raster-CV-Modelle. `MASTER_HANDOVER_BAYESIAN_BACHELOR_V2_FORTSETZUNG_2026-11-11.docx` ist **vor** Optuna datiert.
7. **11.–12.11. V3/V4:** 16 Structural- und 24 Extended-Optuna-Trials; 12 C/Prior-Faktorvergleiche; Strict vs Shared C=0 vs SSL; Seeds, Splits, U-Gewicht-C, 4×3-Nested-CV und ringweise Unsicherheit. `MASTER_HANDOVER_BAYESIAN_BACHELOR_V3_V4_FORTSETZUNG_2026-11-12.docx` wurde erstellt **vor** Abschluss der beiden alternativen C-Dosis-Splits.
8. **12.11. V5:** Drei C-Dosis-Kurven (canonical, 20261114, 20261115) sind im V5-Paket vorhanden und wurden dokumentiert. Vorschlag für robuste Konfiguration K2/P0/M2/C0,2/Priorvarianz 0,385261..., ohne behauptetes Optimum. Neuer `final_candidate_v5`-Wrapper und Code-Audit. Kein finales Model-Freeze.
9. **12.11. V5.1:** `final_candidate_v51`: neue Rohlogger-/Vollständigkeits-/ID-/Zeit-/Plausibilitäts-Gates, separate Completion-Bestätigung; gleiche 10 mathematische Kernmodule, keine neue Modellgüte.

## 3. Hauptmodell V5/V5.1 und Konfiguration

- `freeze_id`: `DM-V5-MAIN-PROPOSED-2026-11-12-v0`; Status `PROVISIONAL_USER_APPROVAL_PENDING`.
- Training nur 8 L + 4 maskierte U, Prozessfenster `F2` mit zusammenhängender aktiver Phase.
- X-Spalten: `F_rad63`, `F_ax64`, `Ist_D150`, `Ist_s151`, `Ist_h152`, `RWG55`. `RWG55` laut Kanalaudit Wachstumsrate m/s, nicht Ringhöhe; Originaldokumentation hierzu noch offen.
- P0-Skalierung: zeitpunktgepooltes X über L+U; physikalische Ringdeskriptoren ringgleichgewichtet mit separaten Scalern; Y-Standardisierung nur aus L. Nicht durcheinanderbringen.
- Late-minus-Early M2-Markov-Occupancy-Momente statt Viterbi-Pfadsurrogat. Zwei Qualitätsköpfe mit individuellen Deskriptorgruppen.
- K=2; C=0,2 in der U-Prozess-ELBO; `physical_var=0.38526108747593474`; 10 Epochen; 8 lokale VI-Schritte; Seed 20261111.
- Alternative Konfigurationen (unverändert archivieren): K2/C0,1 und K3/C2. K3/C2 mit Priorvarianz 0,473291536... aus anderer Optuna-Landschaft.
- V5.1 macht die alten V5-Pickle-Modelle bewusst nicht automatisch im regulären neuen Vorhersagemodus kompatibel.

## 4. Bewertung – stets wissenschaftliche Grenzen nennen

- V4 Nested-Optuna gepoolter RMSE in mm bei acht historisch oft genutzten Entwicklungsringen: Strict [0,6541; 3,1756], SSL [0,5453; 3,2100], Shared-C0 [0,5503; 3,2633] für [Außen; Höhe]. SSL vs Strict: Außen ca. 16,6 % RMSE besser; Höhe ca. 1,1 % schlechter. SSL vs Shared-C0 isoliert den zusätzlichen ELBO-U-Effekt stärker; kleine Verbesserung bei beiden RMSE, jedoch nicht für alle Ringe.
- Nicht behaupten, C=0 sei ein Strict-Supervised-System, da U auch bei gemeinsam berechneten X-Scalern, statischen Deskriptoren und K-Means die Pipeline beeinflusst.
- Nach mehrfacher manueller Modellwahl/Optuna kann Nested-CV auf genau denselben acht früher angesehenen Entwicklungsringen keine unabhängige Externbestätigung liefern.
- Die V1-Outer-Ringbewertung aus 11.11. ist historisch abgeschlossen und darf spätere V2–V5-Entscheidungen nicht motivieren.
- Modellprädiktion ist **End-of-F2**-Prognose, keine validierte Echtzeitregelung oder physikalische Prozessoptimierung.
- Prädiktive SD ist modellbasiert und nicht mit extern validierter 95%-Deckung gleichzusetzen.

## 5. In dieser Sitzung direkt ausgeführte Read-only-Prüfungen

- `DM_16_aktive_Ringe.pkl`: SHA256 stimmt exakt mit festgelegtem Datensatzhash überein.
- V5 / V5.1: 10 `final_core`-Math-Module stimmen in beiden Paketen mit allen 10 Einträgen in `PROPOSED_MODEL_FREEZE.json` SHA256-genau überein.
- `test_input_safety_v51.py`: **18/18 PASS**; ohne Pickle-/Train-/Outer-Label-Zugriff.
- `test_development_logger_equivalence_v51.py --data DM_16_aktive_Ringe.pkl`: **12/12 PASS** bitgenaue F2-Kompatibilität; nur erlaubte L/U-Entwicklungsprozesslogger. Keine neue Qualitätsevaluation und kein neuer Modellfit.
- Original-V5-Audit aus Bericht: Markov M2-Momente vs 32 Pfade bis 2,2e−16, Gradient finite difference bis 4,5e−10; diese alten numerischen Auditprüfungen sind hier aus dem Bericht übernommen, nicht erneut unabhängig getestet worden.

## 6. Offene technische Schritte gemäß aktuellstem V5.1-Bericht

**V5.2**: vollständiges ausführbares Konfigurationsschema; atomare Modellerstellung/Manifesttransaktion; Versionen von Python, NumPy, SciPy, sklearn, Numba und Hashes / Provenienz protokollieren. Unveränderte V5.1- und V5-Originale aufbewahren.

**V5.3**: HMM-/VI-Stabilität, Test numerischer Grenzfälle und gezielte Entwicklungskontrolle des gelabelten lokalen Feedbacks (z. B. local_steps 0 vs 8), weiterhin keinerlei Outer-Test-Reuse.

**V5.4**: nach fachlicher Versionsentscheidung und vollständigen Freigabeprüfungen offizielles Training mit L8+U4 und dokumentierter neuer-Ring-Prädiktion. Kein Anspruch auf unabhängige Prognosegüte ohne neue Ringe.

Offen außerdem: Quellenprüfung `RWG55`, grüne Original-Excel-Targetzellen, unabhängige externe Validierung, prädiktive Kalibrierung, reale physikalische Interpretation und Versuchsplan für neue Ringe.

## 7. Originalwissenschaft und Quellenstatus

Die historische `Quellenstatus_133_Literaturkarten.csv` erfasst 133 Literaturkarten. In der **historischen** Matrix sind 21 Originalvolltexte als identifiziert und 112 nicht identifiziert; nur 5 Einträge waren als „Methodenkern/Original geprüft, aber Beilagen nicht vollständig“ markiert, 16 nur angerissen. Diese alte Klassifikation darf **nicht** als heutiger Volltext-Verfügbarkeitstest verwendet werden. Wichtige früher benutzte Arbeiten/Autoren: Shao et al. 2021/2022, Beal 2003, Bishop 2006, Mowbray et al. 2022, Wang/Shao/Song 2019, Goodfellow et al., Belkin et al., Dai et al., Kang et al., Varma/Simon 2006 und Cawley/Talbot 2010. Vor Seitenzitaten Original-PDF tatsächlich öffnen und Seiten prüfen; wissenschaftliche Literaturbefunde von eigener Modelladaption und eigenen Experimenten trennen.

## 8. Vollständiges Inventar der hier tatsächlich materialisierten ZIP-Archive

Die nachfolgenden ZIPs wurden hier real kopiert, auf sichere Memberpfade geprüft und entpackt. Alle Membernamen und Archiv-SHA256 sind in `RECOVERED_ARCHIVE_MEMBER_MANIFEST.json` protokolliert. **Nicht sämtliche hunderte weiteren historischen Library-ZIPs** wurden in dieser Sitzung entpackt; sie sind bei Bedarf weiterhin über die Dateibibliothek abrufbar.

### `Bayesian_Bachelor_DM_17_Pickledateien_2026-11-10.zip`

SHA256 `a819ff150e34148388824c2757bec9e40892c4067cee0bc43f6308dfc44ef311` · 23 Einträge.

- `DM_16_aktive_Ringe.pkl`
- `DM_alle_17_mit_Status.pkl`
- `README.txt`
- `dateiliste_und_kontrolle.csv`
- `einzelne_ringe/DM01_1638.pkl`
- `einzelne_ringe/DM02_1639.pkl`
- `einzelne_ringe/DM04_1641.pkl`
- `einzelne_ringe/DM05_1642.pkl`
- `einzelne_ringe/DM06_1643.pkl`
- `einzelne_ringe/DM07_1645.pkl`
- `einzelne_ringe/DM08_1646.pkl`
- `einzelne_ringe/DM09_1647.pkl`
- `einzelne_ringe/DM10_1648.pkl`
- `einzelne_ringe/DM11_1649.pkl`
- `einzelne_ringe/DM12_1650.pkl`
- `einzelne_ringe/DM13_1651.pkl`
- `einzelne_ringe/DM14_1655.pkl`
- `einzelne_ringe/DM15_1656.pkl`
- `einzelne_ringe/DM16_1657.pkl`
- `einzelne_ringe/DM17_1658.pkl`
- `einzelne_ringe/DM18_1659.pkl`
- `manifest_pruefbericht.json`
- `DM_excel_zu_pickle.py`

### `Bayesian_Bachelor_DM_Herleitungen_A_bis_D_und_MASTER_HANDOVER_2026-11-11.zip`

SHA256 `a0e2bb04f6e466f02c8ca7b3a955eb85df01f4ebe174114be92b8a4b465fc9c9` · 121 Einträge.

- `01_NEUE_HERLEITUNGEN/Herleitung_A_NEU_Modellstruktur_2026-11-11.docx`
- `01_NEUE_HERLEITUNGEN/Herleitung_A_NEU_Modellstruktur_2026-11-11.md`
- `01_NEUE_HERLEITUNGEN/Herleitung_B_NEU_Strukturierte_VI_2026-11-11.docx`
- `01_NEUE_HERLEITUNGEN/Herleitung_B_NEU_Strukturierte_VI_2026-11-11.md`
- `01_NEUE_HERLEITUNGEN/Herleitung_C_NEU_ELBO_Training_2026-11-11.docx`
- `01_NEUE_HERLEITUNGEN/Herleitung_C_NEU_ELBO_Training_2026-11-11.md`
- `01_NEUE_HERLEITUNGEN/Herleitung_D_NEU_Posterior_Predictive_2026-11-11.docx`
- `01_NEUE_HERLEITUNGEN/Herleitung_D_NEU_Posterior_Predictive_2026-11-11.md`
- `02_ABGLEICH_UND_MASTER/ABGLEICHMATRIX_Original_A-D_zu_finalem_DM_Modell_2026-11-11.docx`
- `02_ABGLEICH_UND_MASTER/ABGLEICHMATRIX_Original_A-D_zu_finalem_DM_Modell_2026-11-11.md`
- `02_ABGLEICH_UND_MASTER/MASTER_HANDOVER_BAYESIAN_BACHELOR_DM_2026-11-11.docx`
- `02_ABGLEICH_UND_MASTER/MASTER_HANDOVER_BAYESIAN_BACHELOR_DM_2026-11-11.md`
- `03_ORIGINALE_UNVERAENDERT/Bayesian_SSR_Ringwalzen_Entscheidungen_Herleitung_A.docx`
- `03_ORIGINALE_UNVERAENDERT/Herleitung_B_Strukturierte_Variationsverteilung_Bayesian_SSR.docx`
- `03_ORIGINALE_UNVERAENDERT/Herleitung_C_Vollstaendiger_Trainingsalgorithmus_Bayesian_SSR.docx`
- `03_ORIGINALE_UNVERAENDERT/Herleitung_D_Posterior_Predictive_Bayesian_SSR.docx`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/README_UPDATE_B1.md`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/compare_smoke_runs.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/full_markov_b1.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/run_dm_smoke.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/test_full_markov.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_B1_FULL_JOINT_Update_Implementierung_BA_2026-11-11/vb_b1.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/README_WICHTIG.md`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/STATUS_UND_WEITERES_VORGEHEN.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/dm_data.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/input/manifest_pruefbericht.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/optuna_inner.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/results/smoke_v1_f2_k3_joint_restricted_c1/smoke_report.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/results/smoke_v1_f2_k3_joint_restricted_c1_3ep/smoke_report.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/results/smoke_v1_f2_k3_off_c1/smoke_report.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/results/smoke_v1_f2_k3_off_c1_3ep/smoke_report.json`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/run_dm_smoke.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/test_data.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/test_math.py`
- `04_PROGRAMM_REFERENZEN_AUS_PATCH_ARCHIVEN/Bayesian_Bachelor_DM_B1_Modellkern_V0_1_2026-11-11/Bayesian_Bachelor_DM_Modellkern_V0_1/vb_b1.py`
- `... weitere 86 Dateien: siehe JSON-Manifest`

### `Bayesian_Bachelor_DM_Pickle_Datenaudit_2026-11-10.zip`

SHA256 `4d2ae92906b61979329e38cd96d5923cff6e2c799c7b4c04891330f37c371751` · 15 Einträge.

- `Datenpruefung_Pickles_ausfuehrlich.md`
- `rings.csv`
- `channels.csv`
- `channel_per_ring.csv`
- `outliers.csv`
- `phases.json`
- `flags.csv`
- `protocol_means.csv`
- `nominal_comparison.csv`
- `audit_data.json`
- `analyze_pickles.py`
- `extra_audit.py`
- `build_report.py`
- `DM01_Zeitreihe_Durchmesser.png`
- `DM12_Zeitreihe_Durchmesser.png`

### `Bayesian_Bachelor_V4_Nested_CV_Einzelringdiagnostik_Kapitel4_5_2026-11-12.zip`

SHA256 `d924d9890be405e7b5f54b67103c1b2f389f1647f921bd509e54f9bb17ad4c2e` · 12 Einträge.

- `FORSCHUNGSNOTIZ_NESTED_OPTUNA_EINZELRINGE_UNSICHERHEIT_KAPITEL4_5_2026-11-12.docx`
- `RINGWEISE_FEHLERDIAGNOSTIK_NESTED_V4.csv`
- `SHA256_QUELLDATEIEN.txt`
- `README_ZUR_EINZELRINGDIAGNOSTIK.md`
- `abs_error_outer.png`
- `abs_error_height.png`
- `delta_squared_ssl_vs_c0.png`
- `analyze.py`
- `create_report.py`
- `originale_quellen/OUTER_PREDICTIONS.csv`
- `originale_quellen/INNER_SELECTED_BY_OUTER_FOLD.csv`
- `originale_quellen/NESTED_SUMMARY.json`

### `Bayesian_Bachelor_V4_Nested_Optuna_4x3_CV_2026-11-12.zip`

SHA256 `795aded6c84b6dd58813a30a7675932f7faae9f7a6c9e385ec4e84c832089c64` · 5 Einträge.

- `nested_optuna_cv_v4.py`
- `test_nested_optuna_cv_v4.py`
- `README_NESTED_OPTUNA_V4.md`
- `FORSCHUNGSNOTIZ_NESTED_OPTUNA_CV_KAPITEL4_5_2026-11-12.docx`
- `SHA256_MANIFEST.txt`

### `Bayesian_Bachelor_V51_Eingangssicherheit_Gepruefter_Code_2026-11-12.zip`

SHA256 `44fd94a2de1e0b989a37f6ea262a1aad63c3318ffb79806d30b4b10bd3cd3c4c` · 21 Einträge.

- `final_candidate_v51/CLOSEOUT_TEMPLATE.json`
- `final_candidate_v51/DEVELOPMENT_INPUT_POLICY_V51.json`
- `final_candidate_v51/PROPOSED_MODEL_FREEZE.json`
- `final_candidate_v51/PRUEFBERICHT_V51_EINGANGSSICHERHEIT_KAPITEL4_5_2026-11-12.docx`
- `final_candidate_v51/README_V51_INPUT_SAFETY.md`
- `final_candidate_v51/RELEASE_SHA256_MANIFEST.json`
- `final_candidate_v51/V51_QA_RESULTS.json`
- `final_candidate_v51/final_core/compare_dynamic_quality_cv4.py`
- `final_candidate_v51/final_core/compare_two_head_cv4.py`
- `final_candidate_v51/final_core/dm_data.py`
- `final_candidate_v51/final_core/dynamic_markov_v2.py`
- `final_candidate_v51/final_core/dynamic_quality_v2.py`
- `final_candidate_v51/final_core/full_markov_b1.py`
- `final_candidate_v51/final_core/fullcov_b1.py`
- `final_candidate_v51/final_core/quality_link_challenge.py`
- `final_candidate_v51/final_core/two_head_b1.py`
- `final_candidate_v51/final_core/vb_b1.py`
- `final_candidate_v51/input_safety_v51.py`
- `final_candidate_v51/main_bayesian_ssr_v51.py`
- `final_candidate_v51/test_development_logger_equivalence_v51.py`
- `final_candidate_v51/test_input_safety_v51.py`

### `Bayesian_Bachelor_V5_C_Dosis_Freeze_Kandidat_Code_und_Kapitel4_5_2026-11-12.zip`

SHA256 `9f9a4863a8f4fa2e58ca9a20029d562cceebac0e4aadcd082a89831135b1751e` · 28 Einträge.

- `final_candidate_v5/README_HAUPTMODELL_KANDIDAT.md`
- `final_candidate_v5/PROPOSED_MODEL_FREEZE.json`
- `final_candidate_v5/main_bayesian_ssr_v1.py`
- `final_candidate_v5/test_main_bayesian_ssr_v1.py`
- `final_candidate_v5/FORSCHUNGSBERICHT_C_DOSIS_MODELFREEZE_V0_KAP4_5_2026-11-12.docx`
- `final_candidate_v5/C_DOSE_VERIFIED_THREE_SPLITS_SUMMARY.json`
- `final_candidate_v5/C_DOSE_VERIFIED_COMPARISON.csv`
- `final_candidate_v5/build_report.py`
- `final_candidate_v5/final_core/compare_dynamic_quality_cv4.py`
- `final_candidate_v5/final_core/compare_two_head_cv4.py`
- `final_candidate_v5/final_core/dm_data.py`
- `final_candidate_v5/final_core/dynamic_markov_v2.py`
- `final_candidate_v5/final_core/dynamic_quality_v2.py`
- `final_candidate_v5/final_core/full_markov_b1.py`
- `final_candidate_v5/final_core/fullcov_b1.py`
- `final_candidate_v5/final_core/quality_link_challenge.py`
- `final_candidate_v5/final_core/two_head_b1.py`
- `final_candidate_v5/final_core/vb_b1.py`
- `final_candidate_v5/final_core/README.txt`
- `final_candidate_v5/sources/C_DOSE_SUMMARY_20261114.json`
- `final_candidate_v5/sources/C_DOSE_SUMMARY_20261115.json`
- `final_candidate_v5/sources/C_DOSE_SUMMARY_canonical.json`
- `final_candidate_v5/figures/C_dose_K13_height.png`
- `final_candidate_v5/figures/C_dose_K13_outer.png`
- `final_candidate_v5/figures/C_dose_K22_height.png`
- `final_candidate_v5/figures/C_dose_K22_outer.png`
- `final_candidate_v5/figures/Pareto_C_dose.png`
- `final_candidate_v5/PACKAGE_SHA256.json`

### `Bayesian_Bachelor_V5_Code_Audit_Mathematik_Daten_IO_2026-11-12.zip`

SHA256 `e6107b263ac0c3af794dd251f0489d261d9d74a5daa4693727544596b9fe88fa` · 9 Einträge.

- `V5_CODE_AUDIT/CODE_AUDIT_BAYESIAN_SSR_V5_KAPITEL4_5_2026-11-12.docx`
- `V5_CODE_AUDIT/AUDIT_FINDINGS.csv`
- `V5_CODE_AUDIT/AUDIT_FINDINGS.json`
- `V5_CODE_AUDIT/audit_v5_math_and_io.py`
- `V5_CODE_AUDIT/AUDIT_MATH_IO_RESULTS_V2.json`
- `V5_CODE_AUDIT/ELBO_V5_QA.png`
- `V5_CODE_AUDIT/EFFEKTIVER_U_ANTEIL.png`
- `V5_CODE_AUDIT/README_CODE_AUDIT_V5.md`
- `V5_CODE_AUDIT/SHA256_MANIFEST.json`

### `DM_Datenbasis_NEU_v2_mit_Herkunft_10-11-2026.zip`

SHA256 `9c51a7368554ab45382b263642b4fef965016041e2fee80475f332e506132700` · 9 Einträge.

- `DM_Loggerdaten_NEU_v2.csv`
- `DM_Zielwerte_NEU_v2_positionsgeprueft.csv`
- `Walzprotokolle_Prozess_Nutzerstand.csv`
- `Walzprotokolle_Geometrie_Nutzerstand.csv`
- `Walzprotokolle_Rohling_Messwerte_Nutzerstand.csv`
- `Walzprotokolle_Qualitaet_Notizen_Nutzerstand.csv`
- `Walzprotokolle_Fertigring_Messwerte_Nutzerstand.csv`
- `README_Quellen_und_Warnungen.txt`
- `DM_Audit_korrigierte_Zielwerte_10-11-2026.xlsx`

## 9. Direkt materialisierte Master-Handovers, Codeaudits und Herleitungen

- `ABGLEICHMATRIX_Original_A-D_zu_finalem_DM_Modell_2026-11-11.docx` – 31104 Byte
- `Bayesian_Bachelor_MASTER_HANDOVER_ABSCHLUSSPHASE_2026-11-10.docx` – 299776 Byte
- `Bayesian_Bachelor_MASTER_HANDOVER_NACHTRAG_2026-11-06.docx` – 62373 Byte
- `Bayesian_Bachelor_ULTIMATIVE_Masteruebergabe_2026-11-05.docx` – 74723 Byte
- `CODE_AUDIT_BAYESIAN_SSR_V5_KAPITEL4_5_2026-11-12.docx` – 147310 Byte
- `FORSCHUNGSBERICHT_C_DOSIS_MODELFREEZE_V0_KAP4_5_2026-11-12.docx` – 405243 Byte
- `FORSCHUNGSNOTIZ_C_DOSIS_DREI_SPLITS_MODELLENTSCHEIDUNG_2026-11-12.docx` – 407809 Byte
- `Herleitung_A_NEU_Modellstruktur_2026-11-11.docx` – 34252 Byte
- `Herleitung_B_NEU_Strukturierte_VI_2026-11-11.docx` – 33075 Byte
- `Herleitung_C_NEU_ELBO_Training_2026-11-11.docx` – 34248 Byte
- `Herleitung_D_NEU_Posterior_Predictive_2026-11-11.docx` – 33679 Byte
- `MASTER_HANDOVER_BAYESIAN_BACHELOR_DM_2026-11-11.docx` – 55468 Byte
- `MASTER_HANDOVER_BAYESIAN_BACHELOR_V2_FORTSETZUNG_2026-11-11.docx` – 56325 Byte
- `MASTER_HANDOVER_BAYESIAN_BACHELOR_V3_V4_FORTSETZUNG_2026-11-12.docx` – 215045 Byte
- `PRUEFBERICHT_V51_EINGANGSSICHERHEIT_KAPITEL4_5_2026-11-12.docx` – 43498 Byte

## 10. Arbeitsprinzip für den nächsten Schritt

1. Zunächst den vom Nutzer angekündigten letzten Stand mit dieser Zusammenführung vergleichen; bei Konflikten entscheidet der neueste nachweislich geprüfte Originalcode bzw. eine explizite neue fachliche Nutzerentscheidung.
2. Kein Training, keine neuen äußeren Qualitätsmetriken, keine Veränderung der 10 mathematischen Kernmodule ohne eigene Freigabe.
3. Alle neuen Änderungen getrennt versionieren, mit synthetischen Prüfungen und positiver/negativer Entwicklungsloggervalidierung versehen und Quellenstatus dokumentieren.
4. Implementierungsanweisungen für den Windows-Rechner auf `.\.venv\Scripts\python.exe` und getrennte Ergebnisordner abstimmen; niemals bestehende Forschungsarchivdaten überschreiben.
5. Für BA-Kapitel 4 und 5 Ergebnisse als explorativ / aus mehrfach verwendeten Entwicklungsringen berichten. Dokumentbearbeitung grundsätzlich mit Word (.docx) und überprüften Quellen-/Seitenangaben.
