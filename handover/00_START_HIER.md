# START HIER – Übergabe an den VS-Code-Agenten

**Projekt:** Bayesian Bachelor · **Stand:** 13.11.2026 · **Status:** Forschungsübergabe, noch kein finaler Model-Freeze.

## So startest du

1. ZIP als Gesamtordner entpacken, idealerweise **neben** dem bisherigen VS-Code-Projekt `Implementierung_BA`. Vorhandene Quelldateien, `input/`, `.venv` und `results/` niemals automatisch überschreiben.
2. In VS Code das bestehende Projekt öffnen und dem ChatGPT-Agenten Zugriff auf den **entpackten Übergabeordner** geben.
3. Die komplette Datei `01_MASTER_PROMPT_VSCODE_AGENT.md` in den Agentenchat einfügen oder diese Datei als explizite Arbeitsanweisung referenzieren; zusammen mit `00_START_HIER.md` und den A–D-Dokumenten.
4. Das neue Modell soll unter einem **eigenständigen neuen Unterordner** (`bayesian_ssr_neubau/`) implementiert werden, nicht in den alten `final_core` hinein.
5. Der große Word-Fahrplan in `08_OPTIMIERUNG_NACH_NEUBAU/` ist ein **Folgeauftrag**, ausdrücklich erst nach fertiger Basismodellimplementierung.

## WICHTIGE Quellenhierarchie

- **A–D neu (11.11.2026)**: generatives DM-Modell, strukturierte VI, ELBO/Training, Posterior-Predictive. Diese Texte bilden den Ausgangspunkt, stammen aber **vor** den mathematischen V5.8–V6.0-Korrekturen.
- **V5.8**: Warmstart und mathematisch geprüfte ELBO-Koordinatenbilanz.
- **V5.9**: K−1-Logits pro lokaler Markov-Wahrscheinlichkeitszeile.
- **V6.0**: Erhalt des induzierten NIG-Priors bei M2-Redundanzreduktion. Diese drei Punkte in der neuen Basis implementieren.
- **V6.1–V6.3**: mathematische und physikalische Diagnostik; V6.3 war zum Übergabezeitpunkt lokal vom Assistenten geprüft, aber der zuletzt angeforderte Windows-V6.3-Test noch **nicht** im Chat bestätigt.
- **Original A–D (27.–29.10.2026)**: historische frühere Modellstruktur mit anderen Targetannahmen, nicht direkt implementieren.
- **V5.2 `final_core`**: alte geprüfte Referenz/Oracle, **kein fertiges neues Modell**. Veränderte Algorithmen nicht ohne mathematische Kontrolle kopieren.
- **Literatur-PDFs**: aus Nutzerbibliothek materialisierte Dateien, Duplikate per SHA256 entfernt. Exakte gedruckte Seitenzahlen **nicht** pauschal für diese Übergabe geprüft.

## Datenvertrag

Daten in bestehendem Projekt: `input/DM_16_aktive_Ringe.pkl` (nicht ins Paket dupliziert). Prüfsumme SHA256 `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`.

Train L: DM04, DM05, DM06, DM09, DM12, DM16, DM17, DM18. Train U: DM01, DM02, DM10, DM14. Historische bereits geöffnete Außenringe DM07, DM08, DM11, DM15; **keine** neue unabhängige Blindprüfung auf ihnen behaupten. DM03 fehlt, DM13 ausgeschlossen.

Zielgrößen **nur** `ovality.ovality_radial_mm`, `planarity.height_deviation_mm`; nicht Innenunrundheit. F2-Zeitfenster. Kanäle `F_rad63`, `F_ax64`, `Ist_D150`, `Ist_s151`, `Ist_h152`, `RWG55` (Reihenfolge verbindlich). Kein Pseudo-Labeling.

## Ordnerübersicht

- `01_MASTER_PROMPT_VSCODE_AGENT.md/.txt`: sehr ausführliche Agentenarbeitsanweisung.
- `02_HERLEITUNGEN_A_B_C_D`: alte und neue Herleitungen vollständig, einschließlich Markdown-Leseansichten der Neufassungen.
- `03_ENTSCHEIDUNGEN_UND_STATUS`: Handovers, 133-Literaturkarten-Status, methodische Pläne.
- `04_REFERENZCODE_READ_ONLY`: V5.2-Kern und V5.8–V6.3-Forschungsimplementierungen plus frühere QA; **nur read-only**.
- `05_DATENSCHEMA_PROVENIENZ`: Datenherkunft / Auditarchive; enthält **nicht** die aktive DM-PKL.
- `06_FORSCHUNGSBEFUNDE_UND_AUDITS`: Dokumentierte Analysen, teils maschinenlesbare JSON/CSV-Traces.
- `07_ORIGINALQUELLEN_PDF`: **82 Byte-eindeutige wissenschaftliche PDFs**, Literaturinventar, SHA256-Manifest. 133 Literaturkarten sind keine 133 bestätigten Volltexte.
- `08_OPTIMIERUNG_NACH_NEUBAU`: separate Word-Datei mit priorisierten Verbesserungen.

## Methoden- und Sicherheitsgrundsätze

**Nicht** aus vorhandenen Altmodellen weitertrainieren; ein neuer, nach mathematischer Konstruktion und getrennter Trainingsschnittstelle entwickelter Code muss frisch initialisieren, eigene Tests bestehen und alle Daten-/Labelzugriffe nachvollziehbar machen. Ein bereits geöffnetes äußeres Testset ist für spätere Architekturänderungen nicht mehr unabhängig. Keine erfundenen Quellenseiten, Null-Fehler-Behauptungen oder „fully Bayesian“-Versprechen ohne ausreichenden Nachweis.
