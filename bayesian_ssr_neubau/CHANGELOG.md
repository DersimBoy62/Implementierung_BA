# Änderungen

## Projektbereinigung – 13.11.2026

Der vollständige ursprüngliche Git-Stand ist auf `archive/vollstand-2026-11-13`
gesichert. `main` enthält den Neubau mit 13 bytegleichen lokalen Testreferenzen,
eigenständigen Prüfskripten und einer ausgabefreien Notebookvorlage. Alte
Workspace-Verzeichnisse und gespeicherte Rechenläufe sind keine Voraussetzung
für die synthetischen oder mathematischen Referenztests mehr. Reale Datentests
überspringen ausschließlich bei fehlender privater Rohdatei mit Begründung.
`BAYESIAN_SSR_DATA` ergänzt den CLI-Datenpfad; explizites `--input` hat Vorrang.
Die mathematische Implementierung und Basishyperparameter bleiben unverändert.

## 0.1.0 – 13.11.2026

Eigenständiger Neubau in `src/bayesian_ring_ssr`, ohne Imports alter Modellkerne.
Neue Prozess-only-Datenverträge und F2-Paritätsprüfung; zwei NIG-Köpfe, volle
Normal-Wishart-Emissionen, Dirichletfaktoren, freie zeitabhängige Markov-VI,
K-1-Logits und exakte M2-Momente mit induzierten Priors. Geschützte lokale
Warmstarts, vollständige gewichtete Blockbilanz, frische Initialisierung,
JSON-Persistenz, Prediction, Diagnostik und vierfache Entwicklungs-CV.

Original-A–D-Konflikte werden ausdrücklich dokumentiert; Quellen mit konkretem
Lesestatus statt pauschaler Volltextbestätigung. Historische Außenringe bleiben
ausgeschlossen. Der separate Optimierungsfahrplan wurde nicht implementiert.
