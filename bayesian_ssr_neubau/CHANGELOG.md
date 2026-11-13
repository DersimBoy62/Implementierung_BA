# Änderungen

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
