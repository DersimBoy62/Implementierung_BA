# Mathematisch belegte Architekturentscheidungen – 13.11.2026

| Entscheidung | Forschungsstand | Neue Basismodell-Policy |
|---|---|---|
| Beobachtungseinheit | DM-Ringsequenz, kein unabhängiger Loggerzeitpunkt | ring-level Split zwingend |
| Targets | ovality.ovality_radial_mm + planarity.height_deviation_mm | exakt 2 Heads, mm |
| Prozess-X | 6 Loggerkanäle in V5.2-Reihenfolge | F2 + identische Filter/Scaler |
| DM03 / DM13 | fehlt/abgebrochen bzw. unvollständige Prozessphase | nicht als aktive Ringe verwenden |
| HMM Emission | D6-Gauß Vollkovarianz, Normal-Wishart | Basismodell beibehalten, AR-HMM erst später |
| Markov VI | zeitabhängige strukturierte q(z) | exakte γ, ξ, Moment 1/2, K−1 gauge |
| U-ELBO | C multipliziert Prozess und U-Entropie | gewichtet/generalisiert, C=.2 als historische Startkonfig |
| Gelabelte Kopplung | beide y-Terme rückwirken in q(z) | keine Pseudo-Labels |
| Local VI | bisheriges Restart-Potenzial; V5.8 fix | Warmstart+FB-Kandidaten und lokaler Annahmecheck |
| HMM Markov Logits | redundant K volle softmax-Logits | V5.9 K−1 je Row |
| M2 Temporalfeatures | ΔK = −ΣΔrest | V6.0 letztes Δ streichen; induzierte NIG-Kovarianz A^T V A |
| Qualitäts-Prior | 2 NIG Köpfe, 9/8 effektive Feature-Dims nach V6.0 | eigenständig posteriorbasierte Inferenz |
| Posterior Predictive | momentbasierte Approximation | kein Versprechen exakter t-Mischungs-Quantile |
| Epochen | 10 zu kurz bei einem geprüften Seed | 30 als Referenzlauf, später convergence policy evaluieren |
| Seed | 20261111 historisch | reproduzierbar; echte Multistarts später |
| HMM Persistenz | Selbsttransition >.994 | keine erzwungene Phasenidentifikation |
| Temporal Emission | Residual ACF lag1 sehr hoch | Forschungsproblem, AR-Erweiterung später |
| äußere Ringe | DM07/08/11/15 einmal geöffnet | keine neue externe Unabhängigkeit behaupten |

Die detaillierten Herleitungen und Quellenstatus stehen in den korrespondierenden Ordnern. Jeder Soll-Ist-Abgleich muss auf konkrete Zeilen/Funktionen/Tests im neuen Code zeigen.
