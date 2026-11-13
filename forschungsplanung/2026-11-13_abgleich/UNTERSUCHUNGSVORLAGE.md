# Vorlage für eine spätere einzelne Forschungsuntersuchung

Status dieser Datei: **nicht ausgefüllt; keine Untersuchung gestartet**.
Die Vorlage ist eine Arbeitsstruktur für einen gesondert vereinbarten nächsten
Schritt. Sie enthält keine pauschale Freigabe des Optimierungsfahrplans.

## 1. Fragestellung und abgrenzbare Hypothese

- Paketnummer aus dem vollständigen Detailfahrplan P1–P12 und konkrete Teilfrage.
- Hypothese, konkurrierende Erklärung und ausdrücklich zulässiges negatives Ergebnis.
- Erkenntnisziel: Mathematik, Prozessbeschreibung, Qualitätsvorhersage,
  Unsicherheitsmodell, Robustheit oder Rechenaufwand.
- Was wird in dieser Untersuchung tatsächlich geändert beziehungsweise nur diagnostiziert?

## 2. Unveränderte Referenz und Entscheidungspunkt

- Referenzmodell, Versions-/Daten-/Quellcodehashes und vorhandener Testnachweis.
- Konkretes Gegenmodell und getrenntes Forschungs-Ausgabeverzeichnis.
- Feste Komponenten: Kanäle, F2, Skalierungspolitik, Köpfe, Priors, K, Seeds,
  Startfaktoren und zulässige Daten, soweit sie nicht selbst der Untersuchungsfaktor sind.
- Handelt es sich um eine größere Architektur-, Prior-, Zielgrößen- oder
  Datenrepräsentationsänderung? Dann die konkrete Nutzerentscheidung dokumentieren,
  bevor abhängiger Code oder ein Trainingslauf begonnen wird.

## 3. Mathematischer Vertrag

- Gemeinsame Verteilung mit allen bedingten Abhängigkeiten und Normierung.
- Anfangsbeobachtung, Ringgrenzen und zeitliche Indizierung.
- Priors mit eindeutiger Skala-/Rate-/Kovarianzkonvention und Gültigkeitsbedingungen.
- Posterior-/Variationsfamilie, exakte gegenüber approximierten Schritten.
- Vollständige Zielgröße einschließlich Entropien, globaler KL und U-Gewichtung.
- Erwartete hinreichende Statistiken, erste/zweite Designmomente, Gradienten
  und konjugierte oder begründet nichtkonjugierte Aktualisierungsgleichungen.
- Grenzfall oder analytische Identität zum Referenzmodell; Bedingungen ihrer Gültigkeit.
- Vorhersagevertrag: Ziel der bedingten Inferenz, integrierte Unsicherheit,
  vorhandene Dichte/Quantile oder nur Momente und ihre Einheit.

## 4. Quellen und eigene Herleitung

- Pro Aussage konkrete Originaldatei und tatsächlich gelesener Abschnitt.
- PDF-Dateiseite und verifizierte gedruckte Seite getrennt angeben.
- Eigene Algebra, Literaturübernahme und empirische Hypothese trennen.
- Fehlende oder ungeprüfte Originalbelege offen lassen; kein ähnliches Ersatzpaper
  als identischen Beleg ausgeben.

## 5. Unabhängige Prüfungen vor realem Vergleich

- Vorab festgelegte numerische Toleranzen.
- Handrechenbarer oder analytischer Kontrollfall.
- Kurze vollständige Enumeration, falls Zustandsmomente betroffen sind.
- Zentrale Gradientenvergleiche beziehungsweise passende unabhängige Identität.
- Synthetische Daten mit bekanntem generativem Verhalten.
- SPD-/Finite-/Randfall-/Split-/Persistenzkontrollen, soweit vom Eingriff betroffen.
- Prüfung, dass Zielfunktion und Gradienten im selben Koordinaten-/Skalenvertrag bleiben.

## 6. Vorab festgelegter experimenteller Vertrag

- Erlaubte L-/U-Ring-IDs, feste Folds und gesperrte historische Außenringe.
- Je Arm dokumentierter TRAIN-only-Fit aller Scaler, Deskriptoren und Initialisierer.
- Tatsächliche Startparameter exportieren; gleiche Startpolitik von gleichen
  realisierten Startpunkten unterscheiden.
- Primäres Maß, sekundäre Maße, negative Kontrolle und vorab definierte Abbruchregel.
- Beide Qualitätsziele in mm; Ringgewichtung und Aggregation explizit festlegen.
- Für Prozessscores: echte prädiktive Dichte beziehungsweise deren erklärte
  Approximation definieren; Trainings-ELBO nicht als austauschbaren Score ausgeben.
- Wenn NLL/CRPS/Intervalle benutzt werden: zuerst die ausgewertete
  Verteilung/Mischung und numerische Integration definieren.
- Regeln für Modellwahl festlegen. Zusätzliche nested CV nur, wenn tatsächlich
  eine Auswahlprozedur bewertet werden soll.

## 7. Tatsächlicher Code- und Ausführungsnachweis

- Isolierte Dateien/Branch/Variante, genaue Diff-Beschreibung und Importgrenzen.
- Tatsächlich ausgeführte Befehle, Exitcodes und Testanzahlen.
- Seeds, Paketversionen, Hardware-/Laufzeitangaben soweit untersucht.
- Unveränderliche Ausgabeordner mit Hashmanifest, Konfiguration und Split.
- Update-/Gradienten-/Abbruchlogs und getrennte TRAIN-/VAL-Ergebnisse.
- Fehlgeschlagene Prüfungen und ihre Korrektur dokumentieren; keine erwarteten
  Ergebnisse als bereits ausgeführt markieren.

## 8. Ergebnisse mit getrennten Schlussfolgerungen

| Dimension | Tatsächlicher Befund | Was daraus folgt | Was daraus offen bleibt |
|---|---|---|---|
| Mathematische Identität/Korrektheit | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Numerik/Konvergenz | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Rechenaufwand | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Prozessbeschreibung | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Außenunrundheit | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Höhenabweichung | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Unsicherheit/Kalibrierung | Noch nicht untersucht | Noch keine Aussage | Noch offen |
| Ringgeneralisation | Noch nicht untersucht | Noch keine Aussage | Noch offen |

Gepaarte Fold-/Ringunterschiede und negative Befunde getrennt ausweisen.
Bei wenigen wiederverwendeten Ringen keine universelle Überlegenheit oder
formale Kalibrierung aus einem günstigen Einzelwert ableiten.

## 9. Begründete Entscheidung

- Beibehalten, verwerfen, begrenzt weiterprüfen oder fachlich ungeklärt?
- Entscheidung anhand der vorher festgelegten Hypothese und Akzeptanzbedingungen.
- Zusätzliche Komplexität und verbleibende Annahmenverletzungen benennen.
- Bei Architekturänderungen Entscheidung des Nutzers dokumentieren.
- Keine automatische Übernahme in die funktionierende Baseline.

## 10. Nachweisbare Übergabe

- Formel–Code–Test–Quelle–Ergebnis-Matrix aktualisieren.
- Herkunft, Leseumfang, Grenzen und Status aller Artefakte angeben.
- Präzise benennen, welche nächste Entscheidung möglich ist und welche
  zusätzliche Evidenz sie voraussetzt.
