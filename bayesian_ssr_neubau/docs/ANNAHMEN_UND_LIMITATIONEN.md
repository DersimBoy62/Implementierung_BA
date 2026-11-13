# Annahmen, Forschungsgrenzen und offene Entscheidungen

Stand 13.11.2026. Dies ist ein mathematisch prüfbarer Basismodell-Neubau, keine
Endfreigabe. Ausführungs- und CV-Status werden im Abschluss-/Testbericht konkret
angegeben; dieses Dokument behauptet keine bereits bestandenen Läufe.

1. **Ringweite Stichprobe.** Die vielen Loggerzeitpunkte sind keine vielen
   unabhängigen Endqualitätslabels. Es gibt acht gelabelte Entwicklungsringe und
   vier Prozess-only-U-TRAIN-Ringe. Die vier historischen Außenringe wurden schon
   2026 ausgewertet. Nachträgliches Vergessen erzeugt keine Blindprüfung.
2. **Metrologie.** Die beiden gespeicherten Zielblattnamen und ihre mm-Werte sind
   verifiziert. Die Gleichsetzung von `planarity.height_deviation_mm` mit einer
   bestimmten normativen Höhenschlagmessung bleibt ohne Messvorschrift offen.
   Keine Innenunrundheit, kein `r_outer_mm`-Ersatz. Der reale Container heißt
   `zielwerte`; nicht mit einem nicht mitgelieferten `all_targets` verwechseln.
3. **Sensoreinheiten.** Originale Pickle-Loggerheader weisen F_rad63/F_ax64 in %,
   Ist_D150/Ist_s151/Ist_h152 in m und RWG55 in m/s (`ring growth rate`) aus.
   Prozentkanäle werden ohne Kalibrierkurve nicht zu Kräften in kN erklärt.
   Korrigierte Loggergeometrien sind nicht automatisch unabhängige Endqualitäten.
4. **Vollständiges Prozessfenster.** Gleiche F2-Zeilen und Zeitachse wie Referenz.
   Die vollständige Sequenz und die End-minus-Anfang-Deskriptoren setzen ein
   abgeschlossenes F2-Fenster voraus. Eine Vorhersage im laufenden Prozess ist
   nicht validiert. Die bekannten 2→1-Loggerphasen sind Kontext, keine zusätzlichen
   Modellfeatures und keine fest zugewiesenen HMM-Zustände.
5. **Serielle Emissionsstruktur.** Vollkovarianz erfasst Kanalkorrelation am selben
   Zeitpunkt. Die Emissionen sind bei gegebenem Zustand und globalen Parametern
   zeitlich unabhängig. Die historische starke Residualautokorrelation ist ein
   relevanter Hinweis auf Fehlstruktur, kein formaler Test mit vorab festgelegtem
   Signifikanzniveau. Ein AR-HMM würde Likelihood, Priors und Anfangsmodell ändern
   und bleibt ein separater Auftrag.
6. **Lokale Zustandsfamilie.** U-FB ist bei festen globalen Faktoren und C>0 im
   gewählten Variationsmodell prozessoptimal. Der globale ringweite Qualitätsterm
   macht den exakten L-Posterior im Allgemeinen nichtmarkovsch; freie zeitabhängige
   Markov-VI ist eine bewusste Approximation. Acht L-BFGS-B-Iterationen und ein
   `success`-Flag beweisen keine Stationarität. Warmstart und Annahmekontrolle
   schützen den Koordinatenvertrag, nicht das globale Optimum.
7. **Numerische Äquivalenz.** K-1-Logits entfernen eine Gauge. M2 entfernt nur die
   letzte Delta-Komponente mit induziertem Prior. Algebraische Gleichheit ist
   nicht gleichheit aller endlichen Optimiererpfade. Flooring beim Umwandeln
   extrem kleiner FB-Wahrscheinlichkeiten ist eine explizite numerische
   Approximation; ungerundete FB-Marginalen bleiben getrennt zugänglich.
8. **C-Gewichtung.** C=.2 ist historisch gewählter Ausgangswert. Das gesamte
   U-Prozessziel einschließlich Entropie wird gewichtet. Dies ist für C≠1 keine
   unveränderte Evidenzschranke des ursprünglichen ungewichteten Modells. Die
   WVI-Gleichungen von Shao/Ge/Song2021 sind nicht identisch damit.
9. **U-Preprocessing.** P0 poolt TRAIN-Zeitpunkte und gewichtet lange Ringe bei
   X-Skalierung stärker. Deskriptor-Scaler sind ringgleichgewichtet. C=0 mit
   gemeinsamem X-/Deskriptorfit kann sich von strengem L-only weiterhin
   unterscheiden. Untersuchungsarme müssen diese Trennung beibehalten.
10. **Regressionsrang.** sum(h)=1 liefert den konstanten Regressionsanteil und
    ist keine entbehrliche lineare Spalte. sum(Delta)=0 erlaubt eine Reduktion.
    Wird die volle .15-Deltavarianz ohne Pushforward übernommen, ändert sich das
    Modell. Die Gleichheit konjugierter effektiver Posterioren bedeutet keine
    Gleichheit beliebiger frei gesetzter alter q(beta)-Familien.
11. **Modellwissen und Priors.** K=2, die zwei Deskriptorgruppen, physikalische
    Priorvarianz .38526108747593474 und 30 Epochen sind historische
    Entwicklungsentscheidungen, keine universell bewiesenen Optima und keine
    mechanistisch gesicherten Konstanten. Prior-Predictive-, Feature- und
    Priorstärke-Sensitivität benötigen spätere kontrollierte Untersuchungen.
12. **Interpretation.** Sehr persistente HMM-Zustände können glatte Trends,
    Ringregime oder echte Phasen abbilden. Zustandsnummern sind permutierbar;
    Vergleiche über Seeds/Folds benötigen eine Zuordnung. Identische K-Means-
    Zentren unter verschiedenen Seeds zeigen keine Initialisierungsrobustheit.
13. **Prädiktive Unsicherheit.** NIG wird bedingt auf den Designvektor analytisch
    integriert. Markov-Mischungsmomente sind unter dem gewählten q exakt.
    Globale HMM-Unsicherheit wird nicht mit erneuter sequenzbedingter
    Parameterintegration vollständig propagiert. Die beiden marginalen SD sind
    approximative Modellunsicherheit, keine kalibrierten 95%-Intervalle.
14. **Zielverteilung.** Gauß-/Student-t-basierte Köpfe haben reelle Unterstützung,
    obwohl die Rohqualitäten physikalisch nichtnegativ sind. Negative
    rücktransformierte Vorhersagen wären zu berichten; stilles Clipping würde
    das Vorhersagemodell ändern. Eine positive Zieltransformation wäre ein
    gesondert herzuleitender Forschungszweig.
15. **Evidenz und Reproduzierbarkeit.** Neue Tests und Läufe sind von alten
    Berichten getrennt. Eine höhere J_C ist kein Nachweis kleinerer mm-Fehler.
    Kleine, mehrfach benutzte Entwicklungs-CV erlaubt keine statistisch sichere
    Überlegenheitsbehauptung. Ladeidentität, Manifestintegrität und gleicher Seed
    prüfen Reproduzierbarkeit innerhalb der dokumentierten Umgebung, nicht
    allgemeine Robustheit gegenüber Initialisierungen oder Verteilungswechseln.

Die nächsten kontrollierten Forschungsfragen sind serielle Emissionen, HMM versus
HMM-freie Regression, volle versus reduzierte Emissionen, echte Startvarianten,
Bayes versus MAP/Ridge, isolierter U-ELBO, Kalibrierung und Ringgeneralisierung.
Der Word-Fahrplan in `08_OPTIMIERUNG_NACH_NEUBAU` bleibt ein gesonderter Auftrag.
Eine Hypothese allein ist kein Grund, diese Erweiterungen gleichzeitig einzubauen.
