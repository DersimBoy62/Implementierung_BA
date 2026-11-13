# Abgleich des Forschungsfahrplans mit dem geprüften Basismodell

Stand: 13.11.2026. Auftrag dieser Prüfung: den gesamten Fahrplan lesen, den
Implementierungs- und Evidenzstand einordnen und die spätere Arbeitsreihenfolge
begründen. Es wurde keine Optimierung, neue Modellvariante oder Trainingsstudie
gestartet. Das Basismodell wird als unveränderte Referenz erhalten.

## Ergebnis und Herkunft

**P0, der beauftragte Modellneuaufbau, ist innerhalb seines festgelegten
mathematischen und numerischen Vertrags abgeschlossen. Die Forschungsfragen
P1–P12 sind dadurch überwiegend vorbereitet, aber noch nicht beantwortet.**

Die neue Download-Datei ist byteidentisch mit dem schon im Übergabepaket
enthaltenen Fahrplan. Ihr SHA256 lautet
`b5f21a4aec64171d6bee6f859f40907c804148c9b958c3abfb4c7cbc5584665b`.
Gelesen wurden alle 185 Haupttextabsätze einschließlich der Tabellenzellen sowie
die Word-Fußzeile. Die vollständige Leseansicht mit stabilen Absatznummern steht
in [OPTIMIERUNGSFAHRPLAN_VOLLSTAENDIG.txt](OPTIMIERUNGSFAHRPLAN_VOLLSTAENDIG.txt).
Diese Nummern bezeichnen Absätze, keine ungeprüften Word- oder Literaturseiten.

Der Master-Prompt, die aktuelle A–D/V6.0-Spezifikation, die tatsächlichen
Produktivmodule, Tests und Ergebnisartefakte wurden dagegen abgeglichen.
[BESTANDSPRUEFUNG.json](BESTANDSPRUEFUNG.json) dokumentiert die dabei neu
ausgeführte Integritätsprüfung: **157 von 157 Release-Dateien unverändert**, der
Datenhash stimmt. Der bestehende Bericht über 105 bestandene Tests gilt damit
weiter für genau diesen Quellcode. In diesem Abgleich wurden die Tests nicht
erneut ausgeführt und keine neuen Testergebnisse behauptet.

Der separate Ordner `forschungsplanung/` ergänzt den versiegelten Release.
Es wurden keine Dateien unter `bayesian_ssr_neubau/` oder alte Laufartefakte
geändert. Die Gesamtmatrix ist zusätzlich als [ABGLEICH_MATRIX.csv](ABGLEICH_MATRIX.csv)
verfügbar.

## 1. Was der abgeschlossene Neuaufbau bereits nachweist

| Gegenstand | Tatsächlich vorhandener Nachweis | Reichweite |
|---|---|---|
| Eigenständiger Code | 23 Produktivmodule, Importaudit ohne alte Modellkerne | Historischer Code bleibt ausschließlich Referenz in separaten Tests. |
| Datenvertrag | Exakter Datenhash, zwei bestätigte Ziele, sechs feste Kanäle, zwölf F2-Sequenzen und Deskriptoren referenzgleich | Kein Nachweis einer vollständig geklärten Messvorschrift oder künftiger Datenverteilung. |
| Bayesian-Faktoren | Dirichlet, Normal-Wishart, zwei NIG-Köpfe; Dichte-/Integral-/Posteriorprüfungen | Variationsapproximation, kein exakter gemeinsamer Posterior. |
| Markov-Mathematik | Freie zeitabhängige lokale Ketten, Forward-Backward, exakte M2-Momente und analytische Gradienten | Exaktheit unter der gewählten lokalen q-Familie. |
| V5.8 | Besserer Warmstart aus altem Faktor und Prozess-FB; lokale Annahme und gemeinsame Blockbilanz | Keine Garantie eines globalen Optimums. |
| V5.9 | K−1 freie Log-Odds pro Zeile, Gauge-/Gradientenprüfungen | Gleicher Verteilungsraum, nicht identische endliche Optimiererwege. |
| V6.0 | Induzierter NIG-Prior bei Entfernung nur des letzten Delta-Merkmals; Köpfe 9/8 | Kein neues Qualitätsmodell und keine dadurch bewiesene Prognoseverbesserung. |
| V6.1–V6.3 | Komponentenbilanz, erwartete Wechsel aus xi, Phasengrenzen, Residual-Lag1, Differenzen und Detrends | Deskriptive Diagnose vorhanden; vollständige Reproduktion aller historischen Diagnosegrößen und eine formale Modellprüfung sind damit nicht automatisch erledigt. |
| Reales frisches Training | 30 Epochen, 8L+4U, 450 kontrollierte Updates, keine Verletzung der absoluten Toleranz 1e−6 | Rechenbefund; keine Kalibrierungs- oder Generalisierungsgarantie. |
| Prediction/Persistenz | Prozess-only-Eingabe, mm-Rückskalierung, Load gleich Memory, Integritäts- und semantische Negativtests | Zwei marginale Momente; keine validierten Mischungsquantile. |
| Entwicklungsevaluation | Vier Ringfolds, fünf Vergleichsarme, alle Scaler im jeweiligen TRAIN gefittet | Explorative Ergebnisse auf acht historisch wiederverwendeten Labelringen. |

Evidenzdateien im unveränderten Release:

- [Mathematisches Modell](../../bayesian_ssr_neubau/docs/MATHEMATISCHES_MODELL.md)
  und [A–D/V6.0-Abgleich](../../bayesian_ssr_neubau/docs/ABGLEICH_A_B_C_D_V60.md).
- [Ausgeführte Tests](../../bayesian_ssr_neubau/docs/verification_001.json),
  [Testprotokoll](../../bayesian_ssr_neubau/docs/TESTPROTOKOLL.md) und
  [V6.0-Parität](../../bayesian_ssr_neubau/docs/V60_PARITAET.md).
- [30-Epochen-Audit](../../bayesian_ssr_neubau/docs/TRAINING_30E_AUDIT.json),
  [Release-Validierung](../../bayesian_ssr_neubau/docs/RELEASE_VALIDATION.json)
  und [CV-Ergebnisse](../../bayesian_ssr_neubau/outputs/development_cv_001/development_cv.json).

Ein dokumentarischer Restpunkt ist ausdrücklich sichtbar:
`model.json` enthält `math_test_status=NOT_RECORDED`. Der abgeschlossene echte
Testnachweis wird durch `RELEASE_VALIDATION.json` an Modell- und Quellcodehashes
gebunden. Das ist kein unbestandener Test. Eine spätere komfortablere Einbettung
des Teststatus gehört zum Artefaktvertrag von P11 und rechtfertigt kein
Überschreiben des vorhandenen Modellbundles.

Die sechs protokollierten lokalen Iterationslimit-Abbrüche bleiben ebenfalls
sichtbar. Der maximale Abschlussgradient über **alle 240 lokalen Aufrufe**
betrug etwa 2,096·10⁻⁴. Diese Zahl ist nicht als alleiniger Gradient der letzten
Epoche zu lesen. Monotone Annahme ersetzt keinen Stationaritätsnachweis;
die weitergehende Konvergenzuntersuchung bleibt P11.

## 2. Korrektur der widersprüchlichen Prioritätsnummern

Die Detailabschnitte sind vollständig und werden hier als maßgebliche
Bezeichnungen verwendet:

| Thema | Kurzüberblick des Word-Dokuments | Vollständiger Detailteil und dieser Abgleich |
|---|---|---|
| Zeit-/Ringgewichtung, Takt und Datenrepräsentation | Kein eigener entsprechender Eintrag | P10 |
| Trainingslogik, lokale VI und Laufzeit | P10 | P11 |
| Wissenschaftliche Nachweise und Traceability | P11 | P12 |

P0–P9 sind hiervon nicht betroffen. Das Originaldokument wird nicht stillschweigend
umgeschrieben; die vollständige Zuordnung bleibt nachvollziehbar.

## 3. Vollständige Statusmatrix P0–P12

„Teilweise“ bedeutet, dass Bausteine oder beschreibende Ergebnisse vorhanden
sind. Es bedeutet nicht, dass der im Fahrplan geforderte Nutzen bereits belegt ist.

| Paket und wissenschaftliche Priorität | Bereits umgesetzt oder belegt | Weiterhin offen | Einordnung |
|---|---|---|---|
| **P0 Basiskonformität** | Eigenständiger Neubau; V5.8–V6.0; T1–T10; frisches 30E-Training; Daten-/Prediction-Grenze; Persistenz; Entwicklungs-CV | Allgemeine Modelladäquatheit, Kalibrierung und weitere Forschungsmodelle gehören nicht zum engen P0-Vertrag | **Abgeschlossen im Basismodellvertrag** |
| **P1 Dynamische Emissionen, sehr hoch** | Lag1-/Differenzen-/phasenweise Detrenddiagnose im neuen Code | Bedingte Prozess-Predictive-Checks, generative Innovationen, Prozesssimulation, AR-Anfangsmodell, konjugierte oder anders begründete AR-Updates, held-out Prozessbewertung | **Diagnose teilweise; AR-Forschungsvariante offen** |
| **P2 Tatsächlicher HMM-Nutzen, sehr hoch** | TRAIN-Mean, physikalische NIG-Regression und HMM in denselben Ringfolds vorhanden | Vorverarbeitung und effektive Priors angleichen; i.i.d.-GMM; h/M2/Deskriptoren getrennt ablatieren; gepaarte Effekte und Auswertungsunsicherheit | **Vergleich vorhanden; eigenständiger HMM-Effekt offen** |
| **P3 Bayes-Nutzen/Kalibrierung, sehr hoch** | Konjugierte Posterioren und korrekte marginale prädiktive Momente | Vergleich mit passend regularisiertem MAP/Ridge/EB; Zustandsmischungsverteilung; NLL/CRPS/PIT/Coverage; Simulationskalibrierung | **Bayesian-Mathematik umgesetzt; Mehrwertstudie offen** |
| **P4 U-Daten/SSL-Effekt, sehr hoch** | A: strict-C0; B: shared-C0; C: shared-Preprocessing plus U-Ziel; vierfache CV; U-Entropie korrekt gewichtet; C0-Tests | Gepaarte Fold-/Ringanalyse als eigenes Forschungspaket; U-Teilmengen; Posterioränderungen; optionaler gesonderter U-Prior-Arm D; Initialisierungsartefakte explizit exportieren | **A/B/C bereits umgesetzt und ausgewertet; Gesamtpaket teilweise** |
| **P5 Generalisierung, sehr hoch und begleitend** | Feste vollständige Ringfolds; TRAIN-only-Fits; acht VAL-Prognosen je Arm; gesperrte historische Außenringe | Vorab fixierter Forschungs-Auswertungsplan; ggf. LORO/repeated Folds; Auswahlprotokoll/nested CV nur bei echter Modellwahl; ehrliche Unsicherheit auf Ringebene | **Baseline-Evaluation fertig; vertiefte Evidenz offen** |
| **P6 Echte Multistarts, hoch** | Gleicher Seed reproduzierbar; anders gesetzter Seed synthetisch ausführbar; frische K-Means-Initialisierung | Tatsächlich verschiedene Zentren/Dirichlets/Startfaktoren; Zustandsalignment; Basins und Vorhersagesensitivität | **Multistartstudie offen** |
| **P7 Physikalische Zustände, hoch** | Erwartete Wechsel aus xi, mittlere Zustandsbesetzung, reale Phasengrenzen und Wechselmasse ±2 s | Zeitlich ausgerichtete gamma-/Kanalplots; uniforme deskriptive Referenz; Permutationsalignment; vollständige Ereigniszuordnung und Mess-/Betreuerwissen | **Numerische Diagnose teilweise; physikalische Validierung offen** |
| **P8 Priors/Features/Kovarianzen, hoch** | Priorerhaltende M2-Reduktion; SPD-/Formeltests; festgelegte Priors und Featuregruppen | Systematische Sensitivität von NIG/NW/Persistenz; Rang-/Konditionsanalyse; Feature-/M2-/Kovarianzablationen; Prior-Predictive-Checks | **Algebraische Basis fertig; Sensitivitätsforschung offen** |
| **P9 Prädiktive Mischung/Multioutput, mittel** | Zwei marginale Mittelwerte/SD; geteilte Zustandsunsicherheit berücksichtigt | Quantile durch kontrollierte Integration; gemeinsame Zielkovarianz aus Zuständen ausgeben; weitergehende globale Parameterintegration; gesondertes korreliertes Residualmodell | **Momente fertig; Verteilungs- und Architekturfragen offen** |
| **P10 Zeit-/Ringgewichtung, mittel** | P0-Zeitpunktpooling, ringgleichgewichtete Deskriptor-Scaler, 0,1-s-Raster und F2-Konvention dokumentiert | Abtast-/Ringgewichtungsstudie; neue probabilistische Bedeutung einer Normierung; Übergangs-/Priortransformation auf anderes Raster | **Ausgangsvertrag festgelegt; Untersuchung offen** |
| **P11 Numerik/Training/Laufzeit, mittel und begleitend** | Lokale Annahme, Blockbilanzen, Gradienten, Abbruchgründe, Checks, Versionslock und Checkpoints | Stationaritäts-/Abbruchstudie; Profiling je Ring; CPU/RAM/Laufzeit; geprüfte Cache-Invalidierung; vollständiger Vergleichsvertrag für Forschungsartefakte | **Korrektheits- und Loggingbasis fertig; Forschungsstudie offen** |
| **P12 Quellen/Traceability, hoch und begleitend** | Aktuelle Modellspezifikation; A–D-Matrix; Inventur; 19 konkret gelesene wissenschaftliche Originalauszüge mit begrenztem Belegumfang | Jede spätere Thesisbehauptung gezielt belegen; Originalseiten/Supplements zusätzlich prüfen; neue Varianten und negative Resultate in die Matrix aufnehmen | **Für Basismodell weitgehend erstellt; vollständiger Thesisnachweis offen** |

### Konkrete Codebelege

| Aussage | Stelle im vorhandenen Produktivcode |
|---|---|
| AR-Emissionen fehlen; jetziger NW-Emissionsterm ist zeitpunktbezogen | `normal_wishart.py:30` |
| Residual-/Phasendiagnostik vorhanden | `diagnostics.py:19` |
| Vorhandene Vergleichsarme und NIG-Baseline | `evaluation.py:16`, `evaluation.py:35`, `evaluation.py:41` |
| Erlaubte Initialisierungsdaten und K-Means-Politik | `coordinate_train.py:58` |
| TRAIN-only-Scaler und U-Policy | `preprocessing.py:97` |
| C multipliziert auch die U-Entropie | `elbo.py:20` |
| Bayesian-Update und Predictive-Momente | `nig_heads.py:46`, `nig_heads.py:74` |
| Prediction gibt marginale Momente aus | `posterior_predictive.py:10` |
| Induzierter M2-Prior | `priors.py:24`, `nig_heads.py:94` |
| Annahme-/Gradienten-/Abbruchvertrag | `local_vi.py:141` |
| Baseline-Konfiguration schützt M2/F2 | `config.py:41` |

Die Pfade in dieser Tabelle beziehen sich jeweils auf
`bayesian_ssr_neubau/src/bayesian_ring_ssr/`.

## 4. Mathematisch besonders kritische Punkte

Die folgenden Präzisierungen sind **Planungs- und Algebrahinweise**. Sie wurden
in diesem Auftrag nicht als neue Verfahren implementiert oder experimentell
bestätigt. Neue empirische oder literaturgestützte Garantien werden daraus
nicht abgeleitet.

### 4.1 P1: Die AR-Emission verändert die Modellklasse

Der vorgeschlagene Term

\[
x_t\mid z_t=k,x_{t-1}\sim N(B_kx_{t-1}+b_k,\Sigma_k)
\]

ersetzt den derzeitigen zustandsabhängigen unabhängigen Emissionsterm.
Die AR-Koeffizienten werden hier B genannt, damit sie nicht mit der globalen
HMM-Übergangsmatrix A verwechselt werden.

Vor einer Architekturentscheidung müssen gemeinsame Dichte, F2-Anfangsbeobachtung,
Prior, erwartete Logdichte, Suffizienzstatistiken, gewichtete Updates und Prediction
festgelegt werden. Für beobachtete Regressoren `(1,x_(t−1))` ist eine konjugierte
Matrix-Regressionskonstruktion ein prüfbarer Kandidat. Eine aus den AR-Parametern
abgeleitete stationäre Anfangsverteilung oder ein beschränkter Stabilitätsprior
kann diese einfache Konjugiertheit verändern. SPD der Innovationskovarianz ist
keine Garantie dynamischer Stabilität; bei schaltenden Systemen muss zusätzlich
klar sein, welcher Stabilitätsbegriff überhaupt benötigt wird.

Die heute berechneten Residuen `x_t−sum_k gamma_tk*mu_k` verwenden ein aus der
gesamten Sequenz geschätztes gamma. Sie sind keine unabhängig aus früheren
Beobachtungen vorhergesagten generativen Innovationen. Hohe Lag1-Werte sind
deshalb eine Motivation zur Modellprüfung, kein fertiger Test zugunsten AR.

Ein AR-HMM muss zuerst in einem ausdrücklich definierten Grenzfall auf das
Basismodell zurückfallen, etwa bei B=0 und passendem Anfangs-/Interceptvertrag.
Auch diese Grenze ist erst herzuleiten; sie folgt nicht aus der Bezeichnung AR-HMM.

### 4.2 P2: Der vorhandene Head-only-Vergleich isoliert HMM nicht

Die physikalische NIG-Baseline nutzt L-only-Scaler und einen expliziten Intercept
mit bedingter Priorvarianz 2. SSL nutzt zusätzlich U-X beim Preprocessing und
latente h-/Delta-Merkmale. Unterschiede in ihren Fehlern lassen sich daher
nicht allein dem HMM zuschreiben.

Eine mögliche eigene symmetrische Zerlegung lautet

\[
\beta_h^Th=\bar\beta+\beta_h^T(h-K^{-1}\mathbf1),
\qquad\bar\beta=K^{-1}\mathbf1^T\beta_h.
\]

Unter dem vorhandenen Prior `beta_h|sigma² ~ N(0,2 sigma² I)` gilt
`Var(beta_bar|sigma²)=2 sigma²/K`, bei K2 also `sigma²`.
Ein gemäß genau dieser Zerlegung induzierter konstanter Prior ist damit nicht
derselbe wie die aktuelle Head-only-Interceptvarianz `2 sigma²`.
Das ist eine begrenzte algebraische Aussage über eine definierte Reduktion,
kein universeller einzig richtiger Head-only-Prior. Für P2 muss der konkrete
verschachtelte Modell-/Priorvertrag vorab entschieden werden.

Auch ein i.i.d.-GMM ist nicht durch das bloße Löschen des Transitionsterms
definiert: gemeinsame Mischgewichte und ihre Zählstatistiken ändern sich.
Der ringweite Qualitätsterm koppelt Zustände weiterhin. Würde zugleich von
freier Markov-q auf zeitlich unabhängige q gewechselt, änderten sich generatives
Modell **und** Variationsapproximation. Diese zwei Effekte müssen getrennt werden.

### 4.3 P3: Bayes kann bei gleichem Design denselben Punktprädiktor liefern

Bei festgehaltenen lokalen Faktoren und identischen Momenten `m_r,H_r` berechnet
der bestehende NIG-Kopf

\[
\mu_N=(V_0^{-1}+\sum_rH_r)^{-1}
       (V_0^{-1}\mu_0+\sum_ry_rm_r).
\]

Dies ist zugleich der Minimierer des entsprechend regularisierten erwarteten
quadratischen Fehlers. Ein passend formulierter Ridge-/MAP-Vergleich kann also
denselben Punktprädiktor besitzen. Gewöhnliches Ridge nur auf `m_r` ersetzt
`H_r` durch `m_rm_r^T` und verändert bereits die Behandlung latenter Unsicherheit.

Die spätere Studie muss Punktprädiktion bei fixierten Momenten, propagierte
Parameter-/Zustandsunsicherheit und das vollständig gekoppelte Lernen von q(z)
unterscheiden. Der V-Term in der Bayesian-Qualitätserwartung beeinflusst die
lokale q-Optimierung; sein Weglassen wäre eine eigene Modell-/Inferenzänderung.
Ein künstlich schlechter regularisiertes Ridge ist kein sinnvoller Bayes-Nachweis.

### 4.4 P4: Generalisierte Zielgröße und U-Vorverarbeitung auseinanderhalten

Die implementierte J_C gewichtet Anfang, Übergang, Emission und Entropie jedes
U-Rings gemeinsam. Bei C=1 entspricht sie der üblichen ELBO des spezifizierten
Missing-Label-Modells; C=0,2 ist eine gewichtete/generaliserte Zielgröße.
Sie wird nicht allein durch einen Literaturverweis zu einem gewöhnlichen
Posterior des ungewichteten Datensatzes.

Die Arme shared-C0 und SSL haben im vorhandenen Code dieselben zulässigen
Scaler- und K-Means-Eingaben, denselben Seed und dieselben übrigen Priors.
Damit ist der gewünschte B/C-Kontrast bereits sinnvoll angelegt. Für stärkere
Reproduktionsnachweise sollten künftige Forschungsartefakte zusätzlich die
tatsächlichen Initialzentren speichern; derzeit wird die Initialisierungspolitik
und nicht die Zentrenmatrix in den CV-Falldateien exportiert.

Die gespeicherten SSL-RMSE sind etwa 0,537/2,922 mm, shared-C0 etwa
0,547/3,230 mm und strict-C0 etwa 0,660/2,921 mm (Außen/Höhe).
Das sind bestehende Ergebnisse, keine neue Studie. Besonders Höhe zeigt, dass
„besser als shared-C0“ nicht automatisch „besser als strict“ bedeutet. Ein
universeller U-Nutzen oder eine belastbare Überlegenheit folgt daraus nicht.

### 4.5 P9: Momentenidentität, Mischungsquantile und Zielrauschen trennen

Die derzeitigen Mittelwerte und Varianzen sind innerhalb des gewählten
Markov-/NIG-Vorhersagevertrags analytisch berechnet. Eine einzelne Student-t-
Verteilung mit diesen Momenten besitzt im Allgemeinen nicht dieselben Quantile
wie die über Zustandsmerkmale gemischte Verteilung. Ein NLL-/CRPS-Vergleich
benötigt deshalb zuerst einen ausdrücklich definierten Verteilungsvertrag.

Monte-Carlo-Quantile aus der **bestehenden** q/NIG-Mischung wären eine andere
Untersuchung als ein neues korreliertes Zwei-Ziel-Rauschmodell. Geteilte
Zustandsunsicherheit erzeugt bereits marginale Abhängigkeit; dies ist nicht
gleichbedeutend mit einer gelernten gemeinsamen Residualkovarianz.

Die aktuellen Kopfdesigns unterscheiden sich in Featuregruppen und Dimension 9/8.
Eine Standard-Matrix-Normal-Inverse-Wishart-Mehrzielregression setzt zunächst
einen gemeinsamen Designvertrag voraus. Ein ungeprüftes Uniondesign oder
maskierte Koeffizienten können Featureannahmen, Prior und Konjugiertheit ändern.
Diese Architektur gehört zu einer gesonderten Nutzerentscheidung.

### 4.6 P10: Ein anderer Takt ist keine rein technische Beschleunigung

Bei Betrachtung nur jedes m-ten Schritts ist für ein gegebenes ursprüngliches A
der grobe Zustandsübergang A^m. Unter zufälligem A gilt im Allgemeinen
`E[A^m] != E[A]^m`. Derselbe Sticky-Dirichlet-Prior auf einem anderen Zeitraster
stellt daher nicht ohne Weiteres dieselbe Persistenzannahme in Sekunden dar.
Zusätzlich sind F2-Anfang, M2-Halbsplit, Deskriptoren und Likelihood-/Priorbalance
festzulegen. Ein willkürlicher Faktor 1/T definiert eine andere Zielgröße.

### 4.7 P5/P11/P12: Ein gültiger Rechenweg ist noch kein empirischer Nutzenbeleg

Monotone J_C schützt den Koordinatenvertrag, beseitigt jedoch weder lokale
Optima noch Modellfehlstruktur. Ein besserer Trainingswert ersetzt keine
gepaarte Ringbewertung. Erneut verwendete Entwicklungslabels werden durch
LORO, repeated oder nested CV nicht nachträglich unabhängig.

Mit acht Labelringen bleiben Unsicherheits- und Coverage-Auswertungen
deskriptiv. R² sollte gegebenenfalls über die vollständigen Out-of-Fold-
Vorhersagen und mit offengelegter Konvention berichtet werden: je Fold sind nur
zwei VAL-Ringe vorhanden, weshalb foldweise R² besonders instabil sein kann.
Konstante oder nahezu konstante Zielwerte benötigen eine definierte Behandlung.

## 5. Empfohlene spätere Reihenfolge und Entscheidungspunkte

Die Reihenfolge übernimmt die wissenschaftliche Priorität des Detailfahrplans.
Begleitende Qualitätsanforderungen werden vorgezogen, ohne sie als bereits
begonnene Forschungsversuche auszugeben.

| Schritt | Später zu bearbeitender Umfang | Warum an dieser Stelle? | Entscheidung vor dem nächsten Eingriff |
|---|---|---|---|
| **0 – erledigt** | P0-Referenz prüfen und erhalten; diesen Abgleich dokumentieren | Verhindert, dass bestehende Korrektheit mit späterer Hypothesenprüfung vermischt wird | Kein erneuter Neubau oder wiederholter Fit allein zur Statusbestätigung erforderlich |
| **1 – gemeinsamer Untersuchungsvertrag** | Aus P5/P11/P12: erlaubte Ringe/Folds, Scores, numerische Toleranzen, Quellenumfang und negative Kontrolle vorab festlegen | Jede folgende Aussage braucht dieselben Vergleichsregeln | Zunächst den nächsten konkreten Untersuchungsauftrag festlegen |
| **2 – P1a** | Bestehende Emissionsannahme durch bedingte Predictive-Checks, Simulation und sauber definierte Innovationen untersuchen | Höchste fachliche Priorität; bisherige gamma-Residuen allein reichen nicht | Ergebnis kann auch „AR derzeit nicht begründet“ sein |
| **3 – P1b nur nach Entscheidung** | Gemeinsame AR-Dichte, Anfangsmodell, Priors, Variationsfamilie, erwartete Logs/Updates/Gradienten und Basismodell-Grenzfall herleiten; anschließend ggf. isolierte Variante | Architektur erst nach Diagnose und mathematischer Prüfung ändern | **Nutzer entscheidet über AR-Architektur und Umsetzungsumfang** |
| **4 – P2** | HMM-/i.i.d.-/h-/M2-/Deskriptorvergleich mit angeglichenen Eingaben und begründeten effektiven Priors | Klärt den tatsächlichen Nutzen der zeitlichen latenten Struktur | Gegenmodelle und erlaubte Variationsfamilien vor dem Fit festlegen |
| **5 – P3, nötigenfalls P9a** | Punktprädiktor-Identitäten, matched Ridge/MAP und Unsicherheitsprüfung; falls nötig erst Quantile der bestehenden Mischung definieren | Bayes-Nutzen lässt sich nicht durch bloß anderes Regularisieren messen | Verteilungs-/Scorevertrag vor der Kalibrierung vereinbaren |
| **6 – P4 vervollständigen** | Vorhandene A/B/C-Ergebnisse gepaart auswerten; danach ggf. U-Teilmengen und Posterioränderungen untersuchen | Vorhandene Studie nutzen; keine automatische neue C-Suche | U-Teilmengenplan und optionalen U-Prior-Arm gesondert festlegen |
| **7 – P6/P7** | Tatsächlich verschiedene Starts und Zustandsalignment; physikalische Interpretation vertiefen | Prüft, ob frühere Effekte an ein Basin oder eine falsche Zustandsdeutung gebunden sind | Frühere Hauptbefunde nur bei tragfähiger Robustheit weiterverfolgen |
| **8 – P8** | Geplante Prior-/Feature-/Kovarianzsensitivität, Rang und Konditionierung | Reaktion auf kleine Labelzahl systematisch untersuchen | Je Variante begrenzte Änderung und Abbruchregel festlegen |
| **9 – P9b** | Falls begründet: gemeinsame Residualkovarianz oder zusätzliche globale Predictive-Integration | Zusätzliche Architektur erst nach Verständnis der vorhandenen Unsicherheit | **Nutzer entscheidet über größere Zwei-Ziel-Architekturänderungen** |
| **10 – P10** | Zeitauflösung/Ringgewichtung bei ausdrücklich geändertem probabilistischen Vertrag | Kein beiläufiges Downsampling als vermeintlich neutraler Geschwindigkeitsfix | **Neue Gewichtungs-/Zeitmodellannahme vorher entscheiden** |

P5, P11 und P12 begleiten **jeden** Schritt; ihre wissenschaftliche Priorität
wird dadurch nicht auf das Ende verschoben. P7-Diagnose kann bereits P1a
unterstützen. Ebenso kann eine durch P1a begründete Taktfrage früher untersucht
werden; diese Abweichung von der Reihenfolge wäre zu dokumentieren und vorher
zu entscheiden. Eine vollständige P6-Studie muss spätestens vor allgemeinen
Robustheitsbehauptungen erfolgen.

Der nächste sachlich passende Vorschlag ist somit **P1a: Modellprüfung der
zeitlichen Emissionsannahme mit vorab festgelegtem Untersuchungsvertrag**.
Dieser Vorschlag wurde hier nur vorbereitet, nicht ausgeführt. Er enthält keine
automatische Erlaubnis, ein AR-HMM einzubauen oder Hyperparameter zu verändern.

## 6. Einordnung der Gates A–F des Fahrplans

| Gate | Stand | Konsequenz |
|---|---|---|
| A mathematische Korrektheit | Im Basismodellvertrag erfüllt: T1–T10 und vollständiger Trainings-/Blocknachweis liegen vor | Neue Gleichungen/Verfahren benötigen eigene Kontrollen; alte Tests ersetzen diese nicht. |
| B Referenz | Erfüllt: 8L+4U, F2/Targets/Scalerpolitik, frischer Fit und vollständige Ring-CV dokumentiert | Referenzartefakte erhalten; Vergleiche in getrennten Forschungsordnern. |
| C eine Hypothese je Experiment | Als Regel gesetzt; noch kein neuer Untersuchungsvertrag ausgefüllt | Vor jedem späteren Fit Primärmaß, Gegenmodell, negative Kontrolle und Stopregel festlegen. |
| D Evidenzstatus | Für vorhandene Ergebnisse korrekt explorativ dokumentiert | Bekannte Außenringe nicht als neue Blindvalidierung verwenden. |
| E Posterior/Kalibrierung | Offen | Analytische Momente und ein ELBO-Plateau erfüllen dieses Gate nicht. |
| F Quellen/Darstellung | Für Basisspezifikation weitgehend erfüllt; für sämtliche Thesis-/Forschungsbehauptungen fortlaufend offen | Neue Aussagen brauchen konkrete Originalbelege oder deutlich eigene Algebra sowie tatsächliche Resultate. |

Eine Erweiterung wird nicht automatisch akzeptiert, weil sie komplexer ist oder
J_C erhöht. Ein negatives Ergebnis ist ein vollständig dokumentierbares
Forschungsergebnis. Wenn Prozessbeschreibung und Qualitätsfehler unterschiedliche
Richtungen zeigen, werden diese Befunde getrennt berichtet und nicht zu einem
beliebigen Gesamterfolg zusammengezogen.

## 7. Fachliche Klärungspunkte für spätere Entscheidungen

Diese Punkte verhindern den vorliegenden Dokumentabgleich nicht. Sie werden
relevant, bevor bestimmte wissenschaftliche Interpretationen oder Änderungen
freigegeben werden:

1. **Messvorschrift und Messunsicherheit:** Die zwei gespeicherten Blattfelder
   sind geprüft. Eine normative Definition der Außenunrundheit und insbesondere
   der als „Höhenschlag“ bezeichneten `planarity.height_deviation_mm` sowie
   Wiederholungsmessungen/Mittelung müssen durch Originalmessvorschriften belegt
   werden. Ohne sie bleibt die metrologische Interpretation begrenzt.
2. **Rolle der Zustände:** Sollen sie statistische Prozessregime erklären oder
   mit konkreten Walzphasen identifiziert werden? Der derzeitige Code erzwingt
   keine Phasenidentität. Mechanische Aussagen benötigen zusätzliche Fachbelege.
3. **Bayes-Ziel und Gewichtung:** C=0,2 ist eine historische Festlegung der
   generalisierten Zielgröße. Die methodische Begründung gegenüber dem Betreuer
   muss vom gewöhnlichen C=1-Modell unterschieden werden.
4. **Forschungsziel:** Der jetzige Nutzerauftrag priorisiert eine verteidigbare
   mathematische Struktur, HMM-/Bayes-/SSL-Verständnis und Generalisierung.
   Ein kleinerer RMSE allein entscheidet daher nicht über die nächste Architektur.
5. **Vertretbarer Unsicherheitsnachweis:** Kalibrierung an synthetisch bekannten
   Fällen und deskriptive Ring-Coverage haben unterschiedliche Rollen. Ohne neue
   Ringe entsteht keine neue unabhängige externe Bestätigung.
6. **Eigenleistung:** Die konkrete Zweikopf-/M2-/VI-Kombination wird durch eigene
   Herleitung und Tests begründet; Literatur liefert gezielt überprüfte Bausteine.
   Eine darüber hinausgehende Neuheitsbehauptung verlangt eine eigene Recherche.

## 8. Quellenumfang und nächste Dokumentation

Die derzeitigen konkreten Originalbelegstellen stehen in
[QUELLENSTATUS.md](../../bayesian_ssr_neubau/docs/QUELLENSTATUS.md).
Dieser Abgleich übernimmt deren ausdrücklich begrenzten Lesestatus; es wurden
hier keine neuen Literaturseiten geprüft oder erfunden. Die 82 PDF-Dateien und
133 Literaturkarten sind keine 82 beziehungsweise 133 vollständig gelesenen
wissenschaftlichen Originale. Beal ist ein Kapitelauszug; eine PDF-Datei ist ein
Laborprotokoll. Die eigene C-Gewichtung wird nicht mit einer abweichenden
Shao-Gewichtung gleichgesetzt.

Für spätere einzelne Untersuchungen liegt eine noch nicht ausgefüllte
[UNTERSUCHUNGSVORLAGE.md](UNTERSUCHUNGSVORLAGE.md) bereit. Sie fordert Hypothese,
mathematischen Vertrag, isolierten Codebezug, unabhängige Prüfungen, reale
Ausführungsevidenz, Grenzen und die begründete Entscheidung. Das Vorhandensein
dieser Vorlage startet keine Untersuchung.
