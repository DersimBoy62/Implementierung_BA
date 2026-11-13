# Soll-/Ist-Abgleich der Herleitungen und des Neubaus

Stand 13.11.2026. Lesebasis sind `00_START_HIER.md`, die vier neuen A–D-Markdown-
Dokumente, die Original-A–D-Word-Dateien einschließlich OMML-Text, die alte
Abgleichmatrix, die Entscheidungsmatrix, der konsolidierte Projektstatus und
die READMEs V5.8–V6.3. Die extrahierten Word-Leseansichten unter `leseansichten/`
sind Hilfsmittel; Originaldateien wurden nicht geändert. SHA256/Leseumfang stehen
in der historischen [Inventur.csv](https://github.com/DersimBoy62/Implementierung_BA/blob/archive/vollstand-2026-11-13/bayesian_ssr_neubau/docs/Inventur.csv)
und [READ_STATUS.json](https://github.com/DersimBoy62/Implementierung_BA/blob/archive/vollstand-2026-11-13/bayesian_ssr_neubau/docs/READ_STATUS.json).
Diese Lesebasis beschreibt den ursprünglichen Neubau. Der vollständige
Gesamtbestand ist auf `archive/vollstand-2026-11-13` archiviert; das aktive
Projekt enthält die daraus benötigten mathematischen Testreferenzen.

Es gelten folgende Vorränge: expliziter Nutzerauftrag vom 13.11.2026; dort
übernommene V5.8-/V5.9-/V6.0-Korrekturen; verifizierter Daten-/Codevertrag;
A–D-Neufassungen vom 11.11.; Originale vom Oktober nur als historische Konzepte.
Eine historische Notiz „kein neues Training ohne Freigabe“ ist durch den jetzigen
ausdrücklichen Neubau-/Trainingsauftrag erfüllt. Ein historisches
`PROPOSED_MODEL_FREEZE.json` ist weiterhin keine Endfreigabe des Neubaus.

## 1. Konkrete Widersprüche und getroffene Entscheidungen

| Fundstelle | Widerspruch | Entscheidung und Folge |
|---|---|---|
| Original A–D | Drei Ziele einschließlich Innenunrundheit, diagonale Emissionen, h-only-Regressionsdesign | Zwei bestätigte Ziele, volle D6-Emissionen mit NW, M2 und zwei verschiedene Deskriptorgruppen. Keine Übernahme der alten Drei-Ziel-Architektur. |
| Neue A.1/A.6, C.1/C.8 | K3,C1,10 Epochen und h-only werden als eingefroren bezeichnet | Aktueller Ausgangsvertrag ist K2,C.2,30 Epochen,M2. Historische Variante bleibt lesbare Evidenz. |
| Neue A.4/B.5/D.1 | Dimensionen 9/8 aus K3+h-only+6/5 Deskriptoren | Neue 9/8 entstehen bei K2 aus K+(K-1)+6/5 nach M2-Reduktion; volle M2-Dimensionen sind 10/9. Gleiche Zahlen bedeuten unterschiedliche Designs. |
| Neue A.4.1 | Physikalische Priorvarianz .25 | Historischer Ausgangswert .38526108747593474; weiterhin Entwicklungsentscheidung, kein bewiesenes Optimum. |
| Auftrag / tatsächliches Pickle | Herkunftsbezeichnung `all_targets`, realer Container `zielwerte` | Verwende ausschließlich die bestätigten Blattfelder aus `zielwerte`, nur für L-TRAIN. Kein Ersatzfeld und kein Umdeuten fehlender Metrologiebelege. |
| Historische Kanalnamen | `force` könnte als kN, `RWG55` als Höhe fehlgelesen werden | Originalheader: Kraftkanäle %, Geometrie m, RWG55 m/s/ring growth rate. Kalibrierte physikalische Kräfte werden nicht erfunden. |
| Original C.7.4 | Adam als anfängliche Untersuchung vorgeschlagen | Basismodell verwendet überprüfbares L-BFGS-B mit analytischem Gradient, besserem Warmstart/FB und Annahmesicherung. Kein stiller Optimiererwechsel im Vergleich. |
| V5.2 lokale Restart-Regel | FB ist besser als voriges q nicht garantiert | V5.8: beide Kandidaten unter aktuellen globalen Faktoren vergleichen; vollständige lokale Funktion prüfen und pro Block dieselbe J_C auswerten. |
| Softmax-Gauge | K volle Logits für K-1 Freiheitsgrade | V5.9: letzten Logit null setzen, korrekter Pullback; gleiche Familie, nicht notwendigerweise gleicher Optimiererpfad. |
| M2-Redundanz | Spalte entfernen könnte den Prior ändern; sum(h)=1 fälschlich linear behandelt | V6.0: nur letztes Delta entfernen, alle h behalten; V_red=A^T V_full A. Bei K2 wirksame Deltavarianz .30 statt .15. |
| Quelle Shao/Ge/Song2021 | Eigene Gewichtung wurde teils als bloße Übernahme interpretiert | Originalgl.24–30 gewichten anders: U-Emission mit C, Zustandsprior ungewichtet. Vollständige U-Prozess+Entropie-Gewichtung ist unsere explizit hergeleitete Konstruktion. |
| Neue D.2 | Die „ideale volle“ Formel integriert p(theta|D) statt p(theta|X*,D) | Im generativen X-Modell ist dies eine modulare Approximation. Original D.2/D.5 trennt beide Fälle korrekter. Neubau beschreibt ehrlich Elog-FB und q-bedingte Mischungsmomente. |
| Alte externe Auswertung | Historische „äußere“ Ringe könnten erneut als blind ausgegeben werden | DM07/08/11/15 werden nicht trainiert oder neu gescort. Entwicklungs-CV bleibt explorativ und historisch beeinflusst. |
| Quellenpaket | 82 Dateien wurden pauschal als Originalpapers bezeichnet | Mindestens die „Formatvorlage Röntgenfluoreszenzanalyse Gruppe2“ ist ein Laborprotokoll. Beal ist ein Kapitel-Auszug. 82 PDF-Dateien sind kein Beleg für82 wissenschaftliche Volltexte. |

## 2. Formelweiser Übergang A: generatives Modell

| Abschnitt | Erhaltener Kern / notwendige Änderung | Neue Implementierung / unabhängige Prüfung |
|---|---|---|
| Original A.1–A.2; neue A.1–A.2 | Vollständiger Ring bleibt Beobachtungseinheit; gemeinsamer homogener HMM, eigene Zustandsfolge je Ring | `schema.py`, `split_guard.py`, `forward_backward.py`; T1/T7/T9 |
| Original A.3; neue A.3 | Diagonal-NIG ersetzen durch Normal-Wishart mit voller Präzision; Wishart-Skala explizit | `normal_wishart.py`; `test_nw_dirichlet_nig.py` und NW-Oracle |
| A.4.1–A.4.3 | h ist zufällige Pfadstatistik; M2 ergänzt early/late-Delta, kein zusätzlicher Intercept | `hmmmoments.py`, `priors.py`; Pfadenumeration, `test_m2_induced_prior.py` |
| A.4/A.5 | Zwei Likelihoods p(y_j|X,z), deterministische s(X) keine dritte Datenlikelihood | `physics_descriptors.py`, `nig_heads.py`; Test der augmentierten Momente und der komprimierten Qualitätsfunktion |
| A.6 | Dirichlet/NW und zwei bedingte NIG-Faktoren statt unabhängiger Punktparameter | `priors.py`, `normal_wishart.py`, `nig_heads.py`; unabhängige Dichte/KL-/Posteriorprüfungen |
| A.7–A.8 | U-Qualitätsdichte integriert zu1; keine künstlichen y; X-only-Datenvertrag | `data_io.py`, `schema.UnlabeledRing`, `preprocessing.py`; T7 und Prozess-only-Export |

Die vollständig aktuelle Faktorisierung und alle verwendeten Parameter stehen in
[MATHEMATISCHES_MODELL.md](MATHEMATISCHES_MODELL.md), §§1–4. Die Gleichungen der
Originale mit drei Outputs oder univariaten Emissionen sind ausdrücklich ersetzt.

## 3. Übergang B: Variationsfamilie, Momente und Gradient

| Abschnitt | Aktuelle mathematische Aussage | Code-/Prüfbezug |
|---|---|---|
| B.1–B.4 | q(pi)q(A)prod NW prod NIG prod q_r; Abhängigkeiten innerhalb NW/NIG bleiben erhalten, global-lokal ist approximativ | Faktorobjekte im neuen Kern; T4 |
| B.5–B.6 | rho und Q_t sind ringspezifisch frei; gamma/xi exakt konsistent; U-FB auf erwarteten Logs | `reduced_markov.py`, `forward_backward.py`; `test_forward_backward_exact.py` |
| B.7–B.8 | Exakte additive erste und zweite M2-Momente unter q, einschließlich zeitlicher Kreuzkorrelation | `hmmmoments.py`; `test_markov_moments_enumeration.py`, K2/3/4-Enumeration |
| B.5 und V5.9 | K-1 Log-Odds je Zeile, letzter Logit0; Gradient erster K-1 Voll-Logitkomponenten nach korrektem Pullback | `reduced_markov.py`, `local_vi.py`; `test_reduced_logits_grad.py` |
| B.9/B.6 neu | Erwarteter Qualitätsterm verwendet E[beta beta^T/v]=V+(a/b)mu mu^T und E[uu^T] | `nig_heads.py`; kontrollierte Likelihood-/Kompressionstests |
| B.10–B.11 | Beide Heads wirken über eine gemeinsame komprimierte Funktion auf denselben q_r zurück | `local_vi.py`; lokale Funktion/Gradient gegen Differenzen und echtes V6.0-Oracle |
| V6.0 | Delta-Lift und induzierter Prior erhalten effektive Funktionen; kein allgemeiner KL-Satz für freie alte q | `priors.py`, `nig_heads.py`; T5 und `test_v60_oracle.py` |

Der neue Momenten-/Gradientenkern verwendet eigenständig formulierte bedingte
Rewardrekursionen. Alte Mathematikmodule sind ausschließlich in separaten
Referenztests als Oracle zulässig; sie sind keine Produktivabhängigkeit.
Die mathematischen Herleitungen stehen in Modell-§§3,5,6.

## 4. Übergang C: Training und ELBO

| Abschnitt | Aktuelle Aussage | Prüfung / Grenze |
|---|---|---|
| C.3–C.4 | J_C addiert Initial, Transition, Emission, Markoventropie und beide L-Qualitäten, minus globale KL | `elbo.py`, Komponentenbilanz in `test_elbo_decomposition.py` |
| C.4.2 | NW-Elog-Emission enthält D/kappa und nu*(x-m)^T W(x-m) sowie Elogdet | unabhängige Normal-Wishart-Dichte/KL und V6.0-Parität |
| C.5.1–C.5.2 | pi/A/NW aus gewichteten aktuellen Suffizienzstatistiken, keine wiederholte Prior-/Datenaddition pro Epoche | Prozessupdate und V6.1-Komponentenbilanz; T4/T6 |
| C.5.3 | Kopfupdates aus sum E[uu^T] und sum y E[u], nur L-TRAIN | NIG-Posteriorformel; induzierter M2-Prior und Oracle |
| C.6 | Globale KL = pi + A-Zeilen + NW-Zustände + zwei NIG; keine ausgelassenen Entropie-/Normalisierkonstanten | getrennte analytische KL-Funktionen und unabhängige Kontrollen |
| C.7/V5.8 | besserer alter Faktor oder FB als Start; Kandidat nur bei nicht schlechter voller lokaler Funktion; vollständige Blockbilanz | lokale Tests, Koordinaten- und synthetische End-to-End-Prüfung; festgelegtes monotonic_atol=1e-6 |
| C.8 | C wirkt auf gesamten U-Beitrag; C0 ohne U-Prozessupdate, Preprocessing-Policy gesondert | gewichtete Statistik-/Entropietests, C0-Regressionstest |
| C.9–C.13 | Fresh init,8L+4U,30E, lokale Gradienten-/Abbruchdiagnose, write-once Artefakte | echte Commands und Resultate ausschließlich im neuen Test-/Abschlussprotokoll |

Historische Trainingszahlen sind Vergleichspunkte, keine zu erzwingenden Zielwerte.
Der neue Lauf muss eigene Blockbilanzen liefern. Ein annähernd gleicher Endwert
beweist keine robuste globale Konvergenz. V6.1–V6.3 diagnostizieren Emissionen,
Persistenz und Phasengrenzen; sie enthalten keinen hergeleiteten AR-HMM-Update.

## 5. Übergang D: Posterior Predictive und Evaluation

| Abschnitt | Aktuelle Aussage | Code-/Prüfbezug und verbleibende Grenze |
|---|---|---|
| D.1–D.3 | Nur neuer Prozess und gespeicherte Trainings-Transformationen; y-unabhängiges Elog-FB | `posterior_predictive.py`, Prozess-only-CLI; Prediction-Boundary-Tests |
| D.4 | M2-Momente der neuen q_*(z|X) exakt, keine Training-q eines anderen Rings wiederverwenden | Momenten-Enumeration und Predictiontests |
| Original D.5/neue D.2 | Voll sequenzbedingte globale HMM-Parameterintegration bleibt offen | ausdrücklich modulare/variationale Näherung; kein heimliches P2/Importance-Sampling |
| D.6–D.8 | NIG bedingt Student-t; nach Zustandsmischung allgemeine Mischung; Mittelwert und Varianz aus m,H | `nig_heads.predictive_moments`, T4/T5/T8; keine exakten Mischungsquantile |
| D.7/D.8 | Zwei marginale SD; gemeinsame Zustandsunsicherheit bedeutet nicht volles korreliertes Rauschmodell | mathematische Kreuzkovarianz nur als Konsequenz beschrieben |
| D.9–D.11 | mm-Rückskalierung; ringgleichgewichtete Entwicklungsauswertung; keine erneute äußere Blindprüfung | `metrics.py`, Entwicklungs-CV; geringe Fallzahl und historische Auswahl bleiben Begrenzungen |
| Persistenz | Konjugierte Faktoren, Feature-/Gatevertrag, train-only Scaler, Split und Herkunft mit Integrität sichern | `persistence.py`; Load-vs-in-memory und write-once-Tests |

Ein weitergehendes „ideales“ generatives Vorhersageziel müsste bei globaler
Integration p(theta|X*,D) proportional p(X*|theta)p(theta|D) benutzen. Die effizientere
Elog-FB-Lösung tut dies nicht und wird nicht als exakte vollständige
Posteriorprädiktive bezeichnet. Auch die acht/zwölf Entwicklungsringe werden
nicht zu unabhängigen neuen Außenringen umetikettiert.

## 6. Was am Neubau tatsächlich mit V6.0 verglichen wurde

Der separate Referenztest wurde am 13.11.2026 mit dem tatsächlichen
`research_math_v60/m2_prior_reduction_v60.py` ausgeführt. Seine Abhängigkeitskette
prüft V5.9/V5.8 und die zehn V5.2-Kernmodule per SHA256. **12 Tests bestanden**
(Exit0;34,27s), und zwar für K2/3/4 jeweils lokale Momente/Gradient, NW, NIG und
drei frisch initialisierte synthetische Trainingsepochen.
Toleranzen, Voraussetzungen und exakter Command stehen in
[V60_PARITAET.md](V60_PARITAET.md). Kein altes Modell, alter Scaler oder
historischer Fit-Zustand war Initialzustand dieser Tests.

Die mathematischen Konstruktionen M2-Reduktion und Gauge sind damit in den
kontrollierten Beispielen numerisch gleichwertig geprüft. Daraus folgen keine
Identität beliebiger alten q-Familien, kein globales Optimum und keine bessere
reale Qualitätsprognose. Reales30E-Training und ringweise CV haben eigene
Artefakte und dürfen nur mit ihrem tatsächlich erreichten Status berichtet werden.

## 7. Offene wissenschaftliche Nachweise

Die externe Originalmetrologie, vollständige HMM-Parameterintegration,
Unsicherheitskalibrierung, echte Initialisierungsrobustheit und unabhängige
zukünftige Ringgeneralisierung sind weiterhin offen. Keine dieser Fragen wird
durch bloß mehr Epochen oder durch identische K-Means-Zentren beantwortet.
Weitere Architektur-/Prior-/AR-Änderungen bleiben dem getrennten Optimierungsauftrag
vorbehalten. Der neue Code und seine eigenen Ausführungsprotokolle bilden den
reviewbaren Basismodellstand; historische Berichte bleiben als solche kenntlich.
