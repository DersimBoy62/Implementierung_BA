# Quellenstatus und tatsächlich gelesene Originalstellen

Stand: 13.11.2026. Maßgeblich sind die unveränderten Dateien in
`handover/07_ORIGINALQUELLEN_PDF`. Vollständige SHA256 und Dateinamen stehen in
[Inventur.csv](Inventur.csv); die bytegleiche Zuordnung zu den fünf Nutzerarchiven
steht in [ARCHIV_HERKUNFT.json](ARCHIV_HERKUNFT.json). Die historischen 133
Literaturkarten sind kein Nachweis für 133 gelesene Volltexte.

Es werden drei Statusklassen unterschieden:

| Klasse | Bedeutung |
|---|---|
| A | Originaldatei vorhanden; der unten bezeichnete konkrete Auszug wurde in diesem Neubau gelesen. Das bedeutet keine Vollprüfung des gesamten Papers oder seiner Supplements. |
| B | PDF vorhanden, aber eine konkrete Seiten-/Methodenprüfung für die beabsichtigte Aussage ist offen. Aus diesem Status wird keine inhaltliche Formelbehauptung abgeleitet. |
| C | Nur bibliografisch bekannt; kein zugeordneter Originalvolltext nachgewiesen. Historische Karten ohne neue Dateizuordnung bleiben hier oder ausdrücklich ungeklärt. |

Alle Seitenangaben in der zweiten Spalte sind **1-basierte PDF-Dateiseiten**.
Gedruckte Seiten werden nur zusätzlich genannt, wenn die Seitenziffer im
ausgelesenen Original sichtbar war. Die PDF-Textschicht wurde mit `pypdf` gelesen;
dies ist kein flächendeckender visueller Faksimile-/Glyphenaudit. Insbesondere die
Gelman-Datei hat teilweise fehlerhafte Schriftkodierung. Hieraus werden keine
wörtlichen Formelzitate und keine ungesicherten gedruckten Seitennummern übernommen.
Der Lesebefehl ist reproduzierbar, zum Beispiel:

```powershell
.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\audit_sources.py pdf --pattern Beal --pages 10-15
```

## Mathematische und methodische Kernbelege, Klasse A

| Original | Gelesene PDF-Seiten / sicher im Text ausgewiesene Druckseiten | Damit gestützte begrenzte Aussage | Nicht daraus ableiten |
|---|---|---|---|
| Rabiner (1989), *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition* | PDF 1–8; Druck 257–264, insbesondere PDF 6–8 / Druck 262–264 | HMM-Pfadfaktorisierung, Forward-/Backward-Rekursionen, gamma/xi und Unterschied zwischen marginalem argmax und Viterbi; Gl. 13–28, 36–37 | Keine Bayes-NIG-Kopfkopplung und keine Ringwalz-Validierung |
| Beal (2003), *Variational Algorithms for Approximate Bayesian Inference*, **Kapitel-3-Auszug** | PDF 10–15; Druck 91–96, insbesondere PDF 11–15 / Druck 92–96 | Globale/lokale VI-Faktorisierung, Dirichletupdates 3.54–3.57, Elog-Potenziale 3.60–3.71 und KL-/Entropieschranke 3.75–3.79 | Die gelieferte Datei hat nur 24 PDF-Seiten und ist nicht die vollständige Dissertation; keine identische ringweite Qualitätsregression |
| Bishop (2006), *Pattern Recognition and Machine Learning* | PDF 710,712,713; Druck 690,692,693; zusätzlich Kontext PDF 711,715–717 | Gaussian-Wishart-Konjugation B.53; Student-t-Momente B.64–B.70; Wishart-Skala, Dichte und Elogdet B.78–B.82 | NIG-Skalenkonventionen werden nicht aufgrund des Wortes „scale“ erraten; die transformierte IG-Dichte und ihr KL werden im Modell explizit angegeben |
| Shao, Xiao, Wang, Zhao & Song (2022), *Real-time estimation ... semisupervised Bayesian HMM* | PDF 1–4 und 6, Teile von PDF 5; Druck 59–62 und 64, Kontext 63 | Dynamischer industrieller Soft-Sensor, getrennte labeled/unlabeled Likelihoods 30–33, Normal-Wishart-Update 40 und Elog-FB 41–43 | Deren y ist zeitpunktbezogen; deren Regressionsprior ist kein identischer ringweiter zweiköpfiger NIG-Prior |
| Shao, Ge & Song (2021), *Semisupervised Bayesian Gaussian Mixture Models for Non-Gaussian Soft Sensor* | PDF 1,3–5; Druck 3455,3457–3459 | Industrieller SSL-Kontext und WVI-Ansatz; konkrete C-Gewichtung 7,24–35 gelesen | **Kein Beleg**, dass unser vollständiger U-Prozess-und-Entropie-Beitrag unverändert dem Paper entspricht |
| Mowbray et al. (2022), *Probabilistic machine learning based soft-sensors for product quality prediction in batch processes* | PDF 1–2; PDF 2 zeigt Druckseite 2, Artikel 104616 | Vollständige Batchtrajektorie mit einer Endqualitätsmessung; Redundanz und wenige Labels; §1.1 | Kein identisches Bayesian-SSR-HMM; dort MPLS mit GP/BNN/HNN |
| Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection* | PDF 1–2; im Artikel ausdrücklich keine zitierfähigen Druckseiten, Artikel 7:91 | Selektionsverzerrung bei Optimierung gegen CV; alle Auswahl-/Fitschritte gehören in die Schleife | Neue Unabhängigkeit historisch vielfach verwendeter Entwicklungslabels entsteht nicht durch erneutes CV |
| Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation* | PDF 1–2; Druck 2079–2080 | Varianz des Auswahlkriteriums ermöglicht Auswahlüberanpassung; Modellauswahl ist Teil der zu evaluierenden Prozedur | Kein Nachweis einer konkreten Ringmodell-Überlegenheit |
| Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure* | PDF 1–2; Druck 913–914 | Abhängige Gruppen erfordern passend blockierte Bewertung; Modellieren von Korrelation beseitigt das CV-Problem nicht automatisch | Datei nennt 2016, **Zeitschriftenausgabe ist Ecography 40 (2017), 913–929**; Online-/Copyrightjahr 2016 unterscheiden |
| Gabry et al. (2019), *Visualization in Bayesian workflow* | PDF 1–2; Druck 389–390 | Modellaufbau, Simulation, Rechendiagnose, posterior predictive checks und CV haben getrennte Rollen | Ein bestandener Rechentest validiert keine reale prädiktive Kalibrierung |
| Gelman et al. (2020), *Bayesian workflow*, arXiv:2011.01808v1 | PDF 1–2, Abstract und Inhaltsübersicht | Bayesian Workflow umfasst Modellprüfung und iterative Diagnose zusätzlich zur Posteriorberechnung | Textschicht teilweise beschädigt; konkrete Formel-/Seitenprüfung tieferer Kapitel offen; kein formelgenauer Modellbeleg |

### Relevanter Widerspruch bei der Gewichtung

Shao/Ge/Song 2021 verwendet in Gl. (24)–(25) für q(Z_U) den C-gewichteten
Emissionslogterm plus **ungewichteten** Elog(pi)-Term. Das Dirichletupdate (30)
enthält N_L+N_U, das Normal-Wishart-Update (32)–(35) dagegen N_L+C N_U.
Unsere Basisspezifikation gewichtet Anfang, Transition, Emission **und Entropie**
jedes U-Rings gemeinsam. Daher sind diese Konstruktionen nicht identisch. Die
eigene Ableitung der J_C und ihr Test sind erforderlich; das Paper ist kein Ersatz
dafür. Auch aus einem im PDF-Text beschädigten Digamma-Ausdruck darf kein Code
kopiert werden: die korrekte Dirichlet-Erwartung ist psi(alpha_k)-psi(sum alpha),
wie Beal 3.68 ausdrücklich zeigt.

### Quellen zur Abgrenzung, Klasse A mit begrenztem Lesenumfang

Hier wurden jeweils Titel, Abstract und Anfang von PDF-Seite 1 gelesen. Diese
Stellen stützen nur die angegebene Einordnung. Methodenformeln und Supplements
wurden für den Basismodell-Neubau nicht vollständig geprüft.

| Original | Konkreter gelesener Beleg und Konsequenz |
|---|---|
| Yuan et al. (2024), *Soft sensor development for mixed oil interface tracking ... knowledge-informed semi-supervised Variational Bayesian Gaussian mixture regression* | Abstract benennt industrial knowledge zur Priorfestlegung und eine semi-supervised Gaussian-mixture-Struktur; kein mechanistischer Ringwalzprior daraus übernommen. DOI 10.1016/j.energy.2024.131516. |
| Li et al. (2026), *A neural networks-aided Bayesian semi-supervised nonlinear hidden Markov model for regressive soft sensing application* | Abstract benennt neuronale Netze im HMM und kombinierte VI/Gradienteninferenz. Ausgabe Measurement257 (2026)118485, online bereits 29.07.2025, DOI 10.1016/j.measurement.2025.118485. Kein Beleg für unsere linear-konjugierte Kopfarchitektur. |
| Dai, Li & Cheng (2023), *Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks* | Abstract/Fig.1 beschreiben Pseudolabels, Unsicherheitskonsistenz und UCVME; unsere U-Integration erzeugt diese Labels nicht. PDF1 weist Druck7304 aus. |
| Kang, Kim & Cho (2016), *Semi-supervised support vector regression based on self-training with label uncertainty ...* | Abstract beschreibt self-training, Schätzung einer Labelverteilung und datenabhängiges Oversampling. Keine unveränderte Bayesian-HMM-Inferenz. PDF1 weist Druck85 aus. |
| van Engelen & Hoos (2020), *A survey on semi-supervised learning* | Abstract sagt ausdrücklich Schwerpunkt semi-supervised **classification**; allgemeine Taxonomie und Annahmen, kein spezifischer Beleg der ringweiten Regressionskonstruktion. PDF1 weist Druck373 aus. |
| Kostopoulos et al. (2018), *Semi-supervised regression: A recent review* | Abstract beschreibt SSR-Taxonomie und begrenzte Literatur; keine Gleichheit unserer Kombination mit einer Einzelmethode behauptet. PDF1 weist Druck1483 aus. |
| Kaufman, Rosset & Perlich (2011), *Leakage in Data Mining: Formulation, Detection, and Avoidance* | Abstract benennt illegitim verfügbare Zielinformation und learn-predict separation. Daraus motiviert sind Prozess-only-Prediction und gesondertes Scoring; PDF1 weist Druck556 aus. |
| Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in machine-learning-based science* | Highlights auf PDF1 benennen überoptimistische Ergebnisse durch Leakage und Model Info Sheets. Keine konkrete Formel oder Seitenbehauptung des Methodenteils. Artikel Patterns4,100804. |

## Vorhanden, aber nicht als geprüfter Beleg benutzt: Klasse B

Die übrigen PDF-Dateien sind per SHA256 inventarisiert. Ohne Eintrag in der obigen
Tabelle ist ihre Seitenprüfung in diesem Arbeitslauf offen. Auch scheinbar ähnliche
Zweitdateien zu Shao2021/2022 werden nicht automatisch als gelesen markiert, nur
weil der Titel ähnlich ist. Byte-Hash, bibliografische Identität und Inhaltsprüfung
sind unterschiedliche Prüfungen. Für eine Bachelorarbeitszitation über den hier
dokumentierten Aussageumfang hinaus muss die konkrete Originalstelle neu geprüft
werden. Vorhandene Supplements wurden nicht pauschal geprüft.

Ein Sonderfall wurde identifiziert: `Formatvorlage Röntgenfluoreszenzanalyse Gruppe 2.pdf`
ist laut tatsächlich gelesener PDF-Seite1 ein studentisches Fachlaborprotokoll zur
additiven Fertigung (26 PDF-Seiten), **kein methodischer Originalbeleg** dieses
Modells. Daher heißt die Paketmenge korrekt „82 PDF-Dateien“, nicht pauschal
„82 wissenschaftliche Originalpaper“. Das Dokument wird nicht als HMM-/Bayesquelle
zitiert.

## Nur bibliografisch: Klasse C und offene Zuordnung

Die historische Datei `Quellenstatus_133_Literaturkarten.csv` bleibt eine historische
Kartei. Einträge ohne nachgewiesene Datei dürfen nur als bibliografisch bekannt
bezeichnet werden. Eine neue vollständige Zuordnung aller 133 Karten zu den 82
Dateien ist nicht abgeschlossen; insbesondere werden die früheren „112 nicht
identifiziert“ nicht als heutiger Fehlbestand ausgegeben. Für die oben benutzten
Kernquellen ist jeweils ein konkreter Originalauszug vorhanden und gelesen.

## Eigenanteil und Reichweite der Belege

Die konkrete Verbindung aus zwei ringweiten NIG-Köpfen, Normal-Wishart-Emissionen,
freier Markov-VI, exakten M2-Momenten und NIG-Priorpushforward wird durch eigene
Algebra und unabhängige synthetische Tests gestützt. Keine Literaturquelle wird
als Beweis genau dieser Gesamtarchitektur oder einer neuen empirischen Überlegenheit
ausgegeben. Äquivalenztests prüfen mathematische Implementierungen. Ring-CV prüft
getrennt davon explorative Vorhersagefehler; die acht vielfach verwendeten
Entwicklungslabels und die vier historisch geöffneten Außenringe können keine neue
unabhängige Bestätigung liefern.
