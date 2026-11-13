# MASTER-PROMPT FÜR DEN VS-CODE-AGENTEN — BAYESIAN BACHELOR

**Übergabe: 13.11.2026 | Projekt: Bayesian Bachelor | Sprache der Kommunikation und Dokumentation: Deutsch**

> **DIESEN GESAMTEN TEXT ALS AUFTRAG AN DEN CHATGPT-AGENTEN IN VS CODE GEBEN.** Ich möchte eine neue, saubere Implementierung von Grund auf in meinem bestehenden lokalen VS-Code-Projekt. Die beigefügten Herleitungen, geprüften Referenzmodule, Originalquellen und früheren Analyseergebnisse sind verbindliche Arbeitsunterlagen, aber NICHT selbst das neue Modell. Arbeite wissenschaftlich, transparent, codezentriert, testgetrieben und mit präzisen Statusangaben. Wo Gleichungen, Code oder Quellen nicht übereinstimmen, löse den Widerspruch ausdrücklich. Keine stillschweigenden Vereinfachungen, keine erfundenen Nachweise, keine voreiligen Erfolgsbehauptungen.

---

## 0. Rolle, Auftrag und nicht verhandelbares Ziel

Du bist mein wissenschaftlicher Python-Entwicklungsagent für meine Bachelorarbeit über **semi-supervised Bayesian-Regression im radial-axialen Ringwalzprozess**. Es geht nicht um ein generisches ML-Beispiel, eine Regression über unabhängige Tabellenzeilen oder Pseudo-Labeling. Mein Ziel ist ein mathematisch und numerisch prüfbares **zweiköpfiges Bayesian Hidden-Markov-Model für vollständige industrielle Loggersequenzen**, das gelabelte und ungelabelte Ringe in einem gemeinsamen probabilistischen Modell verwendet.

Bau die neue Implementierung **wirklich von Grund auf neu**: neue Modulstruktur, eigene implementierte Algorithmen, neue Tests und Persistenz. Verwende die alten Core-Dateien und Forschungsoverlays ausschließlich als lesbare Referenz und unabhängiges numerisches Oracle. **Importiere NICHT** die alten Model-Core-Module in den neuen Produktivcode und kopiere sie nicht unverändert als neue Architektur. Alte Modelle und `results/` dürfen nicht als trainierte Startzustände dienen. Keine alten Gewichte, Posteriorverteilungen, K-Means-Zentren, Skalierer, Checkpoints oder Zielvorhersagen laden, um das neue Modell zu trainieren. Der bekannte Seed darf zur reproduzierbaren Neuinitalisierung verwendet werden.

**Reihenfolge:** (A) Bestandsaufnahme mit Herkunftsnachweisen, (B) mathematische Spezifikation, (C) neues modulares Design, (D) mathematische Einzeltests, (E) integriertes Training von Null, (F) fachlich richtige, **ringgetrennte** Entwicklungsevaluation, (G) reproduzierbare CLI/Artefakte/Handbuch. Ich möchte, dass du eigenständig innerhalb der lokalen Umgebung arbeitest, tatsächlichen Code erzeugst, Tests ausführst, Fehler behebst und Ergebnisse lieferst, nicht lediglich einen Konzeptplan. Falls du eine zwingende fachliche Unklarheit findest, dokumentiere sie mit konkreten Optionen und ihrer Konsequenz; erledige alle unabhängig lösbaren Arbeiten trotzdem.

In diesem ersten Bauauftrag entsteht ein **vollständiges korrektes Basismodell**, noch kein endgültiger Freigabe-/Freeze-Kandidat. Experimentelle Erweiterungen aus dem separaten Word-Optimierungsfahrplan sollen NICHT unkontrolliert gleichzeitig eingeführt werden. Die inzwischen mathematisch als äquivalent nachgewiesenen strukturellen Korrekturen aus V5.8/V5.9/V6.0 hingegen gehören in die neue Basis.

## 1. Was du zuerst im Projektordner prüfen musst

1. Finde das existierende Projekt unter `C:\Users\umutj\OneDrive\Dokumente\PA und BA\Implementierung_BA` oder im aktuell geöffneten VS-Code-Workspace. Verwende relative Pfade ab Workspace-Root. Bestehendes `input/`, `final_candidate_v52/`, `research_math_v58/` … `research_math_v63/`, `results/` und `.venv` nicht löschen oder überschreiben.
2. Lies **zuerst** `00_START_HIER.md`, alle A–D-Herleitungen unter `02_HERLEITUNGEN_A_B_C_D`, die Abgleichmatrix, die Forschungsstatusdateien und die READMEs V5.8–V6.3. Analysiere anschließend den alten `final_candidate_v52/final_core`-Snapshot und die tatsächliche Umgebung. Die Übergabekopien liegen unter `04_REFERENZCODE_READ_ONLY/`.
3. Prüfe, ob `input/DM_16_aktive_Ringe.pkl` existiert und der SHA256 exakt `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5` ist. Bei Abweichung **abbrechen, nicht stillschweigend ein anderes Dataset verwenden**.
4. Erfasse Python-Version (bisher Python 3.11.9), NumPy, SciPy, scikit-learn, pandas, pytest und etwaige lockfiles; schreibe eine neue reproduzierbare `requirements.txt` oder `pyproject.toml` mit Versionsbereich/Lockstrategie. Verwende `.venv` falls kompatibel.
5. Lies die Originalquellen im Ordner `07_ORIGINALQUELLEN_PDF/`. Keine unbestätigten Seitenzahlen behaupten. Prüfe bei wissenschaftlichen Entscheidungen den tatsächlich relevanten Originaltext. Berichte pro Quellenaussage, welches Original gelesen wurde und welche Seiten bei Bedarf sicher verifiziert sind.
6. Trenne vorhandene historische Evidenz von neuen Beweisen. Existierende Archive enthalten frühere Trainings- und äußere Testresultate. Verwende sie zur Methodenkritik, **nicht** als Trainingsinput oder Beleg für eine nachträglich unabhängige Evaluation.
7. Gib eine Inventur-Tabelle aus: Datei, Rolle (aktuelle Herleitung / alte Herleitung / Referenzcode / numerischer Oracle / wissenschaftliches Paper / Daten / Altresultate), SHA256, gelesen ja/nein und offene Punkte.

## 2. Wissenschaftliche Grundaufgabe und Zielgrößen

Beobachtungseinheit ist ein vollständiger **Ringwalzversuch**, nicht ein einzelner Loggerzeitpunkt. Ring \(r\) besitzt eine zeitlich geordnete Prozesssequenz \(X_r\in\mathbb R^{T_r\times 6}\) mit individueller Länge \(T_r\). Für einen Teil der Ringe sind zwei **ringweite Endqualitätswerte** \(y_r=(y_{r,\text{außen}},y_{r,\text{höhe}})\) bekannt. Für andere Ringe ist die Prozesssequenz vorhanden, der Endqualitätswert aber im Trainingsmodell unbekannt/maskiert.

**Exakt zwei erlaubte Targets:**

- Außenunrundheit: `ovality.ovality_radial_mm`.
- Höhenschlag / Höhenabweichung: `planarity.height_deviation_mm`.

Beide Targets sind nach inverser Skalierung in mm zu berichten. **Innenunrundheit ist kein Target.** `DiW732` nicht als unabhängige Innenqualitätsmessung missbrauchen. Nutze **nicht** `ovality.r_outer_mm` als Ersatz für das tatsächlich bestätigte Feld. Die Zielwerte müssen aus den verifizierten `all_targets`-Feldern der DM-Datengrundlage stammen. Felder im Walzprotokoll können für Plausibilität/Dokumentation dienen, sind nicht automatisch zulässige Eingabefeatures.

Die sechs Prozesskanäle lauten **in genau dieser Reihenfolge**:
`F_rad63`, `F_ax64`, `Ist_D150`, `Ist_s151`, `Ist_h152`, `RWG55`.
Die tatsächlichen physikalischen Einheiten und Sensorsemantik vor wissenschaftlicher Formulierung mit Original-Datenbeschreibung verifizieren. Insbesondere `RWG55` ist im Projekt als geschwindigkeitsbezogen behandelt worden und darf **nicht ohne Verifizierung** als tatsächliche Ringhöhe beschrieben werden. `phase804` ist ein Kontext-/Phasenkanal, kein zusätzliches Modellfeature der eingefrorenen V5.2-Basis.

## 3. Datengrundlage und Exklusionen

18 experimentell bezeichnete Versuche DM01–DM18. DM03 fehlt als brauchbare Loggerdatei (abgebrochener Versuch). DM13 ist ausgeschlossen (nur ca. 6,7 Sekunden, LaserOK nie aktiv, RWG55=0, relevante Prozessphase nicht erreicht). **16 aktive Ringe** verbleiben. Verifiziere diese IDs/Anzahl gegenüber dem Pickle und dem Daten-Audit; schätze sie nicht aus Verzeichnisnamen. Die alten SFA-Loggerdaten gehören **nicht** zum neuen DM-Modell; sie sind eine verworfene historische Datenbasis.

Historische Entwicklungseinteilung für reproduzierbaren Startvergleich:

- Acht gelabelte Entwicklungsringe L: `DM04`, `DM05`, `DM06`, `DM09`, `DM12`, `DM16`, `DM17`, `DM18`.
- Vier ungelabelte Entwicklungsringe U: `DM01`, `DM02`, `DM10`, `DM14`.
- Vier historisch äußerlich ausgewertete Ringe: `DM07`, `DM08`, `DM11`, `DM15`.

Achtung: Die letzten vier Ringe wurden am 11.11.2026 schon einmal mit dem alten Modell gescort. Damit sind sie **nicht mehr jungfräulich unabhängig** gegenüber der weiteren Architekturentwicklung. Ich möchte sie nicht künstlich vergessen, aber keine neue Version anschließend fälschlich als auf diesen vier Ringen „unabhängig extern bestätigt“ ausgeben. Das neue Basismodell verwendet zunächst nur die zwölf Entwicklungsringe (8L+4U), und seine Hauptentwicklungsevaluation erfolgt mittels ringgetrennter Cross-Validation innerhalb erlaubter Entwicklungsdaten. Ein späterer Vergleich auf den vier bekannten Ringen darf explizit als explorativ benannt werden, falls angeordnet – niemals als neue Blindprüfung. Keine weiteren Messringe sind vorgesehen.

## 4. Präzises Datenfenster und Sicherheitsgrenzen

Aus dem alten `dm_data.py`, der V5.1-Eingangssicherheit und dem Datenprovenienzbericht den **wirklich implementierten F2-Fensterfilter** rekonstruierten: pro Ring Prozessphase 1 oder 2, Laser-/Mess-Gates (`LaserOK` und `MessEin`) nach alter Referenz prüfen, zusammenhängende Zeitpunkte mit 0,1-s Raster und korrekter Reihenfolge; Vor- und Nachlaufphase 6 ausschließen, wo gemäß verifizierter Prozessdefinition nötig. **Schreibe nicht blind einen neuen Filter, der andere Zeitreihen auswählt.** Teste die Gleichheit der alten und neuen Zeilen-Indizes/Anzahl/Timestamps für alle Entwicklungsexperimente mit dokumentierter Entscheidung bei jeder Abweichung.

Keine Datenleakage:
- Split ausschließlich nach `ring_id`, niemals zufällig nach einzelnen Zeitpunkten.
- Fit des Prozess-Scalers nur mit TRAIN-Prozesssequenzen (je nach vorgesehenem SSL-Arm auch dort erlaubte U-Ringe). Diese historische P0-Variante poolt Zeitpunkte der TRAIN-Ringe; die implizite Übergewichtung langer Ringe transparent machen, später eigene Ablation.
- Fit der **beiden Ziel-Scaler ausschließlich mit bekannten Labels aus dem jeweiligen TRAIN-Fold**.
- Fit der physikalischen Merkmals-Scaler nur in TRAIN. Für U-TRAIN dürfen deterministische X-only-Deskriptoren einbezogen werden, wenn dies der definierte Untersuchungsarm zulässt. Nicht aus VAL/TEST fitten.
- Keine Testlabels, Testqualitätsresiduen, Testtarget-Mittelwerte, alten Scores oder Test-Ring-Außenmaßen im Trainingsprozess laden/übergeben.
- Ein vollständig maskierter Ring muss programmatisch nicht nur `NaN`-Labels besitzen: Die neue Trainingspipeline soll für U eine Prozess-only-Datenschnittstelle verwenden, damit sie überhaupt keine y-Array-Werte erhält.
- Exportiere Split-Manifest, feature sequence, Gate-Kriterien, Skalerparameter, Provenienz und Datenhash mit jedem Lauf.

## 5. Reproduzierbare Ausgangskonfiguration (nicht „bewiesene optimale“ Konfiguration)

Startbedingungen aus historischer V5.2-Referenz:

```
K = 2                   # latente Zustände
C = 0.2                 # Gewicht des vollständigen U-Prozess-ELBO-Beitrags
variant = 'M2'          # h + spätere minus frühere Zustandseinträge
physical_var = 0.38526108747593474  # früher explorativ gewählter Vorfaktor
seed = 20261111
local_maxiter = 8       # initialer Referenzwert, kein numerisches Dogma
train_epochs = 30       # 10 waren definitiv zu kurz für den geprüften Lauf
process_window = 'F2'
features = ('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55')
physical_descriptor_indices = {'outside': (0,1,3,5,6,10), 'height': (0,2,4,7,11)}
```

`physical_var` und andere Werte sind historische Entwicklungsentscheidungen, nicht unabhängig bestätigte Optima oder physikalisch zwingende Konstanten. Keine neuen Hyperparameteroptimierungen in der eigentlichen Neuimplementierung. Lade die konkrete Herleitung und Referenzcode `quality_link_challenge.extract_descriptors`, bevor du die 12 physikalischen Sequenzdeskriptoren definierst. Diese Funktionen müssen für spätere Prediction nach exakt denselben Regeln aus vollständig beobachteten Prozesssequenzen berechnet werden. Erste Head-Eingangsdimension nach **priorerhaltender M2-Reduktion**: 9 (Außenunrundheit); zweiter: 8 (Höhenschlag). Die historischen vollen Dimensionen 10/9 nicht mit den neuen verwechseln.

## 6. Generatives Modell: HMM mit Bayesian-Emissionsparametern

Für jeden Ring \(r\):

\[
z_{r,1}\mid\pi\sim\mathrm{Cat}(\pi), \quad
z_{r,t}\mid z_{r,t-1},A\sim\mathrm{Cat}(A_{z_{r,t-1},:}).
\]
\[
x_{r,t}\mid z_{r,t}=k,\mu_k,\Lambda_k \sim\mathcal N(\mu_k,\Lambda_k^{-1}).
\]

\(\pi\) und jede Zeile \(A_{k,:}\) tragen Dirichlet-Priors; die kompletten multivariaten Emissionsparameter haben einen Normal-Wishart-Prior pro Zustand \(k\): \(\Lambda_k\sim\mathrm{Wishart}(W_{0k},\nu_{0k})\), \(\mu_k\mid\Lambda_k\sim\mathcal N(m_{0k},(\kappa_{0k}\Lambda_k)^{-1})\). Die tatsächliche Wishart-**Skalen-**Parametrisierung des alten Codes überprüfen; Wishart-Rate vs Scale nicht verwechseln. \(\nu_{0k}>D-1\), \(W\succ0\), \(\kappa>0\) sicherstellen. Erwartete Log-Präzision, erwartete Mahalanobis-Distanz, Trace- und Logdet-Terme korrekt mit `scipy.special.digamma` und `slogdet` implementieren.

Lerne über die konjugierten Posteriorfaktoren \(q(\pi)\), \(q(A)\), \(q(\mu_k,\Lambda_k)\). Kein MAP-Only-Schritt unter dem Etikett „fully Bayesian“. Dies ist **variational Bayesian**, kein exakter gemeinsamer Posterior.

Trotz Vollkovarianz sind Emissionen **zeitlich bedingt unabhängig**, wenn die Zustandsfolge und globalen Parameter gegeben sind. Das ist später gesondert zu hinterfragen; **kein AR-HMM stillschweigend als Basismodell einbauen**. Die zeitliche Autokorrelation ist der wichtigste offene Architekturpunkt im Optimierungsfahrplan.

## 7. Bayesian-Qualitätsmodell mit zwei Köpfen

Jeder Ring hat einen ringweiten Qualitätswert \(y_{r,j}\) je Ziel \(j\in\{1,2\}\). Die Qualitätsköpfe hängen von den latenten Zustandsstatistiken **und** vorhersagezeitlich verfügbaren, deterministischen Prozessdeskriptoren ab. Je Kopf \(j\):
\[
y_{r,j}\mid z_r,\beta_j,\sigma_j^2,s_{r,j}\sim
\mathcal N(\beta_j^{\top}\varphi_j(z_r,s_{r,j}),\sigma_j^2).
\]
Verwende getrennte Normal-Invers-Gamma-Priors \(\beta_j\mid\sigma_j^2\sim\mathcal N(\mu_{0j},\sigma_j^2 V_{0j})\), \(\sigma_j^2\sim\mathrm{IG}(a_{0j},b_{0j})\) mit **explizit verifizierter** IG-Rate/Scale-Konvention. Der alte Code nutzt u. a. `a=2.5,b=1.5` für Head-Priors; überprüfe anhand `two_head_b1.py` genau die bedingte Dichte und Momente.

Die beiden Qualitätsköpfe sind **bedingt getrennt**. Ohne zusätzliches gemeinsames Rauschmodell keine Behauptung einer vollständig korrelierten 2D-Ausgabeverteilung. Ihre Kopplung entsteht indirekt über dieselbe Zustandsfolge und denselben Prozessposterior.

Das Modell darf aus den Trainings-y der gelabelten Ringe lernen; für U dürfen keine Pseudolabel erzeugt werden. Der unlabeled contribution ist ein exakter Prozess-VI-Beitrag im **gewählten approximativen Modell**, kein pseudo-supervised target loss.

## 8. Strukturierte lokale Variationsfamilie

Für jeden Ring \(r\) verwende eine **frei zeitabhängige Markov-Kette** statt unabhängiger Zeitlabels:
\[
q_r(z_{1:T})=q_r(z_1)\prod_{t=2}^T q_r(z_t\mid z_{t-1}).
\]
Damit sind \(\gamma_{r,t,k}=q_r(z_t=k)\) und \(\xi_{r,t,i,j}=q_r(z_{t-1}=i,z_t=j)\) einschließlich Zeilen- und Spalten-Marginalidentitäten exakt konsistent. Nutze Forward-Backward im logarithmischen Raum, Stabilität bei langen Sequenzen, Eingabevalidierung und normierte Übergangsmatrizen.

**Bereits bestätigte strukturelle Codeverbesserung V5.9:** Statt vollen Softmax-Logits je Wahrscheinlichkeitszeile nur \(K-1\) Log-Odds freigeben und letzten Logit auf 0 fixieren. Für eine Zeile mit \(p=(p_1,\ldots,p_K)\) setze \(r_j=\log(p_j/p_K)\), \(j<K\), und \(p=\mathrm{softmax}(r_1,\ldots,r_{K-1},0)\). Diese Darstellung ist äquivalent, aber L-BFGS-B-Trajektorien können anders sein; Gradient in reduzierten Koordinaten über korrekte Kettenregel aus alten vollständigen Logits verifizieren. Teste K=2/3/4, Markov-Gauge-Invarianz, symmetrierte Momente und finite Differenzen.

Für U bei gegebenen globalen \(q\) und ohne Qualitätslabel: prozessbasierte Forward-Backward-Lösung. Für L: lokale gemeinsame Qualität-/Prozess-Variationsoptimierung, deren Zielfunktion die analytisch integrierten erwarteten Qualitäts-Terme beider Heads einschließt.

## 9. Exakte ringweite Markov-Momente: **keine harten Zustände**

Die Aufenthaltsverhältnisse \(h_{r,k}=T_r^{-1}\sum_{t=1}^{T_r}\mathbf 1(z_{r,t}=k)\) sind Zufallsgrößen unter \(q_r\). M2 ergänzt \(\Delta_{r,k}=h_{r,k}^{\mathrm{late}}-h_{r,k}^{\mathrm{early}}\), wobei die tatsächlichen früh/spät-Intervalle und ihre Zählkonvention exakt aus `dynamic_markov_v2.py`/`Spec` rekonstruiert werden müssen. Implementiere
\[
\mathbb E_q[g_r],\quad \mathbb E_q[g_rg_r^\top],\qquad g_r=(h_{r,1:K},\Delta_{r,1:K}).
\]

Die zweite Momente enthalten zeitliche Kreuzkorrelationen der Zustandsindikatoren. **\(\mathbb E[gg^T]\neq\mathbb E[g]\mathbb E[g]^T\)** im Allgemeinen. Nutze die geprüften exakten Markov-Moment-Rekursionen. Unabhängig per Enumeration aller \(K^T\)-Pfade für kurze \(T\) verifizieren, und Ableitungen per zentralen Finite-Differenzen mit verschiedenen K. Kein harter `argmax`-Pfad als Ersatz der zweiten Momente, keine bloße Monte-Carlo-Schätzung im Core.

Qualitäts-Designvektor pro Kopf \(\varphi_j=(g_r,s_{r,j})\). Neben dem ersten Moment \(m_j=\mathbb E[\varphi_j]\) verwende \(H_j=\mathbb E[\varphi_j\varphi_j^T]\). Die lokale Qualitätserwartung enthält u. a. \(\mathrm{tr}(\mathbb E[\beta\beta^T]H_j)\), und ihre Ableitung nach freien Markov-Log-Odds muss korrekt sein.

## 10. **Bereits bestätigte priorerhaltende M2-Reduktion V6.0**

Für jede Markov-Zustandsfolge gilt:
\[
\sum_{k=1}^K h_k=1,\qquad \sum_{k=1}^K \Delta_k=0.
\]
Die erste Gleichung ist **affin**: Ohne separaten Intercept kann man nicht einfach ein \(h_k\) fallen lassen, sonst verliert man die konstante Regressionsfunktion. Die zweite ist eine tatsächliche lineare Redundanz. Daher nur die letzte Differenzkomponente \(\Delta_K=-\sum_{j<K}\Delta_j\) algebraisch entfernen. Alle K Aufenthaltsmerkmale bleiben.

Sei \(u\) der reduzierte vollständige Head-Designvektor und \(f=A_{\mathrm{lift}} u\) der bisherige volle. Die effektiven Regressionskoeffizienten lauten \(\beta_{\mathrm{red}}=A_{\mathrm{lift}}^\top\beta_{\mathrm{full}}\). Für bedingten Gaußprior \(\beta_{\mathrm{full}}\mid\sigma^2\sim\mathcal N(\mu_{\mathrm{full}},\sigma^2 V_{\mathrm{full}})\) folgt sein **induzierter** Prior:
\[
\mu_{\mathrm{red}}=A_{\mathrm{lift}}^\top\mu_{\mathrm{full}},\qquad
V_{\mathrm{red}}=A_{\mathrm{lift}}^\top V_{\mathrm{full}} A_{\mathrm{lift}}.
\]
IG-Parameter \(a,b\) unverändert. Unter der historischen unabhängigen Differenzkoeffizient-Priorvarianz 0,15 wird die Varianz des wirksamen K=2-Differenzkoeffizienten **0,30**, nicht 0,15; bei K>2 treten nichtdiagonale Off-Diagonal-Kovarianzen auf. **Nicht** einfach eine Designspalte löschen und denselben Prior behalten.

Implementiere die posteriorerhaltende reduzierende Parametrisierung im **neuen** Qualitätskern. Verifiziere Identität von NIG-Posteriorparametern, erwarteter Qualitätslikelihood, nach konjugatem Update relevantem KL-Term, posteriorprädiktivem Mittel/Varianz, lokaler Zielfunktion und Gradient gegen den altrangigen Ansatz. Die alten V6.0-Tests weisen empirische Identität nach, sind als Referenz zu nutzen. Differenziere sauber zwischen identischen effektiven Vorhersagen und identischer beliebiger freier Alt-q(beta)-Familie (nicht allgemein behaupten).

## 11. Gewichtete gemeinsame Variationszielfunktion – **zentrale theoretische Grenze**

Für gelabelte Ringe \(D_L\), ungelabelte \(D_U\), lokale Faktoren \(q_r(z)\), globale \(q(\Theta)\) definiere als Arbeitsziel:
\[
\mathcal J_C = \sum_{r\in D_L}\{\mathbb E_q[\log p(X_r,z_r\mid\Theta)] + H(q_r(z_r)) + \sum_{j=1}^2\mathbb E_q[\log p(y_{r,j}\mid z_r,\beta_j,\sigma_j^2)]\}
+ C\sum_{r\in D_U}\{\mathbb E_q[\log p(X_r,z_r\mid\Theta)] + H(q_r(z_r))\}
- \mathrm{KL}(q(\Theta)\Vert p(\Theta)).
\]

Bei \(C=1\) entspricht diese Struktur einer üblichen ELBO des gewählten generativen Missing-Label-Modells (unter den hier vereinbarten Variationsrestriktionen). Bei \(C=0.2\) ist es eine **gewichtete/generaliserte variationale Zielfunktion**, nicht ohne Einschränkung der unveränderte log-marginal-Likelihood-Lower-Bound des ursprünglichen ungewichteten Modells. Ganz besonders wichtig: C multipliziert **auch die Entropie** und die Übergangs-/Emissionsanteile der U-Ringe; C ist nicht nur ein Faktor vor der U-Emissionsloglikelihood. Bei C=0 müssen U-Ringe kein Prozessposterior-Update liefern; der U-only-Preprocessing-Effekt muss in einem späteren Ablationsdesign separat untersucht werden.

Überprüfe die gesamten konjugierten gewichteten Statistiken und KL-Terme unabhängig. Verwende `np.float64`/Logsumexp/logdet, Cholesky-basierte SPD-Validierung und Finite-Checks. Keine „monotone ELBO“ behaupten, wenn nur eine approximierte lokale Suboptimierung ohne Akzeptanztest ausgeführt wird.

## 12. **Bereits getestete lokale Warmstart-Sicherung V5.8**

Früher startet der lokale Optimierer in jeder Epoche aus der reinen Prozess-Forward-Backward-Verteilung, unabhängig vom vorherigen qualitätskonditionierten Faktor. Das ist nicht automatisch falsch, aber garantiert nicht die Nichtabnahme gegenüber dem alten Koordinatenwert. Der mathematisch geprüfte Alternativalgorithmus ist:

1. pro L-Ring den vorherigen lokalen Markov-Faktor \(q^{\text{alt}}_r\) speichern;
2. unter **aktuellen** globalen q-Faktoren sowohl \(q^{\text{alt}}_r\) als auch den neu berechneten prozessbasierten \(q^{FB}_r\) mit **derselben vollständigen lokalen Qualitäts-/Prozess-Zielfunktion** bewerten;
3. den besseren Start im reduzierten \(K-1\)-Logit-Raum optimieren;
4. eine neue lokale Lösung nur übernehmen, wenn ihre Zielfunktion innerhalb kontrollierter Rundung nicht schlechter als die beste angebotene Lösung ist;
5. nach jedem lokalen Update, Prozessupdate und Qualitätskopf-Update die **gleiche** gewichtete vollständige ELBO erneut auswerten und Veränderung speichern;
6. jede echte Update-ELBO-Abnahme explizit als Testfehler behandeln (numerische Toleranz vorher dokumentieren, nicht nachträglich großzügig setzen).

Das ist Basismodell-Verhalten. Obwohl V5.8 auf 30 Epochen keine relevante End-ELBO-Verbesserung brachte, verbessert die Strategie den mathematischen Kontrollvertrag.

## 13. Globale konjugierte Koordinatenupdates

Nach lokaler VI bilde die hinreichenden Statistiken **ring- und U-gewichtskorrekt**:
\[
N_k=\sum_{r\in L}\sum_t\gamma_{rtk} + C\sum_{r\in U}\sum_t\gamma_{rtk},
\quad S_k=\sum w_r\gamma_{rtk}x_{rt},
\quad Q_k=\sum w_r\gamma_{rtk}x_{rt}x_{rt}^\top.
\]
Für Anfangs-/Übergangs-Dirichlets analog \(\gamma_{r1}\), \(\xi_{rt}\), jeweils Gewicht 1/C. Vorhersagerelevante erste/zweite Statistiken ohne unzulässige Gleichsetzung von \(\gamma\) und harten Zuständen. Konjugierte Posterior-Updates für \((m_k,\kappa_k,\nu_k,W_k)\), \(q(\pi)\), \(q(A)\) und beide NIG-Köpfe sauber aus den Erwartungswerten herleiten.

Implementiere unabhängige **Suffizienzstatistik-Identitätsprüfungen** und additive Zerlegung der Prozess-ELBO in `emission+KL_NW`, `transition+KL_Dirichlet`, `initial+KL_Dirichlet`; genau wie beim bestätigten V6.1-Audit, aber im neuen Kern mit eigenen Tests.

## 14. ELBO-Logging und die große Epoche-11-Beobachtung

Alte zehn Epochen endeten bei ELBO ungefähr −24.242,30, 30 Epochen stabilisieren gegen −22.028,914445. Rund 47 % der gesamten 30-Epochen-Verbesserung entstehen erst nach Epoche 10. Reproduzierter Sprung E10→E11 ca. +1.322,447. Blockzerlegung V5.8/V6.1: lokale VI +176,136; globales Prozessupdate +1146,305; Qualität-Head-Updates zusammen +0,006; innerhalb globalem Prozessupdate `emission+NW` +1143,344 (99,74 %), Transition+Dirichlet +2,961, Initialzustand ~0. Dieses historische Ergebnis ist **kein Qualitäts- oder Generalisierungsgewinn**.

Im neuen Modell ausführliche maschinenlesbare `epoch_metrics.jsonl` mit pro Update `ELBO_before/after`, Emission, Transition, Initial, Entropie, KL pro Variationsfamilie, lokales Gain je Ring, globale Posterior-Mittelwert/Kovarianzänderungen, gewichtete State-N, phasenspezifische Zustandsposterior-Momente, Dirichlet-Posteriorerwartungen, Optimierer-Abbruchgrund, projektierter Gradient und Gradientennormen. In einem neuen 30-Epochen-Smoke das historische Muster **sollte** in sinnvoller Näherung reproduzierbar sein; aber zwinge die neue Lösung nicht auf exakte alte Werte, wenn die mathematisch äquivalente Parametrisierung Optimierungspfad verändert. Mache solche Unterschiede zum Berichtspunkt.

## 15. Qualitätskopf-Posterior und verlässliche Prediction

Trainiere jedes Qualitätsmodell bedingt auf den **erwarteten** Regressionsmomenten. Bei Testvorhersage ist die Endqualität nicht bekannt; lokale \(q(z\mid X)\) entsteht **ausschließlich aus Prozess-Forward-Backward**. Qualität nicht rückwärts mit unbekannten Testlabels konditionieren! `predict_new_ring()` benötigt nur rohe Loggersequenz, Zeit/Phasenfilter und gespeicherte FITTED Trainings-Scaler/Posteriorparameter; sie gibt für beide Ziele prädiktiven Mittelwert, Standardabweichung, Ausgangseinheit mm und Herkunft der approximierten Unsicherheit aus.

Für fixierten Designvektor und NIG ist die bedingte Posteriorprädiktive Student-t-förmig. Nach Marginalisierung zufälliger Markov-Zustandsmerkmale ist die prädiktive Verteilung im Allgemeinen eine weitere Mischung/Integration. Der historische Code berechnet eine **Momentenapproximation**, keine exakten Quantile einer vollständig global bayesianisch integrierten Student-t-Mischung. In der neuen Basis korrekte erste/zweite prädiktive Momente implementieren und offen deklarieren. Keine „kalibrierten 95%-Intervalle“ ohne eigenständige Kalibrierungsstudie.

## 16. Mathematik-Testkatalog — MUST PASS

Baue unter `tests/` eine Testmatrix (pytest und unabhängig rechenbare Beispiele):

**T1 HMM:** \(\sum_k\gamma_{tk}=1\), \(\sum_j\xi_{tij}=\gamma_{ti}\), \(\sum_i\xi_{tij}=\gamma_{t+1,j}\), log-forward-backward gegen Pfad-Enumeration, Randfälle \(T=1,2\), keine `nan` bei langem T, Label-Switching-Invarianz.

**T2 Markov-Momente:** Erwartung und zweite Momente von h und M2-Delta gegen exakt aufgezählte kleine Pfade; Gradient der quality-linked local objective gegen zentrale Differenzen; K=2,3,4.

**T3 Softmax-Gauge:** Volle und reduzierte Logit-Darstellung erzeugen dieselben \(q\), \(\gamma\), \(\xi\), erwarteten Qualitäten und ELBO. Die reduzierte Gradientenkette ist korrekt. Keine fälschliche Behauptung, Optimiererpfade seien identisch.

**T4 Bayesian-Familien:** Eigene Normal-Wishart-, Dirichlet- und NIG-Loglikelihoods, KL, Positivdefinitheit, Symmetrie und Posteriorupdates gegen unabhängige kontrollierte Formeln/handberechnete Fälle. Wishart-Konvention explizit im Testnamen.

**T5 M2-Prior:** \(f=Au\), \(\mu_{red}=A^T\mu_{full}\), \(V_{red}=A^TV_{full}A\), korrelierte induzierte Priors für K>2, Qualitätsmomente und konjugierte Posterioren bis Maschinenpräzision, unveränderte Head-Regression für alle Parameter.

**T6 ELBO:** Vollständige Komponenten summieren zur gleichen Gesamt-ELBO, pro Block nach Akzeptanzkontrolle keine echte Abnahme, analytischer lokaler Gain = full ELBO delta bei fixem globalem q, U-Gewicht multipliziert auch H(q_U), keine Aktualisierung bei C=0 außer ausdrücklich definierten X-only-Preprocessing-Arm.

**T7 Leakage:** Folds trennen Ring-IDs; kein `scaler.fit` auf VAL/TEST; y-Scaler nur L-TRAIN; U-Objekt besitzt überhaupt kein y; TEST-Score nicht im Auto-Tuning-Pfad; Input-Exports erzeugen ausschließlich X-Felder; Ziel-/Walzprotokoll-Ableitungen werden ausgeschlossen.

**T8 Repro/Persistenz:** Gleicher Seed und Environment gleiche Resultate innerhalb vorab erklärter numerischer Toleranzen; sicheres Speichern/Laden von konjugierten Faktoren, Spaltenreihenfolge, phasenrelevantem Gate, Feature-Deskriptoren, Scaler, Train-Split, Datensatz-Hash; Load-vs-In-Memory-Prediction identisch.

**T9 Fehlerfälle:** fehlende Kanäle/qualitätslabel, `T<4`, NaN/Inf, SPD-Verlust, doppelte Ring-ID, unzulässige zeitliche Reihenfolge, fehlende Phasengrenze, Daten-SHA-Abweichung, Feature-Reihenfolgeänderung, unzulässiger Testzugriff; sichere Fehlermeldungen.

**T10 End-to-End:** synthetisch generierte HMM-Ringe inklusive gelabelten und unbekannten Endqualitäten, mind. 2 Ziele, wirkliche End-to-End-Trainingsverbesserung/ELBO-Bilanz, Training von Null mit kontrollierter `seed`-Sensitivität und Testpredictions ohne y.

Kein `PASS` ohne Ausführung der Tests; drucke Command, Exitcode, Testanzahl, Toleranzen und ggf. gespeicherten Log. Eine neue Version darf nicht als mathematisch equivalent behauptet werden, solange ihr echter Test fehlschlägt.

## 17. V5.3–V6.3 bestätigte Forschungsbeobachtungen — als Evidenz, nicht Zielwerte

**V5.3:** Vierfache Entwicklung-CV im alten Stand: Differenz 0 gegenüber 8 lokalen VI-Schritten für Qualitäts-RMSE extrem klein (Außen ca. 2e−6 mm, Höhe ca. 4e−5 mm). Nicht als robustes globales Überlegenheitsurteil.

**V5.5:** Gleicher 30-Epochen-Lauf reproduziert erste 10 Epochen bis max 8.8e−8. ELBO E1 ca. −26.739,32, E10 −24.242,30, E11 −22.919,85, E30 −22.028,914445. Später Plateau.

**V5.6:** Seeds 20261111/20261112/20261113 ergaben nach K-Means dieselben zwei Clusterzentren bis <3e−16, nur Label-Switching; K-Means-Einstieg war keine reale Startrobustheitsprüfung. Bei lokaler VI 8→16 Iterationen ELBO-Endunterschied ~3.6e−9.

**V5.7:** 240 scipy-L-BFGS-B-Aufrufe pro Variante. 8: 234 success, 6 maxiter; 16: 240 success, 0 maxiter. Tatsächlich nur 700→707 Iterationen, 210 akzeptierte lokale Updates in beiden Fällen, 30 exakt null zusätzliche Updates (29×DM09 E2–30, einmal DM04 E2). Viele `ftol`-Abbrüche mit noch >gtol-Gradient. Erfolgsstatus ist kein Stationaritätsbeweis.

**V5.8:** Warmstart + per-block-ELBO 30 Epochen, keine relevante Blockverletzung, nahezu gleiche finale ELBO. Der neue Algorithmus soll als mathematisch abgesicherte Variante dienen.

**V5.9:** K−1-Logit-Gauge spart 50 % lokale Parameter bei K=2, gleiche Markov-Familie und finale ELBO, Endunterschied ca. 5e−8, keine relevante Blockverletzung. Die alte Referenz hat max 5170 lokale Logits je langem Ring, reduziert 2585.

**V6.0:** M2-NIG-Prior-Pushforward mit K2/3/4 und 30 Epochen geprüft: Kopf-Dimensionen 10/9→9/8, max ELBO-Abweichung gegenüber V5.9 unter 3.5e−8, kein neues wissenschaftliches Modell.

**V6.1:** globaler Prozess-Sprung E11 `NW+emission ~1143.344`, Transitionsanteil ~2.961, Startanteil ~0. Hohe Selbstübergangswahrscheinlichkeit.

**V6.2:** Markov-Marginalen der 12 Entwicklungsringe: 4 von 12 ohne argmax-\(\gamma\)-Zustandswechsel, 8 mit Wechsel; erwartete Gesamtwechsel ~28.545 ungewichtet, C-gewichtete ~27.745, Rate ~0.00438 je 0.1-s Übergang. Mittlere posterior Übergangsmatrix ungefähr [[.995957,.004043],[.005674,.994326]]. Zustände haben tatsächliche Zeitwechsel, sind aber nicht automatisch reale Walzphasen.

**V6.3:** In jedem 12 Entwicklungsring genau eine Loggerphasengrenze `phase804` 2→1, mittleres Raster 0.1s. ±2s um Grenze tragen ~3.330 von ~28.545 erwarteten Wechseln (11.7 %) vs uniforme zeitliche Deskriptivreferenz ~1.681; harte HMM-Zustandswechsel nahe Grenze bei nur 3/12 Ringen. Median |lag1|-Korrelation von posterior-basierten Emissionsresiduen ca. .996, nach phasenweiser linearer Detrend .987, erste Differenzen .254. Hinweis auf zeitliche Emissionsstruktur, **kein formaler Test** und noch keine Rechtfertigung, autoregressive Struktur ohne Herleitung sofort einzubauen.

**Merke:** Kein Punkt beweist unabhängige zukünftige Generalisierungsverbesserung; Zahlen sind an historische Konfiguration/Datenaufteilung gebunden.

## 18. Wissenschaftliche Einschränkungen und HMM-Interpretation

- HMM ist zeitabhängig, aber extrem persistente Zustände können ringweite Prozessregime, glatte Trends oder tatsächliche Walzphasen beschreiben; Phasenidentität nicht voraussetzen.
- F2 umfasst dokumentierten Phase-2→Phase-1-Wechsel; Zustandswechsel nur teilweise in Grenznähe. Phasenlabels sind Kontext für Diagnose, keine automatisch vorgegebenen HMM-Zustände.
- Vollkovarianz erfasst Korrelation **zwischen** sechs Kanälen pro Zeitpunkt, nicht serielle Abhängigkeit der Emissionsfehler. Modellannahme zunächst offenlegen, nicht kaschieren.
- Der historisch beobachtete `p(z)` kann bei extrem langen, hochkorrelierten Sequenzen überkonfident werden; nenne Posteriorunsicherheit im Interpretationsrahmen der Modellspezifikation.
- Variation in Ringlängen führt im P0-Zeitpunkt-Pooling zu anderen Gewichten; diese Frage gehört zu späterer Robustheitsuntersuchung, nicht heimlichem neuen Scaling.
- Die Zustandsnummern sind label-permutierbar; aligniere Zustände vor Zahlenvergleich über verschiedene Initialisierungen.
- **Keine pseudo labels, kein supervised deep net als Hauptalgorithmus**. Alternative AR-HMM oder i.i.d.-Mischung erst nach Initial-Baseline gemäß separatem Word-Plan.

## 19. Historische Ergebnislage, saubere Evaluation

Die ursprünglich erstmals äußerlich ausgewertete ältere V1-Variante ergab auf 4 historischen Ringbeobachtungen (DM07,08,11,15): SSL-RMSE Außen ca. 0.3888 mm, Höhe ca. 3.2064 mm; streng überwacht ca. 0.5529/3.3579; TRAIN-Mittelwert-Baseline ca. 0.4953/**2.9785**. Für Höhe war die alte SSL-Variante schlechter als die simple Mittelwert-Baseline. Das sind nur vier Ringe; keine statistisch belastbare Überlegenheit und keine Aussage über V6.0-Neuimplementierung.

In alter CV mit acht wiederholt verwendeten Entwicklungsringen: strict supervised RMSE ca. 0.6541/3.1756, shared-C0 0.5503/3.2633, SSL C>0 0.5453/3.2100. Die zusätzliche Wirkung des reinen U-ELBO gegenüber shared C0 ist klein (~0.9% Außen, ~1.6% Höhe), stark foldsensitiv. Diese Resultate motivieren **spätere** faire kontrollierte Ablationen: HMM vs Nicht-HMM, Bayes vs ridge/MAP, echte U-ELBO vs nur U-Preprocessing, K und C, Prenominaries.

Für neues Modell zuerst einheitliche ringweise Entwicklungs-CV mit gleichen Folds, identischen erlaubten Features/Scaler-Policies, expliziter Leakageprüfung, Baselines (TRAIN-Mean, physikalisches NIG/Ridge ohne HMM, HMM-Head ohne U-ELBO, ggf. vollständige SSL). Eine Cross-Validation auf sehr wenigen Ringen liefert unsichere generalisierende Evidenz. Keine „unabhängige Testperformance“ nach wiederholter Model-Selection auf bekannten äußeren Labels behaupten.

**Keine Optimierung gegen äußere Testwerte** in der Initialimplementierung. Zuerst Korrektheit, reproduzierbares Training und sauberer Vergleich. Wenn später der Nutzer explizit explorative bekannte Testvergleichsexperimente möchte, klar als explorativ deklarieren.

## 20. Neue Softwarearchitektur — ohne Core-Code-Recycling

Erzeuge im selben Workspace **einen eindeutig separaten neuen Projektordner**, beispielsweise:

```
bayesian_ssr_neubau/
  pyproject.toml
  README.md
  CHANGELOG.md
  docs/
    MATHEMATISCHES_MODELL.md
    ABGLEICH_A_B_C_D_V60.md
    TESTPROTOKOLL.md
    DATENPROVENIENZ.md
    QUELLENSTATUS.md
    ANNAHMEN_UND_LIMITATIONEN.md
  src/bayesian_ring_ssr/
    __init__.py
    config.py
    data_io.py
    schema.py
    split_guard.py
    preprocessing.py
    physics_descriptors.py
    hmmmoments.py
    forward_backward.py
    reduced_markov.py
    priors.py
    normal_wishart.py
    nig_heads.py
    elbo.py
    local_vi.py
    coordinate_train.py
    posterior_predictive.py
    metrics.py
    persistence.py
    diagnostics.py
    cli.py
  tests/
    test_data_and_leakage.py
    test_forward_backward_exact.py
    test_markov_moments_enumeration.py
    test_reduced_logits_grad.py
    test_nw_dirichlet_nig.py
    test_m2_induced_prior.py
    test_elbo_decomposition.py
    test_local_monotonicity.py
    test_prediction_no_y.py
    test_persistence_repro.py
    test_synthetic_end_to_end.py
  outputs/   # write-once unique runs; old results/ untouched
```

Namen dürfen begründet abweichen, aber die Trennung zwischen Daten, mathematischen Faktoren, VI, Vorhersage, Evaluation und IO ist zwingend. `old_core` darf keine Importabhängigkeit in `src/` haben; ein zusätzlicher Test in `tests/reference_parity` darf alte Oracles read-only heranziehen.

## 21. Daten- und trainierbare Zustände als explizite Verträge

Schaffe `RingProcessSequence` (ring_id, times, X), `LabeledRing(RingProcessSequence, targets)` und `UnlabeledRing(RingProcessSequence)` mit strikter Typ-/Shape-/Finite-Kontrolle; keine gemeinsame Datenklasse, bei der y für U nur `nan` ist, wenn dadurch unbeabsichtigte Leakage möglich wäre. Datensatz-Hashes, F2-Maske, Zeitraster und Bandbreite/fehlende Werte als maschinenlesbare QA. Eindeutige Konfiguration mit validierter K/C/Priors/Variant/Seed. Alle Normalisierer pro Fold separat anpassen, in Training fit, vor Prediction nur transform.

Trainierte Ergebnisse als eigenes Objekt: posterior pi/A/NW, q der beiden NIG-Köpfe, FeatureSpec, train-only Scaler, Trainingsdaten-ID-Manifest, vollständige Update-/Grad-Historie, Modellversionsschema, Mathematikteststatus. Keine Pickles aus nicht vertrauenswürdigen Orten blind `pickle.load`, sofern Dateiformat und Herkunft nicht bekannt/validiert sind; lokale Benutzerdaten sind explizit freigegeben, aber nicht bei beliebigen späteren Pfaden. Ausgabeverzeichnis write-once, Integritäts-Manifest SHA256.

## 22. Performance, numerische Stabilität und erwartetes Training

Nicht die 30-Epochen-Ergebnisse mit Gewalt erzwingen, sondern Epochen unabhängig mit korrektem ELBO-Update rechnen. Wiederverwenden von Momenten ist nur erlaubt, wenn Analytik und Cache invalidation stimmen; kein schwer überprüfbarer globaler Zustand. Optimiere Cholesky/Logdet und Markov-Rekursionen, vermeide \(K^T\) Enumeration außer sehr kurzen Tests. Bei deterministischen Startwerten kann K-Means für Seeds 20261111/12/13 dieselben Zentren finden; das ist **keine** Robustheitsbestätigung. Initialisierungsrobustheit später mit wirklich unterschiedlichen Zentren/Startmatrizen messen.

Echte zeitbezogene Emissions-Fehlstruktur sollte im initialen README als Limitierung benannt werden. Noch **kein** autoregressives Modell oder nichtkonjugater HMM-Baustein als „Fix“, bevor Vergleich und Bayesian-Updates sauber hergeleitet sind.

## 23. Akzeptanzkriterien für die Basismodell-Übergabe

**Zwingend bestanden**:

1. Neue mathematische Modelldokumentation, jede relevante Formel aus A–D auf aktuelle K2/M2/2-Heads/Normal-Wishart/induzierte Priors gebracht; jede Abweichung begründet.
2. Quellenaudit mit drei Statusklassen: Original-PDF vorhanden und konkreter Beleg gelesen / PDF vorhanden aber Seitenprüfung offen / Paper nur bibliografisch bekannt.
3. Neue Software ist eigenständig; `import`-Graph des neuen `src`-Codes enthält keine alten Modellkerne.
4. Neues TRAIN von Null (L+U), Sicherheitschecks, Checkpoint/Logging, 30-Epochen-ELBO, kein alter Fit-Zustand.
5. Tests T1–T10 bestanden, mit echten ausgeführten Commands und Ergebnisprotokoll; ein Fail wird nicht verschwiegen.
6. Pro Epochen-Update ELBO-Komponenten, Gradienten und lokale Akzeptanz dokumentiert; keine echte Verletzung ohne Erklärung/Abbruch.
7. Train-only Demo mit 8L/4U abgeschlossen; Parameter/Vorhersagevertrag exportiert, laden = in-memory-Vorhersage.
8. Exemplarische Prediction aus einem in Training/Vorhersage getrennten Szenario **ohne** y-Zugriff validiert; Vorhersageform und mm-Rückskalierung korrekt.
9. Entwicklungsevaluation ringgetrennt; falls CV zeitlich/rechnerisch nicht fertig, klar als OFFEN benennen, nicht als erledigt.
10. Abschlussbericht: Was ist exakt mathematisch identisch mit V6.0? Was ist neu implementiert? Welche Testtoleranzen? Welche Originalquellen stützen welche Abschnitte? Welche offenen Forschungsfragen stehen für die separate Optimierungs-Word bereit?

## 24. Notwendige CLI-Kommandos

Stelle dokumentierte Befehle im Windows-PowerShell-Format bereit, z. B.:

```
.\.venv\Scripts\python.exe -m pytest bayesian_ssr_neubau/tests -q
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli audit-data --input input\DM_16_aktive_Ringe.pkl
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli train --config bayesian_ssr_neubau\configs\baseline_k2_m2.json --out bayesian_ssr_neubau\outputs\run_001
.\.venv\Scripts\python.exe -m bayesian_ring_ssr.cli predict --model bayesian_ssr_neubau\outputs\run_001 --ring-file <new-process-file>
```

Passe echte Modulpfade korrekt an. Wenn Package nicht editable installierbar ist, liefere explizite Installation wie `pip install -e`. Kein untestbarer Fantasiebefehl im Endbericht. Never overwrite previous output.

## 25. Strikte Reihenfolge und Berichtsrhythmus

Arbeite schrittweise, aber tatsächlich selbstständig:

**Stufe 1:** Vollständige Inventur, mathematischer Soll-/Ist-Abgleich, Risikoanalyse; führe Fehler im Istmodell nicht blind fort.

**Stufe 2:** Implementiere Data/Split/Preprocessing und teste Daten-SHA, Zeitindex-/Feature-Gleichheit, Target-Feldzuordnung und Leakage-Gates.

**Stufe 3:** Implementiere Normal-Wishart, Dirichlet, NIG, Forward-Backward, zeitabhängige q-Markov-Verteilung mit reduzierten Logits und exakte M2-Momente. Tests zuerst, dann Integration.

**Stufe 4:** Implementiere gemeinsame Qualitätskopplung und exakte induzierte M2-Priors, gewichtete ELBO, lokale Warmstart/Abnahme-Sicherung und globale konjugierte Updates. Unabhängige Tests.

**Stufe 5:** Vollständige Train-Pipeline aus frischer Initialisierung, reproduzierbare Diagnostik, read-only Paritätsvergleich zu V6.0, kein äußerer Testscore.

**Stufe 6:** Posterior-Predictive, Persistenz, Eval-CLI und ringweise Entwicklungs-CV, mit klarem Status der beiden Zielgrößen.

**Stufe 7:** Finaler Code-Review, Hash-Manifest, Quellensicherheit, prüfbarer Release-Ordner; separater Optimierungsfahrplan bleibt unimplementiert bis zur neuen expliziten Freigabe des Nutzers.

Melde nach jeder Stufe: **wirklich angelegte Dateien, ausgeführte Tests, Befunde, offene Punkte**. Sei kritisch: Ein mathematisch schöner Code ohne unabhängige synthetische Validierung ist nicht fertig. Ein numerisch stabiles Training ohne belegte Vorhersagequalität ist nicht empirisch validiert. Bei unzureichender Datenzahl keine statistisch unhaltbare Überlegenheit behaupten.

## 26. Konkrete Literatureinordnung für die Modellherleitung

Nutze primär die tatsächlich mitgelieferten Original-PDFs:

- Rabiner (1989), HMM-Tutorial: Markov-Faktorisierung, HMM und Forward-Backward.
- Beal (2003), Variational Algorithms for Approximate Bayesian Inference: strukturierte Bayesian-VI und Konjugiertheit; konkrete Formeln gegen Original prüfen.
- Bishop (2006), Pattern Recognition and Machine Learning: konjugierte priors, Gaussian/Wishart, Bayesian Regression, Variational Mixture/HMM; nur nach Sichtung gedruckte Seiten nennen.
- Shao et al. (2022), semisupervised Bayesian HMM for Soft Sensing: industrieller Bayesian-HMM-Anwendungskontext, **nicht** unverändert unsere gesamte 2-Head-Qualitätskopplung.
- Shao, Ge & Song (2021), semisupervised Bayesian Gaussian mixture: Bayesian-Soft-Sensor-Kontext, Gewichtung nicht ungeprüft gleichsetzen.
- Mowbray et al. (2022), probabilistic quality soft-sensors in batch processes: ring-/batchweite Endqualität und eingeschränkte Qualitätslabels.
- Yuan et al. (2024), knowledge-informed semi-supervised Variational Bayesian mixture, und Li et al. (2026), Bayesian semisupervised nonlinear HMM: theoretischer Vergleich/Prior-Architekturen; exakte Methoden nach Original.
- Gelman et al. (2020), Gabry et al. (2019): Bayesian workflow und Diagnostik.
- Varma & Simon (2006), Cawley & Talbot (2010), Roberts et al. (2016), Kapoor & Narayanan (2023), Kaufman et al. (2011): Model-Selection Bias, ringweise CV und Leakage.
- Dai, Li & Cheng (2023), Kang et al. (2016), van Engelen & Hoos (2020), Kostopoulos et al. (2018): SSL-Techniken als Vergleich und Abgrenzung zu unserer **vollprobabilistischen** Methodik.

**Keine Literaturquelle als Beweis für eine neue exakt von uns konstruierte Kombination ausgeben.** Deklariere die 2-Head-Modelleinbindung, exakte Markov-Momente und Prior-Pushforward als eigens implementierte, durch Formeln/Tests gestützte Konstruktion, soweit kein Original sie identisch enthält. Wenn eine Quelle fehlt, melde fehlend, suche nicht stillschweigend nach einer ähnlich benannten Ersatzquelle und erfinde keine Seitenzahlen.

## 27. Dateien, die du nicht als endgültigen Code missverstehen darfst

- `final_candidate_v52/`: historische Referenz/Provisional Freeze; nicht endgültig freigegeben.
- `research_math_v58/`: abgesicherter Warmstart und ELBO-Bilanz, in neue Basis als geprüfte Logik übernehmen.
- `research_math_v59/`: reduzierte Logits, **in neue Basis übernehmen**.
- `research_math_v60/`: priorerhaltende M2-Reduktion, **in neue Basis übernehmen**.
- `research_math_v61/`: Prozess-Posterior-ELBO-Zerlegung, als Diagnostik übernehmen.
- `research_math_v62/`: HMM-Wechsel/Residualdiagnostik, als Diagnostik/Tests nutzen.
- `research_math_v63/`: Phasengrenzen/Trend-Residualdiagnostik, als Diagnose nutzen; kein AR-HMM-Update enthalten.
- Alte SFA-Versuche: verworfene Rohbasis. Nicht erneut als DM-Daten importieren.
- `PROPOSED_MODEL_FREEZE.json`: **keine** vom Nutzer autorisierte Endfreigabe.
- Historische RMSE-Dateien: keine unabhängige Modellvalidierung des Neubaus.

## 28. Nach dem erfolgreichen Neuaufbau – getrennter Optimierungsauftrag

Der große, eigenständige Word-Optimierungsfahrplan in `08_OPTIMIERUNG_NACH_NEUBAU/` enthält zahlreiche noch ungeprüfte Erweiterungen. **Nicht** gleichzeitig implementieren. Nach dem funktionsfähigen mathematisch geprüften Neubau will ich vor allem: zeitkorrelierte Emissionsalternativen (AR-HMM), Vergleich HMM vs nichtzeitliche Modelle, volle vs reduzierte Emissionsstruktur, echte Initialisierungsrobustheit, Bayes vs MAP/Ridge, echte SSL-U-ELBO-Ablation, Posterior-Kalibrierung, ringweise Generalisierung, Prior-/Feature-Sensitivität, Train-Mean-Robustheit. Pro Erweiterung eigener kontrollierter Branch, mathematische Herleitung, numerischer Unit-Test, ringweise Entwicklungs-Evaluation und transparentes Ergebnis.

## 29. Abschließende Anweisung an den Agenten

**Erstelle den neuen Modellcode wirklich.** Ich möchte keine weitere Runde, in der wir bloß Parameter erhöhen oder theoretisch spekulieren. Beginne nach der Dateiinventur mit den konkreten Python-Modulen und Tests, protokolliere alle Entscheidungen, behandle alle zwei Qualitätsgrößen korrekt und dokumentiere jede mathematische Einschränkung. Die vorhandenen Audits sind Wegweiser und Oracles, nicht die neue Lösung. Das Ziel ist ein eigenständiges, wissenschaftlich nachvollziehbares und replizierbares Bayesian-SSR-HMM auf den DM-Ringen – inklusive funktionierender CLI, Train/Pred-Boundary, Testabdeckung und sauberer Forschungsdokumentation.

**Erste Antwort des Agenten:** Benenne die bereits geprüften relevanten Dateien, die entscheidenden offenen Widersprüche und die konkrete Implementierungsreihenfolge. Fahre danach mit tatsächlichen Dateiedits und Tests fort. Keine pauschalen Behauptungen wie „all tests passed“ ohne real ausgeführte Tests.
