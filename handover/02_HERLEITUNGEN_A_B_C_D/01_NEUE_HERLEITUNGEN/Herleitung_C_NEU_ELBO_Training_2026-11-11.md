---
title: "Herleitung C (Neufassung) – ELBO, globale Updates und vollständiger Trainingsalgorithmus"
subtitle: "Zweiköpfiges Bayesian-SSR-HMM mit Normal-Wishart-Emissionen und FULL_JOINT-Markov-VI"
date: "11. November 2026 | Implementierungsabgleich"
lang: de-DE
---

# C.0 Status und Quellengrenzen

Dieses Dokument überarbeitet die Originaldatei `Herleitung_C_Vollstaendiger_Trainingsalgorithmus_Bayesian_SSR.docx` unter Verwendung der eingefrorenen DM-Abschlussimplementierung `two_head_b1.py` sowie der zugrunde liegenden `fullcov_b1.py`, `full_markov_b1.py`, `vb_b1.py` und der äußeren Datensperre. Beal (2003) stützt die Variations-HMM-Grundstruktur; Shao et al. (2022) die Normal-Wishart-Emissionen; Shao, Ge & Song (2021) den Literaturkontext gewichteter VI. Die Kopplung **zweier ringweiter Endqualitätsmessungen** ist eine eigene Modelladaption.

# C.1 Welche Größen werden tatsächlich trainiert?

Das trainierte Modell besitzt globale Variationsfaktoren

$$q(\boldsymbol\pi),\quad\{q(A_{k,:})\}_{k=1}^K,\quad\{q(\boldsymbol\mu_k,\Lambda_k)\}_{k=1}^K,\quad\{q(\boldsymbol\beta_j,\sigma_j^2)\}_{j=1}^2,$$

sowie ringspezifische freie Markov-Faktoren $q_i(Z_i)=\rho_i(z_{i1})\prod_tQ_{it}(z_{it},z_{i,t+1})$ aus Herleitung B. Die sechs skalierten Loggerkanäle werden als **Prozess-** und die zwei skalierten Endqualitätswerte als **Qualitätsbeobachtungen** verwendet. Alle Ringkennwerte $\mathbf s_{ij}$ sind deterministische Funktionen der Loggerdaten und werden nicht als dritte stochastische Datengruppe behandelt.

**Eingefrorene Hauptkonfiguration:** $K=3$, $d=6$, $J=2$, $C=1$, 10 Epochen, höchstens 8 Iterationen pro gelabeltem Ring und lokaler Aktualisierung, Seed 20261111, Normal-Wishart-Emissionen, zwei NIG-Qualitätsköpfe, `FULL_JOINT`-Lokalinferenz.

# C.2 Vollständige ELBO für $C=1$

Für die gemeinsame Verteilung aus Herleitung A schreiben wir

$$\mathcal L(q)=\mathbb E_q[\log p(\Theta,Z,X,Y_L)]-\mathbb E_q[\log q(\Theta,Z)].$$

Unter der angegebenen Faktorzerlegung ergibt sich die auswertbare Form

$$\boxed{\mathcal L(q)=\sum_{i\in L\cup U}\Big[\mathcal P_i(q_i)+\mathcal H(q_i)\Big]+\sum_{i\in L}\sum_{j=1}^{2}\mathcal Q_{ij}(q_i,q_j)-\operatorname{KL}(q_\Theta\Vert p_\Theta)}.$$

Dabei ist $\mathcal H$ die lokale Markov-Entropie (Herleitung B.7), $\mathcal Q_{ij}$ die erwartete Loglikelihood der **einen** Ringqualität, und

$$\begin{aligned}
\mathcal P_i&=\sum_k\gamma_{i1k}\,\mathbb E_q\log\pi_k
+\sum_{t,k,\ell}\xi_{itk\ell}\,\mathbb E_q\log A_{k\ell}\\
&\qquad+\sum_{t,k}\gamma_{itk}\,\mathbb E_q\log\mathcal N_d(\mathbf x_{it};\boldsymbol\mu_k,\Lambda_k^{-1}).
\end{aligned}$$

Die globale KL-Divergenz zerfällt in Dirichlet-, Normal-Wishart- und die beiden NIG-KL-Terme. Alle drei Familien werden **variational**, nicht durch einen einzigen Punktwert, aktualisiert.

## C.2.1 Gewichteter ungelabelter Beitrag und Interpretationsgrenze

Die für die Ablation verwendete verallgemeinerte Zielfunktion lautet

$$\mathcal L_C(q)=\sum_{i\in L}\left[\mathcal P_i+\mathcal H(q_i)+\sum_{j=1}^2\mathcal Q_{ij}\right]+C\sum_{i\in U}[\mathcal P_i+\mathcal H(q_i)]-\operatorname{KL}(q_\Theta\Vert p_\Theta).$$

**Wichtig:** Das Gewicht $C$ multipliziert bei $i\in U$ **den ganzen lokalen Variationsbeitrag inklusive Entropie**, nicht nur die Emissionsloglikelihood. Bei $C=1$ liegt der normale ELBO des Modells mit fehlenden Qualitätswerten vor. Bei $C\ne1$ ist dies eine gewichtete/generalisiert-Bayesian-artige Trainingszielfunktion, nicht ohne weitere Herleitung die exakte ELBO einer unverändert gewöhnlichen Likelihood. Bei fixiertem globalen $q$ und $C>0$ bleibt die optimale *lokale* U-Verteilung durch das positive gemeinsame Gewicht unverändert; die **globalen** Updates werden jedoch mit $C$ gewichtet.

# C.3 Erwartete Emissionslogdichte für vollständige Kovarianzen

Für $q(\boldsymbol\mu_k,\Lambda_k)=\operatorname{NW}(\mathbf m_k,\kappa_k,W_k,\nu_k)$ gilt mit Wishart-Skalenparametrisierung

$$\mathbb E_q[\log|\Lambda_k|]=\sum_{r=1}^{d}\psi\!\left(\frac{\nu_k+1-r}{2}\right)+d\log2+\log|W_k|.$$

Wegen der konjugierten Kopplung von $\mu_k$ und $\Lambda_k$ ergibt sich die tatsächlich implementierte Logpotenzialgleichung

$$\boxed{\ell_{itk}=\frac12\left[\mathbb E\log|\Lambda_k|-d\log(2\pi)-\frac d{\kappa_k}-\nu_k(\mathbf x_{it}-\mathbf m_k)^\top W_k(\mathbf x_{it}-\mathbf m_k)\right].}$$

Es wäre **falsch**, die Kovarianzmatrix einfach anstelle der Präzisionsmatrix einzusetzen oder $d/\kappa_k$ fortzulassen. Für $d=6$ werden die determinanten- und quadratikbezogenen Terme mit stabilen Linearalgebra-Routinen berechnet. Alle posterioren Matrizen müssen positiv definit bleiben.

# C.4 Globale Dirichlet-Updates

Aus den lokalen Zustandsmarginalen werden gewichtete ausreichende Statistiken gebildet. Mit $w_i=1$ für $i\in L$ und $w_i=C$ für $i\in U$ gilt

$$\begin{aligned}
\alpha_k&=\alpha_{0k}+\sum_i w_i\gamma_{i1k},\\
\eta_{k\ell}&=\eta_{0k\ell}+\sum_iw_i\sum_{t=1}^{T_i-1}\xi_{itk\ell}.
\end{aligned}$$

Die posterioren Erwartungen der Log-Wahrscheinlichkeiten folgen aus Digamma-Differenzen der jeweiligen Dirichletparameter. Die Zustandsinitialisierung erfolgt in der aktuellen Implementierung ringweise aus **trainingsexklusiven** K-Means-Zentren. Der Verzicht auf Initialisierung mit Testdaten wird gesondert geprüft.

# C.5 Globaler Normal-Wishart-Update: vollständige Kovarianz

Für jeden Zustand definieren wir ausreichende gewichtete Größen

$$N_k=\sum_iw_i\sum_t\gamma_{itk},\qquad \mathbf S_k=\sum_iw_i\sum_t\gamma_{itk}\mathbf x_{it},\qquad R_k=\sum_iw_i\sum_t\gamma_{itk}\mathbf x_{it}\mathbf x_{it}^\top.$$

Für die Normal-Wishart-Faktoren lauten die Updategleichungen

$$\begin{aligned}
\kappa_k&=\kappa_{0k}+N_k,\\
\mathbf m_k&=\frac{\kappa_{0k}\mathbf m_{0k}+\mathbf S_k}{\kappa_k},\\
\nu_k&=\nu_{0k}+N_k,\\
W_k^{-1}&=W_{0k}^{-1}+R_k+\kappa_{0k}\mathbf m_{0k}\mathbf m_{0k}^\top-\kappa_k\mathbf m_k\mathbf m_k^\top.
\end{aligned}$$

Die letzte Zeile ist die exakte Form der im Modul `fullcov_b1.update_process_full` implementierten hinreichenden Statistik. Das Update benötigt die **vollen Kreuzprodukte** $\mathbf x_{it}\mathbf x_{it}^\top$ und ist daher keine diagonale, kanalweise NIG-Aktualisierung. Der symmetrisierte Rechenweg muss positive Definitheit erhalten; Schlechtkonditionierung wurde in eigenen Priortests ausdrücklich untersucht.

# C.6 Globale NIG-Updates für beide Qualitätsköpfe

Für Kopf $j$ seien $D_j=K+p_j$, $P_{0j}=V_{0j}^{-1}$, $\mathbf u_{ij}=[\mathbf h_i;\mathbf s_{ij}]$, $\widetilde{\mathbf m}_{ij}=\mathbb E_{q_i}[\mathbf u_{ij}]$ und $\widetilde H_{ij}=\mathbb E_{q_i}[\mathbf u_{ij}\mathbf u_{ij}^\top]$. Dann gilt jeweils **nur über tatsächlich gelabelte Trainingsringe**

$$\begin{aligned}
P_j&=P_{0j}+\sum_{i\in L}\widetilde H_{ij},\qquad V_j=P_j^{-1},\\
\boldsymbol\mu_j&=V_j\left(P_{0j}\boldsymbol\beta_{0j}+\sum_{i\in L}y_{ij}\widetilde{\mathbf m}_{ij}\right),\\
a_j&=a_{0j}+\frac{|L|}{2},\\
b_j&=b_{0j}+\frac12\left(\sum_{i\in L}y_{ij}^2+\boldsymbol\beta_{0j}^\top P_{0j}\boldsymbol\beta_{0j}-\boldsymbol\mu_j^\top P_j\boldsymbol\mu_j\right).
\end{aligned}$$

Diese Gleichungen werden in `two_head_b1.fit_head` für beide verschieden dimensionierten Köpfe separat angewendet. Die Matrixinversion wird implementierungsseitig über Cholesky-Lösungen berechnet. Unbekannte U-Qualitäten werden **nicht imputiert**, und U-Ringe erscheinen in den Summen des Qualitätsupdates nicht. Die Korrelation beider Kopfvorhersagen nach Integration über $Z$ ist nicht gleichbedeutend mit einem gemeinsam voll besetzten Qualitätsrauschmodell.

# C.7 Lokale Optimierung gelabelter Ringsequenzen

Zunächst wird jeder lokale Faktor aus der prozessbedingten Forward-Backward-Verteilung initialisiert. Für gelabelte Ringe sind dann bei festen globalen Faktoren alle Zeitpunkte der lokalen initialen und Übergangs-Logits variabel. Die L-BFGS-B-Zielfunktion besteht aus

$$F_i=\underbrace{\mathcal P_i+\mathcal H(q_i)}_{\text{Prozess und lokale Entropie}}+\underbrace{\mathcal Q_{i1}+\mathcal Q_{i2}}_{\text{zwei ringweite Qualitätsfaktoren}}.$$

Die numerischen Gradienten durch die Zählmomentenrekursion wurden gegen finite Differenzen geprüft (unter anderem mit Fehler um $3{,}98\cdot10^{-7}$ in einem dokumentierten Test). Die Qualitätsköpfe werden zu einem linearen Term in $\mathbf m_i$ und einem quadratischen Term in $H_i$ komprimiert. Die beiden Kopfterme wirken gemeinsam auf *dieselbe* Markov-Verteilung zurück. Für einen gelabelten Ring wird der neue lokale Faktor nur akzeptiert, wenn die exakt berechnete **lokale** Zielfunktion gegenüber der initialen prozessbedingten Verteilung gestiegen ist. Diese Absicherung ist nicht identisch mit einem Beweis der Konvergenz des global alternierenden Verfahrens.

**U-Ringe:** Die lokalen Faktoren werden ohne Qualitätslikelihood durch Forward-Backward aktualisiert. Das beschleunigt die numerische Rechnung, ohne die Prinzipien von B1 zu verlassen.

# C.8 Vollständiger implementierter Trainingsablauf

1. **Eingang schützen:** Vertrauenswürdige DM-Pickles mit SHA256 prüfen. Nur 16 aktive Ringe; DM13 nicht trainieren; alte SFA-Daten nicht mischen. Split vor jedem Fitting ringweise festlegen.
2. **Merkmalsfenster festlegen:** F2 und V1-Kern6 aus ausschließlich erlaubten Loggerkanälen; keine Messspalten der Produkt-Endqualität als Eingabe verwenden.
3. **Foldlokale Transformation:** Sechs Loggerkanäle nur auf gewählten Trainingsringen skalieren; zwei Qualitätsziele ausschließlich auf gelabelten Trainingsringen skalieren; physikalische Ringkennwerte aus transformierten X berechnen und pro Kopf **ebenfalls nur auf Trainingsringen** skalieren.
4. **Posteriorinitialisierung:** Dirichlet-, Normal-Wishart- und NIG-Priors setzen; K-Means nur auf den vom Modus zugelassenen Trainingsringen; zustandsspezifische Parameter initialisieren.
5. **Lokale E-Stufe:** Für jeden gelabelten Ring `FULL_JOINT`-Markov-Verteilung mit zwei Qualitätsfaktoren maximieren; für U-Ringe prozessbedingtes Forward-Backward verwenden.
6. **Globale M-/VI-Stufe:** Dirichlet- und Normal-Wishart-Faktoren aus gewichteten $\gamma,\xi,X$ aktualisieren; beiden NIG-Köpfe aus $\widetilde m,\widetilde H$ der **gelabelten** Ringe aktualisieren.
7. **ELBO und Diagnostik:** Zustandssummen, KL-Terme, gewichtete Prozessbeiträge, lokale Gewinne und ELBO protokollieren. 10 Epochen ohne val-selektiertes Early Stopping; keine weiteren Hyperparameteränderungen nach Freeze.
8. **Vorhersage:** Neuen Ring nur aus X inferieren; beide Köpfe und Skalierungen anwenden. Diese Phase wird in Herleitung D präzisiert.

Das Training arbeitet **alternierend** und verwendet approximierte lokale Markov-Faktoren. Im Code befinden sich keine expliziten, aus dem Modell errechneten Pseudo-Qualitätslabels für $U$.

# C.9 Kontrollexperimente und Erkenntnisse

| Modelltest | Gesicherte Beobachtung | Schlussgrenze |
|---|---|---|
| `OFF`, `TILT_JOINT`, `FULL_JOINT` | FULL_JOINT nach 10 Epochen höhere ELBO als OFF, bei sehr ähnlichen Zustandsmassen | ELBO-Verbesserung beweist keine Vorhersageverbesserung |
| 8 vs. 16 lokale Iterationen | Differenz im Experiment höchstens ungefähr $1{,}4\cdot10^{-8}$ | Kein allgemeiner Konvergenzbeweis |
| Diagonal vs. voll | Voll kann Prozesslogscores verbessern; Qualitätsziele teils gegensätzlich | Keine allgemein „beste“ Emission |
| Wishart-Prior-Skalierung / $\nu_0$ | Kondition und Qualitätsfehler reagieren stark auf Priorwahl | Werte nicht nach äußerem Test neu optimieren |
| Lineare zeitabhängige Emission | Kein allgemeiner Gewinn, Außen-RMSE schlechter | Eigener explorativer Vergleichszweig |
| 12 Ringdeskriptoren ohne HMM | Außenunrundheit-Vergleichs-RMSE 0,4803 mm inner-CV; Höhenschlag schlechter | Kein vollgemeinsames SSR; mehrfach genutzte CV-Ringe |
| Hybrid sechs Kennwerte | Außen-RMSE 0,5498 mm inner-CV | Höhenschlag schwächer |
| Zwei Kopfstrukturen | Außen-RMSE 0,5529 mm, Höhenschlag 3,2026 mm inner-CV | Explorative Architekturfindung, N=8 |
| Zwei-Kopf-SSR-Ablation | `C1` gegen `C0`: kleine Verbesserung beider Ziele; `C1` vs. rein supervised beim Höhenschlag schlechter | Vorverarbeitungs- und ELBO-Effekt trennen |

**Numerischer Teststatus:** Originaltests der Zustandsmomente, lokalen analytischen Gradienten, Normal-Wishart-Dichten/KL, augmentierten Kopfstatistiken und der exakten Qualitätslikelihood-Komprimierung wurden auf künstlichen Daten bestanden. Reale Trainingstests und vier innere Folds sind durch Nutzer-Terminalausgaben bestätigt. Eine mathematisch einwandfreie Komponente ersetzt **keine** unabhängige Validierung der Gesamtvorhersage.

# C.10 Leakage-Regeln und versiegelte äußere Bewertung

Die Entwicklung trennte acht gelabelte Ringe $L$, vier künstlich ungelabelte Trainingsringe $U$ und vier echte äußere Testringe. Für die finale Vorhersageroutine wurde ein gesonderter `XOnlyRing`-Zugriff implementiert; Qualitätswerte von $U$ und dem äußeren Test wurden nicht als Trainings-/Vorhersageinputs indiziert. Nach Protokoll-Freeze wurde jede Vorhersage über alle drei Modelle **vor** Öffnung der äußeren Testlabels mit SHA256 gesichert.

Der Vorhersage-Commit lautet `fcaf3cf7874070845641e6f46c5e9dcafe77dde5812427caa3dde87a8178ab26`. Der Report verwendet Protokollhash `ff964627da5c4e04308659d2f7681338966fa834f48049c818ef9b2b5b6ef0b8`. Am 11.11.2026 wurde der äußere Test **einmalig** ausgewertet. Jede weitere Optimierung anhand der vier äußeren Testfehler ist für das bereits behauptete unabhängige Ergebnis ausgeschlossen.

# C.11 Abgleich alt gegen neu

| Original-C-Kapitel | Neufassung |
|---|---|
| C.1–C.4 | Gleiche ELBO-Struktur, jetzt zwei ringweite Köpfe und voller NW-Prozessfaktor. |
| C.5.1 | Dirichlet-Updates erhalten, nach Ringgruppen kontrolliert gewichtet. |
| C.5.2 | Diagonale NIG-Emissionsupdates **ersetzt** durch multivariate Normal-Wishart-Updates. |
| C.5.3 | Einheitsdimensionierte Zustandsregression **ersetzt** durch zwei verschieden dimensionierte NIG-Köpfe mit augmentierten Momenten. |
| C.6 | Globaler KL neu: Dirichlet + Normal-Wishart + zwei NIG-Faktoren. |
| C.7 | Freie $q_0,Q_t$-Markov-Logits mit analytischen Gradienten tatsächlich implementiert. |
| C.8 | Gewicht $C$ einschließlich U-Entropie erhalten; `C0`-Vorverarbeitung explizit kontrolliert. |
| C.9–C.13 | Ausgeführter DM-Pipeline-/Trainings- und Testablauf, numerische Tests und abgeschlossener Freeze. |
| C.14–C.15 | Neu: dokumentierte Evidenz, Ergebnissicherheit und Forschungsgrenzen. |

**Ergebnis von Herleitung C:** Das tatsächliche B1-Verfahren ist mathematisch und algorithmisch bestimmt. Seine globale Inferenz ist vollständig priorbehaftet, aber variational approximiert; seine lokale gelabelte Markov-Inferenz ist frei und differenzierbar, aber mit endlicher Iterationszahl gelöst. Die Qualitätsvorhersage mit dieser Approximation folgt in D.
