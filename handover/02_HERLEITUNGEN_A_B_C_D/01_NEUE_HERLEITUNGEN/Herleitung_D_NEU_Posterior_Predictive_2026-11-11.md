---
title: "Herleitung D (Neufassung) – Posterior Predictive für zwei Qualitätsgrößen"
subtitle: "Exakte bedingte NIG-Integration, Zustandsmischung, implementierte Momentenapproximation und äußerer Test"
date: "11. November 2026 | Abschlussmodell und offene prädiktive Erweiterungen"
lang: de-DE
---

# D.0 Ausgangspunkt und Abgrenzung

Das Original `Herleitung_D_Posterior_Predictive_Bayesian_SSR.docx` war für **drei** Qualitätswerte, eine reine Zustandsaufenthaltsregression und diagonale Emissionen geschrieben. Es diskutierte zudem vollständige prädiktive Simulation, Importance Sampling und Quantilintervalle als **Entwicklungswege**. Der am 11.11.2026 tatsächlich einmalig extern geprüfte Code `two_head_b1.predict` und `outer_predict_frozen.py` berechnet demgegenüber für **zwei** Qualitätsgrößen jeweils den prädiktiven Mittelwert und eine momentbasierte Standardabweichung. Diese Implementierungsgrenze wird im Folgenden explizit hergeleitet.

# D.1 Neues Testobjekt und eingefrorene Modellparameter

Für einen neuen Ring $*$ ist ausschließlich seine vollständig aufgezeichnete Loggersequenz $X_*$ mit Zeitachse $t_*$ verfügbar. Beide Qualitätswerte sind für die Vorhersage unbekannt. F2-Fenster und die sechs Loggerkanäle werden identisch zum Training angewandt. Der Prozess-Scaler $S_X$, die zwei Descriptor-Scaler $S_{s_j}$ und der Qualitäts-Scaler $S_Y$ stammen **ausschließlich aus den jeweiligen Trainingsringen**. Die zugehörigen Ringkennwerte werden deterministisch berechnet:

$$\mathbf s_{*j}=S_{s_j}\!\big(f_j(S_X(X_*),t_*)\big),\qquad j\in\{1,2\}.$$

Die relevante globale Approximation lautet

$$q(\Theta)=q(\boldsymbol\pi)\prod_kq(A_{k,:})\prod_kq(\boldsymbol\mu_k,\Lambda_k)\prod_{j=1}^2q(\boldsymbol\beta_j,\sigma_j^2).$$

Die beiden Qualitätsköpfe besitzen feste, im Protokoll dokumentierte unterschiedliche Eingangsdimensionen $D_1=9$ und $D_2=8$. Die Standardisierung der Zielgröße wird am Ende rückgängig gemacht; tabellierte Außen-Testfehler sind daher **mm**, nicht normierte Einheiten.

# D.2 Der ideale vollständige Posterior Predictive

Formal wäre die volle Bayesian-Vorhersage

$$p(\mathbf y_*\mid X_*,D)=\int\sum_{Z_*}p(\mathbf y_*\mid Z_*,X_*,\Psi)\,p(Z_*\mid X_*,\Theta_X)\,p(\Theta\mid D)\,d\Theta.$$

Unsere Software approximiert $p(\Theta\mid D)$ durch $q(\Theta)$ und den neuen Zustandsfaktor durch eine Prozess-Forward-Backward-Verteilung auf **erwarteten HMM-Logpotenzialen**. Eine vollständige **Monte-Carlo-Marginalisierung über global zufällige HMM-Parameter** findet bei der eingefrorenen Routine nicht statt. Auch die zwei Qualitätskopffaktoren sind im globalen Variationsposterior faktorisiert. Man darf die nachfolgend berechneten Momente deshalb nicht als exakte vollständig integrierte wahre Posterior-Predictive-Momente ausgeben.

# D.3 Induktive lokale Zustandsinferenz ohne Qualitätslabel

Für $X_*$ wird

$$\ell_{*tk}=\mathbb E_{q(\boldsymbol\mu_k,\Lambda_k)}\log\mathcal N_d(\mathbf x_{*t};\boldsymbol\mu_k,\Lambda_k^{-1})$$

sowie $\mathbb E_q\log\boldsymbol\pi$ und $\mathbb E_q\log A$ verwendet. Die neue **prozessbedingte** Zustandsverteilung $q_*(Z_*\mid X_*)$ entsteht mit Forward-Backward. Anders als beim Training eines gelabelten Rings wird hier die Qualitätslikelihood nicht in den lokalen Faktor aufgenommen – sonst würde man das unbekannte Testlabel voraussetzen. Die Momentenrekursion aus B liefert

$$\mathbf m_*=\mathbb E_{q_*}[\mathbf h_*],\quad H_*=\mathbb E_{q_*}[\mathbf h_*\mathbf h_*^\top],\quad \operatorname{Cov}_{q_*}(\mathbf h_*)=H_*-\mathbf m_*\mathbf m_*^\top.$$

Für jeden Qualitätskopf werden wieder die deterministischen X-Kennwerte angefügt:

$$\widetilde{\mathbf m}_{*j}=\begin{bmatrix}\mathbf m_*\\\mathbf s_{*j}\end{bmatrix},\qquad \widetilde H_{*j}=\begin{bmatrix}H_*&\mathbf m_*\mathbf s_{*j}^\top\\\mathbf s_{*j}\mathbf m_*^\top&\mathbf s_{*j}\mathbf s_{*j}^\top\end{bmatrix}.$$

# D.4 Bedingte Qualitätsprädiktion: Normal-Invers-Gamma führt zu Student-t

Für einen **fixierten** augmentierten Qualitätsvektor $\mathbf u_{*j}$ und ein NIG-Posterior $q(\boldsymbol\beta_j,\sigma_j^2)=\operatorname{NIG}(\boldsymbol\mu_j,V_j,a_j,b_j)$ ist die integrierte ein-dimensionale Verteilung

$$y_{*j}\mid \mathbf u_{*j},q(\boldsymbol\beta_j,\sigma_j^2)\sim t_{2a_j}\!\left(\boldsymbol\mu_j^\top\mathbf u_{*j},\quad \frac{b_j}{a_j}\left[1+\mathbf u_{*j}^\top V_j\mathbf u_{*j}\right]\right).$$

Der zweite Parameter ist hier **das Quadrat der Student-t-Skala**. Die Verteilungsvarianz existiert für $a_j>1$ und beträgt

$$\operatorname{Var}(y_{*j}\mid\mathbf u_{*j})=\frac{b_j}{a_j-1}\big(1+\mathbf u_{*j}^\top V_j\mathbf u_{*j}\big).$$

Der für $X_*$ relevante Zustandspfad ist aber weiterhin unbekannt. Daher entsteht **nach Mischung über $q_*(Z_*)$ im Allgemeinen keine einzelne Student-t-Verteilung**, sondern eine Mischung mit zufallsabhängigem $\mathbf h_*(Z_*)$.

# D.5 Tatsächlich implementierter prädiktiver Mittelwert

Der Erwartungswert der unter $q_*$ gemischten bedingten Regressionsprädiktion ist

$$\boxed{\widehat y_{*j}=\boldsymbol\mu_j^\top\widetilde{\mathbf m}_{*j}}.$$

Im Original D war dies $\bar B\mathbf m_h$ für drei Zielgrößen. Neu ist die Summe eines **zustandsbedingten** und eines **direkten beobachteten Kennwertanteils** für jeden der zwei Köpfe:

$$\widehat y_{*j}=\boldsymbol\mu_{hj}^\top\mathbf m_*+\boldsymbol\mu_{sj}^\top\mathbf s_{*j}.$$

Der Code `two_head_b1.predict` verwendet genau diese Gleichung. Der Mittelwert ist auf standardisierter Zielskala berechnet und wird mit dem ausschließlich aus L-Ringen angepassten Y-Scaler nach mm zurücktransformiert.

# D.6 Tatsächlich implementierte prädiktive Varianz

Mit dem Gesetz der totalen Varianz und den exakten Momenten unter dem **lokalen Markov-Faktor** folgt

$$\boxed{v_{*j}=\frac{b_j}{a_j-1}\Big(1+\operatorname{tr}(V_j\widetilde H_{*j})\Big)+\boldsymbol\mu_j^\top\left(\widetilde H_{*j}-\widetilde{\mathbf m}_{*j}\widetilde{\mathbf m}_{*j}^\top\right)\boldsymbol\mu_j.}$$

Die zwei Summanden repräsentieren (i) erwartete bedingte Beobachtungs-/Regressionsunsicherheit und (ii) Varianz des posterioren Zustandsaufenthaltes bei festem Regressionsmittelwert. Der zweite Term betrifft nur den variablen Aufenthaltsblock, da die aus $X_*$ berechneten Descriptoren deterministisch sind. Der Code begrenzt einen durch Rundung leicht negativen Zustandsvarianzterm auf null, ebenso wird die ausgegebene Standardabweichung numerisch positiv gehalten.

$$\widehat{\operatorname{SD}}(y_{*j})=\sqrt{\max(v_{*j},10^{-12})}.$$

**Zentrale Einschränkung:** Die Unsicherheit des globalen HMM-Parameters $\Theta_X$ wird nicht voll durch wiederholtes Simulieren aus $q(\Theta_X)$ und erneute Zustandsinferenz integriert. Die Standardabweichung ist eine **momentbasierte, variational bedingte Approximation**; nicht zwingend die vollständige physikalische Prognoseunsicherheit.

# D.7 Abhängigkeiten zwischen den zwei Qualitätsprognosen

Obwohl $q(\boldsymbol\beta_1,\sigma_1^2)$ und $q(\boldsymbol\beta_2,\sigma_2^2)$ als globale Faktoren separat sind und die Qualitätsrauschkomponenten bedingt unabhängig angenommen werden, teilen beide Köpfe **denselben** zufälligen Aufenthaltsvektor. Daher enthält ihre prädiktive Kreuzkovarianz mindestens den gemeinsamen Zustandsunsicherheitsbeitrag

$$\operatorname{Cov}_{\mathrm{state}}(y_{*1},y_{*2})=\boldsymbol\mu_{h1}^\top\left(H_*-\mathbf m_*\mathbf m_*^\top\right)\boldsymbol\mu_{h2}.$$

Im eingefrorenen äußeren Ergebnis wurden jedoch **nur die zwei marginalen Standardabweichungen** gespeichert, keine vollständige $2\times2$-Qualitäts-Kovarianzmatrix. Die obige Beziehung gehört daher zur mathematischen Konsequenz des Modells, **nicht** zu den bereits berichteten numerisch ausgegebenen Kennzahlen.

# D.8 Quantile, Intervalle und was der Code wirklich berechnet

Das Original D.9 sah prädiktive Samples und Quantile aus der gesamten Zustands-/NIG-Mischung vor. Diese **exakten numerischen Mischungsquantile wurden nicht implementiert**. Stattdessen enthält der Abschlussbericht die deskriptive Normalapproximation

$$I_{*j}^{(95\%,\mathrm{approx})}=[\widehat y_{*j}-1{,}96\widehat{\operatorname{SD}}(y_{*j}),\ \widehat y_{*j}+1{,}96\widehat{\operatorname{SD}}(y_{*j})].$$

Diese Intervalle sind **weder exakte 95-%-Quantilintervalle** der Student-t-Mischung noch nachweislich frequentistisch kalibriert. Auf vier äußeren Ringen liegen die beiden Qualitätswahrheiten jeweils innerhalb dieses approximierten Intervalls (4/4 je Ziel). Die prädiktiven Halbbreiten sind relativ groß; daraus lässt sich keine allgemeine Kalibrierungsaussage ableiten.

Die ursprünglich vorgesehene Klassifikation **P1/P2/P3** ist im finalen Code nicht als systematisch durchgeführte experimentelle Ablation vorhanden. P2 (bedingte global-HMM-Parametersimulation mit Importance-Weighing und ESS) und vollständige Mischungsquantile bleiben mögliche **neue Forschung**, nicht Ergebnis dieser Abschlussauswertung.

# D.9 Definition und dokumentierte Berechnung der äußeren Qualitätsmetriken

Für $N=4$ äußere Test-Ringe wird **jeder Ring gleich gewichtet**, unabhängig von $T_i$:

$$\operatorname{RMSE}_j=\sqrt{\frac1N\sum_{i=1}^N(\widehat y_{ij}-y_{ij})^2},\qquad \operatorname{MAE}_j=\frac1N\sum_{i=1}^N|\widehat y_{ij}-y_{ij}|.$$

Für den versiegelten äußeren Test gelten die **unveränderten**, im hochgeladenen `OUTER_FINAL_SCORE_V1.json` gespeicherten Werte:

| Modell | RMSE Außen [mm] | RMSE Höhenschlag [mm] | MAE Außen [mm] | MAE Höhenschlag [mm] |
|---|---:|---:|---:|---:|
| Zweikopf-Bayesian-SSR (`primary_ssl`) | **0,3888** | 3,2064 | **0,3564** | 2,8053 |
| Zweikopf rein überwacht (`strict_supervised`) | 0,5529 | 3,3579 | 0,4834 | 3,1092 |
| Trainingsmittelwert (`mean_baseline`) | 0,4953 | **2,9785** | 0,4474 | **2,5937** |

Die paarweise Differenz des mittleren quadratischen Fehlers `primary_ssl minus control` lautet in mm²: gegen `strict_supervised` $(-0{,}1545467,-0{,}9943688)$, gegen `mean_baseline` $(-0{,}0941513,+1{,}4098057)$. Ein negativer Wert bevorzugt SSR; beim Höhenschlag ist die Baseline besser.

## D.9.1 Die vier individuellen äußeren Ringprognosen des Hauptmodells

| Ring | Wahrheit Außen [mm] | SSR Außen [mm] | Wahrheit Höhenschlag [mm] | SSR Höhenschlag [mm] | SD Außen / Höhe [mm] |
|---|---:|---:|---:|---:|---:|
| DM07 | 2,5444 | 2,1491 | 16,7152 | 15,4342 | 0,8553 / 3,5650 |
| DM08 | 2,7078 | 2,2471 | 13,6179 | 18,7056 | 0,8513 / 3,8578 |
| DM11 | 2,3657 | 1,8889 | 16,5130 | 13,1317 | 0,6472 / 2,8221 |
| DM15 | 1,8461 | 1,9387 | 17,4662 | 18,9375 | 0,8464 / 4,0325 |

Die numerische Interpretation ist differenziert: Außenunrundheit ist bei vier Ringen **besser** als beide Kontrollmodelle im aggregierten RMSE. Für Höhenschlag ist SSR besser als dieselbe zweiköpfige Architektur ohne U-Ringe, aber **schlechter** als die simple Baseline. Bei DM08 ist der absolute Höhenschlagfehler des SSR-Modells ungefähr 5,09 mm. Es wäre daher wissenschaftlich falsch, eine generell überlegene Qualitätsvorhersage für beide Ziele zu behaupten.

# D.10 Ablauf der einmaligen äußeren Auswertung

**Protokoll-Freeze:** `DM_TWO_HEAD_PROTOCOL_FREEZE.json`; Originaldaten-SHA256 `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`. Die Vorhersagen wurden am 11.11.2026 mit `outer_predict_frozen.py` für alle drei Modelle erstellt und mit SHA256 **vor** Testlabelöffnung versiegelt:

`fcaf3cf7874070845641e6f46c5e9dcafe77dde5812427caa3dde87a8178ab26`.

Danach wurde `outer_score_frozen.py` **einmalig** ausgeführt. Die äußere Testauswertung ist nun **geschlossen**. Die Ringe DM07, DM08, DM11 und DM15 dürfen bei späteren Modellsuchen nicht erneut als unangetasteter Testsatz behandelt werden. Die Kenntnis der Resultate kann selbst bei guter Absicht die nächste Modellentwicklung beeinflussen. Neue unabhängige Bestätigung erfordert neue zuvor ungesehene Ringe oder einen externen Datensatz und ein zuvor festgelegtes Bewertungsprotokoll.

# D.11 Abgleich mit Original D und zukünftige Forschungszweige

| Original-D-Teil | Neue Aussage und Status |
|---|---|
| D.1–D.3 | Neuer Ring nur aus Prozess-X; $J=2$; zeitlich vollständige Sequenz; train-only Skalierung. |
| D.4 | Exakte Aufenthaltsmomente *unter dem prozessbedingten lokalen $q_*$* aus B. |
| D.5 | Vollständige HMM-Parameterintegration **nicht** implementiert; erwartete Prozess-Logpotenziale. |
| D.6–D.7 | Zwei NIG-Köpfe mit unterschiedlichen augmentierten Momenten; Mittelwert und Varianz hergeleitet. |
| D.8 | Mögliche Ziel-Kreuzkovarianz nach Integration von Zustand, nicht als numerischer Output verfügbar. |
| D.9 | Nur $\mu\pm1{,}96\,\mathrm{SD}$, **keine** echten Mischungsquantile. |
| D.10 | P1/P2/P3 noch **kein** abgeschlossenes Vergleichsexperiment. |
| D.11–D.13 | Ringweise Testmetriken, Commit und One-time-Score aus tatsächlich ausgeführtem Pipeline-Code. |
| D.14–D.16 | Numerische Teilprüfungen ausgeführt; Unsicherheitskalibrierung mit $N=4$ offen. |

**Ergebnis der neuen Herleitung D:** Zwei probabilistische Endqualitätsprognosen mit getrennten Bayes-Köpfen, gemeinsamer latenter Zustandsunsicherheit und implementierter momentbasierter posterior-prädiktiver Approximation. Die Formeln unterscheiden ausdrücklich eine mathematisch umfassendere idealisierte Bayesian-Predictive von der tatsächlich ausgeführten, approximativen Abschlussroutine.
