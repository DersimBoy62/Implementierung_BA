---
title: "Herleitung B (Neufassung) – Strukturierte Variational Inference und zwei Qualitätsköpfe"
subtitle: "Freie Markov-Variationsfamilie, exakte Aufenthaltsmomente, augmentierte Qualitätsstatistiken"
date: "11. November 2026 | mathematisch mit Implementierung abgeglichen"
lang: de-DE
---

# B.0 Status und prüfbare Referenz

Die ursprüngliche Herleitung B (`Herleitung_B_Strukturierte_Variationsverteilung_Bayesian_SSR.docx`) beschrieb ein Modell mit diagonalen Prozessemissionen, drei Qualitätswerten und gemeinsamer Zustandsaufenthaltsregression. Der neu implementierte **Zweikopf-Modellkern** weicht davon ab. Folgende Darstellungen wurden mit `full_markov_b1.py`, `fullcov_b1.py` und `two_head_b1.py` abgeglichen. Die exakte Berechnung der ersten/zweiten Momente gilt **innerhalb der Markov-Variationsfamilie**; die globale Posteriorapproximation bleibt eine Variationsapproximation.

# B.1 Ausgangspunkt: exakter gemeinsamer Posterior

Gegeben sind $D=(X_L,Y_L,X_U)$ und globale Parameter $\Theta$ aus Herleitung A. Der exakte Posterior

$$p(\Theta,Z\mid D)=\frac{p(\Theta,Z,X_L,X_U,Y_L)}{p(X_L,X_U,Y_L)}$$

ist wegen der Kopplung der gesamten Sequenz an einen einzigen Qualitätswert und wegen der nichtlinearen Integration über sämtliche Parameter und Zustandsfolgen praktisch nicht geschlossen auswertbar. Die direkte Produktzerlegung der Qualitätslikelihood über einzelne Zeitpunkte ist **nicht** zulässig: Jede $y_{ij}$ ist eine **ringweite** Beobachtung, keine Serie von $T_i$ unabhängigen Labels.

# B.2 Gewählte Variationsfamilie

Wir approximieren den Posterior durch

$$q(\Theta,Z)=q(\boldsymbol\pi)\prod_{k=1}^K q(A_{k,:})\prod_{k=1}^K q(\boldsymbol\mu_k,\Lambda_k)\prod_{j=1}^2 q(\boldsymbol\beta_j,\sigma_j^2)\prod_{i\in L\cup U}q_i(Z_i).$$

Die globalen Faktoren bleiben in den jeweiligen konjugierten Familien: Dirichlet, Normal-Wishart und Normal-Invers-Gamma. Für die **lokalen** Faktoren wird nicht die künstlich vollfaktorisierte Zeitpunktsform $\prod_tq(z_t)$ gewählt, sondern eine frei parametrisierte Markov-Kette:

$$q_i(Z_i)=\rho_i(z_{i1})\prod_{t=2}^{T_i}Q_{i,t-1}(z_{i,t-1},z_{it}).$$

Dabei sind $\boldsymbol\rho_i\in\Delta^{K-1}$ und alle **zeitabhängigen** Zeilen $Q_{i,t-1}(k,:)\in\Delta^{K-1}$ frei. In `full_markov_b1.py` werden sie durch Softmax aus reellen Logits parametrisiert. Dies unterscheidet `FULL_JOINT` grundlegend von der historisch ebenfalls getesteten, auf konstante Zusatzpotenziale beschränkten Variante `TILT_JOINT`.

# B.3 Prozessmarginalen und Übergangsstatistiken

Die prozessbezogene lokale Zielfunktion hängt von den erwarteten Log-Priors, Log-Übergängen und Emissionen ab. Benötigt werden

$$\gamma_{itk}=q_i(z_{it}=k),\qquad \xi_{itk\ell}=q_i(z_{it}=k,z_{i,t+1}=\ell).$$

Für eine explizite lokale Markov-Kette gilt $\gamma_{i1}=\rho_i$ und

$$\xi_{itk\ell}=\gamma_{itk}Q_{it}(k,\ell),\qquad \gamma_{i,t+1,\ell}=\sum_{k=1}^K\xi_{itk\ell}.$$

Die Zustands- und Übergangszählungen sind $n_{ik}=\sum_t\gamma_{itk}$ und $N_{ik\ell}=\sum_{t=1}^{T_i-1}\xi_{itk\ell}$. Ihre Summen lauten $\sum_kn_{ik}=T_i$ und $\sum_{k\ell}N_{ik\ell}=T_i-1$. Für ungelabelte Ringe ist die optimale Markov-Verteilung bei fixierten globalen Faktoren und ohne ringweiten Qualitätsbeitrag durch logstabilen Forward-Backward auf den **erwarteten** HMM-Logpotenzialen berechenbar. Für gelabelte Ringe wird die freie Markov-Verteilung zusätzlich nach dem globalen Qualitätsterm optimiert.

# B.4 Exakte erste und zweite Aufenthaltsmomente

Die relative Aufenthaltsstatistik eines Rings ist

$$h_{ik}(Z_i)=\frac{N_{i,T_i}^k}{T_i},\qquad N_{it}^k=\sum_{s=1}^t\mathbf1\{z_{is}=k\}.$$

Die für die Regression notwendigen Momente sind

$$\mathbf m_i=\mathbb E_{q_i}[\mathbf h_i],\qquad H_i=\mathbb E_{q_i}[\mathbf h_i\mathbf h_i^\top].$$

Nur $\mathbf m_i$ einzusetzen wäre **nicht äquivalent**, denn im Allgemeinen gilt $H_i\neq\mathbf m_i\mathbf m_i^\top$. Der Unterschied $H_i-\mathbf m_i\mathbf m_i^\top$ beschreibt die unter $q_i$ vorhandene Unsicherheit über Zustandsanteile.

## B.4.1 Dynamische Rekursion für die ersten und zweiten Zählmomente

Für einen Ring unter einer lokalen Markov-Variationskette definieren wir unnormierte, auf das aktuelle Endzustandsereignis gewichtete Größen

$$\begin{aligned}
P_t(k)&=q(z_t=k),\\
M_t(k,a)&=\mathbb E[N_t^a\mathbf1\{z_t=k\}],\\
S_t(k,a,b)&=\mathbb E[N_t^aN_t^b\mathbf1\{z_t=k\}].
\end{aligned}$$

Mit $\delta_{a,k}=\mathbf1\{a=k\}$ gilt bei $t=1$:

$$P_1(k)=\rho(k),\quad M_1(k,a)=\rho(k)\delta_{a,k},\quad S_1(k,a,b)=\rho(k)\delta_{a,k}\delta_{b,k}.$$

Für $t\ge2$ folgen die exakten Rekursionen:

$$P_t(k)=\sum_jP_{t-1}(j)Q_{t-1}(j,k),$$

$$M_t(k,a)=\sum_jM_{t-1}(j,a)Q_{t-1}(j,k)+\delta_{a,k}P_t(k),$$

$$\begin{aligned}
S_t(k,a,b)&=\sum_j S_{t-1}(j,a,b)Q_{t-1}(j,k)\\
&\quad+\delta_{a,k}\sum_j M_{t-1}(j,b)Q_{t-1}(j,k)\\
&\quad+\delta_{b,k}\sum_j M_{t-1}(j,a)Q_{t-1}(j,k)\\
&\quad+\delta_{a,k}\delta_{b,k}P_t(k).
\end{aligned}$$

Daraus ergibt sich am Sequenzende:

$$m_{ia}=\frac1{T_i}\sum_kM_{T_i}(k,a),\qquad H_{iab}=\frac1{T_i^2}\sum_kS_{T_i}(k,a,b).$$

Der konkrete Code `vb_b1.moments_occupancy` verwendet eine äquivalente **rückwärts konditionierte** Rekursion auf Basis der Paarmarginalen $\xi$ und $\gamma$. `full_markov_b1._markov_objective_and_grad` verwendet zusätzlich eine vorwärtsgerichtete Zählmomentenrekursion mit analytischem Reverse-Mode-Gradienten. Beide Varianten werden in synthetischen Enumerationstests kontrolliert. Die Rechnung ist für gegebenes $q_i$ exakt; sie ist keine exakte Lösung des gesamten latenten Posteriors.

**Invarianten:** $\mathbf1^\top\mathbf m_i=1$, $H_i\mathbf1=\mathbf m_i$, $\mathbf1^\top H_i\mathbf1=1$, $H_i\succeq0$ und $H_i-\mathbf m_i\mathbf m_i^\top\succeq0$ bis auf numerische Rundungsfehler.

# B.5 Neue augmentierte Momente für die zwei unterschiedlichen Qualitätsköpfe

Aus Herleitung A ist $\mathbf u_{ij}=[\mathbf h_i;\mathbf s_{ij}]$. Die beobachteten Ringkennwerte $\mathbf s_{ij}$ sind bei fixierter Loggersequenz **deterministisch**. Deshalb gelten die folgenden **analytischen** Zusammenhänge ohne weitere Simulation:

$$\widetilde{\mathbf m}_{ij}=\mathbb E_{q_i}[\mathbf u_{ij}]=\begin{bmatrix}\mathbf m_i\\\mathbf s_{ij}\end{bmatrix}$$

und

$$\widetilde H_{ij}=\mathbb E_{q_i}[\mathbf u_{ij}\mathbf u_{ij}^\top]=\begin{bmatrix}H_i&\mathbf m_i\mathbf s_{ij}^\top\\\mathbf s_{ij}\mathbf m_i^\top&\mathbf s_{ij}\mathbf s_{ij}^\top\end{bmatrix}.$$

Das sind die implementierten Inhalte von `two_head_b1.augment`. Für Kopf 1 hat die Matrix Dimension $9\times9$, für Kopf 2 $8\times8$. Insbesondere darf der Block links oben **nicht** durch $\mathbf m_i\mathbf m_i^\top$ ersetzt werden, weil dies die Zustandsunsicherheit unterdrücken würde.

# B.6 Erwarteter ringweiter Qualitätsbeitrag zur ELBO

Für jeden Qualitätskopf gilt im NIG-Faktor

$$q(\boldsymbol\beta_j,\sigma_j^2)=\operatorname{NIG}(\boldsymbol\mu_j,V_j,a_j,b_j),\qquad \mathbb E_q[\sigma_j^{-2}]=\frac{a_j}{b_j},$$

$$\mathbb E_q[\log \sigma_j^{-2}]=\psi(a_j)-\log b_j.$$

Die erwartete Loglikelihood eines **einzigen** Qualitätswerts $y_{ij}$ lautet

$$\begin{aligned}
\mathcal L_{ij}^{(Y)}&=\mathbb E_{q_iq_j}[\log p(y_{ij}\mid\mathbf u_{ij},\boldsymbol\beta_j,\sigma_j^2)]\\
&=-\frac12\log(2\pi)+\frac12[\psi(a_j)-\log b_j]\\
&\quad-\frac12\left\{\frac{a_j}{b_j}\left[y_{ij}^2-2y_{ij}\boldsymbol\mu_j^\top\widetilde{\mathbf m}_{ij}+\boldsymbol\mu_j^\top\widetilde H_{ij}\boldsymbol\mu_j\right]+\operatorname{tr}(V_j\widetilde H_{ij})\right\}.
\end{aligned}$$

Diese Gleichung ist **der zentrale neue Qualitätsbeitrag** gegenüber Original B.9. Beide Terme $\mathcal L_{i1}^{(Y)}$ und $\mathcal L_{i2}^{(Y)}$ werden für einen gelabelten Ring addiert. Es wird kein künstlicher Zeitpunktsverlust für Qualitätslabels erzeugt.

## B.6.1 Komprimierung der Qualitätsrückkopplung

Zerlegen wir $\boldsymbol\mu_j=(\boldsymbol\mu_{hj},\boldsymbol\mu_{sj})$ und $V_j$ in Zustands-, Kennwert- und Kreuzblöcke. Bei gegebenem $y_{ij}$ und $\mathbf s_{ij}$ lässt sich der Qualitätsbeitrag darstellen als

$$\mathcal L_{ij}^{(Y)}(q_i)=c_{ij}+\mathbf b_{ij}^\top\mathbf m_i+\operatorname{tr}(G_{ij}H_i),$$

mit

$$\begin{aligned}
\mathbf b_{ij}&=\frac{a_j}{b_j}y_{ij}\boldsymbol\mu_{hj}-\left(\frac{a_j}{b_j}\boldsymbol\mu_{hj}\boldsymbol\mu_{sj}^\top+V_{j,hs}\right)\mathbf s_{ij},\\
G_{ij}&=-\frac12\left(\frac{a_j}{b_j}\boldsymbol\mu_{hj}\boldsymbol\mu_{hj}^\top+V_{j,hh}\right).
\end{aligned}$$

Die in $c_{ij}$ zusammengefassten restlichen Terme hängen bei festem $y_i$, $\mathbf s_i$ und globalem $q$ **nicht** von der lokalen Markov-Verteilung ab. Die Summe über die beiden Köpfe ergibt $c_i$, $\mathbf b_i$ und $G_i$. Die Implementierung `two_head_b1.both_compressed_coeff` berechnet exakt diese Komprimierung und übergibt sie an den freien lokalen Markov-Optimierer. Durch diesen Schritt wirken beide Zielgrößen **gemeinsam auf dieselbe lokale Zustandsfolge zurück**, ohne die gesamte Ringsequenz enumerieren zu müssen.

# B.7 Lokale Optimierung: Warum Standard-Forward-Backward nicht genügt

Für einen gelabelten Ring sind mit globalem $q(\Theta)$ fest die erwarteten Prozesspotenziale

$$\ell_{itk}=\mathbb E_{q(\mu_k,\Lambda_k)}\log p(\mathbf x_{it}\mid z_{it}=k),\quad g_k=\mathbb E_q\log\pi_k,\quad a_{k\ell}=\mathbb E_q\log A_{k\ell}.$$

Die lokale Zielfunktion lautet

$$\begin{aligned}
F_i(q_i)&=\sum_k\gamma_{i1k}g_k+\sum_{t,k,\ell}\xi_{itk\ell}a_{k\ell}+\sum_{t,k}\gamma_{itk}\ell_{itk}\\
&\quad+\mathcal H[q_i]+\sum_{j=1}^2\mathcal L_{ij}^{(Y)}.
\end{aligned}$$

Die lokale Entropie ist für die parametrisierte Markov-Kette explizit

$$\mathcal H[q_i]=-\sum_k\rho_i(k)\log\rho_i(k)-\sum_{t=1}^{T_i-1}\sum_{k,\ell}\xi_{itk\ell}\log Q_{it}(k,\ell).$$

Da $H_i$ die gesamte Sequenz koppelt, zerfällt $F_i$ nicht in eine bloße Summe gewöhnlicher HMM-Zeitpunktpotenziale. **Ein unverändertes Forward-Backward-Update ist daher für gelabelte Ringe nicht das vollständige lokale B1-Optimum.** Stattdessen optimiert der Code mit L-BFGS-B freie Logits für $\rho_i$ und alle $Q_{it}$. Die Ableitungen der exakt rekursiv berechneten Aufenthaltsmomente werden analytisch mitgerechnet. Ein lokaler Kandidat wird nur übernommen, wenn er die lokale Zielfunktion gegenüber der prozessbasierten Startverteilung nachweislich erhöht. Für das abgeschlossene Experiment waren **höchstens acht lokale Iterationen** vorgesehen – das ist keine Konvergenzgarantie.

Für einen ungelabelten Ring fehlt die letzte Summe und das konditionale lokale Optimum innerhalb der Markov-Familie ist per Forward-Backward zugänglich.

# B.8 Globale Konjugation und ihre Grenze

Die globale Faktorapproximation beachtet die Kopplung von Mittelwert und Präzision innerhalb **jedes** Normal-Wishart-Faktors und von Regressionskoeffizient und Varianz innerhalb **jedes** NIG-Faktors. Dagegen werden Prozessparameter, Qualitätsköpfe und ringweise $q_i$ variational voneinander faktorisiert. Die im Modell enthaltenen Bayesian-Priors machen die Inferenz somit **nicht exakt**. Eine höhere ELBO beweist auch nicht, dass die Qualität des nachgelagerten Ringzielwerts besser wird.

# B.9 Abgleich alt gegen neu und Tests

| Original-B-Abschnitt | Status der Neufassung | Numerischer Prüfnachweis |
|---|---|---|
| B.1–B.4 Posterior, Variationsfaktoren | Übernommen, $q$ um Normal-Wishart und zwei ungleichdimensionierte Köpfe erweitert | `test_fullcov_b1.py`, `test_two_head_b1.py` |
| B.5–B.6 strukturierte Zustandsverteilung | Neu: vollständig freie zeitabhängige Markov-Logits statt nur als Ziel formuliert | `test_full_markov.py` |
| B.7–B.8 Aufenthaltsmomente | Beibehalten, in beiden Algorithmen exakt unter $q$ rekursiv berechnet | Enumerations- und Konsistenztests |
| B.9 erwartete Qualität | Neu: zwei getrennte augmentierte Momentenblöcke | `test_quality_link_challenge.py`, `test_two_head_b1.py` |
| B.10–B.11 lokale ELBO | Gemeinsame Summe beider Qualitätsrückkopplungen; analytische Gradienten | finite Differenzen und Qualitätsfeedback-Test |
| B.12–B.14 Implementierung | Auf echten DM-Ringen ausführbar; lokale Optimierung weiterhin approximativ | 10-Epochen- und CV4-Läufe am 11.11.2026 |

**Mathematische Grenzziehung:** Die Befunde belegen die Korrektheit getesteter Rekursionen und die Ausführbarkeit der spezifizierten Variationsoptimierung. Sie belegen **nicht**, dass das globale Optimum der ELBO gefunden wurde oder dass der wahre Posterior in der Markov-Familie liegt.

**Ergebnis von Herleitung B:** Eine vollständig konkretisierte Variationsfamilie für beide Zielgrößen, exakt berechenbare erste/zweite Zustandsmomente und eine differentiable gemeinsame Rückkopplung zweier unterschiedlich dimensionierter Qualitätsköpfe. Die globalen Updates und das Gesamttraining werden in Herleitung C formuliert.
