# Mathematisches Basismodell, Neubau vom 13.11.2026

Diese Spezifikation ersetzt die überholten Parameterstände in A–D vom 11.11.2026.
Sie beschreibt K=2, M2, zwei Qualitätsköpfe, C=0,2 und 30 Koordinatenepochen.
Die mathematischen Bausteine gelten auch für K>2; K=2/3/4 sind die vorgesehenen
synthetischen Prüffälle. Die Kombination der zwei ringweiten Köpfe, additiven
Markov-Momente und induzierten Priors ist eine eigene Konstruktion. Quellen und
verifizierte Originalstellen stehen in [QUELLENSTATUS.md](QUELLENSTATUS.md).
Aus der Spezifikation folgt kein bereits bestandener Test und keine Modellfreigabe.
Ausführungsnachweise stehen separat im Test-/Trainingsprotokoll.

## 1. Beobachtungen und Vorverarbeitung

Für Ring r sind eine vollständige F2-Sequenz X_r in R^(T_r x 6) und ihre Zeitachse
gegeben. Die sechs Spalten sind verbindlich
`F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55`.
F2 bedeutet in der Referenz `phase804 in {1,2}`, `LaserOK=1`, `MessEin=1`,
zusammenhängend, zeitlich geordnet und mit 0,1-s-Raster. Phase 6 ist dadurch
ausgeschlossen. Die Metadaten dürfen eine Sequenz nicht umsortieren oder ergänzen.
Das Dateninterface validiert Prozesssequenzen und Targets getrennt; ein U-Objekt
besitzt keine Targets. Der tatsächliche Targetcontainer der hashgesicherten Datei
heißt `zielwerte`. Die verifizierten Blattnamen sind
`ovality.ovality_radial_mm` und `planarity.height_deviation_mm`.
Der Unterschied zur Bezeichnung `all_targets` im Auftrag ist in
[DATENPROVENIENZ.md](DATENPROVENIENZ.md) festgehalten.

Pro Fold wird S_X auf den gepoolten TRAIN-Zeitpunkten angepasst; dies gewichtet
längere Ringe bei der Standardisierung stärker. S_Y verwendet ausschließlich die
beiden bekannten Labels der L-TRAIN-Ringe. Aus S_X(X_r) werden zwölf deterministische
Sequenzdeskriptoren berechnet, dann ringgleichgewichtet innerhalb TRAIN skaliert.
Ein konstanter Kanal erhält die Skala 1; Standardabweichungen verwenden ddof=0.
U-TRAIN darf für X-only-Vorverarbeitung verwendet werden, wenn der Untersuchungsarm
dies ausdrücklich vorsieht. VAL und historische Außenringe gehören nie zum Fit.

Die zwölf Deskriptoren sind in folgender Reihenfolge definiert:

| Index | Definition auf transformierter Prozesssequenz |
|---:|---|
| 0 | log(t_end-t_start), Zeit in Sekunden |
| 1,2 | Mittelwert Kanal 0 bzw. 1 |
| 3,4 | Standardabweichung Kanal 0 bzw. 1 |
| 5,6,7 | letzter minus erster Wert Kanal 2,3,4 |
| 8,9 | Mittelwert und Standardabweichung Kanal 5 |
| 10,11 | Maximum Kanal 0 bzw. 1 |

Kopf außen nutzt (0,1,3,5,6,10), Kopf Höhe (0,2,4,7,11). Historisch klingende
Bezeichnungen wie `avg_radial_force` bezeichnen hier Kennwerte eines transformierten
Logger-Prozentkanals, keine physikalischen kN. Alle zwölf Werte sind X-only; keine
Walzprotokoll-Endmessung wird als Feature benutzt. Der Algorithmus benötigt die
vollständig beobachtete F2-Sequenz und ist daher eine End-of-F2-Prognose.

## 2. Generativer Prozess und Prior-Konventionen

Für jeden Ring gilt mit global gemeinsamem pi und A:

\[
p(z_r\mid\pi,A)=\pi_{z_{r1}}\prod_{t=2}^{T_r}A_{z_{r,t-1},z_{rt}},\qquad
x_{rt}\mid z_{rt}=k,\mu_k,\Lambda_k\sim N_D(\mu_k,\Lambda_k^{-1}).
\]

\[
\pi\sim Dir(\alpha_0),\quad A_{k,:}\sim Dir(\eta_{0k}),\quad
\Lambda_k\sim W_D(W_{0k},\nu_{0k}),\quad
\mu_k\mid\Lambda_k\sim N_D(m_{0k},(\kappa_{0k}\Lambda_k)^{-1}).
\]

W ist die **Wishart-Skala der Präzision**, nicht die Rate und nicht die Kovarianz
der Beobachtungen. Die Dichte ist

\[
\log W_D(\Lambda\mid W,\nu)=\frac{\nu-D-1}{2}\log|\Lambda|
-\frac12tr(W^{-1}\Lambda)-Z(W,\nu),
\]
\[
Z(W,\nu)=\frac{\nu D}{2}\log2+\frac\nu2\log|W|+\log\Gamma_D(\nu/2).
\]

Es müssen W positiv definit, kappa>0 und nu>D-1 sein. Die Basis verwendet
D=6, m0=0, kappa0=0,2, nu0=D+3=9, W0=I/9, alpha0=1 und
eta0[k,l]=0,5+4*1{k=l}. E[Lambda]=nu W; daraus folgt nicht
E[Lambda^(-1)]=(nu W)^(-1). Volle Kovarianz erlaubt gleichzeitige Kanalkorrelation,
behält aber **zeitlich bedingte Unabhängigkeit** der Emissionen bei.

## 3. M2-Zustandsstatistik und priorerhaltende Reduktion

Für T>=4 definiert die historische `Spec` n_early=floor(T/2) und
n_late=T-n_early; bei ungeradem T liegt der zusätzliche Zeitpunkt im späten Teil.
Früh ist der Indexbereich 0,...,n_early-1, spät n_early,...,T-1.

\[
h_k=T^{-1}\sum_t1\{z_t=k\},\qquad
\Delta_k=n_{late}^{-1}\sum_{t\in late}1\{z_t=k\}
-n_{early}^{-1}\sum_{t\in early}1\{z_t=k\}.
\]

Für jeden Pfad gilt sum(h)=1 und sum(Delta)=0. Die erste Einschränkung ist affin:
ohne separaten Achsenabschnitt dürfen nicht einfach K-1 Aufenthalte verwendet
werden. Die Basis behält alle h und entfernt ausschließlich Delta_K.
Der reduzierte latente Vektor ist g=(h_1,...,h_K,Delta_1,...,Delta_(K-1)).
Die Deskriptoren s_j werden angehängt, u_j=(g,s_j).

Mit f_j=(h,Delta,s_j)=A_lift u_j gilt für den vollen historischen Prior:

\[
\beta_{red}=A_{lift}^T\beta_{full},\quad
\mu_{0,red}=A_{lift}^T\mu_{0,full},\quad
V_{0,red}=A_{lift}^TV_{0,full}A_{lift}.
\]

Der Delta-Block der Liftmatrix ist `[I_(K-1); -1^T]`; h und s werden identisch
gehoben. Im vollen Prior sind die h-Varianzen 2, Delta-Varianzen 0,15 und
Deskriptor-Varianzen v_phys=0,38526108747593474. Daraus folgt im reduzierten
Delta-Block **0,15*(I+11^T)**, insbesondere 0,30 für K=2 und Kovarianzen 0,15
zwischen unterschiedlichen wirksamen Delta-Koeffizienten bei K>2.
Die IG-Parameter bleiben gleich. Kopf-Dimensionen sind 2K-1+6 bzw. 2K-1+5,
also bei K2 9 und 8 statt voller 10 und 9.

Für Priors und konjugierte Posteriorupdates werden dieselben effektiven
Regressionsfunktionen, prädiktiven Momente und relevanten KL-Werte erhalten.
Es wird keine KL-Identität für beliebige frei gewählte alte q(beta_full) behauptet:
die redundante bedingte Koeffizientenverteilung muss ihre Priorbedingte behalten.

## 4. Die zwei NIG-Qualitätsköpfe

\[
y_{rj}\mid z_r,X_r,\beta_j,v_j\sim N(\beta_j^Tu_{rj},v_j),\quad
v_j\sim IG(a_{0j},b_{0j}),\quad
\beta_j\mid v_j\sim N(\mu_{0j},v_jV_{0j}).
\]

Die hier verwendete IG-Shape-Scale-Dichte ist
`p(v)=b^a/Gamma(a)*v^(-a-1)*exp(-b/v)`, v>0.
Äquivalent gilt 1/v~Gamma(shape=a, **rate=b**). a0=2,5, b0=1,5.
Beide Köpfe sind bei gegebenem Pfad und Parametern bedingt unabhängig. Es gibt
kein zusätzliches korreliertes zweidimensionales Rauschen.

Deterministische s(X) bilden keinen weiteren Likelihoodfaktor. Die gemeinsame
Verteilung lautet

\[
p(\Theta)\prod_{r\in L\cup U}p(z_r\mid\pi,A)p(X_r\mid z_r,\mu,\Lambda)
\prod_{r\in L}\prod_{j=1}^{2}p(y_{rj}\mid z_r,X_r,\beta_j,v_j).
\]

Für U wird die normierte Qualitätsdichte integriert und ergibt 1. U besitzt
keinen Qualitätsverlust und keine Pseudolabels.

## 5. Variationsfamilie und exakte lokale Momente

\[
q=q(\pi)\prod_kq(A_k)\prod_kq(\mu_k,\Lambda_k)
\prod_jq(\beta_j,v_j)\prod_rq_r(z_r),\qquad
q_r(z)=\rho_r(z_1)\prod_{t=1}^{T-1}Q_{rt}(z_t,z_{t+1}).
\]

Q ist ringspezifisch und frei zeitabhängig, im Unterschied zur global homogenen
A. Die Marginalen erfüllen gamma_1=rho, xi_t(i,k)=gamma_t(i)Q_t(i,k) und
gamma_(t+1)(k)=sum_i xi_t(i,k). Die Entropie lautet

\[
H(q_r)=-\sum_k\rho_k\log\rho_k
-\sum_{t,i,k}\xi_t(i,k)\log Q_t(i,k),
\]

mit 0 log0=0. Jede Wahrscheinlichkeitszeile wird mit K-1 Log-Odds
`r_j=log(p_j/p_K)` und letztem Logit null dargestellt. Für strikt positive Zeilen
ist dies eine äquivalente Koordinatenwahl, keine Modellbeschränkung.
Der Pullback lautet

\[
\partial F/\partial r_j=p_j\{\partial F/\partial p_j
-\sum_l p_l\partial F/\partial p_l\},\quad j<K.
\]

Der numerische Übergang eines FB-Faktors in endliche Optimiererkoordinaten verwendet
den dokumentierten historischen Floor 1e-12; die logstabile FB-Inferenz selbst ist
ungefloort. Der Konvertierungsfehler ist numerisch zu prüfen. Algebraische Gauge-
Identität und identische L-BFGS-B-Trajektorien sind unterschiedliche Aussagen.

Die additive M2-Belohnung a_t(k) enthält 1/T für h_k sowie -1/n_early oder
+1/n_late für die jeweilige Delta-Komponente. Für g=sum_t a_t(z_t) definiere

\[
P_t(k)=q(z_t=k),\quad M_t(k)=E[g_{1:t}1\{z_t=k\}],\quad
S_t(k)=E[g_{1:t}g_{1:t}^T1\{z_t=k\}].
\]

Initialisierung: P_1(k)=rho_k, M_1(k)=rho_k a_1(k),
S_1(k)=rho_k a_1(k)a_1(k)^T. Danach:

\[
P_t(k)=\sum_iQ_{t-1}(i,k)P_{t-1}(i),
\]
\[
M_t(k)=\sum_iQ_{t-1}(i,k)[M_{t-1}(i)+P_{t-1}(i)a_t(k)],
\]
\[
S_t(k)=\sum_iQ_{t-1}(i,k)[S_{t-1}(i)+M_{t-1}(i)a_t(k)^T
+a_t(k)M_{t-1}(i)^T+P_{t-1}(i)a_t(k)a_t(k)^T].
\]

Am Ende m_g=sum_k M_T(k), H_g=sum_k S_T(k). Alle zeitlichen Kreuzkorrelationen
sind enthalten. Die Rechnung ist exakt **unter q_r**, nicht unter dem unbekannten
wahren gemeinsamen Posterior. Zweite Momente sind nicht m_g m_g^T.
Die neue Gradientenimplementierung verwendet eigenständige bedingte Präfix-/Suffix-
Rewardmomente. Für den pfadweisen Ausdruck
`F(z)=log p(X,z)-log q(z)+b^Tg+g^TGg+c` gilt wegen E[d log q]=0:

\[
\frac{\partial E_qF}{\partial\ell_{tij}}
=\gamma_t(i)Q_t(i,j)\{E[F\mid z_t=i,z_{t+1}=j]-E[F\mid z_t=i]\},
\]
\[
\frac{\partial E_qF}{\partial\ell_{1j}}
=\rho_j\{E[F\mid z_1=j]-E_qF\}.
\]

Hier sind ell volle Softmax-Logits; die reduzierten Ableitungen sind die ersten
K-1 Komponenten jeder Zeile. Präfix und Suffix sind bei gegebener Zustandskante
bedingt unabhängig, sodass erste und zweite Rewardmomente die quadratische
Qualitätserwartung exakt liefern. Dies ist ein anderer Rechenweg als das historische
Rückwärts-Differenzieren durch die unnormierte Momentenrekursion, mit derselben
Zielfunktion. Zentrale Differenzen, kurze vollständige K^T-Pfadenumeration und
das separate V6.0-Oracle prüfen diese Identität. Harte argmax-Pfade und Monte-Carlo
ersetzen diese Rekursion nicht.

Für deterministische s sind

\[
m_j=\begin{bmatrix}m_g\\s_j\end{bmatrix},\quad
H_j=\begin{bmatrix}H_g&m_gs_j^T\\s_jm_g^T&s_js_j^T\end{bmatrix}.
\]

## 6. Auswertbare Erwartungs- und KL-Terme

Für einen NW-Faktor gelten

\[
E\log|\Lambda|=\sum_{d=1}^{D}\psi((\nu+1-d)/2)+D\log2+\log|W|,
\]
\[
E\log N(x;\mu,\Lambda^{-1})=\tfrac12[E\log|\Lambda|-D\log(2\pi)
-D/\kappa-\nu(x-m)^TW(x-m)].
\]

Für Dir(alpha): Elog p_k=psi(alpha_k)-psi(sum alpha). **exp(Elog A)** ist im
Allgemeinen subnormiert und darf vor FB nicht zeilenweise als Übergangsmatrix
renormiert werden; die numerisch normalisierten lokalen Q werden erst durch die
Ketteninferenz gewonnen (Beal, PDF14, gedruckt95).

Für NIG q=(mu,V,a,b) und m_j,H_j gilt mit tau=a/b:

\[
Q_{rj}=-\tfrac12\log(2\pi)+\tfrac12[\psi(a)-\log b]
-\tfrac12\{\tau[y^2-2y\mu^Tm_j+\mu^TH_j\mu]+tr(VH_j)\}.
\]

E[beta beta^T/v]=V+tau mu mu^T. Die präzisionsgewichtete Matrix ist hier
nicht gleich E[beta beta^T]. Für die lokale Optimierung werden die zwei Köpfe
zu `c+b_g^T m_g+tr(G_g H_g)` komprimiert. Mit
G=-.5*(tau mu mu^T+V), b=tau*y*mu und Partition (g,s) sind
`b_g_eff=b_g+2G_gs s`, `G_eff=G_gg` und
`c_eff=c+b_s^T s+s^T G_ss s`. Beide Köpfe werden addiert.

Zur unabhängigen KL-Prüfung, mit S=sum alpha und S0=sum alpha0:

\[
KL_{Dir}=\log\Gamma(S)-\sum\log\Gamma(\alpha_k)-\log\Gamma(S_0)
+\sum\log\Gamma(\alpha_{0k})
+\sum(\alpha_k-\alpha_{0k})[\psi(\alpha_k)-\psi(S)].
\]

Für NW q und Prior (m0,kappa0,W0,nu0):

\[
KL_W=\tfrac12(\nu-\nu_0)E\log|\Lambda|
+\tfrac\nu2[tr(W_0^{-1}W)-D]+Z(W_0,\nu_0)-Z(W,\nu),
\]
\[
KL_{N\mid W}=\tfrac12\{D[\kappa_0/\kappa-1+\log(\kappa/\kappa_0)]
+\kappa_0\nu(m-m_0)^TW(m-m_0)\}.
\]

Für NIG (Dimension p) gilt

\[
KL_{IG}=a_0\log(b/b_0)+\log\Gamma(a_0)-\log\Gamma(a)
+(a-a_0)\psi(a)+a(b_0/b-1),
\]
\[
KL_{N\mid IG}=\tfrac12[tr(V_0^{-1}V)-p+\log|V_0|-\log|V|
+(a/b)(\mu-\mu_0)^TV_0^{-1}(\mu-\mu_0)].
\]

Jede Familie muss gegen unabhängige Dichte-/Integrations- oder handrechenbare
Kontrollen geprüft werden; das Ausbleiben einer Ausnahme genügt nicht.

## 7. Gewichtetes gemeinsames Trainingsziel

\[
P_r=\gamma_{r1}^TE\log\pi+\sum_{t,i,k}\xi_{rt}(i,k)E\log A_{ik}
+\sum_{t,k}\gamma_{rtk}\ell_{rtk},
\]
\[
\mathcal J_C=\sum_{r\in L}[P_r+H(q_r)+Q_{r1}+Q_{r2}]
+C\sum_{r\in U}[P_r+H(q_r)]-KL(q_\Theta\Vert p_\Theta).
\]

C multipliziert den **gesamten** lokalen U-Prozessbeitrag einschließlich Anfang,
Übergang und Entropie. Für C=1 ist dies eine gewöhnliche ELBO der angegebenen
Missing-Label-Konstruktion. Für C=0,2 heißt das Ziel gewichtete/generaliserte
Variationszielfunktion. Eine unveränderte Evidenzschranke des ursprünglichen
ungewichteten Modells wird nicht behauptet. Shao/Ge/Song (2021) liefert den
Gewichtungskontext, aber seine Gleichungen (24)–(30) sind nicht dieses identische
Gewichtungsmodell; dort sind Zustandsprior und Entropie anders behandelt.

Bei C>0 ist der optimale U-Faktor bei festem q_Theta die Prozess-FB-Verteilung.
Bei C=0 wird kein U-Prozessupdate ausgeführt. Ein optionaler U-Effekt im X-only-
Preprocessing bleibt ein explizit anderer Untersuchungsarm (`shared-C0`).

## 8. Konjugierte globale Updates

Mit w_r=1 für L und C für U:

\[
\alpha_k=\alpha_{0k}+\sum_rw_r\gamma_{r1k},\quad
\eta_{ik}=\eta_{0ik}+\sum_rw_r\sum_t\xi_{rt}(i,k),
\]
\[
N_k=\sum_rw_r\sum_t\gamma_{rtk},\quad
S_k=\sum_rw_r\sum_t\gamma_{rtk}x_{rt},\quad
R_k=\sum_rw_r\sum_t\gamma_{rtk}x_{rt}x_{rt}^T.
\]
\[
\kappa_k=\kappa_{0k}+N_k,\quad\nu_k=\nu_{0k}+N_k,\quad
m_k=(\kappa_{0k}m_{0k}+S_k)/\kappa_k,
\]
\[
W_k^{-1}=W_{0k}^{-1}+R_k+\kappa_{0k}m_{0k}m_{0k}^T-\kappa_km_km_k^T.
\]

Die Updates gelten auch für nichtganzzahlige gewichtete Statistiken. Posterioren
werden stets aus Prior und allen aktuellen Statistiken gebildet, nicht epochenweise
als neue unabhängige Daten aufaddiert. Cholesky prüft positive Definitheit;
Symmetrisieren behebt Rundungsasymmetrie, keine echte negative Eigenrichtung.

Für jeden Kopf mit P0=V0^(-1), ausschließlich über L-TRAIN:

\[
P=P_0+\sum_rH_{rj},\quad V=P^{-1},\quad
\mu=V(P_0\mu_0+\sum_ry_{rj}m_{rj}),\quad a=a_0+|L|/2,
\]
\[
b=b_0+\tfrac12[\sum_ry_{rj}^2+\mu_0^TP_0\mu_0-\mu^TP\mu].
\]

## 9. Lokale Annahme, Initialisierung und Diagnose

Training initialisiert die globalen Faktoren frisch. K-Means-Zentren stammen nur
aus den laut Arm zulässigen TRAIN-Prozessen; sie sind Initialwerte der q-Mittel,
keine geladenen Gewichte. Prior und initialer Variationsfaktor sind verschieden.
Ein vorheriges lokales q aus derselben frischen Trainingsrechnung ist als Warmstart
zulässig; ein historischer Modellzustand ist es nicht.

Für jeden L-Ring werden bisheriges q und aktuelles Prozess-FB unter **denselben**
globalen Faktoren und derselben vollständigen lokalen Zielfunktion bewertet.
Der bessere endliche Kandidat startet L-BFGS-B in reduzierten Log-Odds.
Ein verschlechternder Optimiererkandidat wird verworfen. Acht lokale Iterationen
sind ein Kostenlimit und kein Stationaritätsnachweis. Optimiererstatus, Iterationen,
Funktionsaufrufe, Gradientennormen und Annahmeentscheidung müssen gespeichert werden.

Nach lokalem Update, globalem Prozessupdate und jedem Kopfupdate wird dieselbe
J_C neu berechnet. Eine echte Abnahme unterhalb der **vorab** festgelegten absoluten
Toleranz 1e-6 löst einen Fehler aus. Dies erlaubt Rundungsfehler und ist keine
erlaubte systematische Zielverschlechterung. Die Diagnose speichert neben den
Komponenten `emission-KL_NW`, `transition-KL_A`, `initial-KL_pi` auch Entropie,
Qualität, gewichtete Zustandszahlen und Parameteränderungen. Ein Anstieg dieser
Trainingszielfunktion beweist weder gute Kalibrierung noch bessere Generalisierung.

## 10. Prozess-only-Prediction und mm-Rücktransformation

Für einen neuen Ring werden ausschließlich X und die FITTED TRAIN-Transformationen
verwendet. Erwartete HMM-Logpotenziale erzeugen q_*(z|X) per FB. Qualitätslabels
werden weder übergeben noch zur lokalen Rückkopplung gelesen.

Für ein festes u und NIG q ist die bedingte Qualitätsprädiktive Student-t mit
2a Freiheitsgraden, Lage mu^T u und **Skala zum Quadrat**
`b/a*(1+u^T V u)`. Ihre Varianz ist b/(a-1)*(1+u^T V u), a>1.
Nach Mischung über den Markov-Pfad entsteht im Allgemeinen keine einzelne
Student-t-Verteilung. Die Basis berechnet die exakten Momente der **gewählten
q_*-bedingten Mischung**, nicht die volle wahre Posteriorprädiktive:

\[
\widehat y_j=\mu_j^Tm_j,\qquad
v_j=\frac{b_j}{a_j-1}[1+tr(V_jH_j)]
+\mu_j^T(H_j-m_jm_j^T)\mu_j.
\]

Die zwei Köpfe teilen latente Zustandsunsicherheit. Eine mögliche Kreuzkovarianz
aus dieser gemeinsamen Größe ist mu_g1^T Cov(g) mu_g2; sie bedeutet kein
gemeinsam gelerntes zweidimensionales Qualitätsrauschmodell. Die Basis berichtet
die beiden marginalen Standardabweichungen mit ihrer approximativen Herkunft.
Globale HMM-Parameter werden nicht erneut gezogen und für X_* integriert.

Die Rücktransformation verwendet je Ziel TRAIN-Mittel a_Y und positive Skala s_Y:
`mean_mm=a_Y+s_Y*mean_scaled`, `sd_mm=s_Y*sqrt(v_scaled)`.
Kovarianzen würden beidseitig mit diag(s_Y) transformiert. Es werden keine
kalibrierten 95%-Intervalle und keine exakten Mischungsquantile behauptet.

## 11. Entwicklungsevaluation und Geltungsgrenzen

CV hält vollständige L-Ringe zurück. Alle Fits, Initialisierung und Scaler sind
foldlokal. Innerhalb der zwölf Entwicklungsringe werden dieselben vorab festgelegten
Folds für TRAIN-Mean, physikalischen NIG-/Ridge-Vergleich, shared-C0 und SSL benutzt.
Jeder zurückgehaltene Ring zählt einmal, unabhängig von seiner Sequenzlänge:
`RMSE_j=sqrt(mean((pred_mm_j-y_mm_j)^2))`; MAE analog.
Historische Außenringe DM07/08/11/15 werden weder trainiert noch gescort.
Mehrfach genutzte acht Entwicklungslabels ergeben keine neue unabhängige
Bestätigung, auch wenn alle Foldgrenzen technisch korrekt eingehalten werden.

Serielle Emissionsfehler, kleine Zahl ringweiter Labels, Prior-/Featurewahl,
Initialisierungsrobustheit, Wirkung des U-ELBO und prädiktive Kalibrierung bleiben
eigenständige Forschungsfragen. Ein AR-HMM und unkontrolliertes Hyperparametertuning
gehören nicht zu dieser Basis.
