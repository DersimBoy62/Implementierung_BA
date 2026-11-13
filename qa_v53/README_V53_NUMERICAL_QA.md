# Bayesian Bachelor — V5.3 numerische Vorprüfung (synthetisch)

**Stand:** 12.11.2026  
**Release-Typ:** Ergänzender, nichtinvasiver QA-Prüfzweig für die vorhandene V5.2-Installation. **Keine neue Hauptmodellversion und kein finaler Model-Freeze.**

## Einbau

Das ZIP im Projektordner `Implementierung_BA` entpacken. Es entsteht nur `qa_v53/`. Der bereits vorhandene Ordner `final_candidate_v52/` muss unverändert daneben liegen:

```text
Implementierung_BA/
  .venv/
  final_candidate_v52/         # bisheriger geprüfter Stand, NICHT überschreiben
  qa_v53/
    test_numerical_v53.py
    README_V53_NUMERICAL_QA.md
    V53_SHA256_MANIFEST.json
```

## Ausführen unter Windows/PowerShell

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v52\main_bayesian_ssr_v52.py verify-config
.\.venv\Scripts\python.exe .\qa_v53\test_numerical_v53.py
```

Optional: eine **neue** JSON-Ergebnisdatei im bereits existierenden Verzeichnis `results/` erzeugen (niemals überschreiben):

```powershell
.\.venv\Scripts\python.exe .\qa_v53\test_numerical_v53.py --json-out results\v53_numerik_01.json
```

## Prüfungen

1. Identität der eingefrorenen V5.2-Konfiguration und aller zehn mathematischen SHA256-Dateihashes.
2. M2-Pfadmomente gegen unabhängige vollständige Aufzählung aller 32 K2/T5-Pfade.
3. Lokale Markov-plus-Qualität-Zielfunktion gegen unabhängige Pfadaufzählung (kein Test der vollständigen globalen ELBO).
4. Normierungs-, Symmetrie- und PSD-Eigenschaften von ersten und zweiten M2-Momenten.
5. Analytischer Markov-Gradient gegen zentrierte finite Differenzen.
6. Lange synthetische Markov-Sequenz (2.048 Zeitschritte).
7. Nahezu deterministische Übergänge mit minimalen positiven Wahrscheinlichkeiten 1e-12.
8. KL[q||q]=0 für Normal-Wishart, d=2 und d=6.
9. Starke Konditionierung der Wishart-Skalenmatrix (1e8).
10. Abweisen einer nicht positiv definiten Wishart-Skalenmatrix.
11. Vollkovarianz-Prozessupdate mit hochgradig korrelierten synthetischen Merkmalen.
12. Positivität beider NIG-Qualitätskopf-Predictive-Varianzen.
13. Lokale Qualitätsrückkopplung bei fixierten globalen Parametern: 0 versus 8 lokale VI-Schritte.
14. Umkehrung des künstlichen Qualitätslabels: erwartete Umkehrung des optimierten Zustandsanteils.

## Interpretation des Feedbacktests

In Tests 13/14 werden **absichtlich künstliche Qualitätsköpfe mit starkem Effekt** verwendet; das dient der Funktionsprüfung des lokalen Optimierers und ist **kein auf DM-Ringen trainiertes oder kalibriertes Regressionsmodell**. Eine große Verschiebung der Zustandsbelegung oder ein hoher `local_gain` in diesem Test dürfen **nicht** als empirischer Nutzen von SSL oder als Fortschritt in der realen Qualitätsvorhersage interpretiert werden. Die tatsächliche Wirkung mit den acht gelabelten Entwicklungsringen bleibt für die spätere, gesondert dokumentierte Ablation offen.

## Provenienz und Grenzen

- Verwendet ausschließlich deterministisch generierte synthetische Arrays; kein Laden der DM-Pickles, keine alten Test-Ringe, keine Qualitätsauswertung, kein Training.
- Alle zehn Kernmodule, der V5.1-Eingabeschutz, das V5.2-Hauptskript, seine SHA-Manifestdateien und seine Konfiguration bleiben byte-identisch.
- Der frühere V5.2-Windowslauf hat 19/19 Tests bestanden; dabei wurde der echte Symlink-Test wegen fehlender Berechtigung übersprungen und der Ablehnungspfad simuliert. Diese Ausnahme bleibt separat vermerkt.
- Die Ergebnisse sind keine externe Generalisierung, kein vollständiger Konvergenznachweis und keine Vorhersage über einen neuen Ring.
- Auf dem Entwickler-Linuxstand lief der neue Prüfzweig vollständig durch. Der gesonderte Nutzer-Test mit Python 3.11.9 auf Windows steht aus.

## Weitere Forschungs- und Freigabearbeit

**Erst nach lokaler Prüfung dieses Overlays**: eine sauber getrennte entwicklungsbasierte Ablation mit fixierter Konfiguration und `steps=0` vs. `steps=8`, ohne Zugang zu DM07/DM08/DM11/DM15, mit Speicherung von lokalen Zustaenden, Trajektorien, ELBO und prädiktiver Veränderung. Ergebnisse wegen wiederverwendeter acht Entwicklungslabels ausschließlich explorativ. Danach Model-Freeze und offizieller Lauf nur nach expliziter Entscheidung.
