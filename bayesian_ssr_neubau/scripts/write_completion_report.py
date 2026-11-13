"""Build the German completion report from completed, immutable run artifacts."""
from pathlib import Path
import json

project = Path(__file__).resolve().parents[1]
def read(relative):
    return json.loads((project/relative).read_text(encoding='utf-8'))

training = read('docs/TRAINING_30E_AUDIT.json')
tests = read('docs/verification_001.json')
cv = read('outputs/development_cv_001/development_cv.json')
assert tests['exitcode']==0 and len(cv['folds'])==20
labels = {'train_mean':'TRAIN-Mittelwert','physical_nig':'Physikalische NIG-Regression',
          'strict_c0':'HMM strict-C0','shared_c0':'HMM shared-C0','ssl':'HMM SSL C=0,2'}
table = '\n'.join('| '+labels[arm]+' | '+' | '.join(f'{value:.6f}' for metric in ('rmse_mm','mae_mm') for value in cv['metrics'][arm][metric])+' |' for arm in cv['arms'])
ss,mean,phys,shared = [cv['metrics'][name]['rmse_mm'] for name in ('ssl','train_mean','physical_nig','shared_c0')]
relative = [(ss[j]/shared[j]-1)*100 for j in range(2)]
text = f'''# Abschlussbericht – Basismodell-Neubau vom 13.11.2026

Der eigenständige Neubau ist implementiert, mathematisch geprüft, frisch über
30 Epochen trainiert und in einer vollständigen ringgetrennten Entwicklungs-CV
ausgewertet. Dies ist der beauftragte Basismodellstand, kein endgültiger
Freigabe-/Freeze-Kandidat. Der separate Word-Optimierungsfahrplan bleibt ein
Folgeauftrag. Es wurden keine historischen Außenringe neu gescort.

## 1. Konkrete Liefergegenstände und Stufenstatus

| Stufe | Wirklich vorhandener Nachweis | Status |
|---|---|---|
| 1 Inventur/Abgleich | Inventur.csv:1071 Einträge mit SHA;70 konkrete Lesevermerke;195 Archivdateien bytegleich; mathematische Spezifikation und A–D-Abgleich | abgeschlossen |
| 2 Datenvertrag | Eigenständige schema/data_io/split/preprocessing/descriptors;12/12 F2- und Deskriptorparität;Daten-QA und Metrologiehinweise | abgeschlossen |
| 3 mathematische Faktoren | Eigene Dirichlet/NW/NIG/FB/Markov-Momente/Logits; Enumeration, Dichteintegrale und zentrale Differenzen | abgeschlossen |
| 4 gemeinsame Kopplung | Zwei NIG-Köpfe mit induzierten M2-Priors; ganze U-ELBO einschließlich Entropie gewichtet; Warmstart/Annahme/Blockbilanz | abgeschlossen |
| 5 frisches Training | outputs/baseline_8L4U_001:30E,8L+4U,450 kontrollierte Blöcke,30 Checkpoints,JSONL,Diagnostik | abgeschlossen |
| 6 Prediction/Evaluation | Prozess-only CSV/NPZ/CLI,load==memory,4Folds×5Arme,je8 VAL-Vorhersagen | abgeschlossen |
| 7 Review/Integrität | 105 ausgeführte Tests;23 Produktivmodule ohne Altimporte;Versionslock;Release-Validierung und SHA-Liste | abgeschlossen nach finalem Integritätsskript |

## 2. Mathematische Identität mit V6.0 und eigener Code

Gleich bleiben das K2/M2-Modell, beide Kopfgruppen, die gewichtete J_C,
die frei zeitabhängige Markovfamilie und die effektiven NIG-Priors.
Alle K Aufenthaltsmerkmale bleiben erhalten; nur Delta_K wird entfernt.
Der wirksame K2-Deltakoeffizient hat Priorvarianz0,30. Bei K>2 werden die
induzierten Offdiagonalen0,15 erhalten. Eine willkürlich frei gesetzte volle
q(beta)-Familie muss deshalb nicht dieselbe KL haben.

Neu geschrieben wurden Datenverträge, numerische Faktoren, exakte M2-Rekursion,
bedingter Reward-/Scoregradient, lokaler Optimierungsvertrag, Training,
Vorhersage, Persistenz, Diagnostik, Evaluation, CLI und Tests.
Der Produktivimportgraph umfasst23 Module und keine alten Modellkerne.
Das echte historische V6.0 ist ausschließlich im separaten Referenztest erlaubt.

Zwölf echte Oracle-Paritätstests decken K2/3/4 ab, jeweils Momente/Gradient,
NW,NIG sowie drei frische Trainingsepochen. Unabhängige Enumeration,
Finite-Differenzen, Dichteintegrale und handgerechnete Posterioren verhindern,
dass lediglich derselbe alte Code zweimal geprüft wird.

## 3. Tatsächlich ausgeführtes Training und numerische Grenze

Erste Epochen-J_C:{training['first_epoch_J']:.12f}; nach30 Epochen:
**{training['last_epoch_J']:.12f}**. Kleinste Blockdifferenz:
{training['min_block_delta']:.4g}; keine Verletzung der vorab festgelegten
absoluten Toleranz1e-6 in450 Blöcken.

Nachträglich gelesene historische V6.0-Trainingsreferenz: maximale
Kurvenabweichung{training['post_training_reference']['max_abs_curve_difference']:.4g},
Enddifferenz{training['post_training_reference']['final_difference']:.4g}.
Diese Datei wurde erst nach dem neuen Fit zum Kurvenvergleich gelesen; keine
alten Posterioren,Zentren,Scaler oder Gewichte dienten dem Training.

Der Sprung E10→E11 beträgt{training['epoch11_total_increase']:.6f}; davon
globales Prozessupdate{training['epoch11_process_increase']:.6f}, aufgeteilt in
Emission/NW{training['epoch11_process_components']['emission_NW']:.6f} und
Transition/Dirichlet{training['epoch11_process_components']['transition_Dirichlet']:.6f}.
Dies ist ein Prozess-Zielfunktionsbefund, kein Beweis einer Qualitätsverbesserung.

240 lokale Optimiereraufrufe,191 akzeptierte zusätzliche Optimiererschritte und
6 ohne Erfolgsstatus. Die maximale abschließende Gradientennorm-infinity
beträgt{training['optimizer_max_final_grad_inf']:.4g}; acht Iterationen sind daher
kein Nachweis gtol-genauer Stationarität. Alle Status-/Gradientenwerte und
Annahmeentscheidungen sind protokolliert. Unterschiedliche Annahmezahlen bei
nahezu gleicher J_C sind bei endlichen numerischen Optimiererwegen möglich.

## 4. Vollständige Entwicklungsevaluation in mm

Vier feste Folds innerhalb der acht historischen L-Entwicklungsringe; pro Fold
sechs L-TRAIN,vier U und zwei vollständig zurückgehaltene VAL-Ringe.
Jeder L-Ring wird genau einmal vorhergesagt. Alle Normalisierer und alle
K-Means-Initialisierungen werden innerhalb ihres erlaubten TRAIN-Arms frisch
angepasst. Insgesamt12 HMM-CV-Fits mit je30 Epochen,zusätzlich vier
physikalische NIG-Baselines und vier TRAIN-Mittelwerte. Keine Hyperparametersuche.

| Arm | RMSE Außen mm | RMSE Höhe mm | MAE Außen mm | MAE Höhe mm |
|---|---:|---:|---:|---:|
{table}

SSL gegenüber shared-C0 verändert den gepoolten Entwicklungs-RMSE um
{relative[0]:+.2f}% außen und{relative[1]:+.2f}% Höhe; dieser Vergleich isoliert
den zusätzlichen U-Prozesszielbeitrag bei gemeinsamer X-only-Vorverarbeitung
stärker als der Vergleich mit strict-C0. Einzelringfehler und Foldabhängigkeit
stehen in der vollständigen CV-JSON. Diese Prozentangaben sind deskriptiv.

Die Ergebnisse zeigen die Vorhersagefehler auf **acht mehrfach historisch
verwendeten Entwicklungsringen**. Sie liefern keine unabhängige Bestätigung und
keine statistisch belastbare Überlegenheitsbehauptung. Die vier historischen
Außenringe wurden schon früher gesehen und bleiben in diesem Neubau gesperrt.
Die vollständigen Vorhersagen,TRAIN-Scaler,Konfigurationen und Epochenverläufe
stehen unter `outputs/development_cv_001`.

## 5. Tests, Prediction und reproduzierbare Artefakte

Abschließender echter Testbefehl und Exitcode stehen in `verification_001.json`:
**{tests['tests']} Tests bestanden**,0failures,0errors,0skipped,Exit0.
Pytest-Laufzeit18,25s; gesamte subprocess-Dauer{tests['elapsed_seconds']:.2f}s.
T1–T10 sind in `TESTPROTOKOLL.md` mit Testdateien und Toleranzen zugeordnet.
Die ausführbaren PowerShell-Befehle stehen im README.

Das frische Hauptmodell wurde als prüfbare JSON-Faktoren gespeichert. Laden
verifiziert Hashes,K/Head-/Feature-/Gate-Dimensionen,Split und Scalerherkunft
einschließlich Sidefile-Verträgen. Selbst intern neu versiegelte inkonsistente
Modelle werden von den semantischen Negativtests abgelehnt. Neue Ausgaben sind
write-once; alte results/ und Originalmodelldateien wurden nicht überschrieben.

Die tatsächliche CSV-Prediction-CLI erhält ausschließlich Loggerprozesse und
gibt beide Mittelwerte/SD in mm zurück. DM04 ist dabei ein IO-Beispiel aus TRAIN,
kein neuer Validierungspunkt. Echte Train/Pred-Trennung liefern synthetische
Holdouts und die acht VAL-Ringe der CV. Load gegenüber memory ist exakt gleich.

Der unveränderliche Hauptmodell-Export trägt noch `math_test_status=NOT_RECORDED`,
weil der Fit keine Testausführung selbst attestiert. Die nachträgliche
`RELEASE_VALIDATION.json` bindet den tatsächlich ausgeführten105-Test-Nachweis
an Modell- und Quellcode-SHA. Damit wird kein altes Modellartefakt überschrieben.

## 6. Quellen, Datenentscheidungen und offene Forschung

Die fünf Nutzerarchive sind bytegleich geprüft.19 konkrete wissenschaftliche
Originalauszüge wurden gelesen; QUELLENSTATUS trennt geprüfte Stellen,
vorhandene ungeprüfte PDFs und rein bibliografische Angaben.
Beal ist nur ein Kapitel-Auszug; eine der82 PDF-Dateien ist ein Laborprotokoll.
Quellenverfügbarkeit wird nicht als vollständige Inhaltsprüfung ausgegeben.

Rabiner/Beal stützen HMM und Elog-FB,Bishop die NW-Konvention;Shao2022 liefert
industriellen SsBHMM-Kontext. Shao2021 gewichtet anders als unser J_C.
Die ringweite Zweikopfkopplung,exakten M2-Momente und Priorpushforward sind die
eigene mathematisch/numerisch geprüfte Konstruktion. Keine Quelle wird als
Beweis genau dieser Gesamtkombination bezeichnet.

Das reale Targetwörterbuch heißt `zielwerte`; beide bestätigten Blattfelder
werden nur für L-TRAIN ausgelesen. U erhält eine Klasse ohne Qualitätsattribute.
Das vertrauenswürdige SHA-geprüfte Sammelpickle materialisiert beim Import
physisch auch andere gespeicherte Werte; sie gelangen nicht in die Trainings-
oder Prediction-Schnittstelle. Geometrien werden nicht als Endqualitätsersatz
benutzt. Kraftkanäle sind laut Header Prozentwerte,RWG55 Ringwachstumsrate m/s.
Die normative Gleichsetzung der Planaritätshöhenabweichung mit Höhenschlag
bleibt ohne Messvorschrift offen.

Wesentliche offene Forschungsfragen: serielle Emissionsfehler und AR-HMM-
Alternativen;echte verschiedene Initialisierungen;Prior-/Feature-Sensitivität;
HMM versus HMM-freie Modelle;Bayes versus MAP/Ridge;Kalibrierung und
Ringgeneralisierung. Die Grundemission bleibt zeitlich bedingt unabhängig.
Prädiktive Momente sind unter der gewählten Markov-/NIG-VI exakt,aber globale
HMM-Parameter werden am neuen X nicht vollständig neu integriert. Keine
kalibrierten95%-Mischungsquantile werden behauptet.

Keine zwingende Implementierungsfrage ist ungelöst. Metrologische,
kalibrierungsbezogene und generalisierende Geltungsgrenzen bleiben ausdrücklich
offen und sind keine stillschweigend gelösten Eigenschaften des Basismodells.
'''
with (project/'docs/ABSCHLUSSBERICHT.md').open('x',encoding='utf-8') as handle:
    handle.write(text)
print(table)
