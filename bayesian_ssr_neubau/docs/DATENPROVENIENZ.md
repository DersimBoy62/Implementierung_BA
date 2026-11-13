# Datenvertrag und tatsächliche Prüfung

Stand: 13.11.2026. Der Neubau verwendet ausschließlich die lokal vom Nutzer
freigegebene Datei `input/DM_16_aktive_Ringe.pkl`. SHA256 vor jedem
Deserialisieren: `bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5`.
Ein abweichender Hash führt vor `pickle.load` zum Abbruch. Fremde Pickles werden
nicht als Vorhersageformat unterstützt.

## Beobachtete Struktur und Ziele

Das Pickle enthält genau DM01, DM02, DM04, DM05, DM06, DM07, DM08, DM09, DM10,
DM11, DM12, DM14, DM15, DM16, DM17 und DM18 mit `train_eligible=True`.
DM03 fehlt; DM13 ist ausgeschlossen. Der tatsächlich vorhandene Zielcontainer
heißt **`zielwerte`**, nicht `all_targets`. Der Import verwendet daraus genau
`ovality.ovality_radial_mm` und `planarity.height_deviation_mm` in dieser
Reihenfolge. Die überprüften Einheiten dieser Ergebnisfelder sind mm.
`ovality.r_outer_mm` und eine Innenunrundheit sind keine Targets.

Die Herkunftsbezeichnung `all_targets` im Auftrag wird damit auf die belegte
Excel-/Pickle-Übernahme abgebildet. Die externen ursprünglichen `targets.pkl`
und der gesamte originale 3D-Auswerteprozess sind laut dem vollständig
gelesenen Datenaudit nicht in diesem Paket enthalten; eine erneute unabhängige
Prüfung grüner Originalzellen beziehungsweise der Metrologie bleibt offen.
Die Bezeichnung „Höhenschlag“ ist eine Projektkonvention für die vorhandene
Planaritätshöhenabweichung; eine metrologische Gleichheit ist nicht bewiesen.

Beim vertrauenswürdigen Sammelpickle materialisiert die Python-Deserialisierung
technisch alle enthaltenen Objekte. Nach dieser unvermeidbaren IO-Grenze liest
`load_development` Zielwerte ausschließlich für die registrierten L-Ringe.
U wird unmittelbar als `UnlabeledRing` mit `slots` ohne `y`, `targets` oder
`__dict__` gebaut. Äußere Prozesse und Labels gehen nicht an die Trainings-API.
Ein Prozess-Audit liest auch für L keinerlei Qualitätswerte.

## Originalheader der sechs Loggerkanäle

Die folgende Semantik wurde direkt aus `logger.signal_units` und
`logger.signal_descriptions` des SHA-verifizierten Pickles gelesen, nicht aus
einer vermuteten physikalischen Bedeutung abgeleitet:

| Reihenfolge | Kanal | Originaleinheit | Originalbeschreibung |
|---|---|---|---|
| 1 | F_rad63 | % | radial force |
| 2 | F_ax64 | % | axial force |
| 3 | Ist_D150 | m | acutal OD correctet |
| 4 | Ist_s151 | m | actual wall corrected |
| 5 | Ist_h152 | m | actual height corrected |
| 6 | RWG55 | m/s | ring growth rate |

Die Schreibweise der Originalbeschreibung bleibt hier erhalten. Prozentwerte
der Kraftkanäle sind keine verifizierten kN. `RWG55` bezeichnet nach diesem
Header eine Ringwachstumsrate, keine Ringhöhe. Das ursprüngliche freigegebene
Maschinenhandbuch wurde nicht nachgewiesen; insbesondere die technischen
Phasenbezeichnungen 1 und 2 werden nicht erfunden.

## F2 und Sicherheitsgrenzen

Die Auswahl rekonstruiert exakt `final_candidate_v52/final_core/dm_data.py`:
`phase804` in {1,2} und `LaserOK == 1` und `MessEin == 1`. Keine Phase-6-Zeile
wird in das Modell übernommen. Zeilen bleiben in Originalreihenfolge, ohne
Interpolation, Imputation, Clipping oder Aneinanderhängen getrennter Inseln.
Zusätzlich werden das 0,1-s-Raster mit vorab festgelegten ±0,002 s aus der
V5.1-Eingangssicherheit und die bei allen Entwicklungsringen vorhandene
einmalige 2→1-Grenze geprüft. Diese Prüfungen ändern keine ausgewählte Zeile.
Schema und Regression benötigen mindestens vier beziehungsweise sechs Zeilen;
die tatsächlichen Entwicklungssequenzen sind deutlich länger.

| Ring | F2-Zeilen | Gruppe |
|---|---:|---|
| DM01 | 148 | U |
| DM02 | 419 | U |
| DM04 | 829 | L |
| DM05 | 694 | L |
| DM06 | 1085 | L |
| DM09 | 155 | L |
| DM10 | 381 | U |
| DM12 | 1293 | L |
| DM14 | 424 | U |
| DM16 | 1051 | L |
| DM17 | 663 | L |
| DM18 | 298 | L |

Gesamt: 7440 Zeilen; 6068 L und 1372 U. Bei C=0,2 entspricht dies einer
gewichteten Summe von 6342,4 Prozessbeobachtungen, nicht unabhängigen
Qualitätslabels. Das neue `docs/DATEN_QA.json` enthält Originalindizes,
Zeitgrenzen, Phasengrenze, fehlende Werte und Kanalbandbreiten pro Dev-Ring.

Für vollständige Rohlogger-CSV verlangt `read_process_csv` exakt
`time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin`.
Geprüft werden Phase-6-Vorlauf und ein zeitnaher bleibender inaktiver
Phase-6-Nachlauf entsprechend der strukturellen V5.1-Gates. Der Loggerabschluss
allein belegt keinen erfolgreich gefertigten Ring. Ein bereits F2-validiertes
NPZ dient dem kontrollierten Test-/Datenvertrag; es beweist allein keinen
vollständigen ursprünglichen Rohlogger oder Maschinenabschluss.

## Train-only Skalierung und Deskriptoren

Die acht L-Ringe sind DM04/05/06/09/12/16/17/18; die vier U-Ringe
DM01/02/10/14. DM07/08/11/15 bleiben gesperrt. Vier deterministische
Entwicklungsfolds halten jeweils zwei aufeinanderfolgende sortierte L-IDs aus.
Keine zufällige Aufteilung einzelner Zeitpunkte findet statt.

P0 verwendet einen gepoolten Zeitpunkt-Scaler über erlaubte TRAIN-Prozesse.
Lange Ringe haben dadurch größeren Einfluss auf X-Scaler und Prozessmodell.
Der zweidimensionale Y-Scaler verwendet ausschließlich L-TRAIN-Ziele. Die
zwölf Deskriptoren werden aus dem bereits X-skalierten F2 berechnet und
ringgleichgewichtet auf TRAIN angepasst: Log-Dauer, mittlere Radial-/Axialkraft,
deren Standardabweichungen, End-minus-Start von Außendurchmesser/Wand/Höhe,
Mittelwert/Standardabweichung der Ringwachstumsrate, maximale Radial-/Axialkraft.
Alle Standardabweichungen verwenden `ddof=0`. Die Kopfspalten entsprechen
Außen `(0,1,3,5,6,10)` und Höhe `(0,2,4,7,11)`.

Bei `include_unlabeled=False` werden U weder für X- noch Deskriptor-Scalerfit
verwendet. Für einen expliziten shared-C0-Arm kann X-only-U-Preprocessing
erlaubt werden; dieser Arm wird separat von strict supervised bezeichnet.
Jeder Scaler speichert `mean`, `scale`, `fit_ids`; jede Prediction transformiert
mit gespeicherten Werten und führt keinen Fit aus.

## Ausgeführte Prüfungen

PowerShell mit `PYTHONPATH=bayesian_ssr_neubau/src`:
`.\.venv\Scripts\python.exe -m pytest bayesian_ssr_neubau\tests\test_data_and_leakage.py -q`
ergab **18 passed in 0.50 s**, Exitcode 0. Die Tests prüfen alle zwölf
Entwicklungsringe gegen die separat gelesene historische `dm_data.view`-Funktion:
Zeilenindizes, Zeiten, Kanalwerte und Reihenfolge sind bitgleich. Auch die
zwölf Deskriptoren stimmen pro Ring bitgleich mit der separat gelesenen
historischen Funktion überein. Ein anfänglicher Vergleich zeigte Rundungs-
unterschiede durch `mean/std(axis=0)` statt kanalweiser Summation; die neue
Implementierung erhält jetzt die ursprüngliche kanalweise Summationsfolge.
Weitere Tests prüfen train-only Fits, C0-Ausschluss, fehlende Labels,
NaN/Inf, Kanalreihenfolge, Zeitraster, Phasengrenze, isolierte F2-Inseln,
SHA-Abbruch und write-once Prozess-only-Exporte. Kein äußerer Ring wurde gescort.

Der vollständig gelesene historische Bericht
`Bayesian_Bachelor_DM_Pickle_Datenaudit_2026-11-10.zip/Datenpruefung_Pickles_ausfuehrlich.md`
bleibt historische Evidenz. Seine Rohbestandszahlen und metrologischen Warnungen
werden von neuen Tests getrennt. Alte Skalierer, Zentren, Posterioren oder
Vorhersagen wurden nicht in diese Datenpipeline übernommen.
