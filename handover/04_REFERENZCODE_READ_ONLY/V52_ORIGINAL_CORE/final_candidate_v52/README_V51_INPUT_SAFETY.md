# Bayesian Bachelor V5.1 – Eingangssicherheit / Vor-Freigabe

**Status 12.11.2026:** geprüfter **Entwicklungs-Kandidat**, noch KEIN endgültiger Model-Freeze, keine externe Qualitätsvalidierung. Die Originalfassung `final_candidate_v5` unverändert aufbewahren. Alle zehn mathematischen Module in `final_core/` sind SHA256-identisch zur V5-Fassung.

## 1. Was ist neu?

1. **Historische Ring-IDs**: Groß-/Kleinschreibung, führende Nullen, Trenner und typische Suffixe normalisieren; bestehende DM01–DM18 dürfen nicht als „neu“ vorhergesagt werden. Schutz gegen versehentliches Replaying, kein kryptographischer Identitätsnachweis.
2. **Vollständiger Rohlogger erforderlich**: Spalten GENAU in dieser Reihenfolge: `time,F_rad63,F_ax64,Ist_D150,Ist_s151,Ist_h152,RWG55,phase804,LaserOK,MessEin`. Keine Qualitäts-/Metrologiespalten. Die CSV muss die **Phase-6-Vorphase, ein einziges zusammenhängendes F2-Fenster und den inaktiven Phase-6-Nachlauf** enthalten. Ein ausgeschnittenes F2-CSV aus V5 wird absichtlich ABGELEHNT.
3. **Zeitprüfung**: vollständiger Rohlogger mit `dt=0,1 s` ± `0,002 s`, keine Lücken oder stillen Interpolationen.
4. **Mindestabdeckung**: mindestens 111 F2-Messpunkte bzw. 11,025 s. Diese vorläufige Schwelle stammt aus 75 % der kürzesten zulässigen Entwicklungs-F2-Sequenz (148 Punkte); sie ist KEIN universeller physikalischer Prozessabschlussbeweis.
5. **Trainingsbasierter Bereich**: konservativer Merkmalsbereich `min_train - 25 % * Spannweite` bis `max_train + 25 % * Spannweite`, getrennt je Rohkanal. Plus Prüfung auf positive geometrische Messwerte und Flatline der drei Geometriekanäle. Diese Grenzen erlauben keine extrapolative Qualitätsgarantie; neue legitime Prozesse können abgewiesen werden und müssen untersucht werden.
6. **Zweite Bestätigung**: Die Phase-6-Abschaltung belegt nur das Ende des F2-Prozessfensters. Für echte Prognosen ist zusätzlich eine separate, identifizierbare **End-of-Ring-Abschlussbestätigung** erforderlich (`--completion-record`). Der Datensatz wird über SHA256, Ring-ID, F2-Messpunktzahl und Endzeit gebunden. Ein JSON allein verifiziert nicht, ob die gemachte Erklärung wahr ist.
7. **Modell-Manifest**: Jeder künftige, mit V5.1 trainierte Modellstand speichert die zugehörige Eingangspolicy und ihren Hash. Alte V5-Pickle-Modelle sind im normalen V5.1-Prognosemodus absichtlich NICHT kompatibel. Testreplay durch QA-Monkeypatch ist kein regulärer Einsatz.

**Sonderfall DM06:** Die ersten 14 Zeitpunkte haben Phase 6, aber `LaserOK=MessEin=1`. Deshalb verlangt die Anfangsprüfung nur Phase 6, während der abschließende Nachlauf Phase 6 mit beiden Flags 0 benötigt. Alle 12 zulässigen Rohlogger werden dadurch korrekt akzeptiert.

## 2. Installation ohne Überschreiben

ZIP im Projektordner `Implementierung_BA` entpacken, sodass ein **neuer** Unterordner `final_candidate_v51/` entsteht. Vorhandene V5-Dateien NICHT ersetzen. Für den ersten Test kein neues Modell trainieren.

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v51\test_input_safety_v51.py
.\.venv\Scripts\python.exe .\final_candidate_v51\test_development_logger_equivalence_v51.py --data input\DM_16_aktive_Ringe.pkl
```

Erwartet: **18 synthetische Schutztests** sowie **12/12 bitgenaue F2-Kompatibilitätsprüfungen** mit den zwölf L/U-Entwicklungsringloggern. Der zweite Aufruf liest den bekannten, SHA256-geprüften Pickle, untersucht ausschließlich den Prozesslogger der 12 Entwicklungsringe, erstellt CSV-Dateien nur temporär und verwirft diese am Ende.

Eine zusätzliche freiwillige Policy-Reproduktion auf dem Nutzerrechner (NEUE Zieldatei verwenden):

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v51\main_bayesian_ssr_v51.py prepare-policy --data input\DM_16_aktive_Ringe.pkl --out results\v51\POLICY_REGENERATED_01.json
```

**Hinweis:** Der übergeordnete Ausgabeordner muss bereits existieren (z. B. mit `New-Item -ItemType Directory results\v51 -Force`). Bestehende Ergebnisdateien werden aus Sicherheitsgründen nicht überschrieben.

## 3. Späteres Prüfen eines ganz neuen Prozessloggers

`verify-input` überprüft ausschließlich eine **vollständige Rohlogger-CSV** (inkl. Phase-6-Zeilen) und gibt **keine Qualitätsprognose** aus:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v51\main_bayesian_ssr_v51.py verify-input --policy .\final_candidate_v51\DEVELOPMENT_INPUT_POLICY_V51.json --csv NEUER_RING_VOLLLOG.csv
```

Eine echte neue-Ring-Vorhersage benötigt später ein **neues V5.1-trainiertes und manifestverifiziertes Modell** sowie eine glaubwürdige separate End-of-Ring-Bestätigung:

```powershell
.\.venv\Scripts\python.exe .\final_candidate_v51\main_bayesian_ssr_v51.py predict --model-dir results\v51\EIN_SPATER_VERIFIZIERTER_TRAIN_LAUF --ring-id NEW01 --csv NEUER_RING_VOLLLOG.csv --completion-record CLOSEOUT_NEW01.json --out results\v51\PREDICTION_NEW01.json
```

`CLOSEOUT_TEMPLATE.json` ist absichtlich **nicht gültig**, bevor die tatsächlich belegten Felder durch eine berechtigte Person/Datenerfassung ausgefüllt wurden. **Kein Abschlussrecord wird automatisch erzeugt**, damit diese Bestätigung nicht mit einem mathematischen Loggersignal verwechselt wird. Den Namen `NEW01` niemals zum Umbenennen eines historischen DM-Rings verwenden, um die Sperre zu umgehen.

## 4. Forschungssicherheit / Grenzen

- Der Model-Freeze bleibt **vorläufig**: `K=2, P0, M2, C=0,2, physical_var=0,385261087...`, 8 L/4 U, zwei Targets. Keine Architektur- oder Hyperparameteränderung in V5.1.
- Die 10 mathematischen Kernmodule sind identisch zum V5-Paket. Bestehende CV-/Optuna-Ergebnisse werden weder verändert noch nochmals auf dem geöffneten äußeren Testsatz bewertet.
- Die 18 synthetischen Tests öffnen **keine** Pickle-Datei. Der Kompatibilitätstest lädt die vertrauenswürdige Originalserialisierung in den Speicher, **greift nur auf Loggerdaten der 12 Entwicklungsringe zu** und liest keine Qualitätslabels aus U oder alten äußeren Test-Ringen.
- `RWG55`: laut bisherigem Kanal-Audit Ringwachstumsgeschwindigkeit (m/s), **nicht Ringhöhe**; Maschinendokumentation weiterhin unabhängig prüfen.
- Keine bestätigte Produktionsqualität, keine überprüfte Online-Prozessoptimierung und keine unabhängige Kalibrierung der prädiktiven Standardabweichungen.
- Andere noch offene Auditbereiche: vollständige Konfigurationsvalidierung (V5.2), atomare Modellpersistenz (V5.2), zusätzliche numerische HMM-/VI-Tests und Ablationen (V5.3).
- Modellpickle nur aus vertrauenswürdiger, eigener Erzeugung laden. Hashes ohne externe Authentisierung schützen nicht vor manipulierter Modellquelle.

## 5. Quellenstatus

Grundlage des Implementierungs-/Domänentests: eingefrorener V5-Hauptkandidat samt zehn mathematischen Modulen und bekannte DM-F2-Entwicklungslogger. Originalpublikationen zur SSL-/Bayesian-Modelltheorie wurden in V5.1 **nicht erneut vollständig gelesen oder seitenweise abgeglichen**. Diese Version ist ein Eingabe-/Datenqualitäts-Audit, keine neue Modelltheorie.
