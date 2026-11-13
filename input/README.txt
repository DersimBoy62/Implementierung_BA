BAYESIAN BACHELOR – AKTUELLE DM-ROHDATEN ALS PICKLE

Quelle: Das zuletzt erstellte ZIP mit 17 aktuellen Excel-Dateien inkl. Walzprotokoll.

Ordner einzelne_ringe/: je 1 Pickle pro Ring (17 insgesamt).
DM_16_aktive_Ringe.pkl: OrderedDict mit den 16 regulären Ringen.
DM_alle_17_mit_Status.pkl: OrderedDict mit 17 Ringen, DM13 als ausgeschlossen gekennzeichnet.
DM03: abgebrochener Versuch, keine Loggerdatei; fehlt als Pickle.

Ein Ring-Pickle ist ein Dictionary mit
- ring_id, logger_id, train_eligible, exclusion_reason, source
- logger['matrix']: NumPy-float64-Array (Zeitspalte + 114 Kanäle)
- logger['signal_names'], ['signal_units'], ['signal_descriptions']
- logger['header_rows_excel_1_to_10'] originaler Kopfbereich
- zielwerte: 34 Felder mit Originalnamen und Original-Zellwerten
- walzprotokoll: Liste der ausgefüllten Zeilen ('Bereich', 'Bezeichnung', 'Wert')

Beispiel:
import pickle
with open('DM_16_aktive_Ringe.pkl','rb') as f:
    ringe = pickle.load(f)
ring = ringe['DM01']
X = ring['logger']['matrix']
channel_names = ring['logger']['matrix_columns']
quality = ring['zielwerte']['ovality.ovality_radial_mm']

Die Vektoren der Zielwerte bleiben exakt als Originalstrings erhalten.
Nicht mit echten Geometrie-Koordinatenarrays verwechseln.
Keinerlei Skalierung, Phasenselektion, Datenaufteilung, Label-Rekonstruktion oder Training.
Excel-Styles, Formate und leere Zellen werden nicht in Pickle gespiegelt.

SICHERHEIT: Pickle nur aus vertrauenswuerdigen Quellen laden; das Laden kann Code ausfuehren.

Zum eigenständigen erneuten Aufbau: python DM_excel_zu_pickle.py --input QUELLE.zip --output AUSGABE_ORDNER
