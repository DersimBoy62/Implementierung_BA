LOKALE DM-ROHDATEN

Für Training, Evaluation, reale Datentests und das vollständige Notebook die
freigegebene Datei DM_16_aktive_Ringe.pkl in diesem Ordner bereitstellen.
Sie wird nicht zu GitHub übertragen.

Erwarteter SHA256:
bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5

Alternativ BAYESIAN_SSR_DATA auf den tatsächlichen lokalen Dateipfad setzen
oder bei CLI-Befehlen ausdrücklich --input PFAD angeben. Ein abweichender
Hash wird vor der Deserialisierung abgelehnt.

manifest_pruefbericht.json dokumentiert die historische XLSX-Pickle-Übernahme
des ursprünglichen Datenbestands. Das aktuelle Modell verwendet ausschließlich
die SHA-verifizierte Datei mit den 16 aktiven Ringen. Nicht sämtliche dort
historisch erwähnten Dateien sind Teil dieses Repositorys.

Die vollständigen Daten-, Feature-, Split- und Vorhersageverträge stehen in
bayesian_ssr_neubau/docs/DATENPROVENIENZ.md. Ohne die private Datei laufen
mathematische und synthetische Tests weiter; nur reale Datentests werden mit
ausdrücklicher Begründung übersprungen.
