# Inventur und Herkunft – tatsächlich geprüft

Stand13.11.2026: 1071 Dateien/Archivmitglieder mit SHA256; 70 konkrete Lesevermerke.
Vollständige Tabelle: [Inventur.csv](Inventur.csv). Leseumfang: [READ_STATUS.json](READ_STATUS.json).

| Rolle | Anzahl |
|---|---:|
| Altresultate / historische Modelle; nur Byte-Hash, kein Laden als Modell | 445 |
| Archivmitglied: historische Datengrundlage / Provenienz | 3 |
| Archivmitglied: historische Evidenz / Referenz / Herkunft; nicht automatisch entpackt | 193 |
| Daten / Provenienz (keine Trainingszustände übernommen) | 3 |
| Referenzcode / numerischer Oracle, read-only | 163 |
| Status / wissenschaftliche Dokumentation / Archiv / Umgebung | 136 |
| aktuelle Herleitung vor V5.8–V6.0 | 8 |
| alte Herleitung | 4 |
| historische Evidenz / Altresultate | 33 |
| separater Folgeauftrag, nicht umgesetzt | 1 |
| studentisches Laborprotokoll; kein Modellbeleg | 1 |
| wissenschaftlicher Kapitel-Auszug; kein vollständiges Buch | 1 |
| wissenschaftliches Original-PDF / Quellenkandidat | 80 |

Die fünf bereitgestellten ZIPs wurden vollständig bytegleich gegen handover/ geprüft:
113 Dateien Kernpaket sowie20+21+20+21 PDF-Dateien; insgesamt195 extrahierte Dateien.
Alle82 PDF-Dateien haben unterschiedliche SHA256. Mindestens eine Datei ist ein Laborprotokoll;
Beal liegt als Kapitel-Auszug vor. Verfügbarkeit bedeutet keine vollständige inhaltliche Prüfung.

Alle drei Pasted-text-Anhänge und beide gleichnamigen Download-Masterkopien sind47.120Byte groß
und haben denselben SHA256 e12daff2c5b2f7c09e5c2ebb2bff61a8d4f26a5ca9542d5e5406d520fd9b0682.
Damit ist der vom Nutzer ausdrücklich beauftragte Master-Prompt eindeutig; historische Dokumente
werden als Arbeitsunterlagen mit eigener Quellenhierarchie behandelt.

Historische results/-Dateien wurden ausschließlich gehasht, nicht als trainierte Modelle geladen.
Nicht als gelesen bezeichnete Dateien bleiben für inhaltliche Aussagen offen. Die vollständige
Inventur erfasst das alte Projekt, die kurze handover/-Extraktion und die Nutzeranhänge; die
.venv wird nur über pyvenv.cfg/Versionsnachweise inventarisiert. Die unvollständige lange
uebergabe_2026_11_13/-Erstextraktion bleibt unverändert, ist aber keine maßgebliche Quellenkopie.
Neue Produktivdateien und Laufartefakte erhalten ihre eigene Release-/Lauf-Integritätsliste.
