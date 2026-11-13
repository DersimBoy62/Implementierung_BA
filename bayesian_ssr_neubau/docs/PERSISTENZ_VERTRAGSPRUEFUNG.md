# Semantische Persistenz- und Leakage-Prüfung

Ausgeführt am 13.11.2026:

```powershell
.\.venv\Scripts\python.exe -m pytest bayesian_ssr_neubau\tests\test_persistence_contract.py bayesian_ssr_neubau\tests\test_data_and_leakage.py -q
```

Ergebnis: **40 passed in 4.54 s**, Exitcode 0. Davon sind 22 neue
Persistenz-Vertragsprüfungen und 18 Daten-/Leakage-Prüfungen.

Die Persistenzprüfungen beginnen mit einem tatsächlich frisch trainierten
synthetischen Modell. In unabhängigen temporären Kopien werden semantische
Fehler eingebaut und die SHA256-Prüfsummen anschließend absichtlich korrekt
neu berechnet. Dies prüft die Vertragsvalidierung jenseits bloßer
Bytekorruption: widersprüchliche Zustandszahl, fehlende Köpfe, alte oder falsche
Kopfdimensionen, falsche Scaler-Dimensionen, VAL-Prozesse im X-/Deskriptorfit,
U-Labels im Y-Fit, TRAIN-/VAL-Überlappung, historischer Außenring im Training,
abweichende Feature-/Targetreihenfolge und ein unzulässiges Phase-6-Fenster.
Weitere Fälle verändern den gesonderten Vorhersagevertrag auf erforderliche
Targets, falsche Einheit oder falsche Kopfdimensionen. Alle Fälle werden
bereits beim Laden verworfen. Ein Pfadtraversal im Integritätsmanifest wird
vor dem Lesen außerhalb des Modellordners abgelehnt. Das unveränderte Modell
liefert nach JSON-Laden dieselbe prozessbasierte Prediction wie im Speicher.

Der getrennte AST-Audit wurde mit
`.\.venv\Scripts\python.exe bayesian_ssr_neubau\scripts\audit_importgraph.py`
ausgeführt. `IMPORTGRAPH.json` erfasst **23 Produktivmodule**, ihre Quellhashes
und Imports: **keine Altmodellimporte, keine dynamischen Imports**.
Historische Oracles werden ausschließlich in den dafür benannten Tests gelesen.

Zusätzlich wurde `evaluation.py` unabhängig gelesen: alle vier Folds trennen
Ring-IDs; der Fit erhält sechs L-TRAIN- und die erlaubten U-Prozessobjekte,
keine VAL-Ringe. Zielscaler verwenden ausschließlich L-TRAIN. Strict-C0 und
Shared-C0 sind getrennte Arme mit explizit unterschiedlicher X-only-U-Policy.
Die äußeren vier historischen Ringe werden weder gefittet noch gescort.
Diese Prüfung ist ein Software-/Datenflussaudit, kein Nachweis prädiktiver
Überlegenheit oder kalibrierter Unsicherheiten.
