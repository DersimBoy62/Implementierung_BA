"""Reproduzierbare Leseansichten und SHA256-Inventur, ohne Modelle zu laden."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "bayesian_ssr_neubau/docs"
HANDOVER = ROOT / "handover"
PASTED_FILES = [Path("C:/Users/umutj/.codex/attachments") / identifier / "Pasted text.txt"
                for identifier in ["b6678fb2-413d-492a-88d0-f294901cd196",
                                   "7a2cfbaa-f3cf-4cbf-bcd2-4f7c3851b885",
                                   "b7f68f4b-bad7-4fda-bb05-1fd466723eb4"]]


def extended(path: Path) -> Path:
    """Use Windows extended paths for the supplied long scholarly filenames."""
    value = str(path.absolute())
    return Path("\\\\?\\" + value) if sys.platform == "win32" and not value.startswith("\\\\?\\") else path


def sha256(path: Path) -> str:
    return hashlib.sha256(extended(path).read_bytes()).hexdigest()


def document_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(extended(path)) as archive:
            tree = ET.fromstring(archive.read("word/document.xml"))
        # OMML math text (m:t) is included, unlike plain python-docx paragraphs.
        return "\n".join(
            "".join(node.text or "" for node in p.iter() if node.tag.rsplit("}", 1)[-1] == "t")
            for p in tree.iter() if p.tag.rsplit("}", 1)[-1] == "p"
        )
    return extended(path).read_text(encoding="utf-8-sig")


def extract_documents() -> None:
    destination = DOCS / "leseansichten"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    candidates = [*HANDOVER.rglob("*.docx")]
    for path in sorted(candidates):
        # The separate optimization plan is inventoried, never implemented.
        if "08_OPTIMIERUNG" in str(path):
            continue
        text = document_text(path)
        target = destination / (path.stem + ".txt")
        target.write_text(text, encoding="utf-8")
        records.append({"source": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                        "view": target.relative_to(ROOT).as_posix(), "lines": len(text.splitlines()),
                        "status": "Text extrahiert; nicht automatisch inhaltlich gelesen"})
    (DOCS / "text_extraction_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


def pdf_extract(pattern: str, pages: str | None, search: str | None) -> None:
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    paths = [p for p in (HANDOVER / "07_ORIGINALQUELLEN_PDF").glob("*.pdf") if pattern.casefold() in p.name.casefold()]
    for path in paths:
        reader = PdfReader(io.BytesIO(extended(path).read_bytes()))
        print(f"SOURCE {path.name}; {len(reader.pages)} PDF-Seiten; SHA256 {sha256(path)}")
        if pages:
            selected = []
            for part in pages.split(","):
                if "-" in part:
                    lo, hi = map(int, part.split("-"))
                    selected.extend(range(lo, hi + 1))
                else:
                    selected.append(int(part))
        else:
            selected = range(1, len(reader.pages) + 1)
        for page_no in selected:
            text = reader.pages[page_no - 1].extract_text() or ""
            if search:
                if not any(token.casefold() in text.casefold() for token in search.split("|")):
                    continue
                print(f"MATCH PDF-SEITE {page_no}: {text[:300].replace(chr(10), ' ')}")
            else:
                print(f"\nPDF-SEITE {page_no}\n{text}")


def inventory() -> None:
    states_path = DOCS / "READ_STATUS.json"
    states = json.loads(states_path.read_text(encoding="utf-8")) if states_path.exists() else {}
    candidates = list(HANDOVER.rglob("*"))
    excluded = {".venv", "handover", "uebergabe_2026_11_13", "bayesian_ssr_neubau", "__pycache__"}
    for folder in ROOT.iterdir():
        if folder.is_dir() and folder.name not in excluded and not folder.name.startswith("."):
            candidates.extend(folder.rglob("*"))
        elif folder.is_file() and not folder.name.startswith("."):
            candidates.append(folder)
    candidates.extend([ROOT / "requirements.txt", ROOT / ".venv/pyvenv.cfg"])
    downloads = Path("C:/Users/umutj/Downloads")
    candidates.extend(downloads.glob("Bayesian_Bachelor*2026-11-13.zip"))
    candidates.extend(downloads.glob("MASTER_PROMPT_CHATGPT_VSCODE_AGENT_BAYESIAN_BACHELOR_2026-11-13*.md"))
    candidates.extend(PASTED_FILES)
    rows = []
    for path in sorted(set(candidates)):
        if not extended(path).is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        if "07_ORIGINALQUELLEN" in relative and path.suffix.lower() == ".pdf":
            role = "wissenschaftliches Original-PDF / Quellenkandidat"
            if "Formatvorlage" in path.name:
                role = "studentisches Laborprotokoll; kein Modellbeleg"
            elif "Beal" in path.name:
                role = "wissenschaftlicher Kapitel-Auszug; kein vollständiges Buch"
        elif "01_NEUE_HERLEITUNGEN" in relative:
            role = "aktuelle Herleitung vor V5.8–V6.0"
        elif "03_ORIGINALE_UNVERAENDERT" in relative:
            role = "alte Herleitung"
        elif relative.startswith("results/"):
            role = "Altresultate / historische Modelle; nur Byte-Hash, kein Laden als Modell"
        elif path.suffix.lower() == ".pkl" or relative.startswith("input/"):
            role = "Daten / Provenienz (keine Trainingszustände übernommen)"
        elif path.suffix.lower() == ".py":
            role = "Referenzcode / numerischer Oracle, read-only"
        elif "08_OPTIMIERUNG" in relative:
            role = "separater Folgeauftrag, nicht umgesetzt"
        elif path.suffix.lower() in {".csv", ".json", ".png"} and ("research_" in relative or "06_FORSCHUNGS" in relative or "results" in relative):
            role = "historische Evidenz / Altresultate"
        else:
            role = "Status / wissenschaftliche Dokumentation / Archiv / Umgebung"
        state = states.get(relative, {})
        rows.append({"datei": relative, "rolle": role, "sha256": sha256(path),
                     "gelesen": state.get("gelesen", "nein"),
                     "offene_punkte": state.get("offene_punkte", "inventarisiert; keine inhaltliche Vollprüfung behauptet")})
        if path.suffix.lower() == ".zip" and path.is_relative_to(HANDOVER):
            with zipfile.ZipFile(extended(path)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    member_name = relative + "!" + member.filename
                    member_state = states.get(member_name, {})
                    member_role = "Archivmitglied: historische Evidenz / Referenz / Herkunft; nicht automatisch entpackt"
                    if "DM_Loggerdaten" in member.filename or "Zielwerte" in member.filename:
                        member_role = "Archivmitglied: historische Datengrundlage / Provenienz"
                    rows.append({"datei": member_name, "rolle": member_role,
                                 "sha256": hashlib.sha256(archive.read(member)).hexdigest(),
                                 "gelesen": member_state.get("gelesen", "nein"),
                                 "offene_punkte": member_state.get("offene_punkte", "Archivmitglied gehasht; keine inhaltliche Prüfung behauptet")})
    DOCS.mkdir(parents=True, exist_ok=True)
    with (DOCS / "Inventur.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["datei", "rolle", "sha256", "gelesen", "offene_punkte"])
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["rolle"] for row in rows)
    read_count = sum(row["gelesen"].startswith("ja") for row in rows)
    overview = ["# Inventur und Herkunft – tatsächlich geprüft", "",
                f"Stand13.11.2026: {len(rows)} Dateien/Archivmitglieder mit SHA256; {read_count} konkrete Lesevermerke.",
                "Vollständige Tabelle: [Inventur.csv](Inventur.csv). Leseumfang: [READ_STATUS.json](READ_STATUS.json).", "",
                "| Rolle | Anzahl |", "|---|---:|"]
    overview.extend(f"| {role} | {count} |" for role, count in sorted(groups.items()))
    overview.extend(["", "Die fünf bereitgestellten ZIPs wurden vollständig bytegleich gegen handover/ geprüft:",
                     "113 Dateien Kernpaket sowie20+21+20+21 PDF-Dateien; insgesamt195 extrahierte Dateien.",
                     "Alle82 PDF-Dateien haben unterschiedliche SHA256. Mindestens eine Datei ist ein Laborprotokoll;",
                     "Beal liegt als Kapitel-Auszug vor. Verfügbarkeit bedeutet keine vollständige inhaltliche Prüfung.", "",
                     "Alle drei Pasted-text-Anhänge und beide gleichnamigen Download-Masterkopien sind47.120Byte groß",
                     "und haben denselben SHA256 e12daff2c5b2f7c09e5c2ebb2bff61a8d4f26a5ca9542d5e5406d520fd9b0682.",
                     "Damit ist der vom Nutzer ausdrücklich beauftragte Master-Prompt eindeutig; historische Dokumente",
                     "werden als Arbeitsunterlagen mit eigener Quellenhierarchie behandelt.", "",
                     "Historische results/-Dateien wurden ausschließlich gehasht, nicht als trainierte Modelle geladen.",
                     "Nicht als gelesen bezeichnete Dateien bleiben für inhaltliche Aussagen offen. Die vollständige",
                     "Inventur erfasst das alte Projekt, die kurze handover/-Extraktion und die Nutzeranhänge; die",
                     ".venv wird nur über pyvenv.cfg/Versionsnachweise inventarisiert. Die unvollständige lange",
                     "uebergabe_2026_11_13/-Erstextraktion bleibt unverändert, ist aber keine maßgebliche Quellenkopie.",
                     "Neue Produktivdateien und Laufartefakte erhalten ihre eigene Release-/Lauf-Integritätsliste.", ""])
    (DOCS / "INVENTUR_UEBERBLICK.md").write_text("\n".join(overview), encoding="utf-8")
    print(f"{len(rows)} Dateien mit SHA256 in docs/Inventur.csv")


def record_read_status() -> None:
    """Record the actual 2026-11-13 audit; extraction alone is never 'read'."""
    states = {}
    for relative in ["00_START_HIER.md", "01_MASTER_PROMPT_VSCODE_AGENT.md",
                     "03_ENTSCHEIDUNGEN_UND_STATUS/AKTUELLE_ENTSCHEIDUNGSMATRIX.md",
                     "03_ENTSCHEIDUNGEN_UND_STATUS/PROJEKTUEBERNAHME_BAYESIAN_BACHELOR_2026-11-12.md"]:
        states["handover/" + relative] = {"gelesen": "ja", "offene_punkte": "Historischen Status von aktuellem Nutzerauftrag getrennt; kein neuer wissenschaftlicher Beweis"}
    for path in (HANDOVER / "02_HERLEITUNGEN_A_B_C_D").rglob("*"):
        if path.suffix.lower() == ".md":
            states[path.relative_to(ROOT).as_posix()] = {"gelesen": "ja", "offene_punkte": "Formeln gelesen; veraltete K3/C1/M1/10E und Priorwerte im ABGLEICH korrigiert"}
        elif path.suffix.lower() == ".docx":
            states[path.relative_to(ROOT).as_posix()] = {"gelesen": "ja (Text/OMML; mathematische Abschnitte)", "offene_punkte": "Vollständig extrahiert; keine visuelle Layout-/Objektprüfung; aktuelle Formeln separat fortgeschrieben"}
    relative = "handover/03_ENTSCHEIDUNGEN_UND_STATUS/ABGLEICHMATRIX_Original_A-D_zu_finalem_DM_Modell_2026-11-11.docx"
    states[relative] = {"gelesen": "ja (Text/OMML)", "offene_punkte": "Historischer 11.11.-Stand; aktuelle eigene Abgleichmatrix ersetzt obsolete Parameter"}
    for version in range(58, 64):
        relative = f"handover/04_REFERENZCODE_READ_ONLY/V{version}/research_math_v{version}/README.md"
        states[relative] = {"gelesen": "ja", "offene_punkte": "Historische Tests/Realzahlen nicht als neue Ausführung übernommen; V63 war laut Übergabe Windows-unbestätigt"}
    relative = "handover/04_REFERENZCODE_READ_ONLY/V60/research_math_v60/V60_30E_LOCAL_RESEARCH_SUMMARY.json"
    states[relative] = {"gelesen": "ja", "offene_punkte": "Historischer train-only-Verlauf, ausschließlich Methoden-/Provenienzaudit; kein Trainingsinput"}
    for name in ["BA_V55_Konvergenzanalyse_30_Epochen_und_Folgeexperimente_2026-11-12.docx",
                 "BA_V61_Globaler_Prozessposterior_ELBO_Sprung_2026-11-12.docx"]:
        states["handover/06_FORSCHUNGSBEFUNDE_UND_AUDITS/" + name] = {"gelesen": "ja (Text/OMML)", "offene_punkte": "Historischer Befund; keine neue Trainingsevidenz"}
    data_files = {"dm_data.py": "F2-/Feature-/Target-API; Datenagent",
                  "quality_link_challenge.py": "extract_descriptors, relevant gelesen; Datenagent",
                  "compare_two_head_cv4.py": "Kopf-Featuregruppen, relevant gelesen; Datenagent",
                  "dynamic_markov_v2.py": "Spec/increments und Rewardrekursion; Audit und Mathematikagent",
                  "dynamic_quality_v2.py": "Prior, Initialisierung, lokale/globalen Updates; Audit und Root",
                  "two_head_b1.py": "NIG-Dichte-/Momentkonvention, Kopfkopplung und Update; Audit und Root"}
    core_locations = [ROOT / "final_candidate_v52/final_core", HANDOVER / "04_REFERENZCODE_READ_ONLY/V52_ORIGINAL_CORE/final_candidate_v52/final_core"]
    for directory in core_locations:
        for name, note in data_files.items():
            path = directory / name
            if extended(path).is_file():
                states[path.relative_to(ROOT).as_posix()] = {"gelesen": "ja (relevanter Code)", "offene_punkte": note + "; read-only-Referenz, keine Produktivabhängigkeit"}
    for directory in [ROOT / "final_candidate_v52", HANDOVER / "04_REFERENZCODE_READ_ONLY/V52_ORIGINAL_CORE/final_candidate_v52"]:
        for name in ["input_safety_v51.py", "test_development_logger_equivalence_v51.py"]:
            path = directory / name
            if extended(path).is_file():
                states[path.relative_to(ROOT).as_posix()] = {"gelesen": "ja (Datenagent)", "offene_punkte": "V51-Sicherheits-/Gatevertrag im Neubau unabhängig geprüft"}
    pdf_pages = {"Rabiner": "PDF1–8; konkrete FB/Pfad-Stellen",
                 "Beal": "PDF10–15; Kapitel3-Auszug, Druck91–96",
                 "Bishop 2006": "PDF710–717; NW/Wishart/Student-t; weitere Suchtreffer nicht als gelesen gezählt",
                 "Shao et al. (2022)": "PDF1–4,6 und Teile5; dynamische SSR und NW",
                 "Shao, Ge": "PDF1,3–5; WVI weicht von unserem gesamten U-Beitrag ab",
                 "Mowbray": "PDF1–2; Batch-Endqualität",
                 "Varma": "PDF1–2; CV-Modellauswahl-Bias",
                 "Cawley": "PDF1–2; Auswahlüberanpassung",
                 "Roberts": "PDF1–2; strukturierte CV; Publikationsjahr2017",
                 "Gelman": "PDF1–2; Abstract/Inhaltsübersicht; Schriftkodierung teilweise beschädigt",
                 "Gabry": "PDF1–2; Workflow/Simulation/Predictive Checks",
                 "Yuan et al.": "PDF1; Abstract und bibliografische Einordnung, Methodenteil offen",
                 "Li et al. (2026)": "PDF1; Abstract und Ausgabe2026/online2025, Methodenteil offen",
                 "Dai, Li": "PDF1; Abstract/Fig1 Pseudolabel-Verfahren, Methodenteil offen",
                 "Kang, Kim": "PDF1; self-training-Abstract, Methodenteil offen",
                 "van Engelen": "PDF1; Classification-Survey-Abstract, Methodenteil offen",
                 "kostopoulos": "PDF1; SSR-Survey-Abstract, weitere Details offen",
                 "Kapoor": "PDF1; Leakage-Highlights, konkreter Methodenteil offen",
                 "Kaufman": "PDF1; learn-predict separation im Abstract, weitere Details offen",
                 "Formatvorlage": "PDF1 identifiziert studentisches Fachlaborprotokoll, kein Modellbeleg"}
    for path in (HANDOVER / "07_ORIGINALQUELLEN_PDF").glob("*.pdf"):
        for pattern, note in pdf_pages.items():
            if pattern.casefold() in path.name.casefold():
                states[path.relative_to(ROOT).as_posix()] = {"gelesen": "ja (benannter Auszug)", "offene_punkte": note + "; keine Vollprüfung/Supplementprüfung behauptet"}
                break
    for path in HANDOVER.rglob("*Datenaudit*.zip"):
        with zipfile.ZipFile(extended(path)) as archive:
            for member in archive.namelist():
                if member.endswith("Datenpruefung_Pickles_ausfuehrlich.md"):
                    states[path.relative_to(ROOT).as_posix() + "!" + member] = {"gelesen": "ja (Datenagent vollständig)", "offene_punkte": "Ursprungs-all_targets nicht mitgeliefert; Metrologie axialer Höhenschlag offen; historische Evidenz"}
    states["input/DM_16_aktive_Ringe.pkl"] = {"gelesen": "ja (kontrollierter Daten-/Header-/L-Targetzugriff)", "offene_punkte": "SHA exakt;16IDs,12Dev-Prozesse/L-Ziele geprüft; U-/Außenlabels nicht für Modell/Score verwendet"}
    states[".venv/pyvenv.cfg"] = {"gelesen": "ja", "offene_punkte": "Python3.11.9; Paketversionen im Environment-/Locknachweis"}
    for path in Path("C:/Users/umutj/Downloads").glob("MASTER_PROMPT_CHATGPT_VSCODE_AGENT_BAYESIAN_BACHELOR_2026-11-13*.md"):
        states[path.as_posix()] = {"gelesen": "ja (Root; bitgleiche Kopien)", "offene_punkte": "Vom Nutzer ausdrücklich zur Ausführung beauftragt; beide Downloadkopien SHA-identisch"}
    for path in PASTED_FILES:
        states[path.as_posix()] = {"gelesen": "ja (bitgleich mit gelesenem Master)", "offene_punkte": "Alle drei Pasted-Anhänge per SHA256 mit dem vollständig gelesenen Download-Master identisch"}
    (DOCS / "READ_STATUS.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(states)} tatsächliche Lesevermerke gespeichert")


def verify_archives() -> None:
    records = []
    for archive_path in sorted(Path("C:/Users/umutj/Downloads").glob("Bayesian_Bachelor*2026-11-13.zip")):
        members = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = Path(*Path(member.filename).parts[1:])
                target = HANDOVER / relative
                if not target.resolve().is_relative_to(HANDOVER.resolve()):
                    raise ValueError("Unsafe ZIP member path")
                original_sha = hashlib.sha256(archive.read(member)).hexdigest()
                equal = extended(target).is_file() and sha256(target) == original_sha
                members.append({"member": member.filename, "local": target.relative_to(ROOT).as_posix(),
                                "sha256": original_sha, "byte_equal": equal})
                if not equal:
                    raise AssertionError(f"Missing or different archive member: {target}")
        records.append({"archive": archive_path.as_posix(), "sha256": sha256(archive_path), "members": members})
        print(f"{archive_path.name}: {len(members)} Dateien bytegleich")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ARCHIV_HERKUNFT.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["extract-docx", "pdf", "inventory", "verify-archives", "record-read-status"])
    parser.add_argument("--pattern", default="")
    parser.add_argument("--pages")
    parser.add_argument("--search")
    args = parser.parse_args()
    if args.action == "extract-docx":
        extract_documents()
    elif args.action == "inventory":
        inventory()
    elif args.action == "verify-archives":
        verify_archives()
    elif args.action == "record-read-status":
        record_read_status()
    else:
        pdf_extract(args.pattern, args.pages, args.search)
