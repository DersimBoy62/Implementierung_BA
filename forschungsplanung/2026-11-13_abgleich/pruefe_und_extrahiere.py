"""Nur DOCX-Lesen und Integritätsprüfung; keine Modellimporte oder neuen Fits."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASELINE = ROOT / "bayesian_ssr_neubau"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=Path(
        "C:/Users/umutj/Downloads/BAYESIAN_BACHELOR_OPTIMIERUNGSFAHRPLAN_2026-11-13.docx"))
    args = parser.parse_args()
    text_output = HERE / "OPTIMIERUNGSFAHRPLAN_VOLLSTAENDIG.txt"
    audit_output = HERE / "BESTANDSPRUEFUNG.json"
    if text_output.exists() or audit_output.exists():
        raise FileExistsError("Prüfausgaben existieren bereits; neuer Abgleich benötigt einen neuen Ordner")
    manifest_path = BASELINE / "RELEASE_SHA256.json"
    manifest = read_json(manifest_path)
    mismatches = [name for name, value in manifest["files"].items()
                  if not (BASELINE / name).is_file() or sha(BASELINE / name) != value]
    if mismatches:
        raise ValueError(f"Baseline-Abweichungen: {mismatches}")
    dataset_sha = sha(ROOT / "input/DM_16_aktive_Ringe.pkl")
    if dataset_sha != "bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5":
        raise ValueError("Daten-SHA256-Abweichung")
    handover = ROOT / "handover/08_OPTIMIERUNG_NACH_NEUBAU/OPTIMIERUNGSFAHRPLAN_NACH_NEUBAU_PRIORISIERT_2026-11-13.docx"
    doc_sha = sha(args.document)
    lines = ["Vollständige Leseansicht des Nutzer-DOCX vom 13.11.2026",
             f"Quelle: {args.document}", f"SHA256: {doc_sha}",
             "Absatznummern umfassen Word-Haupttext und Tabellenzellen; keine Word-Seitenangaben.", ""]
    with zipfile.ZipFile(args.document) as archive:
        body = ET.fromstring(archive.read("word/document.xml")).find(W + "body")
        paragraphs = list(body.iter(W + "p"))
        for i, paragraph in enumerate(paragraphs, 1):
            text = "".join(node.text or "" for node in paragraph.iter() if node.tag in (W + "t", M + "t"))
            lines.append(f"P{i:03d}: {text}")
        additional_parts = [name for name in archive.namelist() if name.startswith("word/") and
            name.endswith(".xml") and name.split("/")[-1].startswith(("header", "footer", "footnotes", "endnotes", "comments"))]
        for name in additional_parts:
            part = ET.fromstring(archive.read(name))
            text = " ".join(node.text or "" for node in part.iter() if node.tag in (W + "t", M + "t", W + "instrText"))
            lines.extend(("", f"Zusatzteil {name}: {text}"))
    with text_output.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    tests_path = BASELINE / "docs/verification_001.json"
    tests = read_json(tests_path)
    release = read_json(BASELINE / "docs/RELEASE_VALIDATION.json")
    report = dict(date="2026-11-13", scope="Dokumentabgleich und Integritätsprüfung; keine Optimierung",
        document_path=str(args.document), document_sha256=doc_sha, handover_document_sha256=sha(handover),
        same_document_as_handover=doc_sha == sha(handover), main_body_paragraphs_read=len(paragraphs),
        additional_word_parts_read=additional_parts, baseline_manifest_sha256=sha(manifest_path),
        baseline_manifest_files_checked=len(manifest["files"]), baseline_mismatches=mismatches,
        dataset_sha256=dataset_sha, existing_test_report="bayesian_ssr_neubau/docs/verification_001.json",
        existing_test_report_sha256=sha(tests_path),
        existing_test_counts={k: tests[k] for k in ("tests", "failures", "errors", "skipped", "exitcode")},
        model_sha256=release["model_sha256"], cv_report_sha256=release["cv_report_sha256"],
        new_tests_executed=False, new_training_executed=False, baseline_modified=False)
    with audit_output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
