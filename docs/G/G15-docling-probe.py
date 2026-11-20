"""Compare native-first and existing Docling parsing on public PDFs."""

import argparse
import json
import runpy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from app.rag.parsers.local_slides import parse_local_slides
from app.rag.parsers.pdf import PDFParser


def main(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    docs = Path(__file__).parent
    runpy.run_path(str(docs / "G10-probe.py"))["build"](directory)
    briefing = runpy.run_path(str(docs / "G14-fixtures.py"))
    briefing["make_pdf"](directory / "fixtures" / briefing["FILENAME"])
    native = PDFParser(directory / "docling-native")
    ocr = PDFParser(directory / "docling-ocr", enable_ocr_fallback=True)
    report = {"docling_version": version("docling"), "cases": []}
    for name, parser in (
        (briefing["FILENAME"], native),
        ("Drum-Lab.pdf", native),
        ("Drum-Rotated.pdf", native),
        ("Drum-Scan.pdf", native),
        ("Drum-Scan.pdf", ocr),
    ):
        path = directory / "fixtures" / name
        started = perf_counter()
        base = parse_local_slides(path)
        base_ms = round((perf_counter() - started) * 1000, 3)
        started = perf_counter()
        enhanced = parser.parse(path)
        report["cases"].append({
            "file": name,
            "ocr_enabled": parser.enable_ocr_fallback,
            "native_ms": base_ms,
            "docling_ms": round((perf_counter() - started) * 1000, 3),
            "native": [
                {"id": s.source_id, "kind": s.kind, "state": s.state,
                 "text_chars": len(s.text), "page": s.locator.get("page")}
                for s in base.sources
            ],
            "docling": [[
                {"kind": item.kind, "text_chars": len(item.text), "position": item.bbox is not None,
                 "ocr_text": item.text if "-ocr-" in item.element_id else None}
                for item in slide.elements
            ] for slide in enhanced.slides],
        })
    (directory / "G15-docling.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
