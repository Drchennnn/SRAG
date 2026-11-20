"""Check original offsets, capped reading and actual OCR publication on public inputs."""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureScope, StructureSearcher, StructureSearchRequest
from app.rag.workspace_reading import read_window
from app.schemas.local import SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

COURSE = "public-ordered-reading"


def main(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    docs = Path(__file__).parent
    runpy.run_path(str(docs / "G10-probe.py"))["build"](directory)
    briefing = runpy.run_path(str(docs / "G14-fixtures.py"))
    fixtures = directory / "fixtures"
    briefing["make_pdf"](fixtures / briefing["FILENAME"])
    settings = Settings(
        _env_file=None,
        course_root=fixtures,
        data_dir=directory / "data",
        sqlite_path=directory / "reading.sqlite3",
    )
    library = LocalLibrary(settings)
    snapshots = StructureSnapshotStore(settings)
    report = {"readings": {}, "publish_ms": {}, "ocr": {}}
    for filename in (
        briefing["FILENAME"],
        "Drum-Readings.pptx",
        "v1/Drum-Trend.ipynb",
        "Drum-Scan.pdf",
    ):
        started = perf_counter()
        (item,) = library.import_paths(COURSE, [fixtures / filename])
        report["publish_ms"][filename] = round((perf_counter() - started) * 1000, 3)
        assert item.status == "published" and item.document_id and item.version == 1
        document_id = item.document_id
        request = SnapshotRequest((COURSE,), (document_id,))
        snapshot = snapshots.read(request)
        window = read_window(library.store, snapshot, context_tokens=32768, output_reserve=8192)
        report["readings"][filename] = window.trace()
        for segment in window.segments:
            key = source_key(segment.source.source_id, document_id)
            raw = library.store.read_source(key, course_ids=(COURSE,))
            assert raw.source and raw.source.text[segment.start : segment.end] == segment.text

        if filename == briefing["FILENAME"]:
            coverage = {s.source.locator["page"]: s.text for s in window.segments}
            assert list(coverage) == list(range(1, 9)) and window.full_text_scope
            assert all(
                word in coverage[page]
                for page, words in briefing["EXPECTED"].items()
                for word in words
            )
            preferred = [source_key("page:8", document_id)]
            first = read_window(
                library.store,
                snapshot,
                context_tokens=1600,
                output_reserve=400,
                prompt_reserve=400,
                preferred_sources=preferred,
            )
            assert first.next_cursor and first.segments[0].source.source_id == "page:8"
            resumed = read_window(
                library.store,
                snapshot,
                context_tokens=32768,
                output_reserve=8192,
                preferred_sources=preferred,
                cursor=first.next_cursor,
            )
            parts = {}
            for segment in (*first.segments, *resumed.segments):
                parts.setdefault(segment.source.locator["page"], []).append(segment.text)
            assert {page: "".join(text) for page, text in parts.items()} == coverage
            report["resumed"] = {"first_window": first.trace(), "next_window": resumed.trace()}
        elif filename.endswith(".pptx"):
            (table,) = [s for s in window.segments if s.method == "native_table"]
            assert "Frequency [Hz]" in table.table_headers and "13.75" in table.text
        elif filename.endswith(".ipynb"):
            assert any(s.source.source_id == "cell:0:output:0" for s in window.segments)
            assert any(
                item["source"]["source_id"] == "cell:0:output:1" for item in window.unread_visual
            )
        else:
            searcher = StructureSearcher(snapshots)
            query = StructureSearchRequest(
                query="Overview unscaled",
                scope=StructureScope(course_ids=[COURSE], document_ids=[document_id]),
            )
            before = searcher.search(query, allowed_course_ids={COURSE})
            assert not window.segments and before.status == "no_text_anchor"
            started = perf_counter()
            write = library.enhance_pdf_ocr(COURSE, document_id)
            ocr_ms = round((perf_counter() - started) * 1000, 3)
            after = searcher.search(query, allowed_course_ids={COURSE})
            online = read_window(
                library.store, snapshots.read(request), context_tokens=32768, output_reserve=8192
            )
            assert write["pages"] == {"page:1": "published"}
            (read,) = online.segments
            assert read.method == "ocr" and read.text == "Overview [unscaled]"
            assert any(
                reason.code == "fts_ocr" for hit in after.candidates for reason in hit.reasons
            )
            report["ocr"] = {
                "write": write,
                "elapsed_ms": ocr_ms,
                "search_before": before.status,
                "search_after": after.status,
                "reading": online.trace(),
            }
    (directory / "G15-reading.json").write_text(json.dumps(report, indent=2) + "\n")


def source_key(source_id: str, document_id: str) -> SourceKey:
    return SourceKey(document_id=document_id, version=1, source_id=source_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
