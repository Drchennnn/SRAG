"""Read a published document in order, or OCR its archived scans locally."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.config import get_settings
from app.rag.local_library import LocalLibrary
from app.rag.workspace_reading import ReadingCursor, read_window
from app.schemas.local import SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read")
    ocr = commands.add_parser("ocr")
    for command in (read, ocr):
        command.add_argument("--course", required=True)
        command.add_argument("--document", required=True)
    read.add_argument("--context-tokens", type=int)
    read.add_argument("--output-reserve", type=int, default=8192)
    read.add_argument("--prompt-reserve", type=int, default=1024)
    read.add_argument("--prefer", action="append", default=[], help="version-local source ID")
    read.add_argument("--cursor", help="next_cursor JSON emitted by the previous read")
    args = parser.parse_args()

    settings = get_settings()
    library = LocalLibrary(settings)
    if args.command == "ocr":
        result = library.enhance_pdf_ocr(args.course, args.document)
    else:
        snapshot = StructureSnapshotStore(settings).read(
            SnapshotRequest((args.course,), (args.document,), require_single_document=True)
        )
        if snapshot.resolution_status != "resolved":
            parser.error("当前课程没有此资料")
        version = snapshot.documents[0].version
        ceiling = min(
            settings.answer_context_tokens,
            settings.workspace_input_token_limit + args.output_reserve,
        )
        window = read_window(
            library.store,
            snapshot,
            context_tokens=ceiling if args.context_tokens is None else args.context_tokens,
            output_reserve=args.output_reserve,
            prompt_reserve=args.prompt_reserve,
            preferred_sources=[
                SourceKey(document_id=args.document, version=version, source_id=key)
                for key in args.prefer
            ],
            cursor=ReadingCursor(**json.loads(args.cursor)) if args.cursor else None,
        )
        result = {
            "reading": window.trace(),
            "segments": [
                {
                    "source_id": segment.source.source_id,
                    "version": segment.document.version,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "method": segment.method,
                    "table_headers": segment.table_headers,
                }
                for segment in window.segments
            ],
            "next_cursor": asdict(window.next_cursor) if window.next_cursor else None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
