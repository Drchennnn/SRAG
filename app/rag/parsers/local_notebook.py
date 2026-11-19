"""Read-only, position-stable parsing for locally imported notebooks."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.schemas.local import ParsedSource, SourceDraft

_CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_HEADING_PATTERN = re.compile(r"^\s*(=+|-+)\s*$")
_FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text without evaluating HTML content."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in {"br", "div", "li", "p", "tr"}:
            self.parts.append("\n")
        elif not self.ignored_depth and tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split()).replace(" |  | ", " | ").strip()


def parse_local_notebook(path: Path) -> ParsedSource:
    """Parse notebook JSON without executing cells or resolving external resources."""

    source_path = path.expanduser().resolve()
    if source_path.suffix.lower() != ".ipynb":
        raise ValueError(f"只支持 .ipynb 文件：{source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Notebook 文件不存在：{source_path}")

    try:
        notebook = json.loads(source_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Notebook 不是有效的 UTF-8 JSON：{source_path}") from exc

    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise ValueError("Notebook 顶层 cells 必须是数组")

    valid_cell_ids = _valid_unique_cell_ids(cells)
    sources: list[SourceDraft] = []
    section_path: list[str] = []
    document_title = source_path.stem
    ordinal = 0

    for cell_index, raw_cell in enumerate(cells):
        cell = raw_cell if isinstance(raw_cell, dict) else {}
        cell_type = cell.get("cell_type") if isinstance(cell.get("cell_type"), str) else "unknown"
        cell_id = valid_cell_ids.get(cell_index)
        source = _coerce_text(cell.get("source", ""))
        headings: list[dict[str, Any]] = []
        if cell_type == "markdown":
            headings, section_path = _read_headings(source, section_path)
            if headings and document_title == source_path.stem:
                document_title = str(headings[0]["title"])

        outputs_present = "outputs" in cell
        raw_outputs = cell.get("outputs")
        outputs = raw_outputs if cell_type == "code" and isinstance(raw_outputs, list) else []
        outputs_state = _outputs_state(cell_type, outputs_present, raw_outputs)
        content_state = "available" if source else "empty"
        title = _cell_title(cell_type, cell_index, headings, section_path)
        locator: dict[str, Any] = {
            "cell_index": cell_index,
            "cell_id": cell_id,
            "cell_type": cell_type,
            "section_path": list(section_path),
            "content_state": content_state,
            "outputs_state": outputs_state,
            "json_pointer": f"/cells/{cell_index}",
        }
        if headings:
            locator["headings"] = headings
        if not isinstance(raw_cell, dict):
            locator["parse_error"] = "cell_not_object"

        cell_source_id = f"cell:{cell_index}"
        sources.append(
            SourceDraft(
                source_id=cell_source_id,
                kind="cell",
                ordinal=ordinal,
                title=title,
                text=source,
                locator=locator,
                state="available" if source or headings else "empty",
            )
        )
        ordinal += 1

        for output_index, raw_output in enumerate(outputs):
            output_source, output_state, output_locator = _parse_output(
                raw_output,
                cell_index=cell_index,
                cell_id=cell_id,
                output_index=output_index,
                section_path=section_path,
            )
            sources.append(
                SourceDraft(
                    source_id=f"cell:{cell_index}:output:{output_index}",
                    kind="notebook_output",
                    ordinal=ordinal,
                    parent_id=cell_source_id,
                    title=f"{title} · output {output_index + 1}",
                    text=output_source,
                    locator=output_locator,
                    state=output_state,
                )
            )
            ordinal += 1

    return ParsedSource(title=document_title, sources=sources)


def _valid_unique_cell_ids(cells: list[Any]) -> dict[int, str]:
    candidates: dict[int, str] = {}
    counts: dict[str, int] = {}
    for index, raw_cell in enumerate(cells):
        if not isinstance(raw_cell, dict):
            continue
        value = raw_cell.get("id")
        if isinstance(value, str) and _CELL_ID_PATTERN.fullmatch(value):
            candidates[index] = value
            counts[value] = counts.get(value, 0) + 1
    return {index: value for index, value in candidates.items() if counts[value] == 1}


def _outputs_state(cell_type: str, present: bool, raw_outputs: Any) -> str:
    if cell_type != "code":
        return "unexpected" if present else "not_applicable"
    if not present:
        return "missing"
    if not isinstance(raw_outputs, list):
        return "malformed"
    return "available" if raw_outputs else "empty"


def _cell_title(
    cell_type: str,
    cell_index: int,
    headings: list[dict[str, Any]],
    section_path: list[str],
) -> str:
    if headings:
        return str(headings[-1]["title"])
    if section_path:
        return section_path[-1]
    return f"{cell_type.capitalize()} cell {cell_index + 1}"


def _read_headings(source: str, previous_path: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    headings: list[dict[str, Any]] = []
    current_path = list(previous_path)
    lines = source.splitlines()
    index = 0
    fence_character: str | None = None
    fence_length = 0
    while index < len(lines):
        fence = _FENCE_PATTERN.match(lines[index])
        if fence:
            marker = fence.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            index += 1
            continue
        if fence_character is not None:
            index += 1
            continue
        atx = _ATX_HEADING_PATTERN.match(lines[index])
        setext = (
            _SETEXT_HEADING_PATTERN.match(lines[index + 1])
            if index + 1 < len(lines) and lines[index].strip()
            else None
        )
        if atx:
            level = len(atx.group(1))
            title = atx.group(2).strip()
        elif setext:
            level = 1 if setext.group(1).startswith("=") else 2
            title = lines[index].strip()
            index += 1
        else:
            index += 1
            continue
        current_path = [*current_path[: level - 1], title]
        headings.append({"level": level, "title": title, "section_path": list(current_path)})
        index += 1
    return headings, current_path


def _parse_output(
    raw_output: Any,
    *,
    cell_index: int,
    cell_id: str | None,
    output_index: int,
    section_path: list[str],
) -> tuple[str, str, dict[str, Any]]:
    pointer = f"/cells/{cell_index}/outputs/{output_index}"
    locator: dict[str, Any] = {
        "cell_index": cell_index,
        "cell_id": cell_id,
        "output_index": output_index,
        "section_path": list(section_path),
        "json_pointer": pointer,
        "representations": [],
    }
    if not isinstance(raw_output, dict):
        locator["parse_error"] = "output_not_object"
        return "", "malformed", locator

    raw_output_type = raw_output.get("output_type")
    output_type = raw_output_type if isinstance(raw_output_type, str) else "unknown"
    locator["output_type"] = output_type
    text = ""
    if output_type == "stream":
        text = _coerce_text(raw_output.get("text", ""))
        locator["stream_name"] = raw_output.get("name")
    elif output_type == "error":
        traceback = _coerce_lines(raw_output.get("traceback", []))
        summary = f"{raw_output.get('ename', 'Error')}: {raw_output.get('evalue', '')}".strip()
        text = "\n".join(part for part in (summary, traceback) if part)
    elif output_type in {"display_data", "execute_result"}:
        data = raw_output.get("data")
        if isinstance(data, dict):
            locator["representations"] = [
                _representation_record(pointer, mime_type, value)
                for mime_type, value in data.items()
            ]
            text = _readable_bundle_text(data)
        else:
            locator["parse_error"] = "data_not_object"
    else:
        text = _coerce_text(raw_output.get("text", ""))

    text = _ANSI_PATTERN.sub("", text)
    representations = locator["representations"]
    malformed = any(item.get("status") == "malformed" for item in representations)
    usable = bool(text) or any(item.get("status") == "available" for item in representations)
    if malformed and usable:
        state = "partial"
    elif malformed or locator.get("parse_error"):
        state = "malformed"
    elif usable:
        state = "available"
    else:
        state = "empty"
    return text, state, locator


def _representation_record(pointer: str, mime_type: Any, value: Any) -> dict[str, Any]:
    mime = mime_type if isinstance(mime_type, str) else str(mime_type)
    record: dict[str, Any] = {
        "mime_type": mime,
        "json_pointer": f"{pointer}/data/{_json_pointer_escape(mime)}",
    }
    if mime.startswith("image/"):
        try:
            payload, encoding = _image_bytes(mime, value)
        except (ValueError, binascii.Error, UnicodeEncodeError):
            record.update(status="malformed", encoding="base64", error="invalid_image_data")
            return record
    else:
        payload, encoding = _representation_bytes(value)
    record.update(
        status="available" if payload else "empty",
        encoding=encoding,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return record


def _image_bytes(mime_type: str, value: Any) -> tuple[bytes, str]:
    raw = _coerce_text(value)
    if mime_type == "image/svg+xml":
        return raw.encode("utf-8"), "utf-8"
    compact = "".join(raw.split())
    return base64.b64decode(compact, validate=True), "base64"


def _representation_bytes(value: Any) -> tuple[bytes, str]:
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return rendered.encode("utf-8"), "json"
    return _coerce_text(value).encode("utf-8"), "utf-8"


def _readable_bundle_text(data: dict[str, Any]) -> str:
    if "text/markdown" in data:
        return _coerce_text(data["text/markdown"])
    if "text/html" in data:
        parser = _HTMLTextExtractor()
        parser.feed(_coerce_text(data["text/html"]))
        return parser.text()
    if "text/plain" in data:
        return _coerce_text(data["text/plain"])
    if "application/json" in data:
        return json.dumps(data["application/json"], ensure_ascii=False, sort_keys=True)
    return ""


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _coerce_lines(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return _coerce_text(value)


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
