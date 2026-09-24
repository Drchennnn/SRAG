"""在不执行代码的前提下读取 Jupyter Notebook。"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import nbformat

from app.config import Settings
from app.schemas.document import AssetRef, BlockKind, ContentBlock, ParsedDocument

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
    "image/gif": ".gif",
}


class _HTMLTextExtractor(HTMLParser):
    """将 HTML 输出转换为适合检索的纯文本。"""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split()).replace(" |  | ", " | ").strip()


class NotebookParser:
    """将 Notebook cell 及输出转换为统一文档结构。"""

    def __init__(self, asset_dir: Path, *, output_max_chars: int = 12000) -> None:
        if output_max_chars <= 0:
            raise ValueError("Notebook 输出字符上限必须大于 0")
        self.asset_dir = asset_dir
        self.output_max_chars = output_max_chars

    @classmethod
    def from_settings(cls, settings: Settings) -> NotebookParser:
        """从项目配置创建 Notebook 解析器。"""

        return cls(
            settings.data_dir / "assets",
            output_max_chars=getattr(settings, "notebook_output_max_chars", 12000),
        )

    def parse(self, path: Path, document_id: str | None = None) -> ParsedDocument:
        """只读解析一个 ipynb 文件，不执行任何 cell。"""

        source_path = path.expanduser().resolve()
        if source_path.suffix.lower() != ".ipynb":
            raise ValueError(f"NotebookParser 只支持 .ipynb 文件：{source_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Notebook 文件不存在：{source_path}")

        notebook = nbformat.read(source_path, as_version=4)
        resolved_document_id = document_id or _stable_id("document", source_path.as_posix())
        blocks: list[ContentBlock] = []
        section_path: list[str] = []
        title = source_path.stem

        for cell_index, cell in enumerate(notebook.cells):
            cell_id = str(cell.get("id") or f"cell-{cell_index}")
            cell_type = cell.get("cell_type")
            source = _coerce_text(cell.get("source", ""))

            if cell_type == "markdown":
                markdown_blocks, section_path = self._parse_markdown_cell(
                    source=source,
                    attachments=cell.get("attachments", {}),
                    document_id=resolved_document_id,
                    cell_id=cell_id,
                    cell_index=cell_index,
                    section_path=section_path,
                )
                blocks.extend(markdown_blocks)
                if title == source_path.stem:
                    first_heading = next(
                        (block.section_path[0] for block in markdown_blocks if block.section_path),
                        None,
                    )
                    if first_heading:
                        title = first_heading
                continue

            if cell_type == "code":
                code_block_id = f"{cell_id}:code"
                if source.strip():
                    blocks.append(
                        ContentBlock(
                            block_id=code_block_id,
                            document_id=resolved_document_id,
                            kind=BlockKind.CODE,
                            text=source,
                            section_path=list(section_path),
                            cell_id=cell_id,
                            cell_index=cell_index,
                        )
                    )
                blocks.extend(
                    self._parse_outputs(
                        outputs=cell.get("outputs", []),
                        document_id=resolved_document_id,
                        cell_id=cell_id,
                        cell_index=cell_index,
                        section_path=section_path,
                        related_block_id=code_block_id,
                    )
                )
                continue

            if source.strip():
                blocks.append(
                    ContentBlock(
                        block_id=f"{cell_id}:text",
                        document_id=resolved_document_id,
                        kind=BlockKind.TEXT,
                        text=source,
                        section_path=list(section_path),
                        cell_id=cell_id,
                        cell_index=cell_index,
                    )
                )

        return ParsedDocument(
            document_id=resolved_document_id,
            source_path=source_path.as_posix(),
            title=title,
            blocks=blocks,
        )

    def _parse_markdown_cell(
        self,
        *,
        source: str,
        attachments: dict[str, Any],
        document_id: str,
        cell_id: str,
        cell_index: int,
        section_path: list[str],
    ) -> tuple[list[ContentBlock], list[str]]:
        assets = self._extract_attachments(document_id, cell_id, attachments)
        current_path = list(section_path)
        pieces: list[tuple[str, list[str]]] = []
        buffer: list[str] = []
        buffer_path = list(current_path)

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                pieces.append((text, list(buffer_path)))
            buffer.clear()

        for line in source.splitlines():
            match = _HEADING_PATTERN.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                current_path = current_path[: level - 1]
                current_path.append(heading)
                buffer_path = list(current_path)
            buffer.append(line)
        flush()

        if not pieces and (source.strip() or assets):
            pieces.append((source.strip(), list(current_path)))

        blocks: list[ContentBlock] = []
        for piece_index, (text, piece_path) in enumerate(pieces):
            piece_assets = [
                asset for asset in assets if f"attachment:{Path(asset.path).name}" in text
            ]
            if not piece_assets and piece_index == 0:
                piece_assets = assets
            blocks.append(
                ContentBlock(
                    block_id=f"{cell_id}:markdown:{piece_index}",
                    document_id=document_id,
                    kind=BlockKind.MARKDOWN,
                    text=text,
                    section_path=piece_path,
                    cell_id=cell_id,
                    cell_index=cell_index,
                    assets=piece_assets,
                )
            )
        return blocks, current_path

    def _parse_outputs(
        self,
        *,
        outputs: list[Any],
        document_id: str,
        cell_id: str,
        cell_index: int,
        section_path: list[str],
        related_block_id: str,
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        for output_index, output in enumerate(outputs):
            output_type = output.get("output_type")
            text = ""
            assets: list[AssetRef] = []
            kind = BlockKind.OUTPUT

            if output_type == "stream":
                text = _coerce_text(output.get("text", ""))
            elif output_type == "error":
                traceback = "\n".join(output.get("traceback", []))
                text = "\n".join(
                    part
                    for part in (
                        f"{output.get('ename', 'Error')}: {output.get('evalue', '')}",
                        traceback,
                    )
                    if part.strip()
                )
            elif output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                text, kind = _extract_display_text(data)
                assets = self._extract_output_assets(
                    document_id=document_id,
                    cell_id=cell_id,
                    output_index=output_index,
                    data=data,
                )
                if assets and not text:
                    kind = BlockKind.FIGURE

            text = _truncate(text, self.output_max_chars)
            if text.strip() or assets:
                blocks.append(
                    ContentBlock(
                        block_id=f"{cell_id}:output:{output_index}",
                        document_id=document_id,
                        kind=kind,
                        text=text,
                        section_path=list(section_path),
                        cell_id=cell_id,
                        cell_index=cell_index,
                        related_block_id=related_block_id,
                        assets=assets,
                    )
                )
        return blocks

    def _extract_attachments(
        self,
        document_id: str,
        cell_id: str,
        attachments: dict[str, Any],
    ) -> list[AssetRef]:
        assets: list[AssetRef] = []
        for name, mime_bundle in attachments.items():
            for media_type, data in mime_bundle.items():
                if media_type in _IMAGE_EXTENSIONS:
                    assets.append(
                        self._save_asset(
                            document_id=document_id,
                            cell_id=cell_id,
                            name=name,
                            media_type=media_type,
                            data=data,
                        )
                    )
                    break
        return assets

    def _extract_output_assets(
        self,
        *,
        document_id: str,
        cell_id: str,
        output_index: int,
        data: dict[str, Any],
    ) -> list[AssetRef]:
        return [
            self._save_asset(
                document_id=document_id,
                cell_id=cell_id,
                name=f"output-{output_index}{_IMAGE_EXTENSIONS[media_type]}",
                media_type=media_type,
                data=value,
            )
            for media_type, value in data.items()
            if media_type in _IMAGE_EXTENSIONS
        ]

    def _save_asset(
        self,
        *,
        document_id: str,
        cell_id: str,
        name: str,
        media_type: str,
        data: Any,
    ) -> AssetRef:
        safe_name = _SAFE_NAME_PATTERN.sub("_", Path(name).name)
        extension = _IMAGE_EXTENSIONS[media_type]
        if not safe_name.lower().endswith(extension):
            safe_name = f"{safe_name}{extension}"
        directory = self.asset_dir / document_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{_SAFE_NAME_PATTERN.sub('_', cell_id)}-{safe_name}"

        if media_type == "image/svg+xml":
            target.write_text(_coerce_text(data), encoding="utf-8")
        else:
            target.write_bytes(base64.b64decode(_coerce_text(data)))

        return AssetRef(
            asset_id=_stable_id("asset", document_id, cell_id, name, media_type),
            media_type=media_type,
            path=target.as_posix(),
            caption=Path(name).stem,
        )


def _extract_display_text(data: dict[str, Any]) -> tuple[str, BlockKind]:
    if "text/markdown" in data:
        return _coerce_text(data["text/markdown"]), BlockKind.OUTPUT
    if "text/html" in data:
        html = _coerce_text(data["text/html"])
        parser = _HTMLTextExtractor()
        parser.feed(html)
        kind = BlockKind.TABLE if "<table" in html.lower() else BlockKind.OUTPUT
        return parser.text(), kind
    if "text/plain" in data:
        return _coerce_text(data["text/plain"]), BlockKind.OUTPUT
    if "application/json" in data:
        return json.dumps(data["application/json"], ensure_ascii=False), BlockKind.OUTPUT
    return "", BlockKind.OUTPUT


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[输出已截断，省略 {omitted} 个字符]"


def _stable_id(kind: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{kind}_{digest}"
