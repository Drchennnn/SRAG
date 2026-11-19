"""Read PDF and PPTX snapshots into the B01 local-source contract."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Any

from app.schemas.local import ParsedSource, SourceDraft

_PDF_POSITION_EPSILON = 3.5
_PDFIUM_LOCK = Lock()


def parse_local_slides(path: Path) -> ParsedSource:
    """Parse a PDF or PPTX without rendering pages, running OCR, or executing content."""

    snapshot = path.expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"local slide snapshot does not exist: {snapshot}")
    suffix = snapshot.suffix.casefold()
    if suffix == ".pdf":
        return _parse_pdf(snapshot)
    if suffix == ".pptx":
        return _parse_pptx(snapshot)
    raise ValueError(f"unsupported local slide format: {snapshot.suffix or '<none>'}")


def _parse_pdf(path: Path) -> ParsedSource:
    with _PDFIUM_LOCK:
        return _parse_pdf_locked(path)


def _parse_pdf_locked(path: Path) -> ParsedSource:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - optional installation boundary
        raise RuntimeError("PDF local import requires pypdfium2") from exc

    document = pdfium.PdfDocument(path)
    sources: list[SourceDraft] = []
    ordinal = 0
    try:
        for page_index in range(len(document)):
            page_no = page_index + 1
            page = document[page_index]
            try:
                text_page = None
                text_page = page.get_textpage()
                try:
                    native_text = _normalize_pdf_text(text_page.get_text_range())
                    page_bbox = page.get_bbox()
                    page_rotation = page.get_rotation()
                    has_image = any(
                        True for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])
                    )
                    page_state = (
                        "available"
                        if native_text
                        else ("needs_ocr" if has_image else "no_native_text")
                    )
                    title = _first_nonempty_line(native_text)
                    page_source_id = f"page:{page_no}"
                    sources.append(
                        SourceDraft(
                            source_id=page_source_id,
                            kind="page",
                            ordinal=ordinal,
                            title=title,
                            text=native_text,
                            locator={
                                "format": "pdf",
                                "page": page_no,
                                "native_text": bool(native_text),
                                "rotation_degrees_clockwise": page_rotation,
                                "visible_page_bbox_pdf_units": list(page_bbox),
                                "render_status": "not_attempted",
                                "ocr_status": "not_attempted",
                            },
                            state=page_state,
                        )
                    )
                    ordinal += 1

                    if page_rotation != 0:
                        continue
                    for table_index, table in enumerate(
                        _confirmed_pdf_tables(page, text_page, page_bbox, pdfium), start=1
                    ):
                        sources.append(
                            SourceDraft(
                                source_id=f"page:{page_no}:table:{table_index}",
                                kind="page_object",
                                ordinal=ordinal,
                                parent_id=page_source_id,
                                title=table["title"],
                                text=_table_to_markdown(table["rows"]),
                                locator={
                                    "format": "pdf",
                                    "page": page_no,
                                    "table_index": table_index,
                                    "object_type": "table",
                                    "bbox": table["bbox"],
                                    "native_bbox": table["native_bbox"],
                                    "coordinate_frame": "visible_page_norm_v1",
                                    "native_coordinate_frame": "pdf_page_units_bottom_left_v1",
                                    "native_to_visible_transform": _pdf_transform(page_bbox),
                                    "position_basis": "axis_aligned_native_rule_grid_v1",
                                    "native_content_order": table["native_content_order"],
                                    "columns": table["rows"][0],
                                    "rows": table["rows"],
                                },
                            )
                        )
                        ordinal += 1
                finally:
                    if text_page is not None:
                        text_page.close()
            finally:
                page.close()
    finally:
        document.close()
    return ParsedSource(title=path.stem, sources=sources)


def _parse_pptx(path: Path) -> ParsedSource:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover - optional installation boundary
        raise RuntimeError("PPTX local import requires python-pptx") from exc

    presentation = Presentation(path)
    slide_width = int(presentation.slide_width or 0)
    slide_height = int(presentation.slide_height or 0)
    sources: list[SourceDraft] = []
    ordinal = 0
    for page_no, slide in enumerate(presentation.slides, start=1):
        entries = list(_iter_pptx_shapes(slide.shapes))
        native_text = "\n".join(value for _, shape in entries if (value := _pptx_shape_text(shape)))
        has_picture = any(_pptx_object_type(shape) == "picture" for _, shape in entries)
        title = _pptx_title(slide, entries, slide_height)
        page_source_id = f"page:{page_no}"
        sources.append(
            SourceDraft(
                source_id=page_source_id,
                kind="page",
                ordinal=ordinal,
                title=title,
                text=native_text,
                locator={
                    "format": "pptx",
                    "page": page_no,
                    "native_text": bool(native_text),
                    "render_status": "not_attempted",
                    "slide_size_emu": [slide_width, slide_height],
                    "native_to_visible_transform": _pptx_transform(slide_width, slide_height),
                },
                state=(
                    "available"
                    if native_text
                    else ("needs_ocr" if has_picture else "no_native_text")
                ),
            )
        )
        ordinal += 1

        for shape_id_path, shape in entries:
            object_type = _pptx_object_type(shape)
            text = _pptx_shape_text(shape)
            if not text and object_type not in {"picture", "chart", "table"}:
                continue
            locator: dict[str, Any] = {
                "format": "pptx",
                "page": page_no,
                "shape_id_path": shape_id_path,
                "object_type": object_type,
                "native_coordinate_frame": "pptx_slide_emu_v1",
            }
            bbox, position_basis = _pptx_bbox(
                shape,
                shape_id_path,
                slide_width=slide_width,
                slide_height=slide_height,
            )
            locator["position_basis"] = position_basis
            if bbox is not None:
                locator["bbox"] = bbox
                locator["coordinate_frame"] = "visible_page_norm_v1"
                locator["native_bbox"] = [
                    int(shape.left),
                    int(shape.top),
                    int(shape.left + shape.width),
                    int(shape.top + shape.height),
                ]
                locator["native_to_visible_transform"] = _pptx_transform(slide_width, slide_height)
            if object_type == "table":
                rows = _pptx_table_rows(shape)
                locator["columns"] = rows[0] if rows else []
                locator["rows"] = rows
            sources.append(
                SourceDraft(
                    source_id=(
                        f"page:{page_no}:shape:"
                        + ".".join(str(shape_id) for shape_id in shape_id_path)
                    ),
                    kind="page_object",
                    ordinal=ordinal,
                    parent_id=page_source_id,
                    title=text if object_type == "title" else "",
                    text=text,
                    locator=locator,
                )
            )
            ordinal += 1
    return ParsedSource(title=path.stem, sources=sources)


def _iter_pptx_shapes(shapes: Any, prefix: tuple[int, ...] = ()) -> Iterable[tuple[list[int], Any]]:
    for shape in shapes:
        path = (*prefix, int(shape.shape_id))
        if int(getattr(shape, "shape_type", 0)) == 6:
            yield from _iter_pptx_shapes(shape.shapes, path)
            continue
        yield list(path), shape


def _pptx_title(slide: Any, entries: list[tuple[list[int], Any]], slide_height: int) -> str:
    title_shape = slide.shapes.title
    if title_shape is not None and (text := getattr(title_shape, "text", "").strip()):
        return text
    candidates: list[tuple[int, str]] = []
    for shape_path, shape in entries:
        text = getattr(shape, "text", "").strip()
        if not text or len(text) > 160 or len(shape_path) != 1 or not slide_height:
            continue
        top = int(getattr(shape, "top", slide_height))
        if top / slide_height <= 0.2:
            candidates.append((top, text))
    return min(candidates)[1] if candidates else ""


def _pptx_object_type(shape: Any) -> str:
    if getattr(shape, "has_table", False):
        return "table"
    if getattr(shape, "has_chart", False):
        return "chart"
    if int(getattr(shape, "shape_type", 0)) == 13:
        return "picture"
    if getattr(shape, "is_placeholder", False) and getattr(shape, "placeholder_format", None):
        if getattr(shape.placeholder_format, "type", None) in {1, 3}:
            return "title"
    return "text" if getattr(shape, "text", "").strip() else "other"


def _pptx_shape_text(shape: Any) -> str:
    if getattr(shape, "has_table", False):
        return _table_to_markdown(_pptx_table_rows(shape))
    if getattr(shape, "has_chart", False):
        parts: list[str] = []
        chart = shape.chart
        if chart.has_title and chart.chart_title.has_text_frame:
            parts.append(chart.chart_title.text_frame.text.strip())
        parts.extend(
            series.name.strip()
            for series in chart.series
            if isinstance(series.name, str) and series.name.strip()
        )
        return "\n".join(dict.fromkeys(part for part in parts if part))
    return getattr(shape, "text", "").strip()


def _pptx_table_rows(shape: Any) -> list[list[str]]:
    return [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]


def _pptx_bbox(
    shape: Any,
    shape_id_path: list[int],
    *,
    slide_width: int,
    slide_height: int,
) -> tuple[list[float] | None, str]:
    if len(shape_id_path) != 1:
        return None, "group_transform_unverified"
    if float(getattr(shape, "rotation", 0.0) or 0.0) % 360:
        return None, "rotated_shape_transform_unverified"
    if not slide_width or not slide_height:
        return None, "slide_dimensions_unavailable"
    left = int(shape.left)
    top = int(shape.top)
    right = left + int(shape.width)
    bottom = top + int(shape.height)
    if right < left or bottom < top:
        return None, "invalid_native_extents"
    normalized = [
        _unit_interval(left / slide_width),
        _unit_interval(top / slide_height),
        _unit_interval(right / slide_width),
        _unit_interval(bottom / slide_height),
    ]
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None, "outside_slide_or_degenerate_after_clamp"
    return normalized, "native_shape_extents_v1"


def _confirmed_pdf_tables(
    page: Any,
    text_page: Any,
    page_bbox: tuple[float, float, float, float],
    pdfium: Any,
) -> list[dict[str, Any]]:
    horizontal: list[tuple[float, float, float, int]] = []
    vertical: list[tuple[float, float, float, int]] = []
    for native_order, obj in enumerate(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH])):
        left, bottom, right, top = obj.get_bounds()
        width = max(0.0, right - left)
        height = max(0.0, top - bottom)
        if width >= 12 and height <= _PDF_POSITION_EPSILON:
            horizontal.append((left, right, (bottom + top) / 2, native_order))
        elif height >= 12 and width <= _PDF_POSITION_EPSILON:
            vertical.append(((left + right) / 2, bottom, top, native_order))
    if len(horizontal) < 3 or len(vertical) < 3:
        return []

    tables = [
        table
        for component_horizontal, component_vertical in _pdf_grid_components(horizontal, vertical)
        if (
            table := _pdf_table_from_grid(
                component_horizontal,
                component_vertical,
                text_page,
                page_bbox,
            )
        )
        is not None
    ]
    return sorted(
        tables,
        key=lambda table: (
            -table["native_bbox"][3],
            table["native_bbox"][0],
            table["native_content_order"],
        ),
    )


def _pdf_grid_components(
    horizontal: list[tuple[float, float, float, int]],
    vertical: list[tuple[float, float, float, int]],
) -> list[
    tuple[
        list[tuple[float, float, float, int]],
        list[tuple[float, float, float, int]],
    ]
]:
    neighbors: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for horizontal_index, h_line in enumerate(horizontal):
        h_key = ("h", horizontal_index)
        neighbors.setdefault(h_key, set())
        for vertical_index, v_line in enumerate(vertical):
            if not _lines_intersect(h_line, v_line):
                continue
            v_key = ("v", vertical_index)
            neighbors[h_key].add(v_key)
            neighbors.setdefault(v_key, set()).add(h_key)

    components = []
    remaining = set(neighbors)
    while remaining:
        pending = [remaining.pop()]
        component: set[tuple[str, int]] = set()
        while pending:
            key = pending.pop()
            if key in component:
                continue
            component.add(key)
            new_keys = neighbors[key] - component
            pending.extend(new_keys)
            remaining.difference_update(new_keys)
        component_horizontal = [horizontal[index] for kind, index in component if kind == "h"]
        component_vertical = [vertical[index] for kind, index in component if kind == "v"]
        if len(component_horizontal) >= 3 and len(component_vertical) >= 3:
            components.append((component_horizontal, component_vertical))
    return components


def _lines_intersect(
    horizontal: tuple[float, float, float, int],
    vertical: tuple[float, float, float, int],
) -> bool:
    left, right, y_value, _ = horizontal
    x_value, bottom, top, _ = vertical
    return (
        left - _PDF_POSITION_EPSILON <= x_value <= right + _PDF_POSITION_EPSILON
        and bottom - _PDF_POSITION_EPSILON <= y_value <= top + _PDF_POSITION_EPSILON
    )


def _pdf_table_from_grid(
    horizontal: list[tuple[float, float, float, int]],
    vertical: list[tuple[float, float, float, int]],
    text_page: Any,
    page_bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    x_values = _cluster_values([line[0] for line in vertical])
    y_values = _cluster_values([line[2] for line in horizontal])
    if len(x_values) < 3 or len(y_values) < 3:
        return None
    x_values.sort()
    y_values.sort(reverse=True)
    left, right = x_values[0], x_values[-1]
    bottom, top = y_values[-1], y_values[0]
    if not all(_covers(line[0], line[1], left, right) for line in horizontal):
        return None
    if not all(_covers(line[1], line[2], bottom, top) for line in vertical):
        return None

    rows: list[list[str]] = []
    for row_top, row_bottom in zip(y_values, y_values[1:], strict=False):
        row = [
            _normalize_pdf_text(
                text_page.get_text_bounded(
                    column_left + 0.5,
                    row_bottom + 0.5,
                    column_right - 0.5,
                    row_top - 0.5,
                )
            ).replace("\n", " ")
            for column_left, column_right in zip(x_values, x_values[1:], strict=False)
        ]
        rows.append(row)
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    if sum(bool(value) for value in rows[0]) < 2:
        return None
    if not any(sum(bool(value) for value in row) >= 2 for row in rows[1:]):
        return None

    native_orders = [line[3] for line in horizontal] + [line[3] for line in vertical]
    return {
        "title": " | ".join(value for value in rows[0] if value),
        "rows": rows,
        "bbox": _pdf_bbox_to_visible([left, bottom, right, top], page_bbox),
        "native_bbox": [left, bottom, right, top],
        "native_content_order": min(native_orders),
    }


def _cluster_values(values: list[float]) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and abs(value - clusters[-1][-1]) <= _PDF_POSITION_EPSILON:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _covers(start: float, end: float, expected_start: float, expected_end: float) -> bool:
    return (
        start <= expected_start + _PDF_POSITION_EPSILON
        and end >= expected_end - _PDF_POSITION_EPSILON
    )


def _pdf_bbox_to_visible(
    bbox: list[float], page_bbox: tuple[float, float, float, float]
) -> list[float]:
    page_left, page_bottom, page_right, page_top = page_bbox
    width = page_right - page_left
    height = page_top - page_bottom
    left, bottom, right, top = bbox
    return [
        _unit_interval((left - page_left) / width),
        _unit_interval((page_top - top) / height),
        _unit_interval((right - page_left) / width),
        _unit_interval((page_top - bottom) / height),
    ]


def _pdf_transform(page_bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    page_left, page_bottom, page_right, page_top = page_bbox
    width = page_right - page_left
    height = page_top - page_bottom
    return {
        "version": "pdf_cropbox_bottom_left_to_visible_norm_v1",
        "page_bbox": list(page_bbox),
        "matrix": [1 / width, 0.0, 0.0, -1 / height, -page_left / width, page_top / height],
    }


def _pptx_transform(slide_width: int, slide_height: int) -> dict[str, Any] | None:
    if not slide_width or not slide_height:
        return None
    return {
        "version": "pptx_slide_emu_to_visible_norm_v1",
        "slide_size_emu": [slide_width, slide_height],
        "matrix": [1 / slide_width, 0.0, 0.0, 1 / slide_height, 0.0, 0.0],
    }


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _normalize_pdf_text(value: str) -> str:
    lines = value.replace("\r", "\n").splitlines()
    return "\n".join(line.strip() for line in lines if line.strip())


def _first_nonempty_line(value: str) -> str:
    return next((line for line in value.splitlines() if line.strip()), "")


def _unit_interval(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)
