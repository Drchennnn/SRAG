"""Read PDF and PPTX snapshots into the B01 local-source contract."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Any

from app.rag.parsers.pdf_tables import extract_horizontal_rule_tables
from app.rag.pdf_coordinates import Rect, pdf_visible_matrix, transform_rect
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
    images: list[SourceDraft] = []
    extra_objects: list[SourceDraft] = []
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
                    native_images = list(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))
                    has_image = bool(native_images)
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

                    raster_images = _pdf_image_sources(
                        page, native_images, page_no, page_bbox, page_rotation, pdfium
                    )
                    if page_rotation != 0:
                        images.extend(raster_images)
                        continue
                    page_tables = _confirmed_pdf_tables(page, text_page, page_bbox, pdfium)
                    table_boxes = [tuple(table["native_bbox"]) for table in page_tables]
                    raster_boxes = [
                        tuple(img.locator["native_bbox"])
                        for img in raster_images
                        if isinstance(img.locator.get("native_bbox"), list)
                    ]
                    vector_images = _pdf_vector_image_sources(
                        page,
                        text_page,
                        page_no,
                        page_bbox,
                        page_rotation,
                        table_boxes,
                        raster_boxes,
                        pdfium,
                        start_index=len(raster_images) + 1,
                    )
                    page_images = sorted(
                        [*raster_images, *vector_images],
                        key=lambda item: (
                            item.locator["bbox"][1],
                            item.locator["bbox"][0],
                            item.locator["bbox"][3],
                            item.locator["bbox"][2],
                        ),
                    )
                    images.extend(page_images)
                    image_boxes = [
                        tuple(img.locator["native_bbox"])
                        for img in page_images
                        if isinstance(img.locator.get("native_bbox"), list)
                    ]
                    extra_objects.extend(
                        _pdf_text_object_sources(
                            page,
                            text_page,
                            page_no,
                            page_bbox,
                            page_rotation,
                            [*table_boxes, *image_boxes],
                            pdfium,
                        )
                    )
                    for table in page_tables:
                        table_index = table["source_index"]
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
                                    **(
                                        {"table_rule_style": table["table_rule_style"]}
                                        if "table_rule_style" in table
                                        else {}
                                    ),
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
    # Preserve page/table ordinals. Images and text objects follow visible reading order.
    for image in images:
        image.ordinal = len(sources)
        sources.append(image)
    for obj in extra_objects:
        obj.ordinal = len(sources)
        sources.append(obj)
    return ParsedSource(title=path.stem, sources=sources)


def _pdf_image_sources(
    page: Any, images: list[Any], page_no: int, page_bbox: Rect, rotation: int, pdfium: Any
) -> list[SourceDraft]:
    """Register independent top-level rasters with a reliable visible rectangle."""
    if not images or any(obj.level != 0 for obj in images):
        return []
    try:
        matrix = pdf_visible_matrix(page_bbox, rotation)
    except ValueError:
        return []

    try:
        raster_bounds = [(obj, obj.get_bounds()) for obj in images]
        form_bounds = [
            obj.get_bounds()
            for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_FORM], max_depth=1)
        ]
    except pdfium.PdfiumError:
        return []
    if not all(isfinite(value) for _, bounds in raster_bounds for value in bounds) or not all(
        isfinite(value) for bounds in form_bounds for value in bounds
    ):
        return []

    positioned = []
    for obj, bounds in raster_bounds:
        if not _rectangles_overlap(bounds, page_bbox):
            continue
        if not _pdf_image_has_reliable_bounds(obj, bounds, page_bbox, pdfium):
            return []
        # A competing raster or composited layer would make the apparent object order unreliable.
        if any(_rectangles_overlap(bounds, other) for other in form_bounds) or any(
            _rectangles_overlap(bounds, previous) for _, previous in positioned
        ):
            return []
        visible = [_unit_interval(value) for value in transform_rect(bounds, matrix)]
        positioned.append((visible, bounds))

    positioned.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    return [
        SourceDraft(
            source_id=f"page:{page_no}:image:{image_index}",
            kind="page_object",
            ordinal=0,  # Assigned after the unchanged page/table sequence.
            parent_id=f"page:{page_no}",
            locator={
                "format": "pdf",
                "page": page_no,
                "object_type": "image",
                "image_index": image_index,
                "bbox": visible,
                "native_bbox": list(bounds),
                "coordinate_frame": "visible_page_norm_v1",
                "native_coordinate_frame": "pdf_page_units_bottom_left_v1",
                "position_basis": "pdfium_native_image_extents_v1",
                "native_to_visible_transform": {
                    "version": "pdf_cropbox_bottom_left_to_visible_norm_v1",
                    "page_bbox": list(page_bbox),
                    "rotation_degrees_clockwise": rotation,
                    "matrix": list(matrix),
                },
            },
        )
        for image_index, (visible, bounds) in enumerate(positioned, start=1)
    ]


def _pdf_image_has_reliable_bounds(obj, bounds: Rect, crop: Rect, pdfium) -> bool:
    left, bottom, right, top = bounds
    crop_left, crop_bottom, crop_right, crop_top = crop
    if not (crop_left <= left < right <= crop_right and crop_bottom <= bottom < top <= crop_top):
        return False
    if (right - left) >= 0.9 * (crop_right - crop_left) and (top - bottom) >= 0.9 * (
        crop_top - crop_bottom
    ):
        return (
            False  # A page-sized raster is a page-level object, including scans with an OCR layer.
        )
    try:
        matrix = obj.get_matrix()
    except pdfium.PdfiumError:
        return False
    if not (
        all(
            isfinite(value)
            for value in (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
        )
        and (matrix.b == matrix.c == 0 or matrix.a == matrix.d == 0)
    ):
        return False
    clip_path = pdfium.raw.FPDFPageObj_GetClipPath(obj)
    # PDFium returns -1 for an object without a referenced clip path; 0 can include a text clip.
    return (
        bool(clip_path)
        and pdfium.raw.FPDFClipPath_CountPaths(clip_path) == -1
        and not pdfium.raw.FPDFPageObj_HasTransparency(obj)
    )


def _rectangles_overlap(left: Rect, right: Rect) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1], right[1]
    )


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
    if len(horizontal) < 3:
        return []

    tables: list[dict[str, Any]] = []
    if len(vertical) >= 3:
        tables = [
            table
            for component_horizontal, component_vertical in _pdf_grid_components(
                horizontal, vertical
            )
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
    tables.extend(
        extract_horizontal_rule_tables(
            horizontal,
            vertical,
            text_page,
            page_bbox,
            [tuple(table["native_bbox"]) for table in tables],
            _pdf_bbox_to_visible,
        )
    )
    tables.sort(
        key=lambda table: (
            -table["native_bbox"][3],
            table["native_bbox"][0],
            table["native_content_order"],
        ),
    )
    # Grid tables already have public source keys. Allocate new keys after those,
    # even when the new tables occur earlier on the page.
    for index, table in enumerate(sorted(tables, key=lambda t: "table_rule_style" in t), start=1):
        table["source_index"] = index
    return tables


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


_CAPTION_PREFIX = re.compile(
    r"^(?:figure|fig\.?|图)\s*[0-9]+(?:\.[0-9]+)?\s*[:：.\-]",
    re.IGNORECASE,
)
_TABLE_OR_SECTION_PREFIX = re.compile(
    r"^(?:(?:table|figure|fig\.?|表|图)\s*[0-9]+|[0-9]+(?:\.[0-9]+)*\s+[A-Z\u4e00-\u9fff])",
    re.IGNORECASE,
)
_FORMULA_EQ = re.compile(
    r"[A-Za-z_\\][A-Za-z0-9_()]*\s*(?:=|:=|≈|≤|≥|∝|→)\s*\S"
)
_FORMULA_RELATION = re.compile(r"(?:=|:=|≈|≤|≥|∝|→|≡|∼)")
_RELATION_LHS_TOKEN = re.compile(r"[A-Za-z0-9\u0370-\u03ff)\]}]")
_RELATION_RHS_TOKEN = re.compile(r"[A-Za-z0-9\u0370-\u03ff(\[{∑∫∏√+\-]")
_ISOLATED_EQ_NUMBER = re.compile(
    r"^\s*(?:\([0-9]+(?:\.[0-9]+)*[a-z]?\)|\[[0-9]+(?:\.[0-9]+)*\])\s*$"
)
_TRAILING_EQ_NUMBER = re.compile(
    r"(?:\s+(?:\([0-9]+(?:\.[0-9]+)*[a-z]?\)|\[[0-9]+(?:\.[0-9]+)*\]))\s*$"
)
_ALGORITHM_HEADER_LINE = re.compile(
    r"^\s*algorithm\s+[0-9]+(?:\s*[:.\-\s]|$)",
    re.IGNORECASE,
)
_PSEUDOCODE_PREFIX = re.compile(
    r"^\s*(?:algorithm\s+[0-9]+|require\s*:|ensure\s*:|input\s*:|output\s*:|"
    r"[0-9]{1,3}\s*[:.)]\s+\S|(?:for|while|if|elif|else|return|end\s+(?:for|while|if))\b)",
    re.IGNORECASE,
)
_CONDITION_LINE = re.compile(
    r"^(?:condition|where|for\s+all|for\s+any|for\s+odd|for\s+even|for\s+[A-Za-z]|"
    r"with\s+[A-Za-z]|subject\s+to|适用条件|条件|其中|满足)\b",
    re.IGNORECASE,
)
_PROSE_ENGLISH_WORD = re.compile(r"\b[A-Za-z]{3,}\b")
_PROSE_STOPWORD = re.compile(
    r"\b(?:we|the|and|our|set|use|used|using|with|from|into|that|this|these|those|"
    r"have|has|was|were|are|can|will|may|in|on|of|to|by|for|at|as|an|be|been|being|"
    r"not|also|each|every|all|both|such|than|then|when|while|after|before|during|"
    r"between|through|under|over|about|across|observe|shows|shown|figure|table|section)\b",
    re.IGNORECASE,
)
_PROSE_HAN_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_HAN_CHAR = re.compile(r"[\u3400-\u9fff]")
_MATH_TOKEN_RE = re.compile(r"[=+\-*/^_<>≤≥≈∝→≡∼∑∫∏√()\[\]{}|0-9]")
_PAREN_SNIPPET = re.compile(r"^\s*\([^()]{1,40}\)\s*[.,;:]?\s*$")
_MAX_VECTOR_CLUSTERS_PER_PAGE = 12


@dataclass(frozen=True)
class _LineSegment:
    left: float
    bottom: float
    right: float
    top: float
    col: str
    col_right: float
    text: str
    has_eq_number: bool = False

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def yc(self) -> float:
        return (self.bottom + self.top) / 2


def _rect_area(box: Rect) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: Rect, b: Rect) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _overlap_ratio_min(a: Rect, b: Rect) -> float:
    denom = min(_rect_area(a), _rect_area(b))
    return _intersection_area(a, b) / denom if denom > 0 else 0.0


def _overlap_ratio_self(box: Rect, other: Rect) -> float:
    area = _rect_area(box)
    return _intersection_area(box, other) / area if area > 0 else 0.0


def _clamp_box(box: Rect, bounds: Rect) -> Rect:
    return (
        max(bounds[0], box[0]),
        max(bounds[1], box[1]),
        min(bounds[2], box[2]),
        min(bounds[3], box[3]),
    )


def _path_is_complex(raw_obj: Any, pdfium: Any) -> bool:
    """Return True when a PDF path has more than a simple 4-edge rectangle or line."""
    try:
        handle = getattr(raw_obj, "raw", raw_obj)
        count = int(pdfium.raw.FPDFPath_CountSegments(handle))
        return count > 5
    except (AttributeError, TypeError, ValueError, pdfium.PdfiumError):
        return False


def _inspect_form_children(
    form_obj: Any, pdfium: Any, *, depth: int = 0, max_depth: int = 3
) -> tuple[int, int, bool]:
    """Bounded traversal of a Form XObject returning (path_count, image_count, has_complex_path)."""
    if depth > max_depth:
        return 0, 0, False
    handle = getattr(form_obj, "raw", form_obj)
    try:
        count = int(pdfium.raw.FPDFFormObj_CountObjects(handle))
    except (AttributeError, TypeError, ValueError, pdfium.PdfiumError):
        return 0, 0, False
    paths = 0
    images = 0
    complex_path = False
    for idx in range(max(0, count)):
        try:
            child = pdfium.raw.FPDFFormObj_GetObject(handle, idx)
            obj_type = int(pdfium.raw.FPDFPageObj_GetType(child))
        except (AttributeError, TypeError, ValueError, pdfium.PdfiumError):
            continue
        if obj_type == pdfium.raw.FPDF_PAGEOBJ_PATH:
            paths += 1
            if not complex_path and _path_is_complex(child, pdfium):
                complex_path = True
        elif obj_type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            images += 1
        elif obj_type == pdfium.raw.FPDF_PAGEOBJ_FORM and depth < max_depth:
            sub_p, sub_i, sub_c = _inspect_form_children(
                child, pdfium, depth=depth + 1, max_depth=max_depth
            )
            paths += sub_p
            images += sub_i
            complex_path = complex_path or sub_c
    return paths, images, complex_path


def _is_framed_textbox(
    bounds: Rect,
    path_count: int,
    has_complex_path: bool,
    has_inner_image: bool,
    text_boxes: list[Rect],
    text_page: Any,
) -> bool:
    """V3: Reject simple axis-aligned boxes containing multi-line prose/formula text."""
    if has_complex_path or has_inner_image or path_count > 6:
        return False
    cluster_w = max(1.0, bounds[2] - bounds[0])
    inner_boxes = [tb for tb in text_boxes if _overlap_ratio_self(tb, bounds) >= 0.50]
    if not inner_boxes:
        return False
    inner_boxes.sort(key=lambda b: (-(b[1] + b[3]) / 2, b[0]))
    line_spans: list[tuple[float, float, float]] = []
    for tb in inner_boxes:
        yc = (tb[1] + tb[3]) / 2
        if line_spans and abs(yc - line_spans[-1][2]) <= 4.5:
            lx0, lx1, lyc = line_spans[-1]
            line_spans[-1] = (min(lx0, tb[0]), max(lx1, tb[2]), lyc)
        else:
            line_spans.append((tb[0], tb[2], yc))
    if len(line_spans) < 2:
        return False
    max_width_ratio = max((x1 - x0) / cluster_w for x0, x1, _ in line_spans)
    inner_text = _normalize_pdf_text(text_page.get_text_bounded(*bounds))
    has_prose_or_formula = bool(
        _FORMULA_EQ.search(inner_text)
        or _CONDITION_LINE.search(inner_text)
        or "." in inner_text
        or "。" in inner_text
    )
    return max_width_ratio >= 0.65 or has_prose_or_formula


def _pdf_vector_image_sources(
    page: Any,
    text_page: Any,
    page_no: int,
    page_bbox: Rect,
    rotation: int,
    table_boxes: list[Rect],
    raster_boxes: list[Rect],
    pdfium: Any,
    *,
    start_index: int = 1,
) -> list[SourceDraft]:
    """Register vector-drawn PDF subfigures/plots (paths & Form XObjects) with V1-V5 guardrails."""
    try:
        matrix = pdf_visible_matrix(page_bbox, rotation)
        raw_path_objs = list(
            page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH], max_depth=1)
        )
        raw_form_objs = list(
            page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_FORM], max_depth=1)
        )
        raw_text_objs = list(
            page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_TEXT], max_depth=1)
        )
    except (ValueError, pdfium.PdfiumError):
        return []

    page_left, page_bottom, page_right, page_top = page_bbox
    page_w = max(1.0, page_right - page_left)
    page_h = max(1.0, page_top - page_bottom)
    page_area = page_w * page_h

    text_boxes: list[Rect] = []
    for tobj in raw_text_objs:
        try:
            tb = tobj.get_bounds()
        except pdfium.PdfiumError:
            continue
        if all(isfinite(v) for v in tb) and tb[2] > tb[0] and tb[3] > tb[1]:
            text_boxes.append((tb[0], tb[1], tb[2], tb[3]))

    # V2: Pre-filter dividers, full-page frames, header/footer paths, and table/raster hits.
    filtered: list[tuple[Rect, bool]] = []
    for pobj in raw_path_objs:
        try:
            box = pobj.get_bounds()
        except pdfium.PdfiumError:
            continue
        if not all(isfinite(v) for v in box):
            continue
        left, bottom, right, top = box
        if right <= left or top <= bottom:
            continue
        w, h = right - left, top - bottom
        area = w * h
        if area >= 0.65 * page_area:
            continue
        if (w >= 0.88 * page_w and h >= 0.70 * page_h) or (
            h >= 0.88 * page_h and w >= 0.70 * page_w
        ):
            continue
        if (h <= 3.5 and w >= 0.45 * page_w) or (w <= 3.5 and h >= 0.45 * page_h):
            continue
        if (top >= page_top - 0.06 * page_h or bottom <= page_bottom + 0.06 * page_h) and h < 24.0:
            continue
        rect: Rect = (left, bottom, right, top)
        if any(_rectangles_overlap(rect, tb) for tb in table_boxes):
            continue
        if any(_overlap_ratio_self(rect, rb) > 0.10 for rb in raster_boxes):
            continue
        filtered.append((rect, _path_is_complex(pobj, pdfium)))

    # V4: Tighter clustering gap (5 pt) with two-column gutter barrier.
    mid_x = (page_left + page_right) / 2
    gutter_lo, gutter_hi = page_left + 0.46 * page_w, page_left + 0.54 * page_w
    has_clear_gutter = bool(filtered) and not any(
        b[0] < gutter_lo and b[2] > gutter_hi for b, _ in filtered
    )

    gap = 5.0
    used = [False] * len(filtered)
    accepted_boxes: list[Rect] = []
    for i in range(len(filtered)):
        if used[i]:
            continue
        used[i] = True
        first_box, first_complex = filtered[i]
        queue = [first_box]
        count = 0
        has_complex = first_complex
        cl, cb, cr, ct = first_box
        while queue:
            queue.pop()
            count += 1
            for j in range(len(filtered)):
                if used[j]:
                    continue
                jbox, jcomplex = filtered[j]
                jl, jb, jr, jt = jbox
                if has_clear_gutter:
                    cluster_side = (cl + cr) / 2 < mid_x
                    item_side = (jl + jr) / 2 < mid_x
                    crosses_gutter = (cr <= gutter_hi and jl >= gutter_lo) or (
                        jr <= gutter_hi and cl >= gutter_lo
                    )
                    if cluster_side != item_side and crosses_gutter:
                        continue
                if jl <= cr + gap and jr >= cl - gap and jb <= ct + gap and jt >= cb - gap:
                    used[j] = True
                    queue.append(jbox)
                    has_complex = has_complex or jcomplex
                    cl, cb, cr, ct = min(cl, jl), min(cb, jb), max(cr, jr), max(ct, jt)
        if count < 3 or (cr - cl) < 48.0 or (ct - cb) < 48.0:
            continue
        cluster_rect: Rect = _clamp_box((cl, cb, cr, ct), page_bbox)
        if cluster_rect[2] <= cluster_rect[0] or cluster_rect[3] <= cluster_rect[1]:
            continue
        if _rect_area(cluster_rect) > 0.72 * page_area:
            continue
        if any(_overlap_ratio_min(cluster_rect, tb) > 0.10 for tb in table_boxes):
            continue
        if _is_framed_textbox(cluster_rect, count, has_complex, False, text_boxes, text_page):
            continue
        accepted_boxes.append(cluster_rect)

    # V5: Controlled Form XObject support with >=0.90 bleed clamp, <=0.15 overlap & V3 check.
    for fobj in raw_form_objs:
        try:
            raw_bounds = fobj.get_bounds()
        except pdfium.PdfiumError:
            continue
        if not all(isfinite(v) for v in raw_bounds):
            continue
        fb: Rect = (raw_bounds[0], raw_bounds[1], raw_bounds[2], raw_bounds[3])
        form_area = _rect_area(fb)
        if form_area <= 0.0:
            continue
        if _intersection_area(fb, page_bbox) / form_area < 0.90:
            continue
        clamped = _clamp_box(fb, page_bbox)
        cw, ch = clamped[2] - clamped[0], clamped[3] - clamped[1]
        if cw < 48.0 or ch < 48.0 or _rect_area(clamped) > 0.72 * page_area:
            continue
        competing = [*table_boxes, *raster_boxes, *accepted_boxes]
        if any(_overlap_ratio_min(clamped, other) > 0.15 for other in competing):
            continue
        f_paths, f_images, f_complex = _inspect_form_children(fobj, pdfium, max_depth=3)
        if f_paths == 0 and f_images == 0:
            continue
        if _is_framed_textbox(
            clamped, f_paths, f_complex, f_images > 0, text_boxes, text_page
        ):
            continue
        accepted_boxes.append(clamped)

    if not accepted_boxes or len(accepted_boxes) > _MAX_VECTOR_CLUSTERS_PER_PAGE:
        return []

    positioned: list[tuple[list[float], Rect, str]] = []
    for bounds in accepted_boxes:
        visible = [_unit_interval(value) for value in transform_rect(bounds, matrix)]
        left, bottom, right, top = bounds
        title_text = _normalize_pdf_text(text_page.get_text_bounded(left, bottom, right, top))
        positioned.append((visible, bounds, _first_nonempty_line(title_text)))

    positioned.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    return [
        SourceDraft(
            source_id=f"page:{page_no}:image:{image_index}",
            kind="page_object",
            ordinal=0,
            parent_id=f"page:{page_no}",
            title=title,
            text="",
            locator={
                "format": "pdf",
                "page": page_no,
                "object_type": "image",
                "image_index": image_index,
                "bbox": visible,
                "native_bbox": list(bounds),
                "coordinate_frame": "visible_page_norm_v1",
                "native_coordinate_frame": "pdf_page_units_bottom_left_v1",
                "position_basis": "pdfium_native_vector_extents_v1",
                "native_to_visible_transform": {
                    "version": "pdf_cropbox_bottom_left_to_visible_norm_v1",
                    "page_bbox": list(page_bbox),
                    "rotation_degrees_clockwise": rotation,
                    "matrix": list(matrix),
                },
            },
        )
        for image_index, (visible, bounds, title) in enumerate(positioned, start=start_index)
    ]


def _detect_two_column_gutter_from_lines(
    lines: list[_LineSegment], page_bbox: Rect
) -> tuple[float, float, float] | None:
    """Detect (gutter_x, left_edge, right_edge) ONLY after assembling full text lines."""
    page_left, _, page_right, _ = page_bbox
    page_w = max(1.0, page_right - page_left)
    significant = [
        seg
        for seg in lines
        if seg.width >= max(28.0, 0.08 * page_w) and not _ISOLATED_EQ_NUMBER.match(seg.text)
    ]
    if len(significant) < 8:
        return None
    lo, hi = page_left + 0.44 * page_w, page_left + 0.56 * page_w
    left_only = [seg for seg in significant if seg.right <= hi]
    right_only = [seg for seg in significant if seg.left >= lo]
    crossing = [seg for seg in significant if seg.left < lo and seg.right > hi]
    if len(left_only) < 4 or len(right_only) < 4:
        return None
    if len(crossing) > 0.25 * len(significant):
        return None
    left_rights = sorted(seg.right for seg in left_only)
    right_lefts = sorted(seg.left for seg in right_only)
    left_edge = left_rights[int(0.85 * (len(left_rights) - 1))]
    right_edge = right_lefts[int(0.15 * (len(right_lefts) - 1))]
    if right_edge - left_edge < 8.0:
        return None
    return (left_edge + right_edge) / 2, left_edge, right_edge


def _assign_column(box: Rect, gutter_x: float | None, page_bbox: Rect) -> tuple[str, float]:
    page_left, _, page_right, _ = page_bbox
    if gutter_x is None:
        return "full", page_right
    if box[0] < gutter_x - 4.0 and box[2] > gutter_x + 4.0:
        return "full", page_right
    if box[2] <= gutter_x + 4.0:
        return "left", gutter_x
    if box[0] >= gutter_x - 4.0:
        return "right", page_right
    return "full", page_right


def _detect_algorithm_boxes(
    lines: list[_LineSegment],
    path_objs: list[Rect],
    page_bbox: Rect,
    median_h: float,
) -> list[Rect]:
    """Identify Algorithm pseudocode float regions so pseudocode assignments never emit formulas."""
    page_w = max(1.0, page_bbox[2] - page_bbox[0])
    h_rules = [
        b
        for b in path_objs
        if all(isfinite(v) for v in b)
        and (b[3] - b[1]) <= 3.0
        and (b[2] - b[0]) >= 0.30 * page_w
    ]
    algo_boxes: list[Rect] = []
    ordered = sorted(lines, key=lambda s: (-s.yc, s.left))
    for idx, seg in enumerate(ordered):
        if not _ALGORITHM_HEADER_LINE.match(seg.text):
            continue
        # Look for aligned horizontal rules enclosing the Algorithm float (top + bottom rule).
        top_rules = [
            r
            for r in h_rules
            if r[0] <= seg.left + 20.0
            and r[2] >= seg.right - 20.0
            and seg.top - 2.0 <= (r[1] + r[3]) / 2 <= seg.top + 24.0
        ]
        matched_box: Rect | None = None
        if top_rules:
            tr = min(top_rules, key=lambda r: abs((r[1] + r[3]) / 2 - seg.top))
            aligned_below = sorted(
                (
                    r
                    for r in h_rules
                    if abs(r[0] - tr[0]) <= 12.0
                    and abs(r[2] - tr[2]) <= 12.0
                    and (r[1] + r[3]) / 2 < seg.bottom - 10.0
                ),
                key=lambda r: -(r[1] + r[3]) / 2,
            )
            if aligned_below:
                # Algorithm floats typically have a header rule right below title and a bottom rule.
                bottom_rule = aligned_below[1] if len(aligned_below) >= 2 else aligned_below[0]
                matched_box = (
                    min(tr[0], bottom_rule[0]) - 2.0,
                    bottom_rule[1] - 2.0,
                    max(tr[2], bottom_rule[2]) + 2.0,
                    tr[3] + 2.0,
                )
        if matched_box is None:
            # Fallback without rules: scan contiguous pseudocode/line-numbered rows below header.
            bl, bb, br, bt = seg.left, seg.bottom, seg.right, seg.top
            prev_bottom = seg.bottom
            for nxt in ordered[idx + 1 :]:
                if nxt.col != seg.col and seg.col != "full":
                    continue
                vgap = prev_bottom - nxt.top
                if vgap > max(22.0, 2.4 * median_h):
                    break
                if _CAPTION_PREFIX.match(nxt.text) or _ALGORITHM_HEADER_LINE.match(nxt.text):
                    break
                if _PSEUDOCODE_PREFIX.match(nxt.text) or "←" in nxt.text or ":=" in nxt.text:
                    bl = min(bl, nxt.left)
                    bb = min(bb, nxt.bottom)
                    br = max(br, nxt.right)
                    prev_bottom = nxt.bottom
                elif bb < seg.bottom:
                    break
            if bb < seg.bottom:
                matched_box = (bl - 6.0, bb - 4.0, br + 6.0, bt + 4.0)
        if matched_box is not None:
            algo_boxes.append(matched_box)
    return algo_boxes


def _is_short_math_fragment(seg: _LineSegment, col_width: float) -> bool:
    txt = seg.text.strip()
    if not txt or _CAPTION_PREFIX.match(txt) or _TABLE_OR_SECTION_PREFIX.match(txt):
        return False
    prose_words = len(_PROSE_ENGLISH_WORD.findall(txt))
    han_runs = len(_PROSE_HAN_RUN.findall(txt))
    if prose_words >= 5 or han_runs >= 3:
        return False
    if seg.width >= 0.78 * col_width and prose_words >= 3:
        return False
    return bool(_MATH_TOKEN_RE.search(txt) or len(txt) <= 24)


def _has_complete_formula_relation(body_text: str) -> bool:
    """Require at least one relation operator with non-empty valid LHS and RHS operands."""
    cleaned = _TRAILING_EQ_NUMBER.sub("", body_text).strip()
    for m in _FORMULA_RELATION.finditer(cleaned):
        lhs = cleaned[: m.start()].strip()
        rhs = cleaned[m.end() :].strip()
        if (
            lhs
            and rhs
            and _RELATION_LHS_TOKEN.search(lhs)
            and _RELATION_RHS_TOKEN.search(rhs)
        ):
            return True
    return False


def _passes_standalone_formula_gate(
    blk: list[_LineSegment],
    text: str,
    containing_frame: Rect | None,
    *,
    prev_blk: list[_LineSegment] | None = None,
    next_blk: list[_LineSegment] | None = None,
    col_content_left: float = 0.0,
    median_h: float = 10.0,
) -> bool:
    if _CAPTION_PREFIX.match(text) or _TABLE_OR_SECTION_PREFIX.match(text):
        return False
    if _PSEUDOCODE_PREFIX.match(text):
        return False
    body_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not _CONDITION_LINE.match(line.strip())
    ]
    joined = " ".join(body_lines)
    if not _has_complete_formula_relation(joined):
        return False

    prose_words = len(_PROSE_ENGLISH_WORD.findall(joined)) + len(_PROSE_HAN_RUN.findall(joined))
    stop_count = len(_PROSE_STOPWORD.findall(joined))
    han_count = len(_HAN_CHAR.findall(joined))
    has_eq_num = any(seg.has_eq_number for seg in blk) or bool(_TRAILING_EQ_NUMBER.search(text))

    # Tier 1: Explicit equation number (re-associated or trailing).
    if has_eq_num:
        return prose_words <= 8 and stop_count <= 4 and han_count < 14

    # Tier 2: Explicit X4 formula frame box.
    if containing_frame is not None:
        return len(text) <= 1200 and prose_words <= 18

    # Tier 3: Unframed, unnumbered standalone display formula (strict gate).
    compact = re.sub(r"\s+", "", joined)
    if len(compact) < 6 or _PAREN_SNIPPET.match(joined):
        return False
    if len(blk) > 4:
        return False
    if prose_words > 5 or stop_count >= 3 or han_count >= 8:
        return False
    first_non_cond = body_lines[0] if body_lines else ""
    if prose_words >= 3 and first_non_cond.endswith((".", "。")):
        return False

    bl = min(seg.left for seg in blk)
    bb = min(seg.bottom for seg in blk)
    bt = max(seg.top for seg in blk)

    # Reject wrapped body-prose lines embedded inside a flush-left paragraph.
    if len(blk) == 1 and next_blk is not None:
        next_seg = next_blk[0]
        vgap_next = bb - next_seg.top
        if (
            -4.0 <= vgap_next <= max(10.0, 1.5 * median_h)
            and abs(next_seg.left - bl) <= 6.0
            and not _CONDITION_LINE.match(next_seg.text)
            and not _FORMULA_RELATION.search(next_seg.text)
            and (
                next_seg.text[:1].islower()
                or len(_PROSE_ENGLISH_WORD.findall(next_seg.text)) >= 4
            )
        ):
            return False
    if len(blk) == 1 and prev_blk is not None:
        prev_seg = prev_blk[-1]
        vgap_prev = prev_seg.bottom - bt
        if (
            -4.0 <= vgap_prev <= max(8.0, 1.35 * median_h)
            and abs(prev_seg.left - bl) <= 6.0
            and not _FORMULA_RELATION.search(prev_seg.text)
            and len(_PROSE_ENGLISH_WORD.findall(prev_seg.text)) >= 5
            and not prev_seg.text.rstrip().endswith((".", "。", ":", "："))
        ):
            return False

    # Require standalone display positioning evidence (indented/centered, isolated, or multi-line).
    indented = bl >= col_content_left + 10.0
    gap_above = (prev_blk[-1].bottom - bt) if prev_blk is not None else 999.0
    gap_below = (bb - next_blk[0].top) if next_blk is not None else 999.0
    next_is_cond = bool(next_blk is not None and _CONDITION_LINE.match(next_blk[0].text))
    isolated = gap_above >= max(5.0, 0.55 * median_h) and (
        gap_below >= max(5.0, 0.55 * median_h) or next_is_cond
    )
    return indented or isolated or len(blk) >= 2


def _dedupe_text_objects(
    items: list[tuple[list[float], Rect, str]],
) -> list[tuple[list[float], Rect, str]]:
    kept: list[tuple[list[float], Rect, str]] = []
    for visible, bounds, text in sorted(
        items, key=lambda it: (-_rect_area(it[1]), it[0][1], it[0][0])
    ):
        if any(_overlap_ratio_min(bounds, ex_bounds) > 0.50 for _, ex_bounds, _ in kept):
            continue
        kept.append((visible, bounds, text))
    kept.sort(key=lambda item: (item[0][1], item[0][0]))
    return kept


def _pdf_text_object_sources(
    page: Any,
    text_page: Any,
    page_no: int,
    page_bbox: Rect,
    rotation: int,
    excluded_boxes: list[Rect],
    pdfium: Any,
) -> list[SourceDraft]:
    """Register PDF figure captions and standalone formula blocks with X1-X4 guardrails."""
    try:
        matrix = pdf_visible_matrix(page_bbox, rotation)
        text_objs = [
            obj.get_bounds()
            for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_TEXT], max_depth=1)
        ]
        path_objs = [
            obj.get_bounds()
            for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH], max_depth=1)
        ]
    except (ValueError, pdfium.PdfiumError):
        return []

    page_left, page_bottom, page_right, page_top = page_bbox
    page_w = max(1.0, page_right - page_left)
    page_h = max(1.0, page_top - page_bottom)
    page_area = page_w * page_h

    # X3: Exclude text objects only when >35% of the text box overlaps an image/table box.
    raw_boxes: list[Rect] = [
        (box[0], box[1], box[2], box[3])
        for box in text_objs
        if all(isfinite(v) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
        and not any(
            _overlap_ratio_self((box[0], box[1], box[2], box[3]), ex) > 0.35
            for ex in excluded_boxes
        )
    ]
    if not raw_boxes:
        return []

    body_heights = sorted(b[3] - b[1] for b in raw_boxes if (b[3] - b[1]) >= 4.0)
    if not body_heights:
        body_heights = sorted(b[3] - b[1] for b in raw_boxes)
    median_h = max(6.0, min(24.0, body_heights[len(body_heights) // 2]))

    # Step 1 (Assembly-First): Assemble full horizontal lines across the page BEFORE column
    # detection so LaTeX word-level tokens are never falsely split at x=0.50.
    word_merge_thr = max(11.5, 1.35 * median_h)
    y_bucket_thr = max(3.5, 0.42 * median_h)
    sorted_raw = sorted(raw_boxes, key=lambda b: (-(b[1] + b[3]) / 2, b[0]))
    y_buckets: list[list[Rect]] = []
    for box in sorted_raw:
        yc = (box[1] + box[3]) / 2
        if y_buckets:
            ref = y_buckets[-1][0]
            if abs(yc - (ref[1] + ref[3]) / 2) <= y_bucket_thr:
                y_buckets[-1].append(box)
                continue
        y_buckets.append([box])

    assembled: list[_LineSegment] = []
    for bucket in y_buckets:
        bucket.sort(key=lambda b: b[0])
        groups: list[list[Rect]] = [[bucket[0]]]
        for box in bucket[1:]:
            prev_right = max(item[2] for item in groups[-1])
            if box[0] - prev_right > word_merge_thr:
                groups.append([box])
            else:
                groups[-1].append(box)
        for grp in groups:
            gl = min(x[0] for x in grp)
            gb = min(x[1] for x in grp)
            gr = max(x[2] for x in grp)
            gt = max(x[3] for x in grp)
            txt = _normalize_pdf_text(
                text_page.get_text_bounded(gl - 0.5, gb - 0.5, gr + 0.5, gt + 0.5)
            )
            if txt:
                assembled.append(_LineSegment(gl, gb, gr, gt, "full", page_right, txt))
    if not assembled:
        return []

    # Step 2: Detect two-column gutter from assembled lines (never from word tokens).
    gutter_info = _detect_two_column_gutter_from_lines(assembled, page_bbox)
    gutter_x = gutter_info[0] if gutter_info is not None else None
    gutter_left_edge = gutter_info[1] if gutter_info is not None else None
    gutter_right_edge = gutter_info[2] if gutter_info is not None else None
    page_content_right = max(s.right for s in assembled)
    page_content_left = min(s.left for s in assembled)

    # Promote entire spanning rows (whose ink intrudes into the empty gutter channel) to "full"
    # so justified full-width captions/equations never split into "left" + "full" fragments.
    spanning_ycs: list[float] = []
    if (
        gutter_x is not None
        and gutter_left_edge is not None
        and gutter_right_edge is not None
    ):
        for seg in assembled:
            if _ISOLATED_EQ_NUMBER.match(seg.text):
                continue
            if seg.left < gutter_right_edge - 4.0 and seg.right > gutter_left_edge + 4.0:
                spanning_ycs.append(seg.yc)

    col_raw_segs: dict[str, list[_LineSegment]] = {"full": [], "left": [], "right": []}
    for seg in assembled:
        if any(abs(seg.yc - sy) <= max(4.5, 0.50 * median_h) for sy in spanning_ycs):
            col, col_bound_right = "full", page_right
        else:
            col, col_bound_right = _assign_column(
                (seg.left, seg.bottom, seg.right, seg.top), gutter_x, page_bbox
            )
        col_raw_segs[col].append(
            _LineSegment(
                seg.left, seg.bottom, seg.right, seg.top, col, col_bound_right, seg.text
            )
        )

    # Compute actual content margins per column (critical for 1-inch LaTeX margins!).
    col_content_bounds: dict[str, tuple[float, float]] = {}
    for col, segs in col_raw_segs.items():
        if segs:
            col_content_bounds[col] = (
                min(s.left for s in segs),
                max(s.right for s in segs),
            )
        else:
            col_content_bounds[col] = (page_content_left, page_content_right)

    # Detect Algorithm pseudocode regions to exclude from formulas & captions.
    algorithm_boxes = _detect_algorithm_boxes(assembled, path_objs, page_bbox, median_h)

    # Hold aside isolated right-aligned equation numbers across columns so spanning ("full")
    # equations can also re-associate with a right-margin (1)/(2.3)/[4] token!
    all_tagged = [seg for segs in col_raw_segs.values() for seg in segs]
    eq_num_pool: list[_LineSegment] = []
    segments_by_col: dict[str, list[_LineSegment]] = {"full": [], "left": [], "right": []}
    for col, segs in col_raw_segs.items():
        _, c_right = col_content_bounds[col]
        # Allow matching either the column's content right edge or the page's content right edge.
        right_edge = max(c_right, page_content_right if col in {"full", "right"} else c_right)
        # First pass: bucket by row BEFORE sorting left-to-right (descenders shift yc by 1-2 pt),
        # then merge same-row segments separated by justified spaces or \quad/\qquad within the
        # same column (never crossing gutter_x, and never swallowing right-margin (1)).
        segs.sort(key=lambda s: (-s.yc, s.left))
        col_row_buckets: list[list[_LineSegment]] = []
        row_y_thr = max(4.5, 0.50 * median_h)
        for seg in segs:
            is_right_eq_num = bool(
                _ISOLATED_EQ_NUMBER.match(seg.text) and seg.right >= right_edge - 42.0
            )
            if is_right_eq_num:
                eq_num_pool.append(seg)
                continue
            if col_row_buckets and abs(seg.yc - col_row_buckets[-1][0].yc) <= row_y_thr:
                col_row_buckets[-1].append(seg)
            else:
                col_row_buckets.append([seg])

        row_merged: list[_LineSegment] = []
        same_col_h_thr = max(20.0, 2.2 * median_h)
        for bucket in col_row_buckets:
            bucket.sort(key=lambda s: s.left)
            merged_row: list[_LineSegment] = [bucket[0]]
            for seg in bucket[1:]:
                prev = merged_row[-1]
                if seg.left - prev.right <= same_col_h_thr:
                    gl = min(prev.left, seg.left)
                    gb = min(prev.bottom, seg.bottom)
                    gr = max(prev.right, seg.right)
                    gt = max(prev.top, seg.top)
                    txt = _normalize_pdf_text(
                        text_page.get_text_bounded(gl - 0.5, gb - 0.5, gr + 0.5, gt + 0.5)
                    )
                    merged_row[-1] = _LineSegment(
                        gl, gb, gr, gt, col, prev.col_right, txt or f"{prev.text} {seg.text}"
                    )
                else:
                    merged_row.append(seg)
            row_merged.extend(merged_row)
        row_merged.sort(key=lambda s: (-s.yc, s.left))
        segments_by_col[col] = row_merged

    # Re-associate single-line right-aligned equation numbers where a direct same-row math partner
    # exists (remaining numbers stay in eq_num_pool for post-block fraction matching).
    used_eq_indices: set[int] = set()
    for col in ("full", "left", "right"):
        updated: list[_LineSegment] = []
        for seg in segments_by_col[col]:
            match_idx = next(
                (
                    i
                    for i, num_seg in enumerate(eq_num_pool)
                    if i not in used_eq_indices
                    and (
                        num_seg.col == col
                        or (col == "full" and num_seg.col == "right")
                    )
                    and seg.right <= num_seg.left + 2.0
                    and abs(num_seg.yc - seg.yc) <= max(6.0, 0.75 * seg.height)
                    and (
                        _FORMULA_RELATION.search(seg.text)
                        or _MATH_TOKEN_RE.search(seg.text)
                    )
                ),
                None,
            )
            if match_idx is not None:
                num_seg = eq_num_pool[match_idx]
                used_eq_indices.add(match_idx)
                updated.append(
                    _LineSegment(
                        min(seg.left, num_seg.left),
                        min(seg.bottom, num_seg.bottom),
                        max(seg.right, num_seg.right),
                        max(seg.top, num_seg.top),
                        seg.col,
                        max(seg.col_right, num_seg.col_right),
                        f"{seg.text} {num_seg.text}".strip(),
                        has_eq_number=True,
                    )
                )
            else:
                updated.append(seg)
        segments_by_col[col] = updated

    # Thin short horizontal fraction bars (for LaTeX fraction assembly).
    frac_bars: list[Rect] = [
        (b[0], b[1], b[2], b[3])
        for b in path_objs
        if all(isfinite(v) for v in b)
        and (b[3] - b[1]) <= 2.5
        and 6.0 <= (b[2] - b[0]) <= 0.45 * page_w
    ]

    # X4: Formula frame boxes with explicit size/area upper bounds.
    frame_boxes: list[Rect] = [
        (box[0], box[1], box[2], box[3])
        for box in path_objs
        if all(isfinite(v) for v in box)
        and (box[2] - box[0]) >= 48.0
        and (box[3] - box[1]) >= 18.0
        and (box[2] - box[0]) <= 0.92 * page_w
        and (box[3] - box[1]) <= 0.38 * page_h
        and _rect_area((box[0], box[1], box[2], box[3])) <= 0.35 * page_area
        and not any(
            _overlap_ratio_min((box[0], box[1], box[2], box[3]), ex) > 0.15
            for ex in [*excluded_boxes, *algorithm_boxes]
        )
    ]

    captions: list[tuple[list[float], Rect, str]] = []
    formulas: list[tuple[list[float], Rect, str]] = []
    _ = all_tagged

    for col in ("full", "left", "right"):
        col_segs = segments_by_col[col]
        if not col_segs:
            continue
        col_content_left, col_content_right = col_content_bounds[col]
        col_content_width = max(1.0, col_content_right - col_content_left)
        col_width = (
            (gutter_x - page_left)
            if col == "left" and gutter_x is not None
            else ((page_right - gutter_x) if col == "right" and gutter_x is not None else page_w)
        )

        # Step 3: Group contiguous blocks per column with fraction/piecewise math merging and
        # mid-paragraph inline "Figure N." wrap protection.
        blocks: list[list[_LineSegment]] = []
        prev_vgap: float | None = None
        for seg in col_segs:
            if blocks:
                prev_blk = blocks[-1]
                prev = prev_blk[-1]
                blk_l = min(x.left for x in prev_blk)
                blk_b = min(x.bottom for x in prev_blk)
                blk_r = max(x.right for x in prev_blk)
                blk_t = max(x.top for x in prev_blk)
                vgap = prev.bottom - seg.top
                vgap_blk = blk_b - seg.top
                min_w = max(1.0, min(prev.width, seg.width))
                h_inter = max(0.0, min(prev.right, seg.right) - max(prev.left, seg.left))
                h_overlap_ratio = h_inter / min_w
                blk_h_inter = max(0.0, min(blk_r, seg.right) - max(blk_l, seg.left))
                blk_h_overlap = blk_h_inter / max(1.0, min(blk_r - blk_l, seg.width))
                blk_h_gap = max(0.0, seg.left - blk_r, blk_l - seg.right)
                blk_is_caption = bool(_CAPTION_PREFIX.match(prev_blk[0].text))
                blk_has_formula = any(
                    _FORMULA_RELATION.search(item.text) or item.has_eq_number for item in prev_blk
                )
                seg_is_formula = bool(_FORMULA_RELATION.search(seg.text) or seg.has_eq_number)
                blk_is_short_math = all(
                    _is_short_math_fragment(item, col_width) for item in prev_blk
                )
                seg_math_or_cond = _is_short_math_fragment(seg, col_width) or bool(
                    _CONDITION_LINE.match(seg.text)
                )
                center_dx_blk = abs(((blk_l + blk_r) / 2) - ((seg.left + seg.right) / 2))

                shares_frac_bar = any(
                    (
                        min(blk_b, seg.bottom) - 6.0
                        <= (bar[1] + bar[3]) / 2
                        <= max(blk_t, seg.top) + 6.0
                    )
                    and max(0.0, bar[0] - blk_r, blk_l - bar[2]) <= 18.0
                    and max(0.0, bar[0] - seg.right, seg.left - bar[2]) <= 18.0
                    for bar in frac_bars
                )
                formula_merge = (
                    not blk_is_caption
                    and not _CAPTION_PREFIX.match(seg.text)
                    and not _TABLE_OR_SECTION_PREFIX.match(seg.text)
                    and -max(12.0, 1.4 * median_h) <= vgap_blk <= max(10.0, 1.15 * median_h)
                    and (
                        (blk_has_formula and seg_math_or_cond)
                        or (blk_is_short_math and seg_is_formula)
                        or (
                            shares_frac_bar
                            and blk_is_short_math
                            and _is_short_math_fragment(seg, col_width)
                        )
                    )
                    and (
                        shares_frac_bar
                        or blk_h_overlap >= 0.12
                        or (
                            blk_h_gap <= max(24.0, 2.5 * median_h)
                            and abs(prev.yc - seg.yc) <= max(18.0, 2.0 * median_h)
                        )
                        or center_dx_blk <= 0.25 * col_width
                    )
                )

                # Detect mid-paragraph wrapped prose starting with "Figure N." so it stays in prose.
                is_inline_fig_wrap = (
                    bool(_CAPTION_PREFIX.match(seg.text))
                    and not blk_is_caption
                    and not blk_has_formula
                    and -2.0 <= vgap <= max(7.0, 1.15 * median_h)
                    and (prev_vgap is None or vgap <= prev_vgap * 1.35 + 1.5)
                    and abs(seg.left - prev.left) <= 6.0
                    and prev.width >= 0.65 * col_content_width
                    and not prev.text.rstrip().endswith(
                        (".", "。", "!", "！", "?", "？", ":", "：")
                    )
                )
                gap_jump = prev_vgap is not None and vgap > prev_vgap * 1.6 + 2.0
                left_shift = abs(seg.left - prev.left) > 28.0
                normal_merge = (
                    -4.0 <= vgap <= max(10.0, 1.35 * median_h)
                    and h_overlap_ratio >= 0.35
                    and not gap_jump
                    and not (blk_is_caption and left_shift)
                    and (not _CAPTION_PREFIX.match(seg.text) or is_inline_fig_wrap)
                    and (not _TABLE_OR_SECTION_PREFIX.match(seg.text) or is_inline_fig_wrap)
                    and not (seg_is_formula and not blk_is_caption)
                    and not (blk_has_formula and not _CONDITION_LINE.match(seg.text))
                )
                if formula_merge or normal_merge:
                    prev_blk.append(seg)
                    prev_vgap = max(0.0, vgap)
                    continue
            blocks.append([seg])
            prev_vgap = None

        # Post-block equation-number re-association for vertically centered fraction/multiline eqs.
        for blk in blocks:
            if any(seg.has_eq_number for seg in blk):
                continue
            if not any(_FORMULA_RELATION.search(seg.text) for seg in blk):
                continue
            bl_b = min(x.bottom for x in blk)
            bl_r = max(x.right for x in blk)
            bl_t = max(x.top for x in blk)
            match_idx = next(
                (
                    i
                    for i, num_seg in enumerate(eq_num_pool)
                    if i not in used_eq_indices
                    and (num_seg.col == col or (col == "full" and num_seg.col == "right"))
                    and bl_r <= num_seg.left + 2.0
                    and bl_b - 4.0 <= num_seg.yc <= bl_t + 4.0
                ),
                None,
            )
            if match_idx is not None:
                num_seg = eq_num_pool[match_idx]
                used_eq_indices.add(match_idx)
                blk.append(
                    _LineSegment(
                        num_seg.left,
                        num_seg.bottom,
                        num_seg.right,
                        num_seg.top,
                        col,
                        num_seg.col_right,
                        num_seg.text,
                        has_eq_number=True,
                    )
                )

        for blk_idx, blk in enumerate(blocks):
            bl = min(x.left for x in blk)
            bb = min(x.bottom for x in blk)
            br = max(x.right for x in blk)
            bt = max(x.top for x in blk)
            blk_rect: Rect = (bl, bb, br, bt)
            if any(_overlap_ratio_self(blk_rect, ab) > 0.35 for ab in algorithm_boxes):
                continue
            text = _normalize_pdf_text(text_page.get_text_bounded(bl - 1, bb - 1, br + 1, bt + 1))
            if not text:
                continue
            cap_m = _CAPTION_PREFIX.match(text)
            if cap_m is not None:
                remainder = re.sub(r"\s+", "", text[cap_m.end() :])
                if len(remainder) >= 4:
                    bounds = _clamp_box((bl - 2.0, bb - 2.0, br + 2.0, bt + 2.0), page_bbox)
                    visible = [_unit_interval(v) for v in transform_rect(bounds, matrix)]
                    captions.append((visible, bounds, text))
                continue

            containing_frame = next(
                (
                    fb
                    for fb in frame_boxes
                    if (
                        fb[0] <= bl + 2.0
                        and fb[1] <= bb + 2.0
                        and fb[2] >= br - 2.0
                        and fb[3] >= bt - 2.0
                    )
                ),
                None,
            )
            if containing_frame is not None:
                framed_text = _normalize_pdf_text(text_page.get_text_bounded(*containing_frame))
                if len(framed_text) > 1200:
                    containing_frame = None
                else:
                    text = framed_text
            prev_blk = blocks[blk_idx - 1] if blk_idx > 0 else None
            next_blk = blocks[blk_idx + 1] if blk_idx + 1 < len(blocks) else None
            if _passes_standalone_formula_gate(
                blk,
                text,
                containing_frame,
                prev_blk=prev_blk,
                next_blk=next_blk,
                col_content_left=col_content_left,
                median_h=median_h,
            ):
                if containing_frame is not None:
                    bounds = _clamp_box(containing_frame, page_bbox)
                    if any(existing_bounds == bounds for _, existing_bounds, _ in formulas):
                        continue
                else:
                    bounds = _clamp_box((bl - 2.0, bb - 2.0, br + 2.0, bt + 2.0), page_bbox)
                visible = [_unit_interval(v) for v in transform_rect(bounds, matrix)]
                formulas.append((visible, bounds, text))

    captions = _dedupe_text_objects(captions)
    formulas = _dedupe_text_objects(formulas)

    drafts: list[SourceDraft] = []
    for kind, items in (("caption", captions), ("formula", formulas)):
        for idx, (visible, bounds, text) in enumerate(items, start=1):
            drafts.append(
                SourceDraft(
                    source_id=f"page:{page_no}:{kind}:{idx}",
                    kind="page_object",
                    ordinal=0,
                    parent_id=f"page:{page_no}",
                    title=_first_nonempty_line(text),
                    text="",
                    locator={
                        "format": "pdf",
                        "page": page_no,
                        "object_type": kind,
                        f"{kind}_index": idx,
                        "object_text": text,
                        "bbox": visible,
                        "native_bbox": list(bounds),
                        "coordinate_frame": "visible_page_norm_v1",
                        "native_coordinate_frame": "pdf_page_units_bottom_left_v1",
                        "position_basis": "pdfium_native_text_block_v1",
                        "native_to_visible_transform": {
                            "version": "pdf_cropbox_bottom_left_to_visible_norm_v1",
                            "page_bbox": list(page_bbox),
                            "rotation_degrees_clockwise": rotation,
                            "matrix": list(matrix),
                        },
                    },
                )
            )
    return drafts
