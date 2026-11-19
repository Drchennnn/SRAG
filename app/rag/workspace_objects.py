"""Resolve explicit object questions inside an already authorized publication snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from app.stores.structure_snapshot import SnapshotSource

ObjectKind = Literal["image", "table"]
ObjectOrder = Literal["images", "outputs", "tables"]
Direction = Literal["left", "right", "top", "bottom"]
SourceIdentity = tuple[str, int, str]
Rectangle = tuple[float, float, float, float]

_NUMBER = r"[0-9]{1,4}|[零〇一二两三四五六七八九十]{1,3}"
_CHINESE_DIGITS = {digit: value for value, digit in enumerate("零一二三四五六七八九")}
_CHINESE_DIGITS.update({"〇": 0, "两": 2})
_IMAGE_NUMBER = re.compile(
    rf"第\s*(?P<chinese>{_NUMBER})\s*(?:张|幅|个)\s*(?:图片|图像|子图|插图|图)"
    r"|\b(?:figure|fig\.|image|plot|subplot)\s*#?\s*(?P<english>[0-9]{1,4})\b",
    re.IGNORECASE,
)
_OUTPUT_NUMBER = re.compile(
    rf"第\s*(?P<chinese>{_NUMBER})\s*(?:个)?\s*(?:保存)?输出"
    r"|\boutput\s*#?\s*(?P<english>[0-9]{1,4})\b",
    re.IGNORECASE,
)
_TABLE_NUMBER = re.compile(
    rf"第\s*(?P<chinese>{_NUMBER})\s*(?:张|个)\s*(?:表格|表)"
    r"|\btable\s*#?\s*(?P<english>[0-9]{1,4})\b",
    re.IGNORECASE,
)
_COORDINATED_ORDINAL = re.compile(rf"第\s*(?:{_NUMBER})\s*(?:张|幅|个)\s*(?:和|及|与|、)\s*第")
_IMAGE = re.compile(
    r"子图|图片|图像|插图|图表|原图|什么图|[左右上下]图|图[中上内]|图里的"
    r"|\b(?:figure|image|picture|plot|subplot|chart|panel)\b",
    re.IGNORECASE,
)
_TABLE = re.compile(
    r"表格|(?:参数|配置|汇总|限制|限值|阈值|对照|结果)表|(?<!图)表[中内里]|(?:这|那)张表"
    r"|\btable\b",
    re.IGNORECASE,
)
_FACT = re.compile(
    r"什么|哪[个些种项列行]|多少|如何|为何|怎么|何种|是否|单位|刻度|读出|读取|解读|"
    r"分析|解释|描述|比较|对比|告诉|趋势|数值|结论|"
    r"\b(?:what|which|how|why|read|describe|explain|compare|unit|value|trend|axis)\b",
    re.IGNORECASE,
)
_PREVIEW = re.compile(r"展示|显示|预览|打开|\b(?:show|display|open|preview)\b", re.IGNORECASE)
_AXIS = re.compile(r"横轴|纵轴|坐标轴|\baxis\b", re.IGNORECASE)
_COLOR = re.compile(
    r"(?:红|橙|黄|绿|蓝|紫|灰|黑|白|琥珀|青|粉|棕|褐)色|琥珀子图"
    r"|\b(?:red|orange|yellow|green|blue|purple|grey|gray|black|white|amber|pink|brown)\b",
    re.IGNORECASE,
)
_ENGLISH_OBJECT = r"(?:figure|image|picture|subplot|panel|plot|chart|table)"


def _direction_pattern(chinese: str, english: str) -> re.Pattern[str]:
    return re.compile(
        chinese + r"[^，。！？；\n!?;]{0,9}(?:子图|图表|表格|图|表)"
        rf"|\b{english}(?:\s+[a-z][a-z-]*){{0,2}}\s+{_ENGLISH_OBJECT}\b"
        rf"|\b{_ENGLISH_OBJECT}\s+(?:on|at)\s+(?:the\s+)?{english}\b",
        re.IGNORECASE,
    )


_DIRECTIONS: tuple[tuple[Direction, re.Pattern[str]], ...] = (
    ("left", _direction_pattern(r"左(?:侧|边|半)?", r"(?:leftmost|left(?:[- ]hand)?)")),
    ("right", _direction_pattern(r"右(?:侧|边|半)?", r"(?:rightmost|right(?:[- ]hand)?)")),
    ("top", _direction_pattern(r"上(?:侧|方|半)", r"(?:upper|topmost|top)")),
    ("bottom", _direction_pattern(r"下(?:侧|方|半)", r"(?:lower|bottommost|bottom)")),
)
_NATIVE_POSITION_BASES = ("native_shape_extents_v1", "axis_aligned_native_rule_grid_v1")
_NATIVE_IMAGE_TYPES = {"picture", "image", "chart"}


@dataclass(frozen=True)
class ObjectIntent:
    kind: ObjectKind
    ordinal: int | None = None
    order: ObjectOrder | None = None
    direction: Direction | None = None
    unresolved: bool = False


def object_intent(question: str, *, include_reference: bool = False) -> ObjectIntent | None:
    """Recognize an explicit figure/table question without consuming ordinary previews."""

    outputs = _ordinals(_OUTPUT_NUMBER, question)
    images = _ordinals(_IMAGE_NUMBER, question)
    tables = _ordinals(_TABLE_NUMBER, question)
    if not _FACT.search(question) and (
        _PREVIEW.search(question)
        or not (re.search(r"[?？]", question) or (include_reference and (images or tables)))
    ):
        return None
    has_table = bool(tables or _TABLE.search(question))
    has_image = bool(images or _IMAGE.search(question) or (outputs and _AXIS.search(question)))
    if not (has_image or has_table):
        return None
    kind: ObjectKind = "table" if has_table else "image"
    order: ObjectOrder = "tables" if has_table else ("outputs" if outputs else "images")
    numbers = tables if has_table else (outputs or images)
    directions = [name for name, pattern in _DIRECTIONS if pattern.search(question)]
    ordinal = next(iter(numbers)) if len(numbers) == 1 else None
    direction = directions[0] if len(directions) == 1 else None
    unresolved = (
        has_table
        and has_image
        or len(numbers) > 1
        or bool(_COORDINATED_ORDINAL.search(question))
        or bool(outputs and images)
        or len(directions) > 1
        or bool(has_image and not numbers and not directions and _COLOR.search(question))
    )
    return ObjectIntent(
        kind=kind,
        ordinal=ordinal,
        order=order if numbers else None,
        direction=direction,
        unresolved=unresolved,
    )


def object_choices(
    intent: ObjectIntent,
    sources: Sequence[SnapshotSource],
    *,
    focus: SnapshotSource | None = None,
    region_rects: Mapping[SourceIdentity, Rectangle] | None = None,
) -> list[SnapshotSource]:
    """Return qualified choices. Callers can read only after establishing a unique target."""

    if intent.unresolved or intent.ordinal == 0:
        return []
    regions = region_rects or {}
    candidates = [
        source
        for source in _within_focus(sources, focus)
        if _matches_kind(source, intent.kind, regions)
    ]
    if focus is not None and intent.ordinal is None and intent.direction is None:
        exact = [source for source in candidates if _identity(source) == _identity(focus)]
        if exact:
            return exact

    candidates.sort(key=lambda source: (source.document_id, source.version, source.ordinal))
    if intent.ordinal is not None:
        candidates = _numbered(candidates, intent.ordinal, intent.order)
    if intent.direction:
        positioned = [(source, _rectangle(source, regions)) for source in candidates]
        definite = [
            source
            for source, rect in positioned
            if rect is not None and _on_side(rect, intent.direction)
        ]
        if not definite:
            return []
        # Unlocated siblings can also match, so a single located candidate is ambiguous.
        return definite + [source for source, rect in positioned if rect is None]
    return candidates


def _ordinals(pattern: re.Pattern[str], question: str) -> set[int]:
    return {
        _parse_ordinal(match["chinese"] or match["english"]) for match in pattern.finditer(question)
    }


def _parse_ordinal(number: str) -> int:
    if number.isascii() and number.isdigit():
        return int(number)
    if "十" in number:
        if number.count("十") != 1:
            return 0
        tens, units = number.split("十")
        ten = _CHINESE_DIGITS.get(tens or "一")
        unit = _CHINESE_DIGITS.get(units or "零")
        return ten * 10 + unit if ten is not None and unit is not None else 0
    return _CHINESE_DIGITS.get(number, 0)


def _identity(source: SnapshotSource) -> SourceIdentity:
    return source.document_id, source.version, source.source_id


def _within_focus(
    sources: Sequence[SnapshotSource], focus: SnapshotSource | None
) -> Sequence[SnapshotSource]:
    if focus is None:
        return sources
    parent = focus.source_id if focus.kind in {"page", "cell"} else focus.parent_id
    return [
        source
        for source in sources
        if (source.document_id, source.version) == (focus.document_id, focus.version)
        and (
            source.source_id in {parent, focus.source_id}
            or (parent is not None and source.parent_id == parent)
        )
    ]


def _matches_kind(
    source: SnapshotSource,
    kind: ObjectKind,
    regions: Mapping[SourceIdentity, Rectangle],
) -> bool:
    object_type = source.locator.get("object_type")
    if kind == "table":
        return source.kind == "page_object" and object_type == "table"
    if source.kind == "page_object":
        return isinstance(object_type, str) and object_type in _NATIVE_IMAGE_TYPES
    if source.kind == "notebook_output":
        records = source.locator.get("representations")
        # Preserve declared but unreadable slots to prevent subsequent images from moving up.
        return isinstance(records, list) and any(
            isinstance(record, dict)
            and isinstance(mime_type := record.get("mime_type"), str)
            and mime_type.startswith("image/")
            for record in records
        )
    return source.kind == "page" and _rectangle(source, regions) is not None


def _numbered(
    candidates: list[SnapshotSource], ordinal: int, order: ObjectOrder | None
) -> list[SnapshotSource]:
    if order == "outputs":
        return [
            source
            for source in candidates
            if source.kind == "notebook_output"
            and source.locator.get("output_index") == ordinal - 1
        ]
    positions: dict[tuple[str, int], int] = {}
    matches = []
    for source in candidates:
        document = source.document_id, source.version
        position = positions.get(document, 0) + 1
        positions[document] = position
        if position == ordinal:
            matches.append(source)
    return matches


def _rectangle(
    source: SnapshotSource, regions: Mapping[SourceIdentity, Rectangle]
) -> Rectangle | None:
    registered = regions.get(_identity(source))
    if registered is not None:
        rectangle = registered
    elif (
        source.kind == "page_object"
        and source.locator.get("coordinate_frame") == "visible_page_norm_v1"
        and source.locator.get("position_basis") in _NATIVE_POSITION_BASES
    ):
        rectangle = source.locator.get("bbox")
    else:
        return None
    if not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4:
        return None
    if any(type(value) not in {int, float} or not (0 <= value <= 1) for value in rectangle):
        return None
    x0, y0, x1, y1 = rectangle
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _on_side(rect: Rectangle, direction: Direction) -> bool:
    x0, y0, x1, y1 = rect
    if direction == "left":
        return x1 <= 0.5
    if direction == "right":
        return x0 >= 0.5
    if direction == "top":
        return y1 <= 0.5
    return y0 >= 0.5
