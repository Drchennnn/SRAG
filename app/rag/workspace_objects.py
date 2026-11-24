"""Resolve explicit object questions inside an already authorized publication snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from app.stores.structure_snapshot import SnapshotSource

ObjectKind = Literal["image", "table", "caption", "formula"]
ObjectOrder = Literal["images", "outputs", "tables", "captions", "formulas"]
Direction = Literal["left", "right", "top", "bottom"]
SourceIdentity = tuple[str, int, str]
Rectangle = tuple[float, float, float, float]

_NUMBER = r"[0-9]{1,4}|[零〇一二两三四五六七八九十]{1,3}"
_CHINESE_DIGITS = {digit: value for value, digit in enumerate("零一二三四五六七八九")}
_CHINESE_DIGITS.update({"〇": 0, "两": 2})
_IMAGE_NUMBER = re.compile(
    rf"第\s*(?P<chinese>{_NUMBER})\s*(?:张|幅|个)\s*(?:输出\s*)?(?:图片|图像|子图|插图|图)"
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
    r"|\btable\s*#?\s*(?P<english>[0-9]{1,4})\b"
    r"|(?<![第图])表\s*#?\s*(?P<trailing>[0-9]{1,4})(?![0-9.])",
    re.IGNORECASE,
)
_COORDINATED_ORDINAL = re.compile(rf"第\s*(?:{_NUMBER})\s*(?:张|幅|个)\s*(?:和|及|与|、)\s*第")
_EXPLICIT_TABLE_REF = re.compile(
    r"原表|(?:这|那)张\s*(?:表格|原表|表)"
    r"|(?:刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述)\s*(?:的\s*)?(?:[那这一]\s*)?(?:张\s*)?(?:表格|原表|表)"
)
_EXPLICIT_LOCATION = re.compile(
    rf"第?\s*(?:{_NUMBER})\s*(?:页|张幻灯片|(?:个)?\s*(?:cell|单元格))"
    r"|\b(?:page|slide|cell)\s*#?\s*[1-9][0-9]{0,3}\b",
    re.IGNORECASE,
)
_CURRENT_OBJECT_LOC = re.compile(r"当前|本页|这页|该页|本\s*cell|这个\s*cell|这里|此处", re.I)
_EXPLICIT_CAPTION_TARGET = re.compile(
    rf"(?:图|figure|fig\.?)\s*#?\s*(?:{_NUMBER})\s*(?:的)?\s*(?:图注|图题|注|caption)"
    r"|(?:这|那|该)(?:张图)?(?:的)?\s*(?:图注|图题)"
    r"|\b(?:this|that|the)\s+caption\b",
    re.IGNORECASE,
)
_EXPLICIT_FORMULA_TARGET = re.compile(
    rf"(?:公式|方程|式子)\s*#?\s*(?:{_NUMBER}|[A-Za-z_][A-Za-z0-9_]*)\b"
    rf"|第\s*(?:{_NUMBER})\s*(?:个|条|道)\s*(?:公式|方程|表达式|式子)"
    r"|(?:这|那|该)(?:个|条|道)?\s*(?:公式|方程|表达式|式子)"
    r"|\b(?:eq\.?|equation)\s*#?\s*\(?[0-9.]+\)?"
    r"|\b(?:formula|equation)\s+(?:for|of)\s+[A-Za-z_][A-Za-z0-9_]*\b",
    re.IGNORECASE,
)
_FORMULA_DERIVATION = re.compile(
    r"推导|怎么推|如何推|由来|证明|怎么得到|如何得到|哪里来|从哪来|"
    r"哪几个公式|多个公式|各个公式|\b(?:deriv(?:e|ation)|proof|multiple\s+equations?)\b",
    re.IGNORECASE,
)
_VISUAL_PROSE_IDIOMS = re.compile(
    r"试图|意图|企图|力图|贪图|图谋|版图|蓝图|图灵",
)
_VISUAL_INTENT_QUERY = re.compile(
    r"图片|图像|图表|图示|曲线|直方图|柱状图|折线图|散点图|热力图|框图|示意图|流程图|架构图|"
    r"手写|截图|附图|图例|画图|画出|图中|如图|横轴|纵轴|坐标轴|什么图|哪张图|"
    r"(?:左边|右侧|左侧|右边|上方|下方|上面|下面)(?:那张|的)?\s*图|"
    r"\b(?:figure|fig\.?|plot|chart|diagram|curve|(?:x|y|horizontal|vertical|plot|chart|figure)[-\s]*ax(?:is|es)|"
    r"visual(?:ization)?|illustration|screenshot|handwritten)\b",
    re.IGNORECASE,
)
_DEMONSTRATIVE_TABLE = re.compile(r"(?:这|那)张\s*(?:表格|原表|表)")
_IMAGE = re.compile(
    r"子图|图片|图像|插图|图表|原图|什么图|[左右上下]图|图[中上内]|图里的"
    r"|(?:实验|架构|框|曲线|示意|流程|结构|对比|比较|分布|结果|热力|散点|柱状|折线|直方|系统|原理|网络|训练|损失|性能)图"
    r"|[左右上下](?:侧|边|方|半)(?:的|[那这][张幅个])?\s*图"
    r"|\b(?:figure|image|picture|plot|subplot|subfigure|chart|panel)\b",
    re.IGNORECASE,
)
_COMPARATIVE_PREFIX = re.compile(
    r"(?:与|和|跟|同|较|相比于?|对比|对照|不同于)|"
    r"\bcompared\s+(?:to|with)\b",
    re.IGNORECASE,
)
_EXPLICIT_MULTI_IMAGE = re.compile(
    r"另一[张幅个]|另一[边侧]|旁边那[张幅]|两[张幅个]?(?:子)?图|对比两|"
    r"\b(?:the\s+other|another\s+(?:figure|image|plot|subplot)|"
    r"both\s+(?:figures|plots|subplots)|two\s+(?:figures|plots))\b",
    re.IGNORECASE,
)
_OTHER_OR_MULTI_IMAGE = re.compile(
    r"另一[张幅个]|其余|其它|其他|另一[边侧]|旁边那[张幅]|两[张幅个]?(?:子)?图|对比两|"
    r"(?:变化|差距|差多少|差异|不同)|"
    r"\b(?:the\s+other|another|both\s+(?:figures|plots|subplots)|two\s+(?:figures|plots)|"
    r"compared\s+(?:to|with)|difference|versus|vs\.?)\b",
    re.IGNORECASE,
)
_TABLE = re.compile(
    r"原表|表格|(?:参数|配置|汇总|限制|限值|阈值|对照|结果)表|(?<!图)表[中内里]|(?:这|那)张表"
    r"|\btable\b",
    re.IGNORECASE,
)
_CAPTION = re.compile(
    r"图注|图题|\b(?:figure|fig\.?)\s*(?:[0-9]+\s*)?caption\b|\bcaption\b",
    re.IGNORECASE,
)
_FORMULA = re.compile(
    r"公式|方程|表达式|式子|\b(?:formula|equation)\b",
    re.IGNORECASE,
)
_COLUMN = re.compile(
    rf"表头|列名|(?:这|那|哪)一?列|第\s*(?:{_NUMBER})\s*列"
    r"|(?<![行队排序])列的(?:单位|数值|读值)|\bcolumn\b",
    re.IGNORECASE,
)
_FACT = re.compile(
    r"什么|哪[个些种项列行]|多少|如何|为何|怎么|何种|是否|单位|刻度|读出|读取|解读|"
    r"分析|解释|描述|比较|对比|告诉|趋势|数值|结论|核对|"
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
_ENGLISH_OBJECT = r"(?:figure|image|picture|subplot|subfigure|panel|plot|chart|table)"


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
    (
        "top",
        _direction_pattern(
            r"上(?:侧|方|半|边)|顶部|(?<![如以])上(?=(?:子图|图表|表格|图|表))",
            r"(?:upper|topmost|top)",
        ),
    ),
    (
        "bottom",
        _direction_pattern(
            r"下(?:侧|方|半|边)|底部|(?<![如以])下(?=(?:子图|图表|表格|图|表))",
            r"(?:lower|bottommost|bottom)",
        ),
    ),
)
_NATIVE_POSITION_BASES = (
    "native_shape_extents_v1",
    "axis_aligned_native_rule_grid_v1",
    "pdfium_native_image_extents_v1",
    "pdfium_native_vector_extents_v1",
    "pdfium_native_text_block_v1",
)
_NATIVE_IMAGE_TYPES = {"picture", "image", "chart"}
_CLEAR_OVERLAP_AREA_RATIO = 0.20
_CROSS_AXIS_OVERLAP_RATIO = 0.30


@dataclass(frozen=True)
class ObjectIntent:
    kind: ObjectKind
    ordinal: int | None = None
    order: ObjectOrder | None = None
    direction: Direction | None = None
    unresolved: bool = False


def object_intent(question: str, *, include_reference: bool = False) -> ObjectIntent | None:
    """Recognize an explicit object-viewing/reading question without hijacking RAG queries."""
    from app.rag.evidence_supplement import asks_for_definition_and_experiment
    from app.rag.workspace_composition import asks_for_composition

    outputs = _ordinals(_OUTPUT_NUMBER, question)
    images = _ordinals(_IMAGE_NUMBER, question)
    tables = _ordinals(_TABLE_NUMBER, question)
    has_fact = bool(_FACT.search(question))
    has_preview = bool(_PREVIEW.search(question))
    has_explicit_loc = bool(
        _EXPLICIT_LOCATION.search(question) or _CURRENT_OBJECT_LOC.search(question)
    )
    is_composite = bool(
        asks_for_composition(question) or asks_for_definition_and_experiment(question)
    )
    if is_composite and not include_reference and not (has_preview and has_explicit_loc):
        return None
    has_caption = bool(_CAPTION.search(question))
    has_formula = bool(_FORMULA.search(question))
    has_image = bool(
        not has_caption
        and (images or _IMAGE.search(question) or (outputs and _AXIS.search(question)))
    )
    has_table = bool(
        tables or _TABLE.search(question) or (not has_image and _COLUMN.search(question))
    )
    if has_caption and not has_table:
        explicit_caption = bool(
            has_explicit_loc
            or has_preview
            or _EXPLICIT_CAPTION_TARGET.search(question)
            or include_reference
        )
        return ObjectIntent(kind="caption", order="captions") if explicit_caption else None
    if has_formula and not (has_table or has_image):
        explicit_formula = bool(
            has_explicit_loc
            or has_preview
            or _EXPLICIT_FORMULA_TARGET.search(question)
            or include_reference
        )
        if not explicit_formula:
            return None
        if (
            _FORMULA_DERIVATION.search(question)
            and not has_preview
            and not _EXPLICIT_FORMULA_TARGET.search(question)
        ):
            return None
        return ObjectIntent(kind="formula", order="formulas")
    if not (has_image or has_table):
        return None
    direction_matches = [
        (name, match)
        for name, pattern in _DIRECTIONS
        if (match := pattern.search(question)) is not None
    ]
    directions = [name for name, _ in direction_matches]
    if has_table and not has_image:
        explicit_table = bool(
            tables
            or directions
            or _COLUMN.search(question)
            or _EXPLICIT_TABLE_REF.search(question)
            or _EXPLICIT_LOCATION.search(question)
        )
        if not explicit_table:
            return None
        if not has_fact:
            has_actionable_target = bool(
                tables
                or directions
                or _EXPLICIT_LOCATION.search(question)
                or _DEMONSTRATIVE_TABLE.search(question)
            )
            if has_preview and not has_actionable_target:
                return None
            if not (
                has_actionable_target
                or re.search(r"[?？]", question)
                or (include_reference and tables)
            ):
                return None
    elif not has_fact and (
        has_preview or not (re.search(r"[?？]", question) or (include_reference and images))
    ):
        return None
    kind: ObjectKind = "table" if has_table else "image"
    order: ObjectOrder = "tables" if has_table else ("outputs" if outputs else "images")
    numbers = tables if has_table else (outputs or images)
    ordinal = next(iter(numbers)) if len(numbers) == 1 else None
    direction = directions[0] if len(directions) == 1 else None
    if has_image and not has_table and len(direction_matches) == 1 and ordinal is None:
        _, dir_match = direction_matches[0]
        local_before = question[max(0, dir_match.start() - 24) : dir_match.start()]
        local_after = question[dir_match.end() : min(len(question), dir_match.end() + 16)]
        if (
            _COMPARATIVE_PREFIX.search(local_before)
            or re.search(r"相比|对比|对照", local_after)
        ) and _OTHER_OR_MULTI_IMAGE.search(question):
            direction = None
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
        groups: dict[tuple[str, int, str], list[SnapshotSource]] = {}
        for source in candidates:
            groups.setdefault(_page_group_key(source), []).append(source)
        selected: list[SnapshotSource] = []
        for group in groups.values():
            selected.extend(_select_group_direction(group, intent.direction, regions))
        return selected
    return candidates


def _ordinals(pattern: re.Pattern[str], question: str) -> set[int]:
    return {
        _parse_ordinal(
            match["chinese"] or match["english"] or match.groupdict().get("trailing") or ""
        )
        for match in pattern.finditer(question)
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
    if kind in {"caption", "formula"}:
        return source.kind == "page_object" and object_type == kind
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


def _page_group_key(source: SnapshotSource) -> tuple[str, int, str]:
    if source.parent_id is not None:
        group_id = source.parent_id
    elif source.kind == "page":
        group_id = source.source_id
    elif isinstance(page := source.locator.get("page"), int) and not isinstance(page, bool):
        group_id = f"page:{page}"
    else:
        group_id = source.source_id
    return (source.document_id, source.version, group_id)


def _project_for_direction(
    rect: Rectangle, direction: Direction
) -> tuple[float, float, float, float]:
    """Reflect coords so [0.5, 1.0] on the primary axis is the requested side."""
    x0, y0, x1, y1 = rect
    if direction == "right":
        return (x0, x1, y0, y1)
    if direction == "left":
        return (1.0 - x1, 1.0 - x0, y0, y1)
    if direction == "bottom":
        return (y0, y1, x0, x1)
    return (1.0 - y1, 1.0 - y0, x0, x1)


def _on_side(rect: Rectangle, direction: Direction, *, single: bool = False) -> bool:
    a0, a1, _, _ = _project_for_direction(rect, direction)
    if a0 >= 0.5:
        return True
    span = a1 - a0
    center = (a0 + a1) / 2
    on_side_fraction = max(0.0, a1 - max(0.5, a0)) / span
    if single:
        return center >= 0.58 and on_side_fraction >= 0.75 and a0 >= 0.40
    return center >= 0.55 and on_side_fraction >= 0.65


def _overlap_metrics(
    rect_a: Rectangle, rect_b: Rectangle, direction: Direction
) -> tuple[float, float]:
    a0, a1, b0, b1 = _project_for_direction(rect_a, direction)
    u0, u1, v0, v1 = _project_for_direction(rect_b, direction)
    inter_a = max(0.0, min(a1, u1) - max(a0, u0))
    inter_b = max(0.0, min(b1, v1) - max(b0, v0))
    area_a = (a1 - a0) * (b1 - b0)
    area_b = (u1 - u0) * (v1 - v0)
    min_area = min(area_a, area_b)
    min_axis = min(a1 - a0, u1 - u0)
    area_ratio = (inter_a * inter_b) / min_area if min_area > 0 else 0.0
    axis_ratio = inter_a / min_axis if min_axis > 0 else 0.0
    return area_ratio, axis_ratio


def _select_group_direction(
    group: Sequence[SnapshotSource],
    direction: Direction,
    regions: Mapping[SourceIdentity, Rectangle],
) -> list[SnapshotSource]:
    positioned = [(source, _rectangle(source, regions)) for source in group]
    located = [(source, rect) for source, rect in positioned if rect is not None]
    unlocated = [source for source, rect in positioned if rect is None]
    if not located:
        return []
    single_located = len(located) == 1
    selected_indices = {
        idx
        for idx, (_, rect) in enumerate(located)
        if _on_side(rect, direction, single=single_located)
    }
    if not selected_indices:
        return []
    ambiguous_indices = set(selected_indices)
    for idx in selected_indices:
        a0, _, _, _ = _project_for_direction(located[idx][1], direction)
        crosses_midline = a0 < 0.5
        for peer_idx, (_, peer_rect) in enumerate(located):
            if peer_idx in ambiguous_indices:
                continue
            area_ratio, axis_ratio = _overlap_metrics(located[idx][1], peer_rect, direction)
            if area_ratio >= _CLEAR_OVERLAP_AREA_RATIO or (
                crosses_midline and axis_ratio >= _CROSS_AXIS_OVERLAP_RATIO
            ):
                ambiguous_indices.add(peer_idx)
    # Unlocated siblings can also match, so keep them alongside located candidates.
    return [located[idx][0] for idx in sorted(ambiguous_indices)] + unlocated


def has_visual_question(question: str) -> bool:
    """Return True when the query explicitly asks about visual content rather than prose idioms."""
    cleaned = _VISUAL_PROSE_IDIOMS.sub(" ", question)
    return bool(_VISUAL_INTENT_QUERY.search(cleaned))


def _is_sparse_visual_source(src: SnapshotSource) -> bool:
    if src.kind == "page_object":
        return src.locator.get("object_type") in {"image", "picture", "chart"}
    if src.kind == "notebook_output":
        return any(
            isinstance(rep, dict)
            and rep.get("status") == "available"
            and str(rep.get("mime_type", "")).startswith("image/")
            for rep in src.locator.get("representations", ())
        )
    if src.kind == "page":
        return len(" ".join(src.text.split())) < 160
    return False


def select_s0_visual_targets(
    question: str,
    *,
    ranked: Sequence[SnapshotSource],
    all_sources: Sequence[SnapshotSource],
    visual_candidates: Sequence[SnapshotSource] = (),
    explicit_page: int | None = None,
    explicit_cell: int | None = None,
    max_targets: int = 2,
) -> list[SnapshotSource]:
    """Select bounded (<=2) visual targets for S0 D03 reads when text alone is insufficient."""
    if max_targets <= 0:
        return []
    cap = min(max_targets, 2)
    visual_ask = has_visual_question(question)
    page_targeted = explicit_page is not None or explicit_cell is not None
    if not visual_ask and not page_targeted:
        return []

    by_parent: dict[tuple[str, int, str], list[SnapshotSource]] = {}
    for src in all_sources:
        if src.parent_id and (
            (
                src.kind == "page_object"
                and src.locator.get("object_type") in {"image", "picture", "chart"}
            )
            or (
                src.kind == "notebook_output"
                and _is_sparse_visual_source(src)
            )
        ):
            by_parent.setdefault((src.document_id, src.version, src.parent_id), []).append(src)
    for children in by_parent.values():
        children.sort(key=lambda s: (s.ordinal, s.source_id))

    if not visual_ask and page_targeted:
        # Non-visual targeted page: read only if page/cell is sparse and has registered visuals.
        has_sparse_page_with_visual = any(
            (
                (
                    explicit_page is not None
                    and src.kind == "page"
                    and src.locator.get("page") == explicit_page
                )
                or (
                    explicit_cell is not None
                    and src.kind == "cell"
                    and src.locator.get("cell_index") == explicit_cell
                )
            )
            and len(" ".join(src.text.split())) < 120
            and bool(by_parent.get((src.document_id, src.version, src.source_id)))
            for src in ranked[:4]
        )
        if not has_sparse_page_with_visual:
            return []

    picked: list[SnapshotSource] = []
    seen: set[SourceIdentity] = set()

    def add(src: SnapshotSource) -> None:
        if len(picked) >= cap:
            return
        ident = (src.document_id, src.version, src.source_id)
        if ident in seen:
            return
        seen.add(ident)
        picked.append(src)

    # 1. Explicit visual candidates surfaced by caption_target / visual retrieval.
    for cand in visual_candidates:
        if cand.kind in {"page_object", "notebook_output"} and _is_sparse_visual_source(cand):
            add(cand)

    # 2. Child registered images of the top-ranked pages/cells.
    for anchor in ranked[:4]:
        if len(picked) >= cap:
            break
        if anchor.kind in {"page_object", "notebook_output"} and _is_sparse_visual_source(anchor):
            add(anchor)
            continue
        if anchor.kind in {"page", "cell"}:
            children = by_parent.get((anchor.document_id, anchor.version, anchor.source_id), ())
            for child in children:
                add(child)
                if len(picked) >= cap:
                    break
            if (
                not children
                and anchor.kind == "page"
                and visual_ask
                and len(" ".join(anchor.text.split())) < 160
                and anchor.locator.get("format") in {"pdf", "pptx"}
            ):
                add(anchor)

    return picked[:cap]
