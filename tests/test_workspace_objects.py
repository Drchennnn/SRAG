from dataclasses import replace

import pytest

from app.rag.workspace_objects import ObjectIntent, object_choices, object_intent
from app.stores.structure_snapshot import SnapshotSource


def _source(
    source_id: str,
    kind: str,
    ordinal: int,
    locator: dict,
    *,
    parent_id: str | None = None,
    document_id: str = "public-doc",
    state: str = "available",
) -> SnapshotSource:
    return SnapshotSource(
        course_id="public-course",
        document_id=document_id,
        version=1,
        source_id=source_id,
        parent_id=parent_id,
        ordinal=ordinal,
        kind=kind,
        title="",
        text="",
        locator=locator,
        state=state,
    )


def _output(index: int, *, image: bool = True, state: str = "available") -> SnapshotSource:
    return _source(
        f"cell:1:output:{index}",
        "notebook_output",
        index + 2,
        {
            "cell_index": 1,
            "output_index": index,
            "representations": ([{"mime_type": "image/png", "status": state}] if image else []),
        },
        parent_id="cell:1",
        state=state,
    )


def _page_object(
    number: int, object_type: str, rect: list[float] | None, *, reliable: bool = True
) -> SnapshotSource:
    return _source(
        f"page:1:shape:{number}",
        "page_object",
        number + 1,
        {
            "object_type": object_type,
            "bbox": rect,
            "coordinate_frame": "visible_page_norm_v1",
            "position_basis": "native_shape_extents_v1"
            if reliable
            else "group_transform_unverified",
        },
        parent_id="page:1",
    )


def test_plain_questions_parse_specific_objects_but_keep_existing_previews():
    assert object_intent("显示本 Cell 的两张原件图片") is None
    assert object_intent("刚才第二张图") is None
    assert object_intent("第二张图", include_reference=True) == ObjectIntent(
        kind="image", ordinal=2, order="images"
    )
    assert object_intent("请展示第 2 张图？") is None
    assert object_intent("归纳文中这个推导的结果") is None
    assert object_intent("HarborTrace.ipynb Cell 2 第二幅图的横轴是什么？") == ObjectIntent(
        kind="image", ordinal=2, order="images"
    )
    assert object_intent("Select output 2. What is its axis label?") == ObjectIntent(
        kind="image", ordinal=2, order="outputs"
    )
    assert object_intent("右侧琥珀子图的横轴刻度单位是什么？") == ObjectIntent(
        kind="image", direction="right"
    )
    for question in ("右侧图的单位是什么？", "右边那幅图的横轴是什么？"):
        assert object_intent(question) == ObjectIntent(kind="image", direction="right")
    assert object_intent("What is the unit of the subplot on the right?") == ObjectIntent(
        kind="image", direction="right"
    )
    assert object_intent("定义域右侧有什么限制？图表上的数据是多少？") == ObjectIntent(kind="image")
    assert object_intent("第 1 页表格里的 Pulse gate 设定值是多少？") == ObjectIntent(kind="table")
    assert object_intent("请核对原表的电流列") == ObjectIntent(kind="table")
    assert object_intent("第 2 列的单位是什么？") == ObjectIntent(kind="table")
    assert object_intent("输入序列的单位是什么？") is None
    assert object_intent("打开原表") is None


def test_image_ordinal_preserves_broken_slots_and_differs_from_raw_output_number():
    stream = _output(0, image=False)
    malformed = _output(1, state="malformed")
    usable = _output(2)
    sources = [usable, malformed, stream]
    image_one = ObjectIntent(kind="image", ordinal=1, order="images")
    image_two = ObjectIntent(kind="image", ordinal=2, order="images")
    output_one = ObjectIntent(kind="image", ordinal=1, order="outputs")
    output_two = ObjectIntent(kind="image", ordinal=2, order="outputs")
    assert object_choices(image_one, sources) == [malformed]
    assert object_choices(image_two, sources) == [usable]
    assert object_choices(output_one, sources) == []
    assert object_choices(output_two, sources) == [malformed]
    assert object_choices(replace(image_one, ordinal=0), sources) == []


def test_table_selection_keeps_native_table_distinct_from_page_and_chart():
    page = _source("page:1", "page", 0, {"page": 1})
    table = _page_object(1, "table", None, reliable=False)
    chart = _page_object(2, "chart", [0.55, 0.2, 0.91, 0.5])
    assert object_choices(ObjectIntent(kind="table"), [page, table, chart]) == [table]
    assert object_choices(ObjectIntent(kind="image"), [page, table, chart]) == [chart]


def test_geometry_selects_right_object_without_using_untrusted_boxes():
    left = _page_object(1, "picture", [0.08, 0.2, 0.45, 0.5])
    right = _page_object(2, "chart", [0.55, 0.2, 0.91, 0.5])
    another_right = _page_object(3, "picture", [0.57, 0.55, 0.95, 0.9])
    unlocated = _page_object(4, "chart", [0.57, 0.55, 0.95, 0.9], reliable=False)
    intent = ObjectIntent(kind="image", direction="right")
    assert object_choices(intent, [left, right]) == [right]
    assert object_choices(intent, [left, right, another_right]) == [right, another_right]
    assert object_choices(intent, [left, unlocated]) == []
    assert object_choices(intent, [left, right, unlocated]) == [right, unlocated]


def test_registered_region_must_be_well_formed_even_for_exact_page_key():
    page = _source("page:1", "page", 0, {"page": 1})
    other = _source("page:2", "page", 2, {"page": 2})
    intent = ObjectIntent(kind="image", direction="right")
    registered = {("public-doc", 1, "page:1"): (0.56, 0.2, 0.94, 0.8)}
    assert object_choices(intent, [page, other], region_rects=registered) == [page]
    assert object_choices(intent, [page, other]) == []
    assert not object_choices(
        ObjectIntent(kind="image"),
        [page],
        region_rects={("public-doc", 1, "page:1"): (0.56, 0.8, 0.94, 0.2)},
    )


def test_focus_uses_the_authorized_snapshot_and_current_parent_for_ordinal():
    first = _output(0)
    second = _output(1)
    foreign = replace(first, document_id="different-doc")
    stale_focus = replace(first, version=2)
    assert object_choices(ObjectIntent(kind="image"), [first, second], focus=stale_focus) == []
    assert object_choices(ObjectIntent(kind="image"), [first, second, foreign], focus=first) == [
        first
    ]
    numbered = ObjectIntent(kind="image", ordinal=2, order="images")
    assert object_choices(numbered, [first, second, foreign], focus=first) == [second]


@pytest.mark.parametrize(
    "question",
    (
        "红色子图的纵轴数值是什么？",
        "右侧子图和左侧图片的读数是多少？",
        "第一张和第二张图有什么区别？",
        "比较第两个输出和第二幅图上的读数",
    ),
)
def test_defer_unresolvable_object_descriptions(question):
    intent = object_intent(question)
    assert intent is not None and intent.unresolved
    assert object_choices(intent, [_output(0), _output(1)]) == []


def test_vertical_direction_intent_preserves_history_and_context_phrases():
    assert object_intent("第 4 页上图的横轴是什么？") == ObjectIntent(
        kind="image", direction="top"
    )
    assert object_intent("第 4 页下图的纵轴是什么？") == ObjectIntent(
        kind="image", direction="bottom"
    )
    assert object_intent("第 4 页上方的图表刻度是多少？") == ObjectIntent(
        kind="image", direction="top"
    )
    assert object_intent("第 4 页下方的图趋势如何？") == ObjectIntent(
        kind="image", direction="bottom"
    )
    history_intent = object_intent("上面的第二张图的横轴单位是什么？")
    assert history_intent is not None and history_intent.direction is None
    assert object_intent("上一张图的横轴单位是什么？") is None
    assert object_intent("上一张图片的横轴单位是什么？") == ObjectIntent(kind="image")
    assert object_intent("如下图所示的曲线随上下文长度变化如何？") == ObjectIntent(kind="image")
    diagonal = object_intent("第 1 页右上图的横轴是什么？")
    assert diagonal is not None and diagonal.unresolved


def _page_image(
    page: int,
    number: int,
    rect: list[float] | None,
    *,
    object_type: str = "image",
    reliable: bool = True,
) -> SnapshotSource:
    return _source(
        f"page:{page}:image:{number}",
        "page_object",
        page * 10 + number,
        {
            "page": page,
            "object_type": object_type,
            "bbox": rect,
            "coordinate_frame": "visible_page_norm_v1",
            "position_basis": "pdfium_native_image_extents_v1"
            if reliable
            else "group_transform_unverified",
        },
        parent_id=f"page:{page}",
    )


def test_g25_same_page_relative_positions_and_ambiguity():
    # 1. Right subplot axis label crossing page midline (G20 Q03 shape) + Q10 ambiguity
    q03_right = _page_image(5, 1, [0.444992, 0.688088, 0.800889, 0.863947])
    q03_left = _page_image(5, 2, [0.176471, 0.688093, 0.422344, 0.856755])
    assert object_choices(ObjectIntent(kind="image", direction="right"), [q03_right, q03_left]) == [
        q03_right
    ]
    assert object_choices(ObjectIntent(kind="image", direction="left"), [q03_right, q03_left]) == [
        q03_left
    ]
    assert object_choices(ObjectIntent(kind="image"), [q03_right, q03_left]) == [
        q03_right,
        q03_left,
    ]

    # 2. Left-right mirrored: left figure right label/legend crosses midline
    mirrored_left = _page_image(3, 1, [0.12, 0.28, 0.57, 0.64])
    mirrored_right = _page_image(3, 2, [0.60, 0.28, 0.93, 0.64])
    assert object_choices(
        ObjectIntent(kind="image", direction="left"), [mirrored_left, mirrored_right]
    ) == [mirrored_left]
    assert object_choices(
        ObjectIntent(kind="image", direction="right"), [mirrored_left, mirrored_right]
    ) == [mirrored_right]

    # 3. Top-bottom stacked (including minor y-midline crossing, and refusal for left/right)
    top = _page_image(4, 1, [0.20, 0.14, 0.80, 0.53])
    bottom = _page_image(4, 2, [0.20, 0.55, 0.80, 0.86])
    assert object_choices(ObjectIntent(kind="image", direction="top"), [top, bottom]) == [top]
    assert object_choices(ObjectIntent(kind="image", direction="bottom"), [top, bottom]) == [bottom]
    assert object_choices(ObjectIntent(kind="image", direction="right"), [top, bottom]) == []
    assert object_choices(ObjectIntent(kind="image", direction="left"), [top, bottom]) == []

    # 4. Single centered figure rejects directional queries, accepts non-directional
    centered = _page_image(5, 1, [0.20, 0.28, 0.80, 0.68])
    for direction in ("left", "right", "top", "bottom"):
        assert object_choices(ObjectIntent(kind="image", direction=direction), [centered]) == []
    assert object_choices(ObjectIntent(kind="image"), [centered]) == [centered]

    # 5. Single genuinely right-side figure with minor label crossing still resolves
    single_right = _page_image(6, 1, [0.45, 0.20, 0.86, 0.70])
    assert object_choices(ObjectIntent(kind="image", direction="right"), [single_right]) == [
        single_right
    ]

    # 6. Clearly overlapping candidates (2D overlap, nested inset, and midline-crossing 1D overlap)
    overlap_a = _page_image(7, 1, [0.15, 0.22, 0.65, 0.72])
    overlap_b = _page_image(7, 2, [0.38, 0.25, 0.88, 0.75])
    assert object_choices(
        ObjectIntent(kind="image", direction="right"), [overlap_a, overlap_b]
    ) == [overlap_a, overlap_b]

    inset_outer = _page_image(8, 1, [0.10, 0.15, 0.90, 0.85])
    inset_inner = _page_image(8, 2, [0.55, 0.25, 0.85, 0.55])
    assert object_choices(
        ObjectIntent(kind="image", direction="right"), [inset_outer, inset_inner]
    ) == [inset_outer, inset_inner]

    stacked_a = _page_image(9, 1, [0.25, 0.10, 0.75, 0.45])
    stacked_b = _page_image(9, 2, [0.35, 0.55, 0.85, 0.90])
    assert object_choices(
        ObjectIntent(kind="image", direction="right"), [stacked_a, stacked_b]
    ) == [stacked_a, stacked_b]


def test_g25_multipage_and_registered_region_rects_and_tables():
    # Multi-page grouping keeps page-relative decisions independent per page
    p1_center = _page_image(1, 1, [0.20, 0.25, 0.80, 0.75])
    p2_left = _page_image(2, 1, [0.08, 0.25, 0.40, 0.65])
    p2_right = _page_image(2, 2, [0.44, 0.25, 0.86, 0.65])
    assert object_choices(
        ObjectIntent(kind="image", direction="right"), [p1_center, p2_left, p2_right]
    ) == [p2_right]

    # Registered visible region overrides raw bbox for overlap/crossing checks
    left_tbl = _page_image(1, 1, [0.08, 0.20, 0.44, 0.60], object_type="table")
    right_tbl = _page_image(1, 2, [0.52, 0.20, 0.88, 0.60], object_type="table")
    regions = {("public-doc", 1, "page:1:image:2"): (0.43, 0.20, 0.88, 0.60)}
    assert object_choices(
        ObjectIntent(kind="table", direction="right"),
        [left_tbl, right_tbl],
        region_rects=regions,
    ) == [right_tbl]
    overlap_regions = {("public-doc", 1, "page:1:image:1"): (0.12, 0.20, 0.68, 0.60)}
    assert object_choices(
        ObjectIntent(kind="table", direction="right"),
        [left_tbl, right_tbl],
        region_rects=overlap_regions,
    ) == [left_tbl, right_tbl]
