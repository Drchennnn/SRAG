"""Recover column-aligned PDF tables from native horizontal rules and char coordinates."""

from __future__ import annotations

from typing import Any

_RULE_SPAN_TOLERANCE = 5.0
_LINE_Y_TOLERANCE = 3.5
_MIN_RULE_WIDTH = 24.0
_MIN_TABLE_HEIGHT = 16.0
_UNIFORM_GRID_RATIO = 1.15


def extract_horizontal_rule_tables(
    horizontal: list[tuple[float, float, float, int]],
    vertical: list[tuple[float, float, float, int]],
    text_page: Any,
    page_bbox: tuple[float, float, float, float],
    existing_boxes: list[tuple[float, float, float, float]],
    bbox_to_visible: Any,
) -> list[dict[str, Any]]:
    """Extract borderless (e.g., booktabs) tables bounded by aligned horizontal rules."""
    groups = _group_horizontal_rules(horizontal)
    if not groups:
        return []
    all_chars = _page_char_boxes(text_page)
    if not all_chars:
        return []

    tables: list[dict[str, Any]] = []
    for group in groups:
        ys = sorted((line[2] for line in group), reverse=True)
        if len(ys) < 3 or (ys[0] - ys[-1]) < _MIN_TABLE_HEIGHT:
            continue
        gaps = [ys[i] - ys[i + 1] for i in range(len(ys) - 1)]
        if len(ys) >= 5 and min(gaps) > 0 and max(gaps) / min(gaps) <= _UNIFORM_GRID_RATIO:
            continue
        box = (
            min(line[0] for line in group),
            ys[-1],
            max(line[1] for line in group),
            ys[0],
        )
        if any(_boxes_overlap(box, existing) for existing in existing_boxes):
            continue
        if _has_vertical_lines_inside(vertical, box):
            continue
        parsed = _parse_rule_table(all_chars, text_page, box, ys)
        if parsed is None and len(group) > 3:
            ordered = sorted(group, key=lambda line: -line[2])
            for start in range(1, len(ordered) - 2):
                sub_group = ordered[start:]
                sub_ys = [line[2] for line in sub_group]
                if (sub_ys[0] - sub_ys[-1]) < _MIN_TABLE_HEIGHT:
                    continue
                sub_box = (
                    min(line[0] for line in sub_group),
                    sub_ys[-1],
                    max(line[1] for line in sub_group),
                    sub_ys[0],
                )
                if any(_boxes_overlap(sub_box, existing) for existing in existing_boxes):
                    continue
                if _has_vertical_lines_inside(vertical, sub_box):
                    continue
                parsed = _parse_rule_table(all_chars, text_page, sub_box, sub_ys)
                if parsed is not None:
                    group, ys, box = sub_group, sub_ys, sub_box
                    break
        if parsed is None:
            continue
        rows, title = parsed
        tables.append(
            {
                "title": title,
                "rows": rows,
                "bbox": bbox_to_visible(list(box), page_bbox),
                "native_bbox": list(box),
                "native_content_order": min(line[3] for line in group),
                "table_rule_style": "horizontal_rules",
            }
        )
    return tables


def _group_horizontal_rules(
    horizontal: list[tuple[float, float, float, int]],
) -> list[list[tuple[float, float, float, int]]]:
    groups: list[list[tuple[float, float, float, int]]] = []
    filtered = (item for item in horizontal if item[1] - item[0] >= _MIN_RULE_WIDTH)
    for line in sorted(filtered, key=lambda item: -item[2]):
        for group in groups:
            if (
                abs(group[0][0] - line[0]) <= _RULE_SPAN_TOLERANCE
                and abs(group[0][1] - line[1]) <= _RULE_SPAN_TOLERANCE
            ):
                if not any(abs(existing[2] - line[2]) <= 2.0 for existing in group):
                    group.append(line)
                break
        else:
            groups.append([line])
    return [group for group in groups if len(group) >= 3]


def _has_vertical_lines_inside(
    vertical: list[tuple[float, float, float, int]],
    box: tuple[float, float, float, float],
) -> bool:
    left, bottom, right, top = box
    return any(
        left - 3.0 <= vx <= right + 3.0 and vb <= top + 3.0 and vt >= bottom - 3.0
        for vx, vb, vt, _ in vertical
    )


def _boxes_overlap(
    left_box: tuple[float, float, float, float],
    right_box: tuple[float, float, float, float],
) -> bool:
    return min(left_box[2], right_box[2]) > max(left_box[0], right_box[0]) and min(
        left_box[3], right_box[3]
    ) > max(left_box[1], right_box[1])


def _page_char_boxes(text_page: Any) -> list[tuple[str, tuple[float, float, float, float]]]:
    count = text_page.count_chars()
    chars: list[tuple[str, tuple[float, float, float, float]]] = []
    for index in range(count):
        ch = text_page.get_text_range(index, 1)
        if ch in {"\r", "\n"}:
            continue
        chars.append((ch, text_page.get_charbox(index)))
    return chars


def _extract_phrases(
    all_chars: list[tuple[str, tuple[float, float, float, float]]],
    text_page: Any,
    box: tuple[float, float, float, float],
) -> list[tuple[float, float, list[tuple[str, float, float]]]] | None:
    left, bottom, right, top = box
    inside = []
    for ch, cb in all_chars:
        cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
        if left - 2.0 <= cx <= right + 2.0 and bottom - 2.0 <= cy <= top + 2.0:
            inside.append((ch, cb))
    if not inside:
        return None

    lines: list[dict[str, Any]] = []
    for ch, cb in inside:
        cy = (cb[1] + cb[3]) / 2
        for line in lines:
            if (
                abs(line["y0"] - cb[1]) <= _LINE_Y_TOLERANCE
                or abs(line["y"] - cy) <= _LINE_Y_TOLERANCE
            ):
                line["items"].append((ch, cb))
                break
        else:
            lines.append({"y": cy, "y0": cb[1], "items": [(ch, cb)]})
    lines.sort(key=lambda item: -item["y"])

    # Reject regions that clip running prose mid-word across the left/right rule boundary.
    for line in lines:
        xs = sorted(line["items"], key=lambda item: item[1][0])
        first_x0 = xs[0][1][0]
        if abs(first_x0 - left) <= 3.0:
            for ch, cb in all_chars:
                if (
                    ch != " "
                    and abs((cb[1] + cb[3]) / 2 - line["y"]) <= _LINE_Y_TOLERANCE
                    and left - 8.0 <= cb[2] <= first_x0
                    and first_x0 - cb[2] <= 2.5
                ):
                    return None

    out: list[tuple[float, float, list[tuple[str, float, float]]]] = []
    for line in lines:
        items = sorted(line["items"], key=lambda item: item[1][0])
        heights = sorted(cb[3] - cb[1] for _, cb in items if cb[3] > cb[1])
        med_h = heights[len(heights) // 2] if heights else 8.0
        phrases = _line_phrases(items, text_page, med_h)
        if phrases:
            out.append((line["y"], med_h, phrases))
    return out


def _line_phrases(
    items: list[tuple[str, tuple[float, float, float, float]]],
    text_page: Any,
    med_h: float,
) -> list[tuple[str, float, float]]:
    gap_thr = max(4.2, 0.55 * med_h)
    phrases: list[tuple[str, float, float]] = []
    buf: list[str] = []
    px0: float | None = None
    px1 = 0.0
    py0: float | None = None
    py1 = 0.0

    def flush() -> None:
        nonlocal buf, px0, py0
        if not buf or px0 is None or py0 is None:
            buf, px0, py0 = [], None, None
            return
        bounded = (
            text_page.get_text_bounded(px0 - 0.5, py0 - 0.5, px1 + 0.5, py1 + 0.5)
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        text = " ".join(bounded.split()) if bounded else "".join(buf).strip()
        if text:
            phrases.append((text, px0, px1))
        buf, px0, py0 = [], None, None

    for ch, cb in items:
        if ch == " ":
            if buf and buf[-1] != " ":
                buf.append(" ")
            continue
        if buf and (cb[0] - px1) > gap_thr:
            flush()
        if px0 is None:
            px0, py0 = cb[0], cb[1]
        else:
            py0 = min(py0, cb[1])
        px1 = max(px1, cb[2])
        py1 = max(py1, cb[3])
        buf.append(ch)
    flush()
    return phrases


def _parse_rule_table(
    all_chars: list[tuple[str, tuple[float, float, float, float]]],
    text_page: Any,
    box: tuple[float, float, float, float],
    ys: list[float],
) -> tuple[list[list[str]], str] | None:
    lines = _extract_phrases(all_chars, text_page, box)
    if not lines or len(lines) < 2:
        return None
    midrule = ys[1] if (ys[0] - ys[1]) < 0.55 * (ys[0] - ys[-1]) else None
    header_lines = [line for line in lines if midrule is not None and line[0] > midrule]
    body_lines = [line for line in lines if midrule is None or line[0] < midrule]
    if not header_lines and body_lines:
        header_lines, body_lines = [body_lines[0]], body_lines[1:]
    multi_body = [line for line in body_lines if len(line[2]) >= 2]
    if len(multi_body) < 2:
        return None

    spans = sorted((phrase[1], phrase[2]) for line in multi_body for phrase in line[2])
    cols: list[tuple[float, float]] = []
    for x0, x1 in spans:
        if cols and x0 <= cols[-1][1] + 1.5:
            cols[-1] = (cols[-1][0], max(cols[-1][1], x1))
        else:
            cols.append((x0, x1))
    if len(cols) < 2:
        return None

    bands = [
        (
            box[0] - 2.0 if i == 0 else (cols[i - 1][1] + cx0) / 2,
            box[2] + 2.0 if i == len(cols) - 1 else (cx1 + cols[i + 1][0]) / 2,
        )
        for i, (cx0, cx1) in enumerate(cols)
    ]

    def match_cols(x0: float, x1: float) -> list[int]:
        matched = [
            i
            for i, (bl, br) in enumerate(bands)
            if min(x1, br) - max(x0, bl) >= min(4.0, 0.35 * max(1.0, x1 - x0))
        ]
        if matched:
            return matched
        mid = (x0 + x1) / 2
        return [
            min(
                range(len(bands)),
                key=lambda i: max(0.0, bands[i][0] - mid, mid - bands[i][1]),
            )
        ]

    col_headers = _build_headers(header_lines, cols, bands, match_cols)
    rows: list[list[str]] = [col_headers]
    prev_y: float | None = None
    ambiguous_lines = 0
    for y, med_h, phrases in body_lines:
        row = [""] * len(cols)
        spanning = False
        for text, x0, x1 in phrases:
            matched = match_cols(x0, x1)
            if len(matched) > 1:
                if len(phrases) == 1:
                    row[0] = text
                    spanning = True
                else:
                    ambiguous_lines += 1
                    ci = matched[0]
                    row[ci] = f"{row[ci]} {text}".strip() if row[ci] else text
            else:
                ci = matched[0]
                row[ci] = f"{row[ci]} {text}".strip() if row[ci] else text
        if (
            not spanning
            and len(phrases) == 1
            and len(rows) > 1
            and prev_y is not None
            and (prev_y - y) <= 1.6 * med_h
        ):
            nonempty = [i for i, value in enumerate(row) if value]
            if len(nonempty) == 1 and rows[-1][nonempty[0]]:
                ci = nonempty[0]
                rows[-1][ci] = f"{rows[-1][ci]} {row[ci]}".strip()
                prev_y = y
                continue
        rows.append(row)
        prev_y = y

    if ambiguous_lines >= 2:
        return None
    min_header_cols = 1 if len(rows[0]) == 2 else 2
    if sum(bool(value) for value in rows[0]) < min_header_cols:
        return None
    if not any(sum(bool(value) for value in row) >= 2 for row in rows[1:]):
        return None
    title = " | ".join(value for value in rows[0] if value)
    return rows, title


def _build_headers(
    header_lines: list[tuple[float, float, list[tuple[str, float, float]]]],
    cols: list[tuple[float, float]],
    bands: list[tuple[float, float]],
    match_cols: Any,
) -> list[str]:
    num_cols = len(cols)
    if len(header_lines) >= 2 and len(header_lines[0][2]) < len(header_lines[-1][2]):
        leaf_cols = sorted(
            {
                ci
                for _, _, phrases in header_lines[1:]
                for _, x0, x1 in phrases
                for ci in match_cols(x0, x1)
            }
        )
        upper = header_lines[0][2]
        if upper and len(upper) < len(leaf_cols):
            # Assign each leaf column to its enclosing centered spanner header.
            spanner_by_col: dict[int, str] = {}
            first_spanner_col = min(match_cols(upper[0][1], upper[0][2])[0], leaf_cols[0])
            centers = [(cx0 + cx1) / 2 for cx0, cx1 in cols]
            upper_mids = [(x0 + x1) / 2 for _, x0, x1 in upper]
            for ci in leaf_cols:
                if ci < first_spanner_col:
                    continue
                best = min(range(len(upper)), key=lambda idx: abs(upper_mids[idx] - centers[ci]))
                spanner_by_col[ci] = upper[best][0]
            leaf_headers = [""] * num_cols
            for _, _, phrases in header_lines[1:]:
                for text, x0, x1 in phrases:
                    for ci in match_cols(x0, x1):
                        leaf_headers[ci] = (
                            f"{leaf_headers[ci]} {text}".strip() if leaf_headers[ci] else text
                        )
            return [
                f"{spanner_by_col[ci]}: {leaf_headers[ci]}"
                if ci in spanner_by_col and leaf_headers[ci]
                else (spanner_by_col.get(ci) or leaf_headers[ci])
                for ci in range(num_cols)
            ]

    col_headers = [""] * num_cols
    for _, _, phrases in header_lines:
        for text, x0, x1 in phrases:
            for ci in match_cols(x0, x1):
                col_headers[ci] = f"{col_headers[ci]} {text}".strip() if col_headers[ci] else text
    return col_headers
