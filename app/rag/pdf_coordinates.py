"""Map native PDF page coordinates into the visible, rotated crop box."""

from __future__ import annotations

from math import isfinite

Rect = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]


def pdf_visible_matrix(crop: Rect, rotation: int) -> Matrix:
    left, bottom, right, top = crop
    width, height = right - left, top - bottom
    if not all(isfinite(value) for value in (*crop, width, height)) or width <= 0 or height <= 0:
        raise ValueError("invalid PDF crop box")
    if rotation == 0:
        return 1 / width, 0.0, 0.0, -1 / height, -left / width, top / height
    if rotation == 90:
        return 0.0, 1 / width, 1 / height, 0.0, -bottom / height, -left / width
    if rotation == 180:
        return -1 / width, 0.0, 0.0, 1 / height, right / width, -bottom / height
    if rotation == 270:
        return 0.0, -1 / width, -1 / height, 0.0, top / height, right / width
    raise ValueError("invalid PDF page rotation")


def transform_rect(rect: Rect, matrix: Matrix) -> Rect:
    a, b, c, d, e, f = matrix
    left, bottom, right, top = rect
    xs = (a * left + c * bottom + e, a * right + c * top + e)
    ys = (b * left + d * bottom + f, b * right + d * top + f)
    # These matrices are axis-aligned, with possible axis swaps or sign changes.
    return min(xs), min(ys), max(xs), max(ys)
