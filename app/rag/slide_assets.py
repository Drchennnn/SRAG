"""幻灯片视觉资源的轻量级信息价值判定。"""

from __future__ import annotations

import math
from collections import Counter

from app.rag.chunker import estimate_token_count
from app.schemas.document import (
    Slide,
    SlideAsset,
    SlideAssetRole,
    SlideElementKind,
)


def classify_slide_assets(slides: list[Slide]) -> list[Slide]:
    """使用重复率、位置和面积标注装饰性与信息性资源。"""

    occurrences = Counter(
        asset.sha256
        for slide in slides
        for asset in {item.sha256: item for item in slide.assets}.values()
    )
    repeated_threshold = max(3, math.ceil(len(slides) * 0.3))
    result: list[Slide] = []
    for slide in slides:
        assets = [
            _classify_asset(asset, occurrences[asset.sha256], repeated_threshold)
            for asset in slide.assets
        ]
        result.append(
            slide.model_copy(
                update={
                    "assets": assets,
                    "visual_embedding_candidate": _needs_visual_embedding(slide, assets),
                }
            )
        )
    return result


def _classify_asset(
    asset: SlideAsset, occurrences: int, repeated_threshold: int
) -> SlideAsset:
    if asset.role is not SlideAssetRole.UNKNOWN:
        return asset

    reasons: list[str] = []
    if occurrences >= repeated_threshold:
        reasons.append("在多页重复出现")
    if asset.bbox and asset.bbox.area_ratio <= 0.05 and _is_near_corner(asset):
        reasons.append("位于角落且占比很小")
    if reasons:
        return asset.model_copy(
            update={"role": SlideAssetRole.DECORATIVE, "role_reasons": reasons}
        )

    if asset.bbox and asset.bbox.area_ratio >= 0.2:
        return asset.model_copy(
            update={
                "role": SlideAssetRole.INFORMATIVE,
                "role_reasons": ["单页占比较大且非重复资源"],
            }
        )
    return asset


def _is_near_corner(asset: SlideAsset) -> bool:
    if not asset.bbox:
        return False
    horizontal_edge = asset.bbox.left <= 0.12 or asset.bbox.right >= 0.88
    vertical_edge = asset.bbox.top <= 0.12 or asset.bbox.bottom >= 0.88
    return horizontal_edge and vertical_edge


def _needs_visual_embedding(slide: Slide, assets: list[SlideAsset]) -> bool:
    if any(
        element.kind
        in {SlideElementKind.CHART, SlideElementKind.FORMULA, SlideElementKind.TABLE}
        for element in slide.elements
    ):
        return True
    if any(asset.role is SlideAssetRole.INFORMATIVE for asset in assets):
        return True
    text = "\n".join(element.text for element in slide.elements if element.text)
    meaningful_images = [
        asset for asset in assets if asset.role is not SlideAssetRole.DECORATIVE
    ]
    return bool(meaningful_images) and estimate_token_count(text) < 40
