"""幻灯片图片资源筛选测试。"""

from app.rag.slide_assets import classify_slide_assets
from app.schemas.document import (
    NormalizedBoundingBox,
    Slide,
    SlideAsset,
    SlideAssetRole,
)


def _asset(asset_id: str, digest: str, bbox: NormalizedBoundingBox) -> SlideAsset:
    return SlideAsset(
        asset_id=asset_id,
        media_type="image/png",
        path=f"assets/{asset_id}.png",
        sha256=digest,
        bbox=bbox,
    )


def test_repeated_corner_logo_is_decorative_and_large_unique_image_is_informative() -> None:
    logo_box = NormalizedBoundingBox(left=0.91, top=0.01, right=0.99, bottom=0.08)
    large_box = NormalizedBoundingBox(left=0.15, top=0.2, right=0.85, bottom=0.8)
    slides = [
        Slide(
            slide_id=f"slide-{index}",
            document_id="doc",
            slide_no=index,
            assets=[
                _asset(f"logo-{index}", "same-logo", logo_box),
                *(
                    [_asset("diagram", "unique-diagram", large_box)]
                    if index == 2
                    else []
                ),
            ],
        )
        for index in range(1, 5)
    ]

    classified = classify_slide_assets(slides)

    assert all(slide.assets[0].role is SlideAssetRole.DECORATIVE for slide in classified)
    diagram_slide = classified[1]
    assert diagram_slide.assets[1].role is SlideAssetRole.INFORMATIVE
    assert diagram_slide.visual_embedding_candidate is True

