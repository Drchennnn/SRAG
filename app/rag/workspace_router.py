"""Stage 2: Pure 6-level intent and object routing over an already resolved TurnContext."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.rag.evidence_supplement import asks_for_definition_and_experiment
from app.rag.workspace_composition import asks_for_composition
from app.rag.workspace_objects import (
    _FACT as _OBJECT_FACT,
)
from app.rag.workspace_objects import (
    _IMAGE_NUMBER as _OBJ_IMAGE_NUMBER,
)
from app.rag.workspace_objects import (
    _OUTPUT_NUMBER as _OBJ_OUTPUT_NUMBER,
)
from app.rag.workspace_objects import (
    _PREVIEW as _OBJECT_PREVIEW,
)
from app.rag.workspace_objects import (
    ObjectIntent,
    object_intent,
)
from app.rag.workspace_versions import (
    _CURRENT as _CURRENT_VERSION,
)
from app.rag.workspace_versions import (
    _FILE_TOKEN as _VERSION_FILE_TOKEN,
)
from app.rag.workspace_versions import (
    _NEGATED_COMPARE as _VERSION_NEGATED_COMPARE,
)
from app.rag.workspace_versions import (
    _OLD_CHINESE as _OLD_CH_VERSION,
)
from app.rag.workspace_versions import (
    _OLD_VERSION_EN as _OLD_EN_VERSION,
)
from app.rag.workspace_versions import (
    _PLURAL_OBJECT as _VERSION_PLURAL_OBJECT,
)
from app.rag.workspace_versions import (
    _versions as _extract_version_numbers,
)
from app.rag.workspace_versions import (
    is_version_comparison,
    resolve_current_counterpart,
)
from app.schemas.workspace import ClarificationAnchor, DisplayAnchor
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource

if TYPE_CHECKING:
    from app.rag.workspace import WorkspaceService
    from app.rag.workspace_context import TurnContext


_ORDINAL = re.compile(r"第\s*([0-9]+|[一二两三四五六七八九十]+)\s*张(?:图|图片)")
_PREVIOUS_TURN = re.compile(r"刚才|上一条|前一条|上一轮|前一轮|上次")
_HISTORY_IMAGE = re.compile(r"刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述|此前|先前")
_IMAGE_REFERENCE = re.compile(
    r"(?:刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述|那|这|哪)"
    r"\s*(?:的\s*)?(?:[那这一]\s*)?(?:张\s*)?(?:图片|图像|图)"
    r"|(?:刚才|上一条|前一条|上一轮|前一轮|上次|此前|先前)"
    r"[^，。！？；,;!?\n]{0,16}?(?:读过|看过|展示)[^，。！？；,;!?\n]{0,10}?(?:图片|图像|图)"
)
_OUTPUT_ANAPHORA = re.compile(
    r"(?:刚才|上一条|前一条|上一轮|前一轮|上次|旧版?|此前|先前)"
    r"(?:读过的?|看过的?|展示的?)?(?:那个|这个|对应)?(?:原)?输出"
)
_CELL_OBJECT_QUESTION = re.compile(
    r"输出|图片|图像|图表|柱状图|折线图|表格|(?<![a-zA-Z])图(?![a-zA-Z])|数值|数据|结果|写了|内容|读|提取|识别"
)
_OUTPUT_SPECIFIC = re.compile(r"output[_\s:-]*\d+", re.I)
_TABLE_REFERENCE = re.compile(
    r"(?:刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述|那|这)"
    r"\s*(?:的\s*)?(?:[那这一]\s*)?(?:张\s*)?(?:表格|原表|表)"
)
_CURRENT_IMAGE = re.compile(r"当前|本页|本\s*cell|这里|这页|这个\s*cell", re.IGNORECASE)
_WHOLE_FILE = re.compile(
    r"这份(?:课件|文件|资料)|整份|全文|各页|\b(?:whole|entire) (?:file|document)\b", re.I
)


@dataclass(frozen=True)
class FollowDisplayRoute:
    anchor: DisplayAnchor
    method: str
    question: str


@dataclass(frozen=True)
class ResumeClarificationRoute:
    anchor: ClarificationAnchor


@dataclass(frozen=True)
class CompareVersionsRoute:
    pass


@dataclass(frozen=True)
class CurrentVersionCounterpartRoute:
    status: str
    targets: tuple[SnapshotSource, ...]
    document: SnapshotDocument | None


_EXPLICIT_ONLY_OLD = re.compile(
    r"(?:只看|仅看|只查|仅查|回看|回到|重看)\s*"
    r"(?:旧|上一|前一|上个|前个|历史|老|先前|此前)版(?:本)?",
    re.I,
)


@dataclass(frozen=True)
class HistoryDisplayReferenceRoute:
    ordinal_match: re.Match[str] | None
    explicit_history: bool
    old_version_only: bool = False
    target_versions: frozenset[int] = frozenset()


@dataclass(frozen=True)
class CurrentImageRoute:
    pass


@dataclass(frozen=True)
class StaleFocusRoute:
    pass


@dataclass(frozen=True)
class TargetObjectRoute:
    intent: ObjectIntent
    focus_only: bool = False
    from_demonstrative: bool = False


@dataclass(frozen=True)
class FocusedCellOutputsRoute:
    candidates: tuple[SnapshotSource, ...]


@dataclass(frozen=True)
class FollowTableRoute:
    pass


@dataclass(frozen=True)
class CompositionRoute:
    pass


@dataclass(frozen=True)
class SupplementRoute:
    pass


@dataclass(frozen=True)
class RetrieveOrGenerateRoute:
    mode: str


RouteDecision = (
    FollowDisplayRoute
    | ResumeClarificationRoute
    | CompareVersionsRoute
    | CurrentVersionCounterpartRoute
    | HistoryDisplayReferenceRoute
    | CurrentImageRoute
    | StaleFocusRoute
    | TargetObjectRoute
    | FocusedCellOutputsRoute
    | FollowTableRoute
    | CompositionRoute
    | SupplementRoute
    | RetrieveOrGenerateRoute
)


def route_turn(service: WorkspaceService, ctx: TurnContext) -> RouteDecision:
    """Resolve one deterministic RouteDecision using the 6-level priority ladder."""
    from app.rag.workspace import (
        _EXPERIMENT_TEXT,
        _FILE_MENTION,
        _affirmed_switch,
        _definition_topics,
        _location,
    )

    request = ctx.turn.request
    text = request.text

    # Level 1: Explicit UI anchors (zero regex).
    if isinstance(request.anchor, DisplayAnchor):
        return FollowDisplayRoute(request.anchor, "display_anchor", text)
    if isinstance(request.anchor, ClarificationAnchor):
        return ResumeClarificationRoute(request.anchor)

    # Level 2: Non-ask modes (search / summary).
    if request.kind != "ask":
        return RetrieveOrGenerateRoute(mode=request.kind)

    # Level 3: Cross-version comparison & explicit current-version counterpart recovery.
    if is_version_comparison(text):
        return CompareVersionsRoute()

    ordinal = _ORDINAL.search(text)
    explicit_location = _location(text, None)
    located = explicit_location is not None or bool(_FILE_MENTION.search(text))
    image_reference = bool(ordinal or _IMAGE_REFERENCE.search(text))
    output_anaphora = bool(
        not image_reference
        and not _EXPERIMENT_TEXT.search(text)
        and not _definition_topics(text)
        and _OUTPUT_ANAPHORA.search(text)
    )
    wants_current_counterpart = bool(
        (image_reference or output_anaphora)
        and (not located or _PREVIOUS_TURN.search(text))
        and _CURRENT_VERSION.search(text)
        and not _OLD_CH_VERSION.search(text)
        and not _OLD_EN_VERSION.search(text)
        and ordinal is None
    )
    if wants_current_counterpart:
        status, targets, v2_doc, _ = resolve_current_counterpart(
            service, ctx.owner, ctx.turn, ctx.session, question=text
        )
        if status in {"resolved", "ambiguous", "deleted"}:
            return CurrentVersionCounterpartRoute(status, tuple(targets), v2_doc)

    # Level 4: Explicit historical image/output references (never borrow current focus).
    explicit_history = bool(_HISTORY_IMAGE.search(text))
    cleaned_ver_text = _VERSION_FILE_TOKEN.sub(" ", text)
    explicit_versions = frozenset(_extract_version_numbers(cleaned_ver_text))
    wants_old_version = bool(
        _OLD_CH_VERSION.search(cleaned_ver_text) or _OLD_EN_VERSION.search(cleaned_ver_text)
    )
    explicit_old_num = bool(
        explicit_versions
        and not _CURRENT_VERSION.search(cleaned_ver_text)
        and any(ver < doc.version for doc in ctx.documents.values() for ver in explicit_versions)
    )
    old_version_only = bool(
        _EXPLICIT_ONLY_OLD.search(cleaned_ver_text)
        or (
            wants_old_version
            and not _CURRENT_VERSION.search(cleaned_ver_text)
            and not _VERSION_NEGATED_COMPARE.search(cleaned_ver_text)
        )
        or explicit_old_num
    )
    if (
        (explicit_history and image_reference)
        or output_anaphora
        or (old_version_only and (image_reference or output_anaphora))
    ) and (not located or _PREVIOUS_TURN.search(text) or old_version_only):
        return HistoryDisplayReferenceRoute(
            ordinal_match=ordinal,
            explicit_history=True,
            old_version_only=old_version_only,
            target_versions=explicit_versions if old_version_only else frozenset(),
        )

    # Level 5: Focus-first disambiguation, stale-focus guard, and explicit object/table routing.
    switching = bool(_FILE_MENTION.search(text) and _affirmed_switch(text))
    focus = ctx.focus_state

    # Explicit "当前/本页/本 cell" image reference without explicit file/page location.
    if image_reference and not located and _CURRENT_IMAGE.search(text):
        if focus.status == "stale":
            return StaleFocusRoute()
        return CurrentImageRoute()

    # Bare ordinal image reference ("第一张图" / "第二张图") without explicit location.
    if ordinal is not None and not located:
        if ctx.history.latest_visible_message is not None:
            return HistoryDisplayReferenceRoute(ordinal_match=ordinal, explicit_history=False)
        if focus.status == "stale":
            return StaleFocusRoute()
        return CurrentImageRoute()

    # Demonstrative image reference (e.g. "这张图...") — Focus-first disambiguation (Q04).
    if image_reference and not located:
        if focus.status == "valid":
            latest_msg = ctx.history.latest_visible_message
            history_matches_focus_children = False
            if latest_msg is not None and focus.node is not None and focus.child_images:
                from app.rag.workspace import _focus_matches_display

                visible_ids = set(latest_msg.get("visible_display_ids") or ())
                visible_blocks = [
                    blk
                    for blk in latest_msg.get("blocks", [])
                    if blk.get("type") == "source_image" and blk.get("display_id") in visible_ids
                ]
                child_keys = {
                    (src.document_id, src.version, src.source_id) for src in focus.child_images
                }
                displayed_keys = {
                    (
                        blk["display_source"]["document_id"],
                        blk["display_source"]["version"],
                        blk["display_source"]["source_id"],
                    )
                    for blk in visible_blocks
                    if isinstance(blk.get("display_source"), dict)
                }
                history_matches_focus_children = (
                    bool(visible_blocks)
                    and all(
                        _focus_matches_display(focus.node, latest_msg, blk)
                        for blk in visible_blocks
                    )
                    and child_keys <= displayed_keys
                )
            focused_multi_or_single = (
                focus.source is not None
                and focus.source.kind in {"page", "cell"}
                and len(focus.child_images) >= 1
                and not history_matches_focus_children
            )
            doc_focus_no_history = (
                focus.source is None
                and latest_msg is None
                and len(focus.child_images) >= 1
            )
            if focused_multi_or_single or doc_focus_no_history:
                intent = object_intent(text, include_reference=True) or ObjectIntent(kind="image")
                if intent.kind == "image":
                    return TargetObjectRoute(
                        intent=intent,
                        focus_only=focus.source is not None,
                        from_demonstrative=True,
                    )
        if ctx.history.latest_visible_message is not None:
            return HistoryDisplayReferenceRoute(ordinal_match=None, explicit_history=False)
        if focus.status == "stale":
            return StaleFocusRoute()
        return CurrentImageRoute()

    # Table reference follow-up (falls back in dispatcher if no visible table exists).
    if _TABLE_REFERENCE.search(text) and (not located or _PREVIOUS_TURN.search(text)):
        return FollowTableRoute()

    # Stale focus guard for ordinary new questions (e.g. "看当前 Cell").
    if focus.status == "stale" and not switching:
        return StaleFocusRoute()

    # Cross-modal three-role composition or offline definition+experiment supplement.
    if asks_for_composition(text):
        return CompositionRoute()

    whole_file = bool(_WHOLE_FILE.search(text))
    effective_location = explicit_location or (
        None if whole_file else _location("", focus.node if focus.status == "valid" else None)
    )
    if (
        effective_location is None
        and not service.settings.workspace_answer_enabled
        and asks_for_definition_and_experiment(text)
    ):
        return SupplementRoute()

    # Explicit object intent (image / table / caption / formula) with snapshot existence guard.
    target = object_intent(text)
    if target is not None:
        if target.kind in {"caption", "formula"} and not any(
            src.kind == "page_object" and src.locator.get("object_type") == target.kind
            for src in ctx.scope_snapshot.sources
        ):
            target = None
        if target is not None:
            return TargetObjectRoute(intent=target)

    # Focused multi-output Notebook cell general question (328ae63).
    if (
        explicit_location is None
        and not whole_file
        and focus.status == "valid"
        and focus.source is not None
        and focus.source.kind == "cell"
        and len(focus.child_outputs) >= 2
        and _CELL_OBJECT_QUESTION.search(text)
        and not (_OBJECT_PREVIEW.search(text) and not _OBJECT_FACT.search(text))
        and not _OBJ_OUTPUT_NUMBER.search(text)
        and not _OBJ_IMAGE_NUMBER.search(text)
        and not _OUTPUT_SPECIFIC.search(text)
        and not _VERSION_PLURAL_OBJECT.search(text)
        and not _EXPERIMENT_TEXT.search(text)
        and not _definition_topics(text)
    ):
        file_names = [
            doc.file_name
            for doc in ctx.documents.values()
            if doc.file_name.casefold() in text.casefold()
        ]
        if not file_names or (
            focus.node is not None and set(file_names) == {focus.node.document.file_name}
        ):
            return FocusedCellOutputsRoute(candidates=focus.child_outputs)

    # Level 6: General retrieval / generation pipeline.
    return RetrieveOrGenerateRoute(mode="ask")
