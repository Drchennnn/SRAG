"""Compare a saved historical workspace reference with its current publication."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.rag.source_evidence import DisplayRecord, EvidenceRequest, HistoricalSourceGrant
from app.rag.structure_search import StructureLocation
from app.rag.workspace_objects import ObjectIntent, object_choices, object_intent
from app.schemas.local import SourceDraft, SourceKey
from app.schemas.turn_evidence import TurnEvidence, TurnRead
from app.schemas.workspace import WorkspaceError, inaccessible
from app.stores.structure_snapshot import SnapshotDocument, SnapshotRequest, SnapshotSource

if TYPE_CHECKING:
    from app.rag.workspace import Draft, WorkspaceService
    from app.stores.local_library import HistoricalSource
    from app.stores.workspace import Attachment, Session, Turn

_OLD_STEM = (
    r"(?:旧|历史|老|先前|此前|早先|早前|以前|之前|上一|前一|上个|前个|上一个|前一个)"
    r"(?:的|那个?|这个?)?版(?:本)?"
)
_OLD = re.compile(
    _OLD_STEM + r"|\b(?:old|older|previous|prior|historical|archived|earlier)\b",
    re.I,
)
_OLD_CHINESE = re.compile(_OLD_STEM)
_OLD_VERSION_EN = re.compile(
    r"\b(?:old|older|previous|prior|historical|archived|earlier)\s+"
    r"(?:versions?|editions?|revisions?|releases?|v[1-9][0-9]*)\b",
    re.I,
)
_OLD_COMPARATIVE_PREFIX = re.compile(
    rf"(?:和|跟|与|同|(?<!比)较|(?<![相对])比(?:起)?|相比于?)\s*{_OLD_STEM}"
    r"|\b(?:compar(?:e|ed|ing)\s+(?:with|to|against)|against|from)\s+(?:the\s+)?"
    r"(?:old|older|previous|prior|historical|archived|earlier)\s+"
    r"(?:versions?|editions?|revisions?|releases?|v[1-9][0-9]*)\b",
    re.I,
)
_OLD_LOCATIVE = re.compile(
    rf"{_OLD_STEM}\s*(?:里|中|内|里面|之中|当中)"
    r"|\bin\s+(?:the\s+)?(?:old|older|previous|prior|historical|archived|earlier)\s+"
    r"(?:versions?|editions?|revisions?|releases?)\b",
    re.I,
)
_PLURAL_OBJECT = re.compile(
    r"(?:这两|那两|这二|那二|两|2|双|多)(?:张|幅|个|份|种|组|条|项)?(?:图片|图像|子图|插图|图表|图|表格|表|输出|Cell|公式)"
    r"|(?:图|表|输出|Cell)\s*[0-9一二三四五六七八九十]+\s*(?:和|与|及|跟|vs\.?)\s*(?:图|表|输出|Cell)\s*[0-9一二三四五六七八九十]+"
    r"|\b(?:two|both|multiple|several)\s+(?:figures?|images?|plots?|subplots?|charts?|tables?|outputs?|cells?|panels?)\b",
    re.I,
)
_PREVIOUS = re.compile(
    r"(?:上一|前一|上个|前个|上一个|前一个)(?:的|那个?|这个?)?版(?:本)?|\b(?:previous|prior)\b",
    re.I,
)
_CURRENT = re.compile(
    r"(?:当前|现行|新|最新|现在这|这一)版(?:本)?|\b(?:current|latest|new)\b",
    re.I,
)
_TWO_VERSIONS = re.compile(
    r"(?:前后|新旧)?(?:两|两个|2\s*个)版(?:本)?"
    r"|\b(?:two|both)\s+(?:versions?|editions?|revisions?|releases?)\b",
    re.I,
)
_NON_COMPARATIVE = re.compile(
    r"(?<![和跟与同])比较\s*(?:重要|关键|核心|常见|典型|明显|突出|主要|详细|完整|基础|简单|复杂|大|小|多|少|高|低|快|慢|早|晚|好|差|准确|稳定|显著)"
    r"|变化(?:趋势|曲线|规律|过程|情况(?!(?:如何|怎样|怎么|[？?]|$)))"
    r"|不同(?:的)?(?:版本|参数|配置|条件|情况|场景|阶段|设置|实验|数据集|类型|方式|方法|指标|结果|方面|时)"
)
_NEGATED_COMPARE = re.compile(
    rf"(?:不用|不要|无需|不必|无须|勿|别|不再|不需|不作|不做|并非|不是(?:要|问|在)?|忽略|跳过|抛开)"
    rf"(?:再|去)?"
    rf"(?:(?:和|跟|与|同)?\s*(?:[^，。！？?\n]{{0,14}}?{_OLD_STEM}|(?:和|跟|与|同)\s*[^，。！？?\n]{{0,12}}?))?"
    r"(?:比较|对比|相比|比对|对照|差异|区别|变化|不同|比)"
    r"|\b(?:don['’]t|do not|never|without|not|skip|ignore)\s+"
    r"(?:compar(?:e|ing)|contrast(?:ing)?|diff(?:erences?)?)\b",
    re.I,
)
_COMPARE = re.compile(
    rf"比较|对比|相比|比对|对照|差异|区别|异同|变化|变动|改动|改变|出入|不同|怎么变了?|变了(?:多少|什么|吗)"
    r"|(?:和|跟|与|同)[^，。！？?\n]{0,18}?比(?![如例特重较])"
    rf"|(?:比|较)(?:起)?\s*(?:{_OLD_STEM}|v(?:ersion\s*)?[1-9][0-9]{{0,5}}|第\s*[1-9][0-9]{{0,5}}\s*版)"
    r"|\b(?:compar(?:e|es|ed|ing|ison)|differ(?:s|ed|ing|ence|ences|ent)?|chang(?:e|ed|es|ing)|versus|vs\.?)\b",
    re.I,
)
_VERSION_COMPARE = re.compile(
    r"(?:和|跟|与|同|(?<!比)较|相比|对比|(?<!对)比(?:起)?)\s*(?:v(?:ersion\s*)?[1-9][0-9]{0,5}|第\s*[1-9][0-9]{0,5}\s*版)"
    r"|(?:v(?:ersion\s*)?[1-9][0-9]{0,5}|第\s*[1-9][0-9]{0,5}\s*版)\s*(?:和|跟|与|同)?\s*(?:相比|对比|比)"
    r"|\b(?:compared?\s+(?:with|to|against)|against|from)\s+v(?:ersion\s*)?[1-9][0-9]{0,5}\b",
    re.I,
)
_BARE_COMPARISON = re.compile(
    rf"^(?:和|跟|与|同)?{_OLD_STEM}"
    r"(?:(?:相比|比较|对比|比)(?:起来)?(?:呢|怎样|如何|有什么变化)?|有什么变化)[？?。\s]*$"
)
_VERSION = re.compile(
    r"(?<![A-Za-z0-9])v(?:ersion\s*)?([1-9][0-9]{0,5})(?![A-Za-z0-9])"
    r"|(?:第\s*)?([1-9][0-9]{0,5})\s*版",
    re.I,
)
_FILE_TOKEN = re.compile(
    r"""["'“‘「『][^"'”’」』\n]+\.(?:pdf|pptx|ipynb)["'”’」』]"""
    r"|(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:pdf|pptx|ipynb)(?![a-z0-9_-])"
    r"|(?:(?<=^)(?!(?:请)?(?:比较|对比|相比))|(?<=[\s\"'“‘「『(（:：,，、和与及]))[\w.-]+\.(?:pdf|pptx|ipynb)(?![a-z0-9_-])"
    r"|(?:第\s*)?[1-9][0-9]{0,5}\s*版\.(?:pdf|pptx|ipynb)(?![a-z0-9_-])",
    re.I,
)
_LABEL_JOIN = re.compile(r"\s*(?:[:：=（(]\s*)?")
_OUTPUT = re.compile(r"输出|\b(?:output|stdout)\b", re.I)
_PAGE = re.compile(r"\bpage\s*([1-9][0-9]{0,3})\b", re.I)
_X = re.compile(r"横轴|\bx(?:[- ]?axis)\b", re.I)
_Y = re.compile(r"纵轴|\by(?:[- ]?axis)\b", re.I)
_UNIT = re.compile(r"单位|量纲|\bunits?\b", re.I)
_LABEL = re.compile(r"标签|名称|\b(?:label|title|name)\b", re.I)
_EXCERPT = 900


@dataclass(frozen=True)
class _Previous:
    message_id: str
    evidence: TurnEvidence
    source: SnapshotSource
    original: HistoricalSource
    attachment: Attachment | None


def is_version_comparison(text: str) -> bool:
    request_text = _FILE_TOKEN.sub(" ", text)
    if _BARE_COMPARISON.fullmatch(request_text.strip()):
        return True
    if _NEGATED_COMPARE.search(request_text):
        return False
    comparative_text = _NON_COMPARATIVE.sub(" ", request_text)
    if not _COMPARE.search(comparative_text):
        return False
    versions = _versions(request_text)
    has_current = bool(_CURRENT.search(request_text))
    has_old = bool(_OLD_CHINESE.search(request_text) or _OLD_VERSION_EN.search(request_text))
    has_old_pair = bool(_OLD.search(request_text) and has_current)
    has_single_version_compare = len(versions) == 1 and bool(
        has_current or _VERSION_COMPARE.search(request_text)
    )
    if not (
        has_old_pair
        or has_old
        or len(versions) > 1
        or has_single_version_compare
        or _TWO_VERSIONS.search(request_text)
    ):
        return False
    if (
        not has_current
        and len(versions) < 2
        and not _TWO_VERSIONS.search(request_text)
        and not _OLD_COMPARATIVE_PREFIX.search(request_text)
        and _OLD_LOCATIVE.search(request_text)
        and _PLURAL_OBJECT.search(request_text)
    ):
        return False
    return True


def _versions(text: str) -> set[int]:
    return {int(latin or chinese) for latin, chinese in _VERSION.findall(text)}


def _labeled_versions(text: str, role: re.Pattern[str]) -> set[int]:
    versions = list(_VERSION.finditer(text))
    numbers = set()
    for label in role.finditer(text):
        match = next(
            (
                version
                for version in versions
                if label.end() <= version.start()
                and _LABEL_JOIN.fullmatch(text[label.end() : version.start()])
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    version
                    for version in reversed(versions)
                    if version.end() <= label.start()
                    and _LABEL_JOIN.fullmatch(text[version.end() : label.start()])
                ),
                None,
            )
        if match:
            numbers.add(int(match[1] or match[2]))
    return numbers


def compare_versions(service: WorkspaceService, owner: str, turn: Turn, session: Session) -> Turn:
    from app.rag.workspace import Draft, _doc_ref, _location, _object_file_names

    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()
    question = turn.request.text
    implicit = bool(_BARE_COMPARISON.fullmatch(question))
    draft = Draft(session.session_id)
    trace: dict[str, Any] = {"route": "version_comparison", "external_call_count": 0}
    scope = tuple(session.course_ids)
    snapshot = service.snapshots.read(SnapshotRequest(scope, tuple(session.document_ids)))
    names = _object_file_names(question, snapshot.documents)
    if names is None or len(set(names)) > 1:
        draft.notice("comparison_scope", "找不到可使用的指定文件，请核对课程和资料范围。")
        return service._complete(turn, draft, trace=trace)
    focus = None
    if not names and session.focus_id:
        focus = service._catalog(session.session_id).nodes.get(session.focus_id)
        if focus is not None:
            names = [focus.document.file_name]
    if names:
        snapshot = service.snapshots.read(
            SnapshotRequest(
                scope, tuple(doc.document_id for doc in snapshot.documents), tuple(names)
            )
        )
    if snapshot.resolution_status != "resolved" or len(snapshot.documents) != 1:
        note = (
            "先前定位已随版本变更，请明确要比较的文件及对象。"
            if not names and session.focus_id and focus is None and not implicit
            else "请先选定范围内可以比较的一份资料。"
        )
        draft.notice("comparison_scope", note)
        return service._complete(turn, draft, trace=trace)

    request_text = question
    for name in names:
        request_text = re.sub(re.escape(name), " ", request_text, flags=re.I)
    location = _location(request_text, None)
    if location is None and (page := _PAGE.search(request_text)):
        location = StructureLocation(page=int(page[1]))
    location = location or _location("", focus)
    intent = object_intent(request_text, include_reference=True)
    image = bool(intent and intent.kind == "image")
    versions = _versions(request_text)
    if len(versions) > 2:
        draft.notice("comparison_scope", "请仅指定一个旧版本和当前版本。")
        return service._complete(turn, draft, trace=trace)
    old_labels = _labeled_versions(request_text, _OLD)
    current_labels = _labeled_versions(request_text, _CURRENT)
    docs = {
        doc.document_id: doc
        for doc in snapshot.documents
        if (not current_labels or doc.version in current_labels)
        and doc.version not in old_labels
        and (len(versions) < 2 or doc.version in versions)
    }
    single_baseline_conflict = (
        len(versions) == 1
        and not old_labels
        and not current_labels
        and not _OLD.search(request_text)
        and all(doc.version in versions for doc in docs.values())
    )
    if len(old_labels) > 1 or len(current_labels) > 1 or not docs or single_baseline_conflict:
        draft.notice(
            "comparison_version_mismatch",
            "标明的旧版与当前版不符合当前发布状态，请核对两个版本。",
        )
        return service._complete(turn, draft, trace=trace)
    history = _completed_history(service, owner, turn, session)
    current = [
        source
        for source in snapshot.sources
        if source.document_id in docs
        and _at_location(source, location)
        and (not _OUTPUT.search(request_text) or image or source.kind == "notebook_output")
    ]
    if intent:
        regions = {}
        for source in current:
            key = _key(source)
            identity = key.document_id, key.version, key.source_id
            region_id = service.region_ids.get(identity)
            region = service.evidence.regions.get(region_id) if region_id else None
            if (
                region
                and region.source == key
                and region.content_sha256 == docs[key.document_id].content_sha256
                and region.coordinate_frame == "visible_page_norm_v1"
            ):
                regions[identity] = region.target_rect
        current = object_choices(intent, current, region_rects=regions)
    old = _old_candidates(service, session, history, docs, versions, request_text, image)
    paired = [
        (past, candidate)
        for past in old
        for candidate in current
        if _counterpart(past, candidate, location, intent)
    ]
    if location is None and not image and not intent and len(paired) > 1:
        latest = paired[0][0].message_id
        recent = [(older, newer) for older, newer in paired if older.message_id == latest]
        keys = {
            (_key(older.source).document_id, older.source.version, newer.version)
            for older, newer in recent
        }
        if len(keys) == 1 and len(recent) <= 8:
            paired = recent
            if len(paired) > 1:
                return _compare_native_group(service, turn, draft, docs, paired, trace)
    if len(paired) != 1:
        code = "comparison_missing_history" if not old else "comparison_ambiguous"
        text = (
            "本会话没有范围内、可核对的旧版引用。请从旧回答定位原件后再比较。"
            if not old
            else "无法唯一对应两版对象。请选择一份资料，并明确页、表格或 Cell 内的输出位置。"
        )
        draft.notice(code, text)
        return service._complete(turn, draft, trace=trace)

    past, selected = paired[0]
    document = docs[selected.document_id]
    revision = past.evidence.revision
    history_ref = {
        "document_id": revision.source.document_id,
        "version": revision.source.version,
        "course_id": revision.course_id,
        "publication_epoch": revision.publication_epoch,
        "content_sha256": revision.content_sha256,
    }
    trace.update(
        historical_source=revision.source.model_dump(),
        current_source=_key(selected).model_dump(),
        historical_message_id=past.message_id,
    )
    if not service.store.active(turn):
        raise WorkspaceError("interrupted")
    if image:
        error = _compare_images(service, turn, draft, past, document, selected, history, trace)
        if error:
            return service._finish_invalid_read(turn, error, trace)
    else:
        _compare_native(service, draft, past, document, selected, trace)
        if service.settings.workspace_answer_enabled and len(draft.sources) == 2:
            service._generate_draft(
                turn,
                draft,
                trace,
                "compare_versions",
                refs=[_doc_ref(document)],
                historical=[history_ref],
            )
    return service._complete(
        turn,
        draft,
        refs=[_doc_ref(document)],
        historical=[history_ref],
        trace=trace,
    )


def _completed_history(
    service: WorkspaceService, owner: str, turn: Turn, session: Session
) -> list[tuple[Turn, dict[str, Any]]]:
    _, turns, messages = service.store.timeline(owner, session.session_id)
    by_id = {item.turn_id: item for item in turns if item.status == "completed"}
    history = []
    for message in messages:
        if message["turn_id"] == turn.turn_id:
            break
        previous = by_id.get(message["turn_id"])
        if message["role"] == "assistant" and previous and service._turn_access(previous):
            history.append((previous, message))
    return list(reversed(history))


def _compare_native_group(service, turn, draft, docs, paired, trace):
    from app.rag.workspace import _doc_ref

    if not service.store.active(turn):
        raise WorkspaceError("interrupted")
    older, newer = paired[0]
    revision = older.evidence.revision
    historical = [
        dict(
            course_id=revision.course_id,
            document_id=revision.source.document_id,
            version=revision.source.version,
            publication_epoch=revision.publication_epoch,
            content_sha256=revision.content_sha256,
        )
    ]
    document = docs[newer.document_id]
    refs = [_doc_ref(document)]
    trace["paired_sources"] = []
    for previous, current in paired:
        details: dict[str, Any] = {}
        _compare_native(service, draft, previous, document, current, details)
        trace["paired_sources"].append(
            {
                "historical": _key(previous.source).model_dump(),
                "current": _key(current).model_dump(),
                **details,
            }
        )
    if service.settings.workspace_answer_enabled and draft.sources:
        service._generate_draft(
            turn,
            draft,
            trace,
            "compare_versions",
            refs=refs,
            historical=historical,
        )
    return service._complete(turn, draft, refs=refs, historical=historical, trace=trace)


def _old_candidates(
    service: WorkspaceService,
    session: Session,
    history: list[tuple[Turn, dict[str, Any]]],
    docs: dict[str, SnapshotDocument],
    versions: set[int],
    question: str,
    image: bool,
) -> list[_Previous]:
    records = {}
    for _, message in history:
        if image:
            visible = set(message["visible_display_ids"])
            admitted = {
                b["evidence_id"]: b["display_id"]
                for b in message["blocks"]
                if b.get("type") == "source_image" and b["display_id"] in visible
            }
        else:
            admitted = {
                cited: None
                for b in message["blocks"]
                if b.get("type") == "text"
                for cited in b.get("citations", [])
            }
        for evidence_id, display_id in admitted.items():
            evidence = TurnEvidence.model_validate(message["sources"][evidence_id])
            revision = evidence.revision
            key = revision.source
            document = docs.get(key.document_id)
            if (
                not document
                or document.course_id != revision.course_id
                or document.version == key.version
                or (len(versions) == 2 and document.version not in versions)
                or (len(versions - {document.version}) > 0 and key.version not in versions)
                or (
                    not (versions - {document.version})
                    and _PREVIOUS.search(question)
                    and key.version + 1 != document.version
                )
            ):
                continue
            identity = key.document_id, key.version, key.source_id
            if identity in records:
                continue
            if not image and not any(
                read.method in {"native_text", "native_table"}
                and read.actual_source == key
                and read.status in {"succeeded", "partial"}
                and read.text
                for read in evidence.reads
            ):
                continue
            attachment = (
                service._attachment(session.session_id, message["message_id"], display_id)
                if display_id
                else None
            )
            if attachment and (
                attachment.description.get("fallback_reason") is not None
                or SourceKey.model_validate(attachment.description["display_source"]) != key
            ):
                continue
            original = service._historical_source(
                {"course_id": revision.course_id, "content_sha256": revision.content_sha256}, key
            )
            assert original.source is not None
            source = _as_snapshot(original.source, revision.course_id, key)
            records[identity] = _Previous(
                message["message_id"], evidence, source, original, attachment
            )
    return list(records.values())


def _at_location(source: SnapshotSource, location: StructureLocation | None) -> bool:
    return location is None or (
        (location.page is None or source.locator.get("page") == location.page)
        and (location.cell_index is None or source.locator.get("cell_index") == location.cell_index)
    )


def _counterpart(
    old: _Previous,
    current: SnapshotSource,
    location: StructureLocation | None,
    intent: ObjectIntent | None,
) -> bool:
    source = old.source
    if source.document_id != current.document_id or source.kind != current.kind:
        return False
    prev, now = source.locator, current.locator
    if source.kind in {"cell", "notebook_output"}:
        known_cell = bool(prev.get("cell_id")) and prev["cell_id"] == now.get("cell_id")
        same_explicit_cell = (
            not prev.get("cell_id")
            and not now.get("cell_id")
            and location is not None
            and location.cell_index is not None
            and prev.get("cell_index") == location.cell_index
        )
        if not (known_cell or same_explicit_cell):
            return False
        if source.kind == "cell":
            return prev.get("cell_type") == now.get("cell_type")
        if any(
            prev.get(key) != now.get(key) for key in ("output_index", "output_type", "stream_name")
        ):
            return False
        if intent and intent.order == "images" and intent.ordinal is not None:
            return _old_notebook_image_order(old) == intent.ordinal
        return True
    if prev.get("page") != now.get("page"):
        return False
    if source.kind == "page":
        return bool(
            location
            and location.page is not None
            or (source.title.strip() and source.title == current.title)
        )
    return (
        prev.get("object_type") == now.get("object_type")
        and prev.get("coordinate_frame") == now.get("coordinate_frame") == "visible_page_norm_v1"
        and prev.get("bbox") == now.get("bbox")
        and bool(prev.get("bbox"))
    )


def _old_notebook_image_order(past: _Previous) -> int | None:
    try:
        locator = past.source.locator
        data = json.loads(past.original.original)
        outputs = data["cells"][locator["cell_index"]]["outputs"]
        index = locator["output_index"]
        return sum(
            isinstance(output, dict)
            and isinstance(bundle := output.get("data"), dict)
            and any(mime.startswith("image/") for mime in bundle)
            for output in outputs[: index + 1]
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _as_snapshot(source: SourceDraft, course_id: str, key: SourceKey) -> SnapshotSource:
    return SnapshotSource(
        course_id,
        key.document_id,
        key.version,
        key.source_id,
        source.parent_id,
        source.ordinal,
        source.kind,
        source.title,
        source.text,
        source.locator,
        source.state,
    )


def _key(source: SnapshotSource) -> SourceKey:
    return SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )


def _compare_native(
    service: WorkspaceService,
    draft: Draft,
    past: _Previous,
    document: SnapshotDocument,
    selected: SnapshotSource,
    trace: dict[str, Any],
) -> None:
    old_read = _native_read(past.source)
    key = _key(selected)
    current = service.library.store.read_source(key, course_ids=(document.course_id,))
    if (
        not current.is_current
        or current.content_sha256 != document.content_sha256
        or current.publication_epoch != document.publication_epoch
    ):
        raise WorkspaceError("stale_reference")
    new_read = (
        _native_read(_as_snapshot(current.source, document.course_id, key))
        if current.status == "available" and current.source
        else None
    )
    if old_read:
        item = _include_old(draft, past, old_read)
        item.locator["version_role"] = "historical"
        draft.text(f"旧版 v{past.source.version}\n{old_read.text}", item.evidence_id)
    if new_read:
        item = draft.add_evidence(document, selected, [new_read])
        item.locator["version_role"] = "current"
        draft.text(f"当前版 v{document.version}\n{new_read.text}", item.evidence_id)
    if not old_read or not new_read:
        missing = (
            "旧版和当前版"
            if not old_read and not new_read
            else "旧版"
            if not old_read
            else "当前版"
        )
        draft.notice("comparison_missing_read", f"{missing}缺少该对象的可核对原文，暂不能比较。")
    trace["historical_read_status"] = old_read.status if old_read else "not_attempted"
    trace["current_read_status"] = new_read.status if new_read else "not_attempted"


def _native_read(source: SnapshotSource) -> TurnRead | None:
    text = source.text.strip()
    if not text or source.locator.get("output_type") == "error":
        return None
    clipped = len(text) > _EXCERPT
    return TurnRead(
        actual_source=_key(source),
        method="native_table" if source.locator.get("object_type") == "table" else "native_text",
        status="partial" if clipped else "succeeded",
        text=text[:_EXCERPT] + ("…" if clipped else ""),
        warnings=["excerpt_truncated"] if clipped else [],
    )


def _include_old(draft: Draft, past: _Previous, read: TurnRead | None) -> TurnEvidence:
    item = past.evidence.model_copy(
        update={
            "evidence_id": f"ev-{len(draft.sources) + 1}",
            "reads": [read] if read else [],
            "read_status": read.status if read else "not_attempted",
            "locator": {
                key: value for key, value in past.evidence.locator.items() if key != "text_span"
            },
        }
    )
    draft.sources[item.evidence_id] = item
    return item


def _image_attribute(question: str) -> str | None:
    axes = [name for name, pattern in (("横轴", _X), ("纵轴", _Y)) if pattern.search(question)]
    attrs = [
        name for name, pattern in (("单位", _UNIT), ("标签", _LABEL)) if pattern.search(question)
    ]
    return f"{axes[0]}{attrs[0]}" if len(axes) == len(attrs) == 1 else None


def _previous_visual_read(
    past: _Previous,
    history: list[tuple[Turn, dict[str, Any]]],
    attribute: str | None,
    display: DisplayRecord,
) -> TurnRead | None:
    if attribute is None:
        return None
    for turn, message in history:
        if _image_attribute(turn.request.text) != attribute:
            continue
        question_digest = hashlib.sha256(turn.request.text.strip().encode()).hexdigest()
        for payload in message["sources"].values():
            candidate = TurnEvidence.model_validate(payload)
            if candidate.revision != past.evidence.revision:
                continue
            for read in candidate.reads:
                if (
                    read.actual_source == past.evidence.revision.source
                    and read.method == "vision"
                    and read.status == "succeeded"
                    and read.origin_execution == "real_provider"
                    and read.execution in {"real_provider", "cache_reuse"}
                    and read.text.strip()
                    and read.actual_input_sha256 == display.byte_sha256
                    and read.actual_input_media_type == display.media_type
                    and read.actual_input_rect == display.rect_visible_norm
                    and read.question_sha256 == question_digest
                ):
                    return read
    return None


def _compare_images(
    service: WorkspaceService,
    turn: Turn,
    draft: Draft,
    past: _Previous,
    document: SnapshotDocument,
    selected: SnapshotSource,
    history: list[tuple[Turn, dict[str, Any]]],
    trace: dict[str, Any],
) -> str | None:
    assert past.attachment is not None
    revision = past.evidence.revision
    desc = past.attachment.description
    request = EvidenceRequest.model_validate(desc["request"]).model_copy(
        update={"read_intent": "display_only", "question": None}
    )
    grant = HistoricalSourceGrant(
        revision.source,
        (revision.course_id,),
        revision.content_sha256,
        past.message_id,
        past.evidence.evidence_id,
    )
    rendered = service.evidence.render_historical(request, grant)
    if rendered.status in {"not_accessible", "stale_reference"}:
        raise WorkspaceError(rendered.status)
    display = rendered.display
    if past.source.kind == "page":
        # Registered subregions share the page key; compare their actual normalized crop.
        current_preview = service.evidence.render_current(
            service._render_request(document, selected)
        )
        if current_preview.status in {"stale_reference", "not_accessible"}:
            raise WorkspaceError(current_preview.status)
        if (
            not display
            or not current_preview.display
            or display.rect_visible_norm is None
            or display.rect_visible_norm != current_preview.display.rect_visible_norm
        ):
            draft.notice("comparison_ambiguous", "两版页内裁区无法可靠对应。")
            return None
    attribute = _image_attribute(turn.request.text)
    old_read = None
    if (
        display is not None
        and display.actual_source == revision.source
        and display.fallback_reason is None
        and display.byte_sha256 == desc["byte_sha256"]
    ):
        old_read = _previous_visual_read(past, history, attribute, display)
        item = _include_old(draft, past, old_read)
        draft.add_image(
            item, rendered, request, f"{past.evidence.file_name} · 旧版 v{past.source.version}"
        )
        if old_read:
            draft.text(
                f"旧版 v{past.source.version} {attribute} · {old_read.text}", item.evidence_id
            )
    if not old_read:
        draft.notice(
            "historical_image_unread", "旧版该图没有已核对的同属性读取，暂不能断言其内容。"
        )
    trace["historical_read_status"] = old_read.status if old_read else "not_attempted"

    if attribute is None:
        current_request = service._render_request(document, selected)
        display_only = service.evidence.render_current(current_request)
        if display_only.status in {"stale_reference", "not_accessible"}:
            raise WorkspaceError(display_only.status)
        item = draft.add_evidence(document, selected, [])
        draft.add_image(
            item,
            display_only,
            current_request,
            f"{document.file_name} · 当前版 v{document.version}",
        )
        draft.notice("comparison_attribute", "请指定要比较的轴标签或单位等属性。")
        trace["current_read_status"] = "not_attempted"
        return None

    draft.text(f"当前版 v{document.version} · {attribute}")
    trace["directed_attribute"] = attribute
    error = service._read_selected_image(
        turn, draft, document, selected, f"当前图中的{attribute}是什么？", trace
    )
    trace["current_read_status"] = trace.get("read_status", "not_attempted")
    return error
