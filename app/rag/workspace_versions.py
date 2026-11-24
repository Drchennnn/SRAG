"""Compare a saved historical workspace reference with its current publication."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.rag.source_evidence import DisplayRecord, EvidenceRequest, HistoricalSourceGrant
from app.rag.structure_search import StructureLocation
from app.rag.workspace_clarification import (
    ClarificationOptionRecord,
    ComparisonResumeContext,
    HistoricalComparisonRef,
    build_clarification,
)
from app.rag.workspace_evidence import format_object_option_label, readable_source_text
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
_METRIC_ATTRS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "耗时与延迟",
        re.compile(
            r"耗时|延迟|时延|时间|毫秒|(?<![A-Za-z0-9_])ms(?![A-Za-z0-9_])|秒",
            re.I,
        ),
    ),
    ("准确率与精度", re.compile(r"准确率|精度|\b(?:accuracy|precision)\b", re.I)),
    ("吞吐量", re.compile(r"吞吐|\b(?:throughput|qps|tps)\b", re.I)),
)
_VALUE_HINT = re.compile(
    r"数值|峰值|趋势|大小|读数|数据|\b(?:values?|peaks?|trends?|numbers?)\b",
    re.I,
)
_MIXED_CELL_TEXT_WITH_IMAGE = re.compile(
    r"指标|参数|代码|文本|文字|stdout|保存(?:的)?输出|运行(?:的)?输出|实验(?:结果|输出|数值)|"
    r"实测(?:结果|数值|数据)|准确率|召回率|延迟|误差|"
    r"\b(?:metrics?|code|text|stdout|saved\s+outputs?|accuracy|recall|latency|rmse|mae)\b|"
    r"(?:数值|结果|数据)\s*(?:与|和|及|以及|还有|并)\s*"
    r"第\s*[0-9一二三四五六七八九十两]+\s*(?:张|幅|个)?(?:输出)?图|"
    r"第\s*[0-9一二三四五六七八九十两]+\s*(?:张|幅|个)?(?:输出)?图\s*"
    r"(?:与|和|及|以及|还有|并)\s*(?:数值|结果|数据)",
    re.I,
)
_CODE_TEXT_CUE = re.compile(r"代码|cell|单元格|\b(?:code|cell)\b", re.I)
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


def compare_versions(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
    *,
    question: str | None = None,
    forced_file_name: str | None = None,
) -> Turn:
    from app.rag.structure_search import StructureScope, StructureSearcher, StructureSearchRequest
    from app.rag.workspace import Draft, _doc_ref, _location, _object_file_names

    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()
    question = question if question is not None else turn.request.text
    implicit = bool(_BARE_COMPARISON.fullmatch(question))
    draft = Draft(session.session_id)
    trace: dict[str, Any] = {"route": "version_comparison", "external_call_count": 0}
    scope = tuple(session.course_ids)
    snapshot = service.snapshots.read(SnapshotRequest(scope, tuple(session.document_ids)))
    names: list[str] | None = (
        [forced_file_name] if forced_file_name else _object_file_names(question, snapshot.documents)
    )
    if names is None or len(set(names)) > 1:
        draft.notice("comparison_scope", "找不到可使用的指定文件，请核对课程和资料范围。")
        return service._complete(turn, draft, trace=trace)
    focus = None
    target_doc: SnapshotDocument | None = None
    multi_ids = service.snapshots.multi_version_document_ids(
        [doc.document_id for doc in snapshot.documents]
    )
    multi_docs = [doc for doc in snapshot.documents if doc.document_id in multi_ids]
    if not names and session.focus_id:
        focus = service._catalog(session.session_id).nodes.get(session.focus_id)
        if focus is not None:
            if focus.document.document_id in multi_ids or not multi_docs:
                target_doc = focus.document
                names = [focus.document.file_name]
            else:
                focus = None
        else:
            doc_map = {
                doc.document_id: doc
                for doc in (multi_docs if multi_docs else snapshot.documents)
            }
            for _, msg in _completed_history(service, owner, turn, session):
                msg_docs = {
                    ev["revision"]["source"]["document_id"]
                    for ev in msg["sources"].values()
                    if ev["revision"]["source"]["document_id"] in doc_map
                }
                if len(msg_docs) == 1:
                    target_doc = doc_map[next(iter(msg_docs))]
                    names = [target_doc.file_name]
                    break
    if not names and multi_docs:
        if len(multi_docs) == 1:
            target_doc = multi_docs[0]
            names = [target_doc.file_name]
        else:
            cleaned_q = _COMPARE.sub(
                " ", _CURRENT.sub(" ", _OLD.sub(" ", _VERSION.sub(" ", question)))
            )
            cleaned_q = re.sub(r"[，。！？；,;!?\s]+", " ", cleaned_q).strip()
            if cleaned_q:
                probe = StructureSearcher(service.snapshots).search(
                    StructureSearchRequest(
                        query=cleaned_q,
                        top_k=8,
                        include_adjacency=False,
                        include_page_fallback=False,
                        scope=StructureScope(
                            course_ids=list(scope),
                            document_ids=[doc.document_id for doc in multi_docs],
                        ),
                    ),
                    allowed_course_ids=service.allowed_courses,
                )
                hit_docs = list(
                    dict.fromkeys(
                        cand.source.document_id
                        for cand in probe.candidates
                        if cand.match_kind != "browse"
                    )
                )
                if len(hit_docs) == 1:
                    hit_doc = next(
                        (d for d in multi_docs if d.document_id == hit_docs[0]), None
                    )
                    if hit_doc is not None:
                        target_doc = hit_doc
                        names = [hit_doc.file_name]
        if not names:
            multi_map = {doc.document_id: doc for doc in multi_docs}
            return build_clarification(
                service,
                turn,
                "有多份多版本资料，请先选定要比较的一份资料。",
                service._document_choices(session.session_id, multi_map),
                refs=[_doc_ref(doc) for doc in multi_docs],
                notice_code="comparison_scope",
                trace=trace,
            )
    if not names and not multi_docs and len(snapshot.documents) > 1:
        draft.notice(
            "comparison_missing_history",
            "本会话范围内没有可比较的多版本资料。请先更新版本并从旧回答定位原件后再比较。",
        )
        return service._complete(turn, draft, trace=trace)
    if target_doc is not None:
        snapshot = service.snapshots.read(SnapshotRequest(scope, (target_doc.document_id,)))
    elif names:
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
    full_history = _completed_history(service, owner, turn, session)
    candidate_history = full_history
    attribute = _image_attribute(question)
    inherited_image = False
    if full_history and (
        implicit
        or (
            not intent
            and re.search(
                r"张图|幅图|那图|这图|刚才|读过|看过|图片|图像|图表|原图",
                request_text,
            )
        )
        or (image and attribute is None)
    ):
        matched_old = _latest_old_visual_entry(full_history, docs, versions, request_text)
        if matched_old is not None:
            previous_turn, message, read_visual_ids, old_display_ids = matched_old
            prior_intent = object_intent(previous_turn.request.text, include_reference=True)
            if len(read_visual_ids) == 1 or (
                implicit and prior_intent and prior_intent.kind == "image"
            ):
                attribute = attribute or _image_attribute(previous_turn.request.text)
                single_display = len(old_display_ids) == 1
                if implicit and len(read_visual_ids) != 1 and (not single_display or not attribute):
                    draft.notice("comparison_ambiguous", "请指定要比较的图，以及轴标签或单位。")
                    return service._complete(turn, draft, trace=trace)
                candidate_history = [(previous_turn, message)]
                intent = intent or prior_intent or ObjectIntent(kind="image")
                image = inherited_image = True
    current = [
        source
        for source in snapshot.sources
        if source.document_id in docs
        and _at_location(source, location)
        and (not _OUTPUT.search(request_text) or image or source.kind == "notebook_output")
    ]
    if intent:
        current = object_choices(
            intent, current, region_rects=_region_rects(service, docs, current)
        )
    old = _old_candidates(
        service, session, candidate_history, docs, versions, request_text, image, intent=intent
    )
    if (
        not old
        and not image
        and location is None
        and not intent
        and full_history
    ):
        matched_old = _latest_old_visual_entry(full_history, docs, versions, request_text)
        if matched_old is not None and len(matched_old[2]) == 1:
            previous_turn, message, _, _ = matched_old
            candidate_history = [(previous_turn, message)]
            prior_intent = object_intent(previous_turn.request.text, include_reference=True)
            attribute = attribute or _image_attribute(previous_turn.request.text)
            intent = prior_intent or ObjectIntent(kind="image")
            image = inherited_image = True
            current = [
                source
                for source in snapshot.sources
                if source.document_id in docs
                and _at_location(source, location)
                and (not _OUTPUT.search(request_text) or image or source.kind == "notebook_output")
            ]
            current = object_choices(
                intent, current, region_rects=_region_rects(service, docs, current)
            )
            old = _old_candidates(
                service,
                session,
                candidate_history,
                docs,
                versions,
                request_text,
                True,
                intent=intent,
            )
    doc_sources_by_doc: dict[str, list[SnapshotSource]] = {}
    for src in snapshot.sources:
        doc_sources_by_doc.setdefault(src.document_id, []).append(src)
    paired = [
        (past, candidate)
        for past in old
        for candidate in current
        if _counterpart(
            past,
            candidate,
            location,
            intent,
            current_sources=doc_sources_by_doc.get(candidate.document_id),
        )
    ]
    has_intent_old = bool(
        old
        and intent is not None
        and any(
            item.source.document_id == current[0].document_id
            and (
                intent.order != "images"
                or intent.ordinal is None
                or _old_notebook_image_order(item) == intent.ordinal
            )
            for item in old
        )
    )
    if (
        not paired
        and not has_intent_old
        and image
        and intent is not None
        and len(current) == 1
        and current[0].kind == "notebook_output"
        and full_history
    ):
        recovered = _recover_authorized_cell_visual_old(
            service,
            session,
            full_history,
            docs,
            versions,
            request_text,
            existing_old=old,
            selected=current[0],
            location=location,
            intent=intent,
            current_sources=doc_sources_by_doc.get(current[0].document_id),
        )
        if recovered:
            if not old:
                old = list(recovered)
            paired = [
                (past, current[0])
                for past in recovered
                if _counterpart(
                    past,
                    current[0],
                    location,
                    intent,
                    current_sources=doc_sources_by_doc.get(current[0].document_id),
                )
            ]
    if (
        not paired
        and location is None
        and not image
        and not intent
        and old
        and any(item.source.kind == "page" for item in old)
    ):
        paired = [
            (past, candidate)
            for past in old
            for candidate in current
            if past.source.document_id == candidate.document_id
            and candidate.kind == "page"
            and past.source.locator.get("page") is not None
            and past.source.locator.get("page") == candidate.locator.get("page")
        ]
    if paired:
        msg_recency = {msg["message_id"]: idx for idx, (_, msg) in enumerate(full_history)}
        dedup_paired: dict[
            tuple[str, int, str, str, int, str], tuple[_Previous, SnapshotSource]
        ] = {}
        for past_item, cand_item in paired:
            pair_key = (
                past_item.source.document_id,
                past_item.source.version,
                past_item.source.source_id,
                cand_item.document_id,
                cand_item.version,
                cand_item.source_id,
            )
            existing = dedup_paired.get(pair_key)
            if existing is None or (
                msg_recency.get(past_item.message_id, len(full_history))
                < msg_recency.get(existing[0].message_id, len(full_history))
            ):
                dedup_paired[pair_key] = (past_item, cand_item)
        paired = list(dedup_paired.values())
    if location is None and not image and not intent and len(paired) > 1:
        latest = paired[0][0].message_id
        recent = [(older, newer) for older, newer in paired if older.message_id == latest][:8]
        keys = {
            (_key(older.source).document_id, older.source.version, newer.version)
            for older, newer in recent
        }
        if len(keys) == 1:
            paired = recent
            if len(paired) > 1:
                return _compare_native_group(service, turn, draft, docs, paired, trace)
    if len(paired) != 1:
        reordered = _reordered_comparison_options(
            service,
            session,
            old,
            snapshot.sources,
            docs,
            question=question,
            is_image=image,
            attribute=attribute,
            inherited_image=inherited_image,
        )
        if reordered is not None:
            options, pub_refs, hist_refs = reordered
            return build_clarification(
                service,
                turn,
                "新版输出顺序已变化，无法自动对齐两版对象。请选择当前版本中要对比的输出。",
                options,
                refs=pub_refs,
                historical_refs=hist_refs,
                trace={
                    "route": "version_comparison",
                    "clarification_reason": "reordered_comparison",
                    "external_call_count": 0,
                },
            )
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
    return execute_paired_comparison(
        service,
        owner,
        turn,
        session,
        past=past,
        document=document,
        selected=selected,
        history=full_history,
        question=question,
        is_image=image,
        attribute=attribute,
        inherited_image=inherited_image,
    )


def _latest_old_visual_entry(
    history: list[tuple[Turn, dict[str, Any]]],
    docs: dict[str, SnapshotDocument],
    versions: set[int],
    request_text: str,
) -> tuple[Turn, dict[str, Any], list[str], list[str]] | None:
    """Walk completed history to find the most recent old-version visual read or display."""

    def _is_old_source(src: dict[str, Any]) -> bool:
        doc_id = src.get("document_id")
        ver = src.get("version")
        if not isinstance(doc_id, str) or not isinstance(ver, int):
            return False
        doc = docs.get(doc_id)
        if doc is None or ver >= doc.version:
            return False
        old_target = versions - {doc.version}
        if old_target and ver not in old_target:
            return False
        if not old_target and _PREVIOUS.search(request_text) and ver + 1 != doc.version:
            return False
        return True

    fallback_display: tuple[Turn, dict[str, Any], list[str], list[str]] | None = None
    for prev_turn, message in history:
        visible_ids = set(message.get("visible_display_ids") or ())
        old_display_ids = [
            b["display_id"]
            for b in message.get("blocks", [])
            if b.get("type") == "source_image"
            and b.get("display_id") in visible_ids
            and _is_old_source(b.get("display_source") or {})
        ]
        old_read_ids = [
            eid
            for eid, ev in (message.get("sources") or {}).items()
            if _is_old_source((ev.get("revision") or {}).get("source") or {})
            and any(
                r.get("method") == "vision" and r.get("status") in {"succeeded", "partial"}
                for r in ev.get("reads", [])
            )
        ]
        if len(old_read_ids) == 1:
            return prev_turn, message, old_read_ids, old_display_ids
        if fallback_display is None and old_display_ids:
            fallback_display = (prev_turn, message, old_read_ids, old_display_ids)
    return fallback_display


def execute_paired_comparison(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
    *,
    past: _Previous,
    document: SnapshotDocument,
    selected: SnapshotSource,
    history: list[tuple[Turn, dict[str, Any]]],
    question: str,
    is_image: bool,
    attribute: str | None,
    inherited_image: bool,
    resumed_from_turn_id: str | None = None,
) -> Turn:
    """Shared bilateral v1 + v2 comparison executor for direct pairing and b2 resume."""
    from app.rag.workspace import Draft, _doc_ref

    draft = Draft(session.session_id)
    revision = past.evidence.revision
    history_ref = {
        "document_id": revision.source.document_id,
        "version": revision.source.version,
        "course_id": revision.course_id,
        "publication_epoch": revision.publication_epoch,
        "content_sha256": revision.content_sha256,
    }
    trace: dict[str, Any] = {
        "route": "version_comparison",
        "external_call_count": 0,
        "historical_source": revision.source.model_dump(),
        "current_source": _key(selected).model_dump(),
        "historical_message_id": past.message_id,
    }
    if resumed_from_turn_id is not None:
        trace["resumed_from_clarification"] = resumed_from_turn_id
    if not service.store.active(turn):
        raise WorkspaceError("interrupted")
    if is_image:
        mixed_cell_text = bool(
            selected.kind == "notebook_output" and _MIXED_CELL_TEXT_WITH_IMAGE.search(question)
        )
        if mixed_cell_text:
            _attach_paired_cell_text(
                service,
                draft,
                past,
                document,
                selected,
                question=question,
                trace=trace,
            )
        error = _compare_images(
            service,
            turn,
            draft,
            past,
            document,
            selected,
            history,
            trace,
            attribute,
            question=question,
            inherited_image=inherited_image,
            resumed_from_clarification=resumed_from_turn_id is not None,
            mixed_cell_text=mixed_cell_text,
        )
        if error:
            return service._finish_invalid_read(turn, error, trace)
        if (
            mixed_cell_text
            and service.settings.workspace_answer_enabled
            and service.settings.deepseek_api_key.strip()
            and len(draft.sources) >= 2
        ):
            service._generate_draft(
                turn,
                draft,
                trace,
                "compare_versions",
                refs=[_doc_ref(document)],
                historical=[history_ref],
                question=question,
            )
    else:
        _compare_native(service, draft, past, document, selected, trace)
        if (
            service.settings.workspace_answer_enabled
            and service.settings.deepseek_api_key.strip()
            and len(draft.sources) == 2
        ):
            service._generate_draft(
                turn,
                draft,
                trace,
                "compare_versions",
                refs=[_doc_ref(document)],
                historical=[history_ref],
                question=question,
            )
    resolution = (
        {
            "method": "clarification_compare_versions",
            "resolution": "clarification_compare_versions",
            "historical_source": revision.source.model_dump(),
            "current_source": _key(selected).model_dump(),
        }
        if resumed_from_turn_id is not None
        else None
    )
    return service._complete(
        turn,
        draft,
        refs=[_doc_ref(document)],
        historical=[history_ref],
        resolution=resolution,
        trace=trace,
    )


def _reordered_comparison_options(
    service: WorkspaceService,
    session: Session,
    old: list[_Previous],
    current: Sequence[SnapshotSource],
    docs: dict[str, SnapshotDocument],
    *,
    question: str,
    is_image: bool = False,
    attribute: str | None = None,
    inherited_image: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    from app.rag.workspace import _can_display, _doc_ref, _label

    if len(old) != 1 or old[0].source.kind != "notebook_output":
        return None
    past = old[0]
    old_cell_id = past.source.locator.get("cell_id")
    same_cell = [
        src
        for src in current
        if src.document_id == past.source.document_id
        and src.kind == "notebook_output"
        and (not is_image or _can_display(src))
        and (
            (old_cell_id and src.locator.get("cell_id") == old_cell_id)
            or (
                not old_cell_id
                and src.locator.get("cell_index") == past.source.locator.get("cell_index")
            )
        )
    ]
    if not same_cell:
        return None
    catalog = service._catalog(session.session_id)
    nodes_by_key = {
        (entry.source.document_id, entry.source.version, entry.source.source_id): entry
        for entry in catalog.nodes.values()
        if entry.source is not None
    }
    rev = past.evidence.revision
    historical_ref = HistoricalComparisonRef(
        message_id=past.message_id,
        evidence_id=past.evidence.evidence_id,
        display_id=past.attachment.display_id if past.attachment is not None else None,
        source=rev.source,
        course_id=rev.course_id,
        publication_epoch=rev.publication_epoch,
        content_sha256=rev.content_sha256,
    )
    comp_context = ComparisonResumeContext(
        original_question=question,
        historical_ref=historical_ref,
        is_image=is_image,
        attribute=attribute,
        inherited_image=inherited_image,
    )
    options: list[dict[str, Any]] = []
    for cand in same_cell[:8]:
        entry = nodes_by_key.get((cand.document_id, cand.version, cand.source_id))
        doc = docs.get(cand.document_id)
        if entry is None or doc is None:
            continue
        is_disp = _can_display(cand)
        label = format_object_option_label(
            cand,
            doc,
            same_cell,
            base_label=_label(cand),
            is_displayable=is_disp,
        )
        record = ClarificationOptionRecord(
            option_id=uuid4().hex,
            label=label,
            kind="document",
            resume_action="compare_versions",
            node_id=entry.entry["node_id"],
            target_source=_key(cand),
            current_object=is_disp,
            comparison_context=comp_context,
        )
        options.append(record.model_dump(mode="json"))
    if not options:
        return None
    pub_refs = [_doc_ref(doc) for doc in docs.values()]
    hist_refs = [
        {
            "document_id": rev.source.document_id,
            "version": rev.source.version,
            "course_id": rev.course_id,
            "publication_epoch": rev.publication_epoch,
            "content_sha256": rev.content_sha256,
        }
    ]
    return options, pub_refs, hist_refs


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
    *,
    intent: ObjectIntent | None = None,
) -> list[_Previous]:
    records = {}
    for prev_turn, message in history:
        if image:
            visible = set(message["visible_display_ids"])
            admitted: dict[str, tuple[str, str] | None] = {
                b["evidence_id"]: (message["message_id"], b["display_id"])
                for b in message["blocks"]
                if b.get("type") == "source_image" and b["display_id"] in visible
            }
            if not admitted:
                res = prev_turn.outcome.get("resolution") or {}
                if (
                    isinstance(res.get("message_id"), str)
                    and isinstance(res.get("display_id"), str)
                ):
                    for eid, ev in message["sources"].items():
                        if any(
                            r.get("method") == "vision"
                            and r.get("status") in {"succeeded", "partial"}
                            and r.get("actual_source") == ev["revision"]["source"]
                            for r in ev.get("reads", [])
                        ):
                            admitted[eid] = (res["message_id"], res["display_id"])
            else:
                # Prefer the single visually read display when multiple displays appeared together.
                read_only = {
                    eid: val
                    for eid, val in admitted.items()
                    if any(
                        r.get("method") == "vision" and r.get("status") in {"succeeded", "partial"}
                        for r in message["sources"].get(eid, {}).get("reads", [])
                    )
                }
                if len(read_only) == 1:
                    admitted = read_only
        else:
            admitted = {
                cited: None
                for b in message["blocks"]
                if b.get("type") == "text"
                for cited in b.get("citations", [])
            }
        added_this_turn = 0
        for evidence_id, display_ref in admitted.items():
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
            attachment = None
            attach_msg_id = message["message_id"]
            if display_ref:
                attach_msg_id, disp_id = display_ref
                attachment = service._attachment(session.session_id, attach_msg_id, disp_id)
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
            record_evidence = evidence
            record_msg_id = attach_msg_id
            if display_ref and attach_msg_id != message["message_id"] and attachment is not None:
                ev_id = attachment.description.get("evidence_id")
                if isinstance(ev_id, str):
                    record_evidence = evidence.model_copy(update={"evidence_id": ev_id})
            records[identity] = _Previous(
                record_msg_id, record_evidence, source, original, attachment
            )
            added_this_turn += 1
        if (
            image
            and added_this_turn == 1
            and (intent is None or (intent.ordinal is None and intent.direction is None))
            and any(
                r.get("method") == "vision" and r.get("status") in {"succeeded", "partial"}
                for ev in message["sources"].values()
                for r in ev.get("reads", [])
            )
        ):
            break
    return list(records.values())


def _same_notebook_cell(a: SnapshotSource, b: SnapshotSource) -> bool:
    if a.document_id != b.document_id or a.kind not in {"cell", "notebook_output"}:
        return False
    a_id, b_id = a.locator.get("cell_id"), b.locator.get("cell_id")
    if a_id and b_id:
        return a_id == b_id
    return (
        not a_id
        and not b_id
        and a.locator.get("cell_index") is not None
        and a.locator.get("cell_index") == b.locator.get("cell_index")
    )


def _recover_authorized_cell_visual_old(
    service: WorkspaceService,
    session: Session,
    full_history: list[tuple[Turn, dict[str, Any]]],
    docs: dict[str, SnapshotDocument],
    versions: set[int],
    request_text: str,
    *,
    existing_old: Sequence[_Previous],
    selected: SnapshotSource,
    location: StructureLocation | None,
    intent: ObjectIntent,
    current_sources: Sequence[SnapshotSource] | None,
) -> list[_Previous]:
    """Recover an authorized v1 notebook output image via same-cell historical citations."""
    text_old = _old_candidates(
        service, session, full_history, docs, versions, request_text, False, intent=intent
    )
    msg_recency = {msg["message_id"]: idx for idx, (_, msg) in enumerate(full_history)}
    anchors = sorted(
        (
            item
            for item in (*existing_old, *text_old)
            if _same_notebook_cell(item.source, selected)
        ),
        key=lambda item: msg_recency.get(item.message_id, len(full_history)),
    )
    if not anchors:
        return []
    anchor = anchors[0]
    v1_cell_idx = anchor.source.locator.get("cell_index")
    if not isinstance(v1_cell_idx, int):
        return []
    try:
        v1_outputs = json.loads(anchor.original.original)["cells"][v1_cell_idx]["outputs"]
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    if not isinstance(v1_outputs, list):
        return []

    rev = anchor.evidence.revision
    candidate_indices: list[int] = []
    target_out_idx = selected.locator.get("output_index")
    if isinstance(target_out_idx, int) and 0 <= target_out_idx < len(v1_outputs):
        candidate_indices.append(target_out_idx)
    for idx, raw_out in enumerate(v1_outputs):
        if idx not in candidate_indices and isinstance(raw_out, dict):
            data = raw_out.get("data")
            if isinstance(data, dict) and any(
                isinstance(k, str) and k.startswith("image/") for k in data
            ):
                candidate_indices.append(idx)

    recovered: list[_Previous] = []
    for out_idx in candidate_indices:
        v1_key = SourceKey(
            document_id=selected.document_id,
            version=rev.source.version,
            source_id=f"cell:{v1_cell_idx}:output:{out_idx}",
        )
        hist = service._historical_source(
            {"course_id": rev.course_id, "content_sha256": rev.content_sha256},
            v1_key,
        )
        if hist.source is None or not _output_image_mimes(
            hist.source.locator.get("representations")
        ):
            continue
        snap_src = _as_snapshot(hist.source, rev.course_id, v1_key)
        synth_ev = TurnEvidence(
            evidence_id=anchor.evidence.evidence_id,
            revision=rev.model_copy(update={"source": v1_key}),
            file_name=anchor.evidence.file_name,
            locator=dict(hist.source.locator),
            read_status="not_attempted",
            reads=[],
        )
        prev = _Previous(anchor.message_id, synth_ev, snap_src, hist, None)
        if _counterpart(prev, selected, location, intent, current_sources=current_sources):
            return [prev]
        if not recovered:
            recovered.append(prev)
    return recovered


def _attach_paired_cell_text(
    service: WorkspaceService,
    draft: Draft,
    past: _Previous,
    document: SnapshotDocument,
    selected: SnapshotSource,
    *,
    question: str,
    trace: dict[str, Any],
) -> None:
    """Attach reliable paired v1+v2 text/stdout evidence from the same Notebook cell."""
    v1_cell_idx = past.source.locator.get("cell_index")
    v2_cell_idx = selected.locator.get("cell_index")
    v2_cell_id = selected.locator.get("cell_id")
    if not isinstance(v1_cell_idx, int) or not isinstance(v2_cell_idx, int):
        return
    try:
        v1_cell = json.loads(past.original.original)["cells"][v1_cell_idx]
        v1_outputs = v1_cell.get("outputs") or []
    except (KeyError, IndexError, TypeError, ValueError):
        return

    all_v2 = service.library.store.current_sources(
        document.document_id, course_ids=(document.course_id,)
    )
    v2_same_cell = [
        src
        for src in all_v2
        if (
            (v2_cell_id and src.locator.get("cell_id") == v2_cell_id)
            or (not v2_cell_id and src.locator.get("cell_index") == v2_cell_idx)
        )
    ]
    v2_text_outputs = sorted(
        (
            src
            for src in v2_same_cell
            if src.kind == "notebook_output"
            and src.locator.get("output_type") != "error"
            and bool(readable_source_text(_as_snapshot(src, document.course_id, SourceKey(
                document_id=document.document_id, version=document.version, source_id=src.source_id
            ))).strip())
        ),
        key=lambda s: (int(s.locator.get("output_index", s.ordinal)), s.ordinal),
    )
    v2_code_cells = [src for src in v2_same_cell if src.kind == "cell"]

    pairs_to_attach: list[tuple[SourceKey, SourceDraft]] = []
    rev = past.evidence.revision
    for v2_out in v2_text_outputs:
        out_idx = v2_out.locator.get("output_index")
        if not isinstance(out_idx, int) or out_idx >= len(v1_outputs):
            continue
        raw_v1_out = v1_outputs[out_idx]
        if not isinstance(raw_v1_out, dict):
            continue
        v1_stream = raw_v1_out.get("name") if raw_v1_out.get("output_type") == "stream" else None
        if (raw_v1_out.get("output_type"), v1_stream) != (
            v2_out.locator.get("output_type"),
            v2_out.locator.get("stream_name"),
        ):
            continue
        v1_key = SourceKey(
            document_id=document.document_id,
            version=rev.source.version,
            source_id=f"cell:{v1_cell_idx}:output:{out_idx}",
        )
        pairs_to_attach.append((v1_key, v2_out))

    if (
        (not pairs_to_attach or _CODE_TEXT_CUE.search(question))
        and v2_code_cells
        and len(pairs_to_attach) < 2
    ):
        v2_c = v2_code_cells[0]
        if (
            v1_cell.get("cell_type") == v2_c.locator.get("cell_type")
            and isinstance(v1_cell.get("source"), (str, list))
        ):
            v1_cell_key = SourceKey(
                document_id=document.document_id,
                version=rev.source.version,
                source_id=f"cell:{v1_cell_idx}",
            )
            pairs_to_attach.append((v1_cell_key, v2_c))

    attached_pairs: list[dict[str, Any]] = []
    for v1_key, v2_draft in pairs_to_attach[:2]:
        v2_key = SourceKey(
            document_id=document.document_id,
            version=document.version,
            source_id=v2_draft.source_id,
        )
        hist = service._historical_source(
            {"course_id": rev.course_id, "content_sha256": rev.content_sha256},
            v1_key,
        )
        if hist.status != "available" or hist.source is None:
            continue
        v1_snap = _as_snapshot(hist.source, rev.course_id, v1_key)
        v2_snap = _as_snapshot(v2_draft, document.course_id, v2_key)
        if _native_read(v1_snap) is None or _native_read(v2_snap) is None:
            continue
        synth_ev = TurnEvidence(
            evidence_id=past.evidence.evidence_id,
            revision=rev.model_copy(update={"source": v1_key}),
            file_name=past.evidence.file_name,
            locator=dict(hist.source.locator),
            read_status="not_attempted",
            reads=[],
        )
        past_txt = _Previous(past.message_id, synth_ev, v1_snap, hist, None)
        details: dict[str, Any] = {}
        _compare_native(service, draft, past_txt, document, v2_snap, details)
        attached_pairs.append(
            {
                "historical": v1_key.model_dump(),
                "current": v2_key.model_dump(),
                **details,
            }
        )
    if attached_pairs:
        trace["paired_cell_text_sources"] = attached_pairs


def _region_rects(
    service: WorkspaceService,
    docs: dict[str, SnapshotDocument],
    sources: Sequence[SnapshotSource],
) -> dict[tuple[str, int, str], tuple[float, float, float, float]]:
    regions: dict[tuple[str, int, str], tuple[float, float, float, float]] = {}
    for source in sources:
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
    return regions


def _at_location(source: SnapshotSource, location: StructureLocation | None) -> bool:
    return location is None or (
        (location.page is None or source.locator.get("page") == location.page)
        and (location.cell_index is None or source.locator.get("cell_index") == location.cell_index)
    )


def _output_image_mimes(representations: Any) -> tuple[str, ...]:
    if not isinstance(representations, list):
        return ()
    return tuple(
        sorted(
            item["mime_type"]
            for item in representations
            if isinstance(item, dict)
            and isinstance(item.get("mime_type"), str)
            and item["mime_type"].startswith("image/")
        )
    )


def _raw_output_image_hashes(raw_output: dict[str, Any]) -> tuple[tuple[str, ...], set[str]]:
    data = raw_output.get("data")
    if not isinstance(data, dict):
        return (), set()
    mimes = tuple(sorted(k for k in data if isinstance(k, str) and k.startswith("image/")))
    digests: set[str] = set()
    for mime in mimes:
        raw_val = data.get(mime)
        raw_text = "".join(raw_val) if isinstance(raw_val, list) else raw_val
        if isinstance(raw_text, str):
            try:
                decoded = base64.b64decode(raw_text, validate=False)
                digests.add(hashlib.sha256(decoded).hexdigest())
            except Exception:
                pass
    return mimes, digests


def _cell_outputs_reliable(
    old: _Previous,
    current: SnapshotSource,
    current_sources: Sequence[SnapshotSource] | None,
) -> bool:
    if old.source.kind != "notebook_output" or current_sources is None:
        return True
    old_cell_idx = old.source.locator.get("cell_index")
    new_cell_idx = current.locator.get("cell_index")
    if not isinstance(old_cell_idx, int) or not isinstance(new_cell_idx, int):
        return True
    try:
        old_nb = json.loads(old.original.original)
        old_outputs = old_nb["cells"][old_cell_idx]["outputs"]
    except (KeyError, IndexError, TypeError, ValueError):
        return True
    if not isinstance(old_outputs, list):
        return True
    new_outputs = sorted(
        (
            src
            for src in current_sources
            if src.document_id == current.document_id
            and src.kind == "notebook_output"
            and src.locator.get("cell_index") == new_cell_idx
        ),
        key=lambda s: int(s.locator.get("output_index", -1)),
    )
    if len(old_outputs) != len(new_outputs):
        return False
    old_hash_to_indices: dict[str, set[int]] = {}
    for idx, (o_raw, n_src) in enumerate(zip(old_outputs, new_outputs, strict=True)):
        if not isinstance(o_raw, dict):
            return False
        o_type = o_raw.get("output_type")
        o_stream = o_raw.get("name") if o_type == "stream" else None
        o_mimes, o_hashes = _raw_output_image_hashes(o_raw)
        for digest in o_hashes:
            old_hash_to_indices.setdefault(digest, set()).add(idx)
        n_type = n_src.locator.get("output_type")
        n_stream = n_src.locator.get("stream_name")
        n_mimes = _output_image_mimes(n_src.locator.get("representations"))
        if (o_type, o_stream, o_mimes) != (n_type, n_stream, n_mimes):
            return False
    for n_idx, n_src in enumerate(new_outputs):
        reps = n_src.locator.get("representations") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            sha = rep.get("sha256")
            if isinstance(sha, str) and sha in old_hash_to_indices:
                if n_idx not in old_hash_to_indices[sha]:
                    return False
    return True


def _counterpart(
    old: _Previous,
    current: SnapshotSource,
    location: StructureLocation | None,
    intent: ObjectIntent | None,
    *,
    current_sources: Sequence[SnapshotSource] | None = None,
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
        if not _cell_outputs_reliable(old, current, current_sources):
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
    text = readable_source_text(source).strip()
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
    if len(axes) == len(attrs) == 1:
        return f"{axes[0]}{attrs[0]}"
    metric_attrs = [name for name, pattern in _METRIC_ATTRS if pattern.search(question)]
    return "、".join(metric_attrs) if metric_attrs else None


def _previous_visual_read(
    past: _Previous,
    history: list[tuple[Turn, dict[str, Any]]],
    attribute: str | None,
    display: DisplayRecord,
    *,
    allow_general: bool = False,
) -> TurnRead | None:
    """Resolve a verified historical real_provider vision read bound to exact display bytes.

    Priority:
    1. Exact single-attribute match (when attribute is specified and prior turn asked same attr).
    2. General verified read reuse (when allow_general=True and either current attribute is None
       or the prior turn performed a general whole-image read rather than a conflicting attribute).
    Never invents or returns a read for an image that was only displayed (not_attempted).
    """
    if attribute is None and not allow_general:
        return None
    general_fallback: TurnRead | None = None
    for turn, message in history:
        prior_attr = _image_attribute(turn.request.text)
        question_digest = hashlib.sha256(turn.request.text.strip().encode()).hexdigest()
        for payload in message["sources"].values():
            candidate = TurnEvidence.model_validate(payload)
            if candidate.revision != past.evidence.revision:
                continue
            for read in candidate.reads:
                if not (
                    read.actual_source == past.evidence.revision.source
                    and read.method == "vision"
                    and read.status in {"succeeded", "partial"}
                    and read.origin_execution == "real_provider"
                    and read.execution in {"real_provider", "cache_reuse"}
                    and read.text.strip()
                    and read.actual_input_sha256 == display.byte_sha256
                    and read.actual_input_media_type == display.media_type
                    and read.actual_input_rect == display.rect_visible_norm
                ):
                    continue
                if attribute is not None and prior_attr == attribute:
                    if read.question_sha256 in {None, question_digest}:
                        return read
                elif allow_general and (attribute is None or prior_attr is None):
                    if general_fallback is None:
                        general_fallback = read
    return general_fallback


def _single_image_v2_prompt(attribute: str | None, question: str) -> str:
    if attribute is not None:
        if _VALUE_HINT.search(question):
            return f"请读取当前图中的{attribute}、指标名称及核心数值。"
        return f"当前图中的{attribute}是什么？"
    return "请描述当前图中可见的坐标轴名称、单位、图例及核心数值。"


def _compare_images(
    service: WorkspaceService,
    turn: Turn,
    draft: Draft,
    past: _Previous,
    document: SnapshotDocument,
    selected: SnapshotSource,
    history: list[tuple[Turn, dict[str, Any]]],
    trace: dict[str, Any],
    attribute: str | None,
    *,
    question: str = "",
    inherited_image: bool,
    resumed_from_clarification: bool = False,
    mixed_cell_text: bool = False,
) -> str | None:
    revision = past.evidence.revision
    if past.attachment is not None:
        desc = past.attachment.description
        request = EvidenceRequest.model_validate(desc["request"]).model_copy(
            update={"read_intent": "display_only", "question": None}
        )
        expected_sha: str | None = desc["byte_sha256"]
    else:
        hist_doc = SnapshotDocument(
            course_id=revision.course_id,
            document_id=revision.source.document_id,
            version=revision.source.version,
            publication_epoch=revision.publication_epoch,
            content_sha256=revision.content_sha256,
            source_path="",
            file_name=past.evidence.file_name,
            title=past.evidence.file_name,
            document_format=document.document_format,
        )
        request = service._render_request(hist_doc, past.source).model_copy(
            update={"read_intent": "display_only", "question": None}
        )
        expected_sha = None
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
    allow_general = inherited_image or resumed_from_clarification
    old_read = None
    if (
        display is not None
        and display.actual_source == revision.source
        and display.fallback_reason is None
        and (expected_sha is None or display.byte_sha256 == expected_sha)
    ):
        old_read = _previous_visual_read(
            past, history, attribute, display, allow_general=allow_general
        )
        item = _include_old(draft, past, old_read)
        item.locator["version_role"] = "historical"
        draft.add_image(
            item, rendered, request, f"{past.evidence.file_name} · 旧版 v{past.source.version}"
        )
        if old_read:
            label = attribute or "已读取内容"
            draft.text(
                f"旧版 v{past.source.version} {label} · {old_read.text}", item.evidence_id
            )
    if not old_read:
        draft.notice(
            "historical_image_unread", "旧版该图没有已核对的同属性读取，暂不能断言其内容。"
        )
    trace["historical_read_status"] = old_read.status if old_read else "not_attempted"
    if inherited_image and not old_read and not resumed_from_clarification:
        return None

    if attribute is None and not (allow_general and old_read) and not mixed_cell_text:
        current_request = service._render_request(document, selected)
        display_only = service.evidence.render_current(current_request)
        if display_only.status in {"stale_reference", "not_accessible"}:
            raise WorkspaceError(display_only.status)
        item = draft.add_evidence(document, selected, [])
        item.locator["version_role"] = "current"
        draft.add_image(
            item,
            display_only,
            current_request,
            f"{document.file_name} · 当前版 v{document.version}",
        )
        draft.notice("comparison_attribute", "请指定要比较的轴标签或单位等属性。")
        trace["current_read_status"] = "not_attempted"
        return None

    target_attr = attribute or "内容与关键数值"
    draft.text(f"当前版 v{document.version} · {target_attr}")
    trace["directed_attribute"] = target_attr
    prompt_q = _single_image_v2_prompt(attribute, question)
    before_keys = set(draft.sources)
    error = service._read_selected_image(turn, draft, document, selected, prompt_q, trace)
    for eid in draft.sources:
        if eid not in before_keys:
            draft.sources[eid].locator["version_role"] = "current"
    trace["current_read_status"] = trace.get("read_status", "not_attempted")
    return error


def resolve_current_counterpart(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
    *,
    question: str,
) -> tuple[str, list[SnapshotSource], SnapshotDocument | None, _Previous | None]:
    """Map the single previously-read historical visual object to its current version."""

    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()
    history = _completed_history(service, owner, turn, session)
    if not history:
        return "none", [], None, None
    scope = tuple(session.course_ids)
    snapshot = service.snapshots.read(SnapshotRequest(scope, tuple(session.document_ids)))
    if snapshot.resolution_status != "resolved" or not snapshot.documents:
        return "none", [], None, None
    docs = {doc.document_id: doc for doc in snapshot.documents}
    old_list = _old_candidates(service, session, history, docs, set(), question, True)
    if not old_list:
        return "none", [], None, None
    if len(old_list) > 1:
        return "ambiguous", [], None, None
    past = old_list[0]
    document = docs.get(past.source.document_id)
    if document is None or document.version <= past.source.version:
        return "none", [], None, past
    doc_sources = [src for src in snapshot.sources if src.document_id == document.document_id]
    current_cell_matches = [
        src
        for src in doc_sources
        if _counterpart(past, src, None, None, current_sources=doc_sources)
    ]
    if len(current_cell_matches) == 1:
        return "resolved", current_cell_matches, document, past
    same_cell_candidates = [
        src
        for src in doc_sources
        if src.kind == past.source.kind
        and src.locator.get("cell_id") == past.source.locator.get("cell_id")
        and _output_image_mimes(src.locator.get("representations"))
    ]
    if not same_cell_candidates:
        return "deleted", [], document, past
    return "ambiguous", same_cell_candidates, document, past
