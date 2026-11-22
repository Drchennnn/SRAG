"""Persistent local study sessions with versioned display and reading evidence."""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.rag.evidence_answer import EvidenceAnswerer
from app.rag.evidence_supplement import SupplementRetriever, asks_for_definition_and_experiment
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import (
    EvidenceRequest,
    EvidenceResult,
    HistoricalSourceGrant,
    RegionRegistry,
    RenditionKind,
    SourceEvidenceService,
)
from app.rag.source_vision import SourceVisionService
from app.rag.structure_search import (
    StructureLocation,
    StructureScope,
    StructureSearcher,
    StructureSearchRequest,
)
from app.rag.workspace_composition import asks_for_composition, compose_evidence
from app.rag.workspace_generation import (
    OUTPUT_RESERVE,
    estimate_prompt_reserve,
    generate_response,
)
from app.rag.workspace_initial_focus import answer_current_image, resolve_current_image
from app.rag.workspace_objects import (
    _FACT as _OBJECT_FACT,
)
from app.rag.workspace_objects import (
    _PREVIEW as _OBJECT_PREVIEW,
)
from app.rag.workspace_objects import (
    ObjectIntent,
    object_choices,
    object_intent,
)
from app.rag.workspace_reading import (
    ReadingCursor,
    _has_visual,
    read_window,
)
from app.rag.workspace_versions import compare_versions, is_version_comparison
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import SourceRevision, TurnEvidence, TurnRead, TurnScope
from app.schemas.workspace import (
    ClarificationAnchor,
    CreateSession,
    DisplayAnchor,
    TurnRequest,
    UpdateFocus,
    UpdateScope,
    WorkspaceError,
    inaccessible,
)
from app.stores.structure_snapshot import (
    SnapshotDocument,
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshot,
    StructureSnapshotStore,
)
from app.stores.workspace import Attachment, Session, Turn, WorkspaceStore, json_text

_ORDINAL = re.compile(r"第\s*([0-9]+|[一二两三四五六七八九十]+)\s*张(?:图|图片)")
_PREVIOUS_TURN = re.compile(r"刚才|上一条|前一条|上一轮|前一轮|上次")
_EXPERIMENT_TEXT = re.compile(r"实验|实测|试验|\b(?:experiments?|measurements?)\b", re.I)
_EXPERIMENT_REFERENCE = re.compile(
    r"(?:同一|那项|那次|上述)(?:配套)?(?:实验|实测|试验)"
)
_EXPERIMENT_FILLER = re.compile(
    r"配套|实验|实测|试验|现在|还能|能否|可以|如何|核对|确认|结果|数据|记录|"
    r"分别|处理|测出的?|测得的?|是多少|多少|几次"
)
_DEFINITION_REFERENCE = re.compile(
    r"(?P<reference>(?:那|这|该)(?:一)?(?:条|个|项|种)?"
    r"(?:判定规则|规则|概念|门槛|机制|方法|窗口)|它|前者|后者)"
    r"\s*(?=(?:的定义|是?怎么定义|是?如何定义|指什么|是什么意思|是什么))"
)
_DEFINITION_REQUEST = re.compile(
    r"(?:怎么定义|如何定义|请定义|什么是)\s*(?P<after>.+)$"
    r"|(?P<before>.+?)(?:的定义(?:是)?什么|是?怎么定义的?|是?如何定义的?|是什么意思)$"
)
_DEFINITION_PREFACE = re.compile(
    r"^(?:请问|请|在|据|按|这份|上面|讲义|课件|论文|资料|中|里|的|把|给出|说明|一下|\s)+"
)
_TOPIC_SEPARATOR = re.compile(r"和|与|及|、|或|\band\b|\bor\b", re.I)
_HISTORY_IMAGE = re.compile(r"刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述")
_IMAGE_REFERENCE = re.compile(
    r"(?:刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述|那|这|哪)"
    r"\s*(?:的\s*)?(?:[那这一]\s*)?(?:张\s*)?(?:图片|图像|图)"
)
_TABLE_REFERENCE = re.compile(
    r"(?:刚才|上一条|前一条|上一轮|前一轮|上次|上面|上述|那|这)"
    r"\s*(?:的\s*)?(?:[那这一]\s*)?(?:张\s*)?(?:表格|原表|表)"
)
_CURRENT_IMAGE = re.compile(r"当前|本页|本\s*cell|这里|这页|这个\s*cell", re.IGNORECASE)
_BARE_REFERENCE = re.compile(r"^(?:刚才\s*)?第\s*\S+\s*张(?:图|图片)[。！？?\s]*$")
_PAGE = re.compile(r"第?\s*([1-9][0-9]{0,3})\s*(?:页|张幻灯片)")
_PAGE_PREFIX = re.compile(r"\b(?:page|slide)\s+#?([1-9][0-9]{0,3})\b", re.IGNORECASE)
_CELL = re.compile(r"第?\s*([1-9][0-9]{0,3})\s*(?:个)?\s*cell", re.IGNORECASE)
_CELL_PREFIX = re.compile(r"\bcell\s+([1-9][0-9]{0,3})\b", re.IGNORECASE)
_FILE_MENTION = re.compile(r"\.(?:pdf|pptx|ipynb)(?=$|[^a-z0-9_])", re.IGNORECASE)
_WHOLE_FILE = re.compile(
    r"这份(?:课件|文件|资料)|整份|全文|各页|\b(?:whole|entire) (?:file|document)\b", re.I
)
_REWRITE_PRIOR = re.compile(
    r"两句|[一两三二四]句话|简短|简要|简述|再说一遍|精简|summari[sz]e", re.I
)
_SUMMARY_FILE = re.compile(r"总结|概括|概要|summari[sz]e", re.I)
_CONTINUE_TEXT = re.compile(r"^(?:请)?(?:继续读|继续往后读|续读|读下一段)[。！？?\s]*$")
_SWITCH_FILE = re.compile(r"切换到|改问|改看|\bswitch to\b", re.I)
_SWITCH_DENIAL = re.compile(r"不要|请勿|禁止|别|\b(?:not|never|don['’]t)\b", re.I)
_FILE_EXCLUSION = re.compile(
    r"(?:不要|请勿|禁止|不用|不许|无需|不得|(?<!\w)别)(?:再|去)?\s*"
    r"(?:使用|用|查|看|读取|读|打开|引用|选择|选|切换到|改问|改看)"
    r"|\b(?:do not|don['’]t|must not|never)\s+(?:use|read|open|select|cite|switch to)\b",
    re.IGNORECASE,
)
_FILE_ARGUMENT = (
    r"""\s*["'“‘「『]?([^"'”’」』,，。！？；;!?\n]{1,180}?\.(?:pdf|pptx|ipynb))"""
    r"""(?=$|[\s"'”’」』,，。！？；;!?.和与、及])["'”’」』]?"""
)
_ONLY_FILE = re.compile(
    r"(?:只(?:在|查(?:看)?|看|用)|仅(?:在|查(?:看)?|看|用)|\bonly\s+(?:in|use|read))"
    r"\s*[:：]?" + _FILE_ARGUMENT,
    re.IGNORECASE,
)
_NEXT_FILE = re.compile(r"\s*(?:以及|和|与|、|,|\band\b)" + _FILE_ARGUMENT, re.IGNORECASE)
_POSITION_KEYS = {"page", "cell_index", "cell_id", "output_index", "section_path", "bbox"}
_TEXT_LIMIT = 900
_IMAGE_LIMIT = 3
_IMAGE_ATTEMPT_LIMIT = 12
_SUPPLEMENT_BUDGET = 2
_SUMMARY_SOURCE_LIMIT = 6
_BASE_KINDS = {"page", "cell"}
_HAN_QUERY = re.compile(r"[\u3400-\u9fff]")
_HAN_QUERY_RUN = re.compile(r"[\u3400-\u9fff]+")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _related_experiment_question(previous: str, current: str) -> bool:
    def clauses(text):
        if not asks_for_definition_and_experiment(f"定义。{_EXPERIMENT_TEXT.sub('实验', text)}"):
            return []
        return [
            clause for clause in re.split(r"[。！？!?；;\n]", text)
            if _EXPERIMENT_TEXT.search(clause)
        ]

    old_clauses, new_clauses = clauses(previous), clauses(current)
    if not old_clauses or not new_clauses:
        return False
    if any(_PREVIOUS_TURN.search(c) or _EXPERIMENT_REFERENCE.search(c) for c in new_clauses):
        return True

    def terms(items):
        source = _EXPERIMENT_FILLER.sub(" ", " ".join(items)).casefold()
        latin = set(re.findall(r"[a-z][a-z0-9_-]{2,}", source))
        han = {
            word[index:index + 3]
            for word in _HAN_QUERY_RUN.findall(source)
            for index in range(len(word) - 2)
        }
        return latin, han

    old_latin, old_han = terms(old_clauses)
    new_latin, new_han = terms(new_clauses)
    return bool(old_latin & new_latin or len(old_han & new_han) >= 2)


def _definition_topics(question: str) -> set[str]:
    topics = set()
    for clause in re.split(r"[，,。！？?!；;：:\n]", question):
        match = _DEFINITION_REQUEST.search(clause.strip())
        if not match:
            continue
        names = _DEFINITION_PREFACE.sub("", match["after"] or match["before"])
        for name in _TOPIC_SEPARATOR.split(names):
            name = name.strip(" \t`‘’“”'\"「」《》")
            if (
                2 <= len(name) <= 80
                and not _PREVIOUS_TURN.search(name)
                and not _DEFINITION_REFERENCE.search(name + "怎么定义")
                and re.fullmatch(r"[\w ()+./×-]+", name)
            ):
                topics.add(name)
    return topics


def _contains_topic(text: str, name: str) -> bool:
    name = " ".join(name.split())
    text = " ".join(text.split())
    if name.isascii():
        pattern = rf"(?<![a-z0-9_-]){re.escape(name)}(?![a-z0-9_-])"
        return re.search(pattern, text, re.I) is not None
    return name.casefold() in text.casefold()


def _has_latin_corpus(sources: Sequence[SnapshotSource]) -> bool:
    han_by_doc: dict[str, int] = {}
    latin_by_doc: dict[str, int] = {}
    for source in sources:
        if not source.text.strip():
            continue
        han_by_doc[source.document_id] = han_by_doc.get(source.document_id, 0) + len(
            _HAN_QUERY.findall(source.text)
        )
        latin_by_doc[source.document_id] = latin_by_doc.get(source.document_id, 0) + sum(
            len(word) for word in _LATIN_WORD.findall(source.text)
        )
    return any(
        latin_by_doc[doc_id] > han_by_doc.get(doc_id, 0) for doc_id in latin_by_doc
    )


@dataclass(frozen=True)
class CatalogNode:
    document: SnapshotDocument
    entry: dict[str, Any]
    source: SnapshotSource | None = None

    def key(self) -> SourceKey:
        assert self.source is not None
        return _key(self.source)


@dataclass
class Catalog:
    payload: list[dict[str, Any]]
    nodes: dict[str, CatalogNode]


@dataclass
class Draft:
    session_id: str
    message_id: str = field(default_factory=lambda: uuid4().hex)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, TurnEvidence] = field(default_factory=dict)
    images: list[Attachment] = field(default_factory=list)

    def text(self, text: str, *citations: str) -> None:
        self.blocks.append({"type": "text", "text": text, "citations": list(citations)})

    def notice(self, code: str, text: str) -> None:
        self.blocks.append({"type": "notice", "code": code, "text": text})

    def add_evidence(
        self, doc: SnapshotDocument, source: SnapshotSource, reads: list[TurnRead]
    ) -> TurnEvidence:
        evidence = TurnEvidence(
            evidence_id=f"ev-{len(self.sources) + 1}",
            revision=SourceRevision(
                source=_key(source),
                course_id=doc.course_id,
                publication_epoch=doc.publication_epoch,
                content_sha256=doc.content_sha256,
            ),
            file_name=doc.file_name,
            locator=_positions(source.locator),
            read_status=_status(reads),
            reads=reads,
        )
        self.sources[evidence.evidence_id] = evidence
        return evidence

    def add_image(
        self, evidence: TurnEvidence, result: EvidenceResult, request: EvidenceRequest, label: str
    ) -> None:
        display = result.display
        if result.status != "available" or display is None:
            self.notice(result.status, " · ".join([label, "图片不可用", *result.warnings[:1]]))
            return
        image = base64.b64decode(display.data_base64, validate=True)
        if _sha(image) != display.byte_sha256:
            self.notice("rendition_unavailable", "显示图片校验失败")
            return
        display_id = uuid4().hex
        source = display.display_source.model_dump()
        entry = {
            "type": "source_image",
            "display_id": display_id,
            "evidence_id": evidence.evidence_id,
            "display_source": source,
            "rendition_kind": display.rendition_kind,
            "caption": label,
            "width": display.width,
            "height": display.height,
            "rect_visible_norm": display.rect_visible_norm,
            "read_status": result.read_status,
            "fallback_reason": display.fallback_reason,
            "media_url": (
                f"/api/workspace/sessions/{self.session_id}/messages/"
                f"{self.message_id}/media/{display_id}?view=inline"
            ),
        }
        self.images.append(
            Attachment(
                display_id,
                {
                    **entry,
                    "content_sha256": evidence.revision.content_sha256,
                    "publication_epoch": evidence.revision.publication_epoch,
                    "course_id": evidence.revision.course_id,
                    "file_name": evidence.file_name,
                    "mime_type": display.media_type,
                    "byte_sha256": display.byte_sha256,
                    "request": request.model_dump(mode="json"),
                },
                image,
            )
        )
        self.blocks.append(entry)

    def answer(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "blocks": self.blocks,
            "sources": {key: value.model_dump(mode="json") for key, value in self.sources.items()},
            "visible_display_ids": [image.display_id for image in self.images],
        }


def make_workspace(
    settings: Settings,
    *,
    regions: RegionRegistry | None = None,
    region_ids: Mapping[tuple[str, int, str], str] | None = None,
) -> WorkspaceService:
    """Build the shared CLI/HTTP logic. Region injection is server-owned."""

    allowed = frozenset(
        item.strip() for item in settings.workspace_course_ids.split(",") if item.strip()
    )
    library = LocalLibrary(settings)
    store = WorkspaceStore(settings)
    store.initialize()
    snapshots = StructureSnapshotStore(settings)
    evidence = SourceEvidenceService(
        library.store,
        cache_dir=settings.data_dir / "source_evidence_cache",
        current_course_ids=sorted(allowed),
        soffice_path=settings.srag_soffice,
        regions=regions,
        max_inline_bytes=settings.source_evidence_max_inline_bytes,
    )
    vision = SourceVisionService(
        evidence,
        library.store,
        cache_dir=settings.data_dir / "source_vision_cache",
        enabled=settings.source_vision_enabled,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_vision_model,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_image_bytes=settings.dashscope_visual_max_image_bytes,
    )
    return WorkspaceService(settings, store, library, snapshots, vision, allowed, region_ids or {})


class WorkspaceService:
    def __init__(
        self,
        settings: Settings,
        store: WorkspaceStore,
        library: LocalLibrary,
        snapshots: StructureSnapshotStore,
        vision: SourceVisionService,
        allowed_courses: frozenset[str],
        region_ids: Mapping[tuple[str, int, str], str],
    ) -> None:
        self.settings = settings
        self.store = store
        self.library = library
        self.snapshots = snapshots
        self.searcher = StructureSearcher(snapshots)
        self.answerer = EvidenceAnswerer(SupplementRetriever(library.store, snapshots))
        self.vision = vision
        self.evidence = vision.evidence
        self.allowed_courses = allowed_courses
        self.region_ids = region_ids
        from app.clients.jev import JevClient
        from app.rag.workspace_retrieval import BilingualRewriter

        self.rewriter = BilingualRewriter(settings)
        self.scorer = JevClient(settings)

    def create(self, owner: str | None, request: CreateSession) -> tuple[dict[str, Any], str]:
        courses, documents = self._scope(
            request.course_ids or sorted(self.allowed_courses), request.document_ids
        )
        raw_owner = secrets.token_urlsafe(32) if owner is None else owner
        session = self.store.create(
            raw_owner,
            fresh=owner is None,
            title=request.title.strip() or "学习会话",
            course_ids=courses,
            document_ids=documents,
        )
        return {**self._header(session), "focus": None, "turns": [], "messages": []}, raw_owner

    def list_sessions(self, owner: str) -> dict[str, Any]:
        return {"sessions": [self._header(session) for session in self.store.sessions(owner)]}

    def session(self, owner: str, session_id: str) -> dict[str, Any]:
        session, turns, messages = self.store.timeline(owner, session_id)
        by_id = {turn.turn_id: turn for turn in turns}
        access = {turn.turn_id: self._turn_access(turn) for turn in turns}
        catalog = self._catalog(session_id) if session.focus_id else None
        focus = catalog.nodes.get(session.focus_id) if catalog and session.focus_id else None
        return {
            **self._header(session),
            "focus": _focus_entry(focus) if focus else None,
            "turns": [self._turn_entry(turn, access[turn.turn_id]) for turn in turns],
            "messages": [
                self._project_message(message, access[by_id[message["turn_id"]].turn_id])
                for message in messages
            ],
        }

    def update_scope(self, owner: str, session_id: str, request: UpdateScope) -> dict[str, Any]:
        session = self._session_revision(owner, session_id, request.scope_revision)
        courses, docs = self._scope(request.course_ids, request.document_ids)
        self.store.update_context(owner, session, courses, docs, None)
        return self.session(owner, session_id)

    def update_focus(self, owner: str, session_id: str, request: UpdateFocus) -> dict[str, Any]:
        session = self._session_revision(owner, session_id, request.scope_revision)
        if request.node_id is not None:
            node = self._node(session, request.node_id)
            if not _in_scope(session, node.document):
                raise WorkspaceError("scope_changed")
        self.store.update_context(
            owner, session, session.course_ids, session.document_ids, request.node_id
        )
        return self.session(owner, session_id)

    def catalog(self, owner: str, session_id: str) -> dict[str, Any]:
        session = self.store.session(owner, session_id)
        return {
            "courses": self._catalog(session_id).payload,
            "scope_revision": session.scope_revision,
        }

    def catalog_media(self, owner: str, session_id: str, node_id: str) -> tuple[bytes, str]:
        session = self.store.session(owner, session_id)
        node = self._node(session, node_id)
        if not _in_scope(session, node.document) or node.source is None:
            raise inaccessible()
        response = self.evidence.render_current(self._render_request(node.document, node.source))
        return _image_response(response)

    def authorize_import(
        self, owner: str, session_id: str, course_id: str, document_id: str | None
    ) -> None:
        self.store.session(owner, session_id)
        self._scope([course_id], [document_id] if document_id else [])

    def import_file(
        self, owner: str, session_id: str, course_id: str, path: Path, document_id: str | None
    ) -> dict[str, Any]:
        self.authorize_import(owner, session_id, course_id, document_id)
        result = self.library.import_paths(course_id, [path], document_id=document_id)[0]
        if result.status == "failed":
            return {"file_name": path.name, "status": "failed", "error": "import_failed"}
        value = result.model_dump(mode="json", exclude={"source_path", "error"})
        return {"file_name": path.name, **value}

    def submit(self, owner: str, session_id: str, request: TurnRequest) -> dict[str, Any]:
        turn, started = self.store.start(owner, session_id, request)
        if started:
            turn = self._run(owner, turn)
        return self._envelope(turn)

    def get_turn(self, owner: str, session_id: str, client_turn_id: str) -> dict[str, Any]:
        return self._envelope(self.store.turn(owner, session_id, client_turn_id))

    def cancel(self, owner: str, session_id: str, client_turn_id: str) -> dict[str, Any]:
        return self._envelope(self.store.cancel(owner, session_id, client_turn_id))

    def retry(self, owner: str, session_id: str, client_turn_id: str) -> dict[str, Any]:
        turn, started = self.store.retry(owner, session_id, client_turn_id)
        if started:
            turn = self._run(owner, turn)
        return self._envelope(turn)

    def historical_media(
        self, owner: str, session_id: str, message_id: str, display_id: str, *, view: str
    ) -> tuple[bytes, str]:
        self.store.session(owner, session_id)
        item = self._attachment(session_id, message_id, display_id)
        details = item.description
        key = SourceKey.model_validate(details["display_source"])
        self._historical_source(details, key)
        if view == "inline":
            if _sha(item.image) != details["byte_sha256"]:
                raise WorkspaceError("rendition_unavailable", 409)
            return item.image, details["mime_type"]
        if details["rendition_kind"] == "notebook_output":
            raise WorkspaceError("rendition_unavailable", 409)
        # D01 handles same-version parent selection even when the display is a page crop.
        request = EvidenceRequest(
            source=key,
            expected_content_sha256=details["content_sha256"],
            rendition="original_page",
        )
        grant = HistoricalSourceGrant(
            key,
            (details["course_id"],),
            details["content_sha256"],
            message_id,
            details["evidence_id"],
        )
        return _image_response(self.evidence.render_historical(request, grant))

    def historical_evidence(
        self, owner: str, session_id: str, message_id: str, evidence_id: str
    ) -> dict[str, Any]:
        stored = self.store.completed_answer(owner, session_id, message_id)
        if stored is None:
            raise inaccessible()
        turn, message = stored
        if not self._turn_access(turn) or not any(
            block["type"] == "text" and evidence_id in block.get("citations", [])
            for block in message["blocks"]
        ):
            raise inaccessible()
        payload = message["sources"].get(evidence_id)
        if payload is None:
            raise inaccessible()
        evidence = TurnEvidence.model_validate(payload)
        key = evidence.revision.source
        read = next(
            (
                item
                for item in evidence.reads
                if item.actual_source == key
                and item.method in {"native_text", "native_table", "ocr"}
                and item.status in {"succeeded", "partial"}
                and item.text
            ),
            None,
        )
        if read is None:
            raise inaccessible()
        current = self._historical_source(
            {
                "course_id": evidence.revision.course_id,
                "content_sha256": evidence.revision.content_sha256,
            },
            key,
        )
        source = current.source
        if source is None or not self._matches_text(evidence, read, source.text):
            raise WorkspaceError("original_unavailable", 409)
        if not self._turn_access(turn):
            raise inaccessible()
        return {
            "source": key.model_dump(),
            "file_name": evidence.file_name,
            "locator": _positions(source.locator),
            "text": read.text,
            "is_current": current.is_current,
            "has_preview": _can_display(source),
        }

    def _run(self, owner: str, turn: Turn) -> Turn:
        session = self.store.session(owner, turn.session_id)
        try:
            if session.scope_revision != turn.request.scope_revision:
                raise WorkspaceError("scope_changed")
            if isinstance(turn.request.anchor, DisplayAnchor):
                return self._follow_display(
                    turn, session, turn.request.anchor, "display_anchor", turn.request.text
                )
            if isinstance(turn.request.anchor, ClarificationAnchor):
                return self._choose(turn, session, turn.request.anchor)
            if turn.request.kind == "ask" and is_version_comparison(turn.request.text):
                return compare_versions(self, owner, turn, session)
            ordinal = _ORDINAL.search(turn.request.text)
            located = _location(turn.request.text, None) is not None or _FILE_MENTION.search(
                turn.request.text
            )
            image_reference = ordinal or _IMAGE_REFERENCE.search(turn.request.text)
            if (
                turn.request.kind == "ask"
                and image_reference
                and (not located or _PREVIOUS_TURN.search(turn.request.text))
            ):
                return self._resolve_reference(owner, turn, session, ordinal)
            if (
                turn.request.kind == "ask"
                and _TABLE_REFERENCE.search(turn.request.text)
                and (not located or _PREVIOUS_TURN.search(turn.request.text))
            ):
                resolved = self._resolve_table_reference(owner, turn, session)
                if resolved is not None:
                    return resolved
            return self._retrieve(turn, session, turn.request.text, owner=owner)
        except WorkspaceError as error:
            # Authorized absence of a stored reference is a turn failure, with no source metadata.
            return self.store.finish(
                turn,
                "failed",
                {
                    "error": error.code,
                    "retryable": error.code
                    in {
                        "stale_reference",
                        "interrupted",
                        "temporarily_unavailable",
                    },
                },
                None,
                (),
                (),
                self.allowed_courses,
            )
        except (OSError, RuntimeError, ValueError):
            return self.store.finish(
                turn,
                "failed",
                {"error": "temporarily_unavailable", "retryable": True},
                None,
                (),
                (),
                self.allowed_courses,
            )

    def _resolve_reference(
        self, owner: str, turn: Turn, session: Session, match: re.Match[str] | None
    ) -> Turn:
        explicit_history = bool(_HISTORY_IMAGE.search(turn.request.text))
        if not explicit_history and _CURRENT_IMAGE.search(turn.request.text):
            return resolve_current_image(self, turn, session)
        _, turns, messages = self.store.timeline(owner, session.session_id)
        completed = {item.turn_id for item in turns if item.status == "completed"}
        boundary = next(
            message["ordinal"]
            for message in messages
            if message["turn_id"] == turn.turn_id and message["role"] == "user"
        )
        previous = next(
            (
                item
                for item in reversed(messages)
                if item["role"] == "assistant"
                and item["turn_id"] in completed
                and item["ordinal"] < boundary
                and item["visible_display_ids"]
            ),
            None,
        )
        if previous is None:
            if not explicit_history:
                return resolve_current_image(self, turn, session)
            return self._clarify(turn, "找不到带图片的已完成回答，请从原件卡片选择。", [], [])
        ids = previous["visible_display_ids"]
        index = _number(match.group(1)) if match else (1 if len(ids) == 1 else 0)
        if not explicit_history and session.focus_id:
            catalog = self._catalog(session.session_id)
            focus = catalog.nodes.get(session.focus_id)
            relevant = {ids[index - 1]} if 1 <= index <= len(ids) else set(ids)
            displays = [
                block for block in previous["blocks"] if block.get("display_id") in relevant
            ]
            if (
                focus is None
                or any(not _focus_matches_display(focus, previous, block) for block in displays)
                or (
                    match is not None
                    and len(displays) == 1
                    and not _focus_ordinal_matches(catalog, focus, index, displays[0])
                )
            ):
                return self._clarify(
                    turn, "当前焦点与此前展示不同，请说明要读“当前原件”还是“刚才的图”。", [], []
                )
        if 1 <= index <= len(ids):
            anchor = DisplayAnchor(
                type="display",
                message_id=previous["message_id"],
                display_id=ids[index - 1],
            )
            return self._follow_display(
                turn,
                session,
                anchor,
                "visible_ordinal" if match else "visible_single",
                turn.request.text,
            )
        options, refs = [], []
        for order, display_id in enumerate(ids[:6], 1):
            try:
                item = self._attachment(session.session_id, previous["message_id"], display_id)
            except WorkspaceError:
                continue
            options.append(
                {
                    "option_id": uuid4().hex,
                    "label": f"第 {order} 张图 · {item.description['caption']}",
                    "kind": "display",
                    "message_id": previous["message_id"],
                    "display_id": display_id,
                }
            )
            refs.append(_attachment_ref(item))
        return self._clarify(turn, "请选择你指的图片。", options, refs, historical=True)

    def _choose(self, turn: Turn, session: Session, anchor: ClarificationAnchor) -> Turn:
        previous = self.store.clarification(session.session_id, anchor.turn_id)
        if (
            previous is None
            or previous.request.scope_revision != session.scope_revision
            or not self._turn_access(previous)
        ):
            raise inaccessible()
        option = next(
            (
                item
                for item in previous.outcome.get("options", [])
                if item["option_id"] == anchor.option_id
            ),
            None,
        )
        if option is None:
            raise inaccessible()
        if option["kind"] == "display":
            selected = DisplayAnchor(
                type="display", message_id=option["message_id"], display_id=option["display_id"]
            )
            return self._follow_display(
                turn,
                session,
                selected,
                "clarification_choice",
                previous.request.text,
            )
        node = self._node(session, option["node_id"])
        if not _in_scope(session, node.document):
            raise inaccessible()
        if option.get("current_object"):
            return answer_current_image(
                self, turn, session, node, previous.request.text, "current_choice"
            )
        return self._retrieve(
            turn, session, previous.request.text, forced=node, kind=previous.request.kind
        )

    def _follow_display(
        self, turn: Turn, session: Session, anchor: DisplayAnchor, how: str, question: str
    ) -> Turn:
        item = self._attachment(session.session_id, anchor.message_id, anchor.display_id)
        desc = item.description
        key = SourceKey.model_validate(desc["display_source"])
        stored = self._historical_source(desc, key)
        ref = _attachment_ref(item)
        message = {
            "method": how,
            "resolution": how,
            "message_id": anchor.message_id,
            "display_id": anchor.display_id,
            "actual_source": key.model_dump(),
            "ordinal": self._ordinal(session.session_id, item, anchor.message_id),
        }
        draft = Draft(session.session_id)
        if ref["course_id"] not in session.course_ids or (
            session.document_ids and key.document_id not in session.document_ids
        ):
            draft.notice("outside_current_scope", "图片在当前范围之外。请切到该文件后再追问。")
            return self._complete(turn, draft, historical=[ref], resolution=message)
        if not stored.is_current or stored.publication_epoch != ref["publication_epoch"]:
            draft.notice(
                "historical_only",
                "这是先前回答中的原图。可以回看旧图，或从目录另选当前版本。",
            )
            return self._complete(turn, draft, historical=[ref], resolution=message)
        if not self.store.active(turn):
            raise WorkspaceError("interrupted")
        request = EvidenceRequest.model_validate(desc["request"])
        if request.source != key:
            # A parent-page fallback has become the actual display and follow-up object.
            request = EvidenceRequest(
                source=key,
                expected_content_sha256=ref["content_sha256"],
                expected_publication_epoch=ref["publication_epoch"],
                rendition="original_page",
            )
        same_display = self.evidence.render_current(request)
        if same_display.display is None:
            raise WorkspaceError(same_display.status)
        if (
            same_display.display.display_source != key
            or same_display.display.byte_sha256 != desc["byte_sha256"]
        ):
            raise WorkspaceError("stale_reference")
        if not self.store.active(turn):
            raise WorkspaceError("interrupted")
        prompt = question.strip()
        if not prompt or _BARE_REFERENCE.fullmatch(prompt):
            prompt = "请描述图中实际可见的内容，尤其是坐标轴名称和单位（如有）。"
        started = perf_counter()
        response = self.vision.process_current(
            EvidenceRequest.model_validate(
                {
                    **request.model_dump(),
                    "read_intent": "directed_question",
                    "question": prompt,
                }
            )
        )
        elapsed_ms = (perf_counter() - started) * 1000
        actual_display = response.display
        error = response.status
        if actual_display is not None and (
            actual_display.actual_source != key or actual_display.byte_sha256 != desc["byte_sha256"]
        ):
            error = "stale_reference"
        if error in {"stale_reference", "not_accessible", "original_unavailable"}:
            return self._finish_invalid_read(turn, error, _vision_trace(response, elapsed_ms))
        reads = _visual_reads(response)
        source = stored.source
        assert source is not None
        evidence = TurnEvidence(
            evidence_id="ev-1",
            revision=SourceRevision(
                source=key,
                course_id=ref["course_id"],
                publication_epoch=ref["publication_epoch"],
                content_sha256=ref["content_sha256"],
            ),
            file_name=desc["file_name"],
            locator=_positions(source.locator),
            read_status=response.read_status,
            reads=reads,
        )
        draft.sources["ev-1"] = evidence
        readable = next(
            (
                record.text
                for record in reads
                if record.status in {"succeeded", "partial"}
                and record.origin_execution == "real_provider"
                and record.text
            ),
            None,
        )
        if readable:
            draft.text(readable, evidence.evidence_id)
        elif response.read_status == "failed":
            draft.notice("vision_failed", "已定位到原图，视觉服务读取失败。可重试或直接查看原件。")
        else:
            draft.notice(
                response.capability_reason or "not_attempted",
                f"已定位到第 {message['ordinal']} 张图。当前没有可用的读图结果，可打开原图查看。",
            )
        return self._complete(
            turn,
            draft,
            refs=[ref],
            resolution=message,
            trace=_vision_trace(response, elapsed_ms),
            failed=response.read_status == "failed",
        )

    def _retrieve(
        self,
        turn: Turn,
        session: Session,
        text: str,
        *,
        forced: CatalogNode | None = None,
        kind: str | None = None,
        owner: str | None = None,
    ) -> Turn:
        mode = kind or turn.request.kind
        node = forced
        switching = bool(mode == "ask" and _FILE_MENTION.search(text) and _affirmed_switch(text))
        explicit_location = _location(text, None)
        whole_file = _WHOLE_FILE.search(text)
        definition_reference = (
            _DEFINITION_REFERENCE.search(text) if not _definition_topics(text) else None
        )
        followup = bool(
            self.settings.workspace_answer_enabled
            and owner
            and mode == "ask"
            and not explicit_location
            and not whole_file
            and (definition_reference or not _FILE_MENTION.search(text))
            and (
                definition_reference
                or _PREVIOUS_TURN.search(text)
                or _CONTINUE_TEXT.fullmatch(text)
                or (len(text) <= 28 and _REWRITE_PRIOR.search(text))
            )
        )
        check_experiment_scope = (
            self.settings.workspace_answer_enabled
            and mode == "ask"
            and bool(session.document_ids)
            and bool(_EXPERIMENT_TEXT.search(text))
        )
        latest = (
            self._latest_reply(owner, turn)
            if owner and (followup or check_experiment_scope)
            else None
        )
        topic_document = None
        if followup and latest and latest[0].request.scope_revision == turn.request.scope_revision:
            topic_document = (latest[0].outcome.get("trace") or {}).get("topic_document_id")
        if (
            node is None
            and session.focus_id
            and mode == "ask"
            and not (switching or topic_document)
        ):
            node = self._node(session, session.focus_id, stale=True)
        if not set(session.course_ids) <= self.allowed_courses:
            raise inaccessible()
        doc_ids = (
            (node.document.document_id,)
            if node
            else (
                (topic_document,)
                if topic_document
                else (() if switching else tuple(session.document_ids))
            )
        )
        snapshot = self.snapshots.read(SnapshotRequest(tuple(session.course_ids), doc_ids))
        if snapshot.resolution_status != "resolved":
            if doc_ids:
                raise WorkspaceError("stale_reference")
            draft = Draft(session.session_id)
            draft.notice("empty_library", "这个范围没有已发布资料。请导入文件或切换范围。")
            return self._complete(turn, draft)
        documents = {doc.document_id: doc for doc in snapshot.documents}
        if node and (
            not _in_scope(session, node.document)
            or _doc_ref(documents[node.document.document_id]) != _doc_ref(node.document)
        ):
            raise WorkspaceError("stale_reference")
        refs = [_doc_ref(doc) for doc in documents.values()]
        if switching:
            names = _object_file_names(text, documents.values())
            selected = (
                self.snapshots.read(
                    SnapshotRequest(
                        tuple(session.course_ids),
                        tuple(documents),
                        tuple(names or ()),
                        require_single_document=True,
                    )
                )
                if names
                else None
            )
            if selected is None or selected.resolution_status != "resolved":
                draft = Draft(session.session_id)
                draft.notice("scope_not_found", "无法确认要切换的文件，请从资料目录选择。")
                return self._complete(turn, draft)
            if any(
                _doc_ref(doc) != _doc_ref(documents[doc.document_id]) for doc in selected.documents
            ):
                raise WorkspaceError("stale_reference")
            snapshot = selected
            documents = {doc.document_id: doc for doc in snapshot.documents}
            refs = [_doc_ref(doc) for doc in documents.values()]
            topic_document = selected.documents[0].document_id
        if mode == "summary" and len(documents) != 1:
            options = self._document_choices(session.session_id, documents)
            return self._clarify(turn, "请选择一份资料进行总结。", options, refs)
        sources = {(source.document_id, source.source_id): source for source in snapshot.sources}
        location = explicit_location or (None if whole_file else _location("", node))
        file_names = [
            doc.file_name
            for doc in documents.values()
            if doc.file_name.casefold() in text.casefold()
        ]
        if self.settings.workspace_answer_enabled and mode in {"ask", "summary"}:
            names = _object_file_names(text, documents.values())
            if names is None:
                draft = Draft(session.session_id)
                draft.notice("scope_not_found", "指定文件不在本轮范围，请确认资料或明确切换。")
                return self._complete(turn, draft, refs=refs)
            file_names = names
        if mode == "ask" and asks_for_composition(text):
            return self._answer_composition(turn, session, text, snapshot)
        if (
            mode == "ask"
            and location is None
            and not self.settings.workspace_answer_enabled
            and asks_for_definition_and_experiment(text)
        ):
            file_names = _explicit_filenames(text) or file_names
            return self._answer_with_supplement(turn, session, text, documents, file_names)
        if mode == "ask" and (target := object_intent(text)):
            return self._answer_object(turn, session, text, target, documents, node, location)
        previous = None
        resolved_topic = None
        if followup:
            current_documents = {
                key: doc for key, doc in documents.items()
                if not file_names or doc.file_name in file_names
            }
            same_scope = bool(
                latest and latest[0].request.scope_revision == turn.request.scope_revision
            )
            if same_scope:
                previous = self._scoped_previous(latest, current_documents)
            if definition_reference and previous is None:
                if not same_scope:
                    resolved_topic = self._resolve_definition_topic(latest, current_documents)
                if resolved_topic is None:
                    return self._clarify(
                        turn, "无法确定你指的是哪个定义。请写出概念或规则名称。", [], refs,
                    )
                start, end = definition_reference.span("reference")
                text = f"{text[:start]}{resolved_topic}{text[end:]}"
            if _REWRITE_PRIOR.search(text):
                if previous:
                    return self._rewrite_previous(turn, previous, refs, topic_document)
                draft = Draft(session.session_id)
                draft.notice(
                    "prior_evidence_unavailable", "本范围没有可核对的上一轮回答。请重述问题。"
                )
                return self._complete(turn, draft, refs=refs)
        if mode == "summary":
            ranked = sorted(
                (source for source in snapshot.sources if source.text.strip()),
                key=lambda source: source.kind not in _BASE_KINDS,
            )
            trace: dict[str, Any] = {
                "effective_scope": _effective(documents),
                "available_sources": len(ranked),
                "max_sources": _SUMMARY_SOURCE_LIMIT,
            }
        else:
            request = StructureSearchRequest(
                query=text,
                location=location,
                top_k=12,
                scope=StructureScope(
                    course_ids=session.course_ids,
                    document_ids=list(documents),
                    file_names=[] if node else file_names,
                ),
            )
            use_retrieval = (
                mode == "ask"
                and not _CONTINUE_TEXT.fullmatch(text)
                and not (whole_file and _SUMMARY_FILE.search(text))
            )
            plan = (
                self._plan_search(turn, snapshot, request, previous=previous, file_names=file_names)
                if use_retrieval
                else None
            )
            search = (
                plan.initial
                if plan
                else self.searcher.search(request, allowed_course_ids=self.allowed_courses)
            )
            if search.status == "ambiguous_scope":
                match_ids = {item.document_id for item in search.options}
                selected = {
                    key: doc for key, doc in documents.items() if not match_ids or key in match_ids
                }
                return self._clarify(
                    turn,
                    "有多份资料符合范围，请选择一份。",
                    self._document_choices(session.session_id, selected),
                    refs,
                )
            expected = {key: _doc_ref(doc) for key, doc in documents.items()}
            for doc in search.snapshot:
                if _doc_ref(doc) != expected.get(doc.document_id):
                    raise WorkspaceError("stale_reference")
            if not file_names and {doc.document_id for doc in search.snapshot} != set(documents):
                raise WorkspaceError("stale_reference")
            chosen = plan.selected if plan else search.candidates
            ranked = [
                sources[(hit.source.document_id, hit.source.source_id)]
                for hit in chosen
                if (hit.source.document_id, hit.source.source_id) in sources
            ]
            visual_ranked = [
                sources[(hit.source.document_id, hit.source.source_id)]
                for hit in (getattr(plan, "visual_candidates", ()) if plan else ())
                if (hit.source.document_id, hit.source.source_id) in sources
            ]
            trace = {
                "effective_scope": _effective(documents),
                "search_status": search.status,
                "coverage": search.coverage.model_dump(mode="json"),
                "candidates": [
                    {
                        "source": hit.source.model_dump(),
                        "file_name": hit.file_name,
                        "locator": _positions(hit.locator),
                        "match_kind": hit.match_kind,
                        "reasons": [reason.model_dump() for reason in hit.reasons],
                    }
                    for hit in chosen
                ],
            }
            if plan:
                trace["retrieval"] = plan.trace
                trace["external_call_count"] = plan.trace.get("external_call_count", 0)
        if topic_document:
            trace["topic_document_id"] = topic_document
        if resolved_topic:
            trace["definition_topic"] = resolved_topic
        if mode in {"ask", "summary"} and self.settings.workspace_answer_enabled:
            if self.settings.deepseek_api_key.strip():
                answer_mode = "summary" if whole_file and _SUMMARY_FILE.search(text) else mode
                return self._generate_snapshot(
                    turn,
                    snapshot,
                    ranked,
                    trace,
                    mode=answer_mode,
                    location=location,
                    names=file_names,
                    previous=previous,
                    question=text,
                    extra_media=visual_ranked if mode == "ask" else (),
                    selected_file_scope=bool(session.document_ids or file_names),
                    preceding_reply=latest if check_experiment_scope else None,
                )
        draft = Draft(session.session_id)
        if mode == "summary":
            draft.text("当前文件摘要（部分来源）")
        failures = self._read_text(
            turn, draft, documents, ranked, _SUMMARY_SOURCE_LIMIT if mode == "summary" else 2
        )
        if mode == "summary":
            coverage = _summary_coverage(snapshot.sources, draft.sources, failures)
            trace.update(coverage)
            if coverage["full_text_scope"]:
                draft.blocks[0]["text"] = "当前文件摘要"
            else:
                draft.notice("partial_summary", "部分页、Cell 或摘录未完整读取，请参照来源与范围。")
        elif mode == "ask":
            media_focus = None if explicit_location else node
            chosen = _media_candidates(ranked, snapshot.sources, media_focus)
            trace["display_attempts"] = self._attach_images(turn, draft, documents, chosen)
        if not draft.sources:
            draft.notice("no_readable_evidence", "在所选范围内没有读到支持本题的原文或图片。")
        elif mode in {"ask", "summary"} and any(b["type"] == "text" for b in draft.blocks):
            draft.notice("excerpt_fallback", "未启用回答模型或缺少配置。本条为原文摘录。")
        return self._complete(turn, draft, refs=refs, trace=trace)

    def _plan_search(
        self,
        turn: Turn,
        snapshot: StructureSnapshot,
        request: StructureSearchRequest,
        *,
        previous=None,
        file_names: Sequence[str] = (),
    ):
        settings = self.settings
        strategy = settings.workspace_retrieval_strategy
        if (
            request.location is not None
            or not settings.workspace_answer_enabled
            or not settings.deepseek_api_key.strip()
        ):
            return None
        budget = min(
            settings.workspace_input_token_limit, settings.answer_context_tokens - OUTPUT_RESERVE
        )
        if budget <= 0:
            return None
        if file_names:
            allowed = {name.casefold() for name in file_names}
            target_docs = tuple(
                doc for doc in snapshot.documents if doc.file_name.casefold() in allowed
            )
            doc_ids = {doc.document_id for doc in target_docs}
            target_sources = tuple(
                src for src in snapshot.sources if src.document_id in doc_ids
            )
        else:
            target_docs = snapshot.documents
            target_sources = snapshot.sources
        if not target_docs:
            return None
        docs_by_id = {doc.document_id: doc for doc in target_docs}
        readable_sources = [src for src in target_sources if src.text.strip()]
        if not readable_sources:
            return None
        context = self._prior_context(previous) if previous else None
        prompt_mode = self._prompt_mode("ask", request.query, request.location)
        unread_visual = sum(1 for src in target_sources if _has_visual(src))
        fixed_reserve = estimate_prompt_reserve(
            request.query,
            mode=prompt_mode,
            context=context,
            coverage={"full_text_scope": False, "unread_visual_total": unread_visual},
        )
        if budget <= fixed_reserve:
            return None
        from app.rag.workspace_reading import (
            estimate_embedded_json_budget,
            estimate_embedded_text_budget,
        )

        def _skel(idx: int, src: SnapshotSource) -> dict[str, Any]:
            method = (
                "native_table" if src.locator.get("object_type") == "table" else "native_text"
            )
            return {
                "id": f"ev-{idx}",
                "source": _key(src).model_dump(),
                "file_name": docs_by_id[src.document_id].file_name,
                "locator": {
                    k: src.locator[k]
                    for k in ("page", "cell_index", "output_index", "section_path", "bbox")
                    if k in src.locator
                },
                "reads": [{"method": method, "status": "succeeded", "text": ""}],
            }

        min_source_bytes = min(
            estimate_embedded_json_budget(_skel(1, src)) for src in readable_sources
        )
        if budget - fixed_reserve <= min_source_bytes:
            return None
        if len(target_docs) == 1:
            reserve_true = estimate_prompt_reserve(
                request.query,
                mode=prompt_mode,
                context=context,
                coverage={"full_text_scope": True, "unread_visual_total": unread_visual},
            )
            total_single_bytes = reserve_true
            for text_idx, src in enumerate(readable_sources, 1):
                total_single_bytes += (
                    estimate_embedded_json_budget(_skel(text_idx, src))
                    + (1 if text_idx > 1 else 0)
                    + estimate_embedded_text_budget(src.text)
                )
                if total_single_bytes > budget:
                    break
            if total_single_bytes <= budget:
                return None  # Full-text reading already covers this selected short document.
        if strategy == "fts":
            if not _HAN_QUERY.search(request.query) or not _has_latin_corpus(target_sources):
                return None
            strategy = "bilingual"
        from app.rag.workspace_retrieval import plan_retrieval

        candidate_budget = request.top_k if strategy == "bilingual" else 4
        return plan_retrieval(
            settings,
            self.searcher,
            snapshot,
            request,
            self.allowed_courses,
            strategy=strategy,
            budget=candidate_budget,
            library=self.library.store,
            active=lambda: self.store.active(turn),
            rewrite=self.rewriter,
            scorer=self.scorer,
            supplement_saved_outputs=prompt_mode == "definition_and_experiment",
        )

    def _matches_text(self, item: TurnEvidence, read: TurnRead, original: str) -> bool:
        if read.method == "ocr":
            original = self.library.store.read_ocr_text(
                item.revision.source, course_ids=(item.revision.course_id,)
            )
        if span := item.locator.get("text_span"):
            start, end = span
            return 0 <= start < end <= len(original) and original[start:end] == read.text
        excerpt = read.text.removesuffix("…") if "excerpt_truncated" in read.warnings else read.text
        return original.startswith(excerpt) or original.strip().startswith(excerpt)

    def _latest_reply(self, owner: str, turn: Turn) -> tuple[Turn, dict[str, Any]] | None:
        latest = self.store.previous_reply(owner, turn)
        return latest if latest and self._turn_access(latest[0]) else None

    def _resolve_definition_topic(self, latest, documents) -> str | None:
        if not latest:
            return None
        topics = _definition_topics(latest[0].request.text)
        if not topics:
            return None
        current = self._scoped_previous(latest, documents, allow_subset=True)
        if not current:
            return None
        confirmed = {
            name for name in topics
            if any(_contains_topic(read.text, name)
                   for item in current[2].values() for read in item.reads)
        }
        return next(iter(confirmed)) if len(confirmed) == 1 else None

    @staticmethod
    def _excluded_previous_experiment(turn, preceding_reply, document_ids, question) -> bool:
        if not preceding_reply:
            return False
        previous, message = preceding_reply
        if (
            previous.request.scope_revision == turn.request.scope_revision
            or not _related_experiment_question(previous.request.text, question)
        ):
            return False
        # Only use the persisted identities of citations from the visible answer.
        for block in message["blocks"]:
            if block["type"] != "text":
                continue
            for key in block.get("citations", []):
                item = message["sources"].get(key)
                if not item or not isinstance(item["locator"].get("output_index"), int):
                    continue
                identity = item["revision"]["source"]
                if identity["document_id"] not in document_ids and any(
                    read["actual_source"] == identity
                    and read["method"] == "native_text"
                    and read["status"] in {"succeeded", "partial"}
                    for read in item["reads"]
                ):
                    return True
        return False

    def _scoped_previous(
        self, latest: tuple[Turn, dict[str, Any]], documents: dict[str, SnapshotDocument],
        *, allow_subset: bool = False,
    ) -> tuple[Turn, dict[str, Any], dict[str, TurnEvidence]] | None:
        previous, message = latest
        identifiers = {
            key
            for block in message["blocks"]
            if block["type"] == "text"
            for key in block.get("citations", [])
        }
        sources = {}
        for key in message["sources"]:
            if key not in identifiers:
                continue
            item = TurnEvidence.model_validate(message["sources"][key])
            if allow_subset and isinstance(item.locator.get("output_index"), int):
                continue
            identity = item.revision.source
            doc = documents.get(identity.document_id)
            if (
                doc is None
                or doc.version != identity.version
                or doc.publication_epoch != item.revision.publication_epoch
                or doc.content_sha256 != item.revision.content_sha256
            ):
                if allow_subset:
                    continue
                return None
            raw = self._historical_source(_doc_ref(doc), identity)
            reads = [
                read
                for read in item.reads
                if raw.source
                and read.text
                and read.status in {"succeeded", "partial"}
                and read.actual_source == identity
                and read.method in {"native_text", "native_table", "ocr"}
                and self._matches_text(item, read, raw.source.text)
            ]
            if reads:
                sources[key] = item.model_copy(update={"reads": reads})
        return (previous, message, sources) if sources else None

    @staticmethod
    def _prior_context(previous) -> dict[str, str]:
        turn, message, _ = previous
        body = "\n".join(block["text"] for block in message["blocks"] if block["type"] == "text")
        return {"previous_question": turn.request.text[:800], "previous_answer": body[:1800]}

    def _rewrite_previous(self, turn, previous, refs, topic_document=None) -> Turn:
        _, _, sources = previous
        draft = Draft(turn.session_id, sources=sources)
        for evidence_id, item in sources.items():
            draft.text(item.reads[0].text, evidence_id)
        trace = {
            "route": "generated_answer",
            "reading": {
                "mode": "previous_citations",
                "revalidated_evidence_ids": list(sources),
                "visual_read_status": "not_attempted",
            },
        }
        if topic_document:
            trace["topic_document_id"] = topic_document
        self._generate_draft(
            turn, draft, trace, "rewrite", refs=refs, context=self._prior_context(previous)
        )
        return self._complete(turn, draft, refs=refs, trace=trace)

    @staticmethod
    def _prompt_mode(mode, question, location):
        if mode == "ask" and location is None and asks_for_definition_and_experiment(question):
            return "definition_and_experiment"
        return mode

    def _generate_snapshot(
        self,
        turn,
        snapshot,
        ranked,
        trace,
        *,
        mode,
        location,
        names,
        previous,
        question,
        extra_media=(),
        selected_file_scope=False,
        preceding_reply=None,
    ):
        if names:
            allowed = {name.casefold() for name in names}
            docs = tuple(doc for doc in snapshot.documents if doc.file_name.casefold() in allowed)
            if not docs:
                return self._complete(turn, Draft(turn.session_id), trace=trace)
        else:
            docs = snapshot.documents
        identifiers = {doc.document_id for doc in docs}
        sources = tuple(
            source
            for source in snapshot.sources
            if source.document_id in identifiers
            and (
                location is None
                or (
                    (location.page is None or source.locator.get("page") == location.page)
                    and (
                        location.cell_index is None
                        or source.locator.get("cell_index") == location.cell_index
                    )
                )
            )
        )
        snapshot = replace(snapshot, sources=sources, documents=docs)
        source_keys = {(source.document_id, source.source_id) for source in sources}
        ranked = [
            source for source in ranked if (source.document_id, source.source_id) in source_keys
        ]
        preferred = [_key(source) for source in ranked] if mode == "ask" else []
        # Course-wide search keeps the lexical/adjacency scope; selected files may fit in full.
        if len(docs) > 1 and location is None:
            snapshot = replace(snapshot, sources=tuple(ranked))
        last_trace = previous[0].outcome.get("trace", {}).get("reading", {}) if previous else {}
        cursor = None
        if _CONTINUE_TEXT.fullmatch(question):
            if last_trace.get("next_cursor"):
                cursor = ReadingCursor(**last_trace["next_cursor"])
                preferred = [SourceKey.model_validate(key) for key in last_trace["preferred"]]
            else:
                draft = Draft(turn.session_id)
                draft.notice("no_reading_cursor", "本轮范围没有可续读的位置。")
                return self._complete(turn, draft, refs=[_doc_ref(doc) for doc in docs])
        context = self._prior_context(previous) if previous else None
        prompt_mode = self._prompt_mode(mode, question, location)
        unread_visual_total = sum(1 for source in snapshot.sources if _has_visual(source))
        coverage = {"unread_visual_total": unread_visual_total}
        if self._excluded_previous_experiment(turn, preceding_reply, identifiers, question):
            coverage["assess_experiment_coverage"] = True
        reserve_false = estimate_prompt_reserve(
            question,
            mode=prompt_mode,
            context=context,
            coverage={**coverage, "full_text_scope": False},
        )
        reserve_true = estimate_prompt_reserve(
            question,
            mode=prompt_mode,
            context=context,
            coverage={**coverage, "full_text_scope": True},
        )
        reserve = reserve_false
        ceiling = min(
            self.settings.answer_context_tokens,
            self.settings.workspace_input_token_limit + OUTPUT_RESERVE,
        )
        input_budget = ceiling - OUTPUT_RESERVE
        draft = Draft(turn.session_id)
        trace["route"] = "generated_answer"
        trace.setdefault("external_call_count", 0)
        try:
            if len(docs) == 1 and location is None and cursor is None:
                window = read_window(
                    self.library.store,
                    snapshot,
                    context_tokens=ceiling,
                    output_reserve=OUTPUT_RESERVE,
                    prompt_reserve=reserve_true,
                    preferred_sources=preferred,
                    cursor=cursor,
                    active=lambda: self.store.active(turn),
                )
                if window.full_text_scope:
                    reserve = reserve_true
                else:
                    window = read_window(
                        self.library.store,
                        snapshot,
                        context_tokens=ceiling,
                        output_reserve=OUTPUT_RESERVE,
                        prompt_reserve=reserve_false,
                        preferred_sources=preferred,
                        cursor=cursor,
                        active=lambda: self.store.active(turn),
                    )
            else:
                window = read_window(
                    self.library.store,
                    snapshot,
                    context_tokens=ceiling,
                    output_reserve=OUTPUT_RESERVE,
                    prompt_reserve=reserve_false,
                    preferred_sources=preferred,
                    cursor=cursor,
                    active=lambda: self.store.active(turn),
                )
        except ValueError:
            budget_reason = (
                "fixed_overhead_exceeded" if input_budget <= reserve else "no_evidence_fits"
            )
            trace["reading"] = {
                "status": "budget_exceeded",
                "budget_reason": budget_reason,
                "input_budget": max(0, input_budget),
                "prompt_reserve": reserve,
                "budget": max(0, input_budget - reserve),
                "estimated_input": 0,
                "visual_read_status": "not_attempted",
            }
            trace["generation"] = {
                "provider": "deepseek",
                "model": self.settings.deepseek_chat_model,
                "status": "budget_exceeded",
                "budget_reason": budget_reason,
                "input_budget": max(0, input_budget),
                "fixed_overhead": reserve,
                "estimated_input": reserve,
                "output_reserve": OUTPUT_RESERVE,
                "input_evidence_ids": [],
                "external_call_count": 0,
                "attempts": [],
            }
            draft.notice("reading_budget", "模型上下文预算无法容纳原文，请缩小范围或调整模型配置。")
            return self._complete(turn, draft, refs=[_doc_ref(doc) for doc in docs], trace=trace)
        full_text_scope = window.full_text_scope and len(docs) == 1 and location is None
        trace["reading"] = window.trace() | {
            "mode": mode,
            "input_budget": max(0, input_budget),
            "prompt_reserve": reserve,
            "estimated_message_input": reserve + window.estimated_input,
            "preferred": [key.model_dump() for key in preferred],
            "full_text_scope": full_text_scope,
            "location": location.model_dump(mode="json") if location else None,
        }
        for segment in window.segments:
            source = segment.source
            read = TurnRead(
                actual_source=_key(source),
                method=segment.method,
                status="succeeded",
                text=segment.text,
            )
            item = draft.add_evidence(segment.document, source, [read])
            item.locator["text_span"] = [segment.start, segment.end]
            draft.text(segment.text, item.evidence_id)
        refs = [_doc_ref(doc) for doc in docs]
        self._generate_draft(
            turn,
            draft,
            trace,
            prompt_mode,
            refs=refs,
            question=question,
            context=context,
            coverage={
                **coverage,
                "full_text_scope": full_text_scope,
                "unread_visual_total": len(window.unread_visual),
                "selected_file_scope": (
                    prompt_mode == "definition_and_experiment" and selected_file_scope
                ),
            },
        )
        selected = ranked or [segment.source for segment in window.segments]
        media_seeds = list(selected)
        seen_media = {(item.document_id, item.source_id) for item in media_seeds}
        for item in extra_media:
            key = (item.document_id, item.source_id)
            if item.document_id in identifiers and key not in seen_media:
                seen_media.add(key)
                media_seeds.append(item)
        if mode == "ask" and media_seeds:
            media = _media_candidates(media_seeds, snapshot.sources, None)
            self._attach_images(turn, draft, {doc.document_id: doc for doc in docs}, media)
        return self._complete(turn, draft, refs=refs, trace=trace)

    def _generate_draft(
        self,
        turn,
        draft,
        trace,
        mode,
        *,
        refs,
        historical=(),
        question=None,
        context=None,
        coverage=None,
    ) -> None:
        if not self.store.active(turn):
            raise WorkspaceError("interrupted")
        if not all(
            self.store.filter_accessible(refs, self.allowed_courses, current=True)
        ) or not all(self.store.filter_accessible(historical, self.allowed_courses)):
            raise WorkspaceError("stale_reference")
        result = generate_response(
            self.settings,
            question or turn.request.text,
            draft.sources,
            mode=mode,
            context=context,
            coverage=coverage,
        )
        trace["generation"] = result.trace
        trace["external_call_count"] = (
            trace.get("external_call_count", 0) + result.trace["external_call_count"]
        )
        if result.status in {"succeeded", "partial", "insufficient"}:
            if result.input_evidence:
                non_text = {
                    key: item
                    for key, item in draft.sources.items()
                    if key not in result.input_evidence
                    and not any(
                        read.method in {"native_text", "native_table", "ocr"}
                        for read in item.reads
                    )
                }
                draft.sources = {**result.input_evidence, **non_text}
            draft.blocks = [block for block in draft.blocks if block["type"] != "text"]
            for statement in result.statements:
                draft.text(statement.text, *statement.citations)
            excluded_experiment = (
                coverage
                and coverage.get("assess_experiment_coverage")
                and result.trace.get("experiment_missing") is True
                and result.missing
            )
            if result.missing:
                missing = result.missing
                if coverage and coverage.get("selected_file_scope") and not excluded_experiment:
                    missing = f"本轮仅查所选文件。{missing}"
                draft.notice("missing_evidence", missing)
            if excluded_experiment:
                draft.notice(
                    "experiment_outside_scope",
                    "本轮选定范围已改变，上一轮引用的实验依据已超出当前所选文件。"
                    "本轮能核对的内容以上述当前文件的回答和引用为准。"
                    "如需核对该实验，请在资料目录重新选择实验文件，"
                    "或切回课程全部资料后再提问。",
                )
        else:
            draft.notice("excerpt_fallback", "回答模型未能生成可核对的答案。本条保留已读原文摘录。")

    def _answer_composition(
        self, turn: Turn, session: Session, text: str, snapshot: StructureSnapshot
    ) -> Turn:
        draft = Draft(session.session_id)
        refs = {doc.document_id: _doc_ref(doc) for doc in snapshot.documents}
        if _location(text, None) is not None or _PAGE_PREFIX.search(text):
            return self._clarify(
                turn,
                "三类证据需要分别核对。请先选择要使用的文件，或单独检查指定页和 Cell。",
                [],
                list(refs.values()),
            )
        names = _object_file_names(text, snapshot.documents)
        if names is None:
            draft.notice("scope_not_found", "无法确认指定文件，请明确可使用的资料名称。")
            return self._complete(turn, draft, refs=list(refs.values()))
        if names:
            snapshot = self.snapshots.read(
                SnapshotRequest(tuple(session.course_ids), tuple(refs), tuple(names))
            )
            if any(_doc_ref(doc) != refs.get(doc.document_id) for doc in snapshot.documents):
                raise WorkspaceError("stale_reference")
            narrowed = {doc.document_id: doc for doc in snapshot.documents}
            if snapshot.resolution_status == "ambiguous_scope":
                return self._clarify(
                    turn,
                    "有多份资料符合范围，请选择一份。",
                    self._document_choices(session.session_id, narrowed),
                    [_doc_ref(doc) for doc in narrowed.values()],
                )
            if snapshot.resolution_status != "resolved":
                draft.notice("scope_not_found", "在所选范围内找不到指定的文件。")
                return self._complete(turn, draft, refs=list(refs.values()))
            refs = {key: _doc_ref(doc) for key, doc in narrowed.items()}
        run = compose_evidence(
            text,
            snapshot,
            self.searcher,
            self.library.store,
            allowed_courses=self.allowed_courses,
            active=lambda: self.store.active(turn),
        )
        draft.sources.update(run.evidence)
        for statement in run.statements:
            draft.text(statement.text, *statement.citations)
        for code, note in run.notices:
            draft.notice(code, note)
        refs = list(refs.values())
        if (
            self.settings.workspace_answer_enabled
            and run.read_evidence
            and run.trace["answer_status"] != "supported"
            and not run.trace["conflicts"]
        ):
            generated = Draft(session.session_id, sources=run.read_evidence)
            self._generate_draft(turn, generated, run.trace, "compose", refs=refs)
            status = run.trace["generation"]["status"]
            if status in {"succeeded", "partial", "insufficient"}:
                cited = {key for block in generated.blocks for key in block.get("citations", [])}
                generated.sources = {
                    key: item for key, item in generated.sources.items() if key in cited
                }
                run.trace["final_evidence_ids"] = list(generated.sources)
                draft = generated
            else:
                draft.notice(
                    "generation_unavailable", "回答模型未生成可核对的答案，已保留本地核对结果。"
                )
        return self._complete(turn, draft, refs=refs, trace=run.trace)

    def _answer_object(
        self,
        turn: Turn,
        session: Session,
        text: str,
        target: ObjectIntent,
        documents: dict[str, SnapshotDocument],
        node: CatalogNode | None,
        location: StructureLocation | None,
    ) -> Turn:
        draft = Draft(session.session_id)
        names = _object_file_names(text, documents.values())
        if names is None:
            draft.notice("scope_not_found", "无法确认指定文件，请明确可使用的资料名称。")
            return self._complete(turn, draft, refs=[_doc_ref(doc) for doc in documents.values()])
        narrowed = self.snapshots.read(
            SnapshotRequest(
                tuple(session.course_ids),
                tuple(documents),
                tuple(names),
                require_single_document=location is not None,
            )
        )
        if any(
            doc.document_id not in documents
            or _doc_ref(doc) != _doc_ref(documents[doc.document_id])
            for doc in narrowed.documents
        ):
            raise WorkspaceError("stale_reference")
        docs = {doc.document_id: doc for doc in narrowed.documents}
        refs = [_doc_ref(doc) for doc in docs.values()]
        if narrowed.resolution_status == "ambiguous_scope":
            return self._clarify(
                turn,
                "有多份资料符合范围，请选择一份。",
                self._document_choices(session.session_id, docs),
                refs,
            )
        if narrowed.resolution_status != "resolved":
            draft.notice("scope_not_found", "在所选范围内找不到指定的文件。")
            return self._complete(turn, draft, refs=refs)
        sources = [
            source
            for source in narrowed.sources
            if location is None
            or (
                (location.page is None or source.locator.get("page") == location.page)
                and (
                    location.cell_index is None
                    or source.locator.get("cell_index") == location.cell_index
                )
            )
        ]
        focus = node.source if node and (location is None or node.source in sources) else None
        region_rects = {}
        for source in sources:
            key = (source.document_id, source.version, source.source_id)
            region_id = self.region_ids.get(key)
            if region_id and (region := self.evidence.regions.get(region_id)):
                if (
                    region.coordinate_frame == "visible_page_norm_v1"
                    and region.source == _key(source)
                    and region.content_sha256 == docs[source.document_id].content_sha256
                ):
                    region_rects[key] = region.target_rect
        choices = object_choices(target, sources, focus=focus, region_rects=region_rects)
        if target.kind == "table" and len(choices) > 1:
            choices = _disambiguate_table_choices(text, choices, narrowed.sources)
        trace: dict[str, Any] = {
            "route": "target_object",
            "effective_scope": _effective(docs),
            "object_candidates": [_key(choice).model_dump() for choice in choices],
            "external_call_count": 0,
        }
        if len(choices) != 1:
            if choices and target.kind == "table":
                note = _format_table_ambiguity(choices, docs)
            elif choices:
                note = "有多个可能的对象，请指定原件位置或图序。"
            else:
                note = "找不到能核对位置的指定对象，请补充文件、页或 Cell 及对象位置。"
            draft.notice("object_ambiguous" if choices else "object_not_found", note)
            return self._complete(turn, draft, refs=refs, trace=trace)
        source = choices[0]
        trace["selected_source"] = _key(source).model_dump()
        if target.kind == "table":
            self._render_table_crop(turn, draft, docs, source)
        else:
            error = self._read_selected_image(
                turn, draft, docs[source.document_id], source, text, trace
            )
            if error:
                return self._finish_invalid_read(turn, error, trace)
        return self._complete(turn, draft, refs=refs, trace=trace)

    def _render_table_crop(
        self,
        turn: Turn,
        draft: Draft,
        docs: Mapping[str, SnapshotDocument],
        source: SnapshotSource,
    ) -> None:
        failures = self._read_text(turn, draft, docs, [source], 1)
        if failures or not draft.sources:
            draft.notice("object_unreadable", "已定位表格，未读到可核对的原生表格内容。")
            return
        item = next(iter(draft.sources.values()))
        doc = docs[source.document_id]
        # Use the native table bounds so the requested column keeps its headers and units.
        request = self._render_request(doc, source).model_copy(
            update={"registered_region_id": None}
        )
        result = self.evidence.render_current(request)
        if result.status in {"stale_reference", "not_accessible"}:
            raise WorkspaceError(result.status)
        display = result.display
        if (
            display is not None
            and display.actual_source == item.revision.source
            and display.rendition_kind == "object_crop"
        ):
            label = f"{doc.file_name} · v{doc.version} · {_label(source)}"
            draft.add_image(item, result, request, label)
        else:
            draft.notice("table_crop_unavailable", "表格裁图不可用，可从依据打开原表。")

    def _resolve_table_reference(self, owner: str, turn: Turn, session: Session) -> Turn | None:
        _, turns, messages = self.store.timeline(owner, session.session_id)
        completed = {item.turn_id: item for item in turns if item.status == "completed"}
        boundary = next(
            message["ordinal"]
            for message in messages
            if message["turn_id"] == turn.turn_id and message["role"] == "user"
        )
        table_entries: list[dict[str, Any]] = []
        for message in reversed(messages):
            if (
                message["role"] != "assistant"
                or message["turn_id"] not in completed
                or message["ordinal"] >= boundary
            ):
                continue
            sources = message["sources"]
            visible_ids = {key for block in message["blocks"] for key in block.get("citations", ())}
            for block in message["blocks"]:
                if block["type"] != "source_image":
                    continue
                key = block["evidence_id"]
                if (
                    block["display_id"] in message["visible_display_ids"]
                    and key in sources
                    and block["display_source"] == sources[key]["revision"]["source"]
                ):
                    visible_ids.add(key)
            entries = [
                item
                for key, item in sources.items()
                if key in visible_ids and _is_table_evidence(item)
            ]
            if entries and self._turn_access(completed[message["turn_id"]]):
                table_entries = entries
                break
        if not table_entries:
            if _HISTORY_IMAGE.search(turn.request.text):
                return self._clarify(
                    turn, "找不到带表格的已完成回答，请从原件目录选择表格。", [], []
                )
            return None
        unique = {}
        refs = {}
        for item in table_entries:
            rev = item["revision"]
            src = rev["source"]
            if rev["course_id"] not in session.course_ids or (
                session.document_ids and src["document_id"] not in session.document_ids
            ):
                continue
            unique[(src["document_id"], src["version"], src["source_id"])] = item
            refs[(src["document_id"], src["version"])] = {
                "course_id": rev["course_id"],
                "document_id": src["document_id"],
                "version": src["version"],
                "publication_epoch": rev["publication_epoch"],
                "content_sha256": rev["content_sha256"],
            }
        draft = Draft(session.session_id)
        if not unique:
            draft.notice("outside_current_scope", "表格在当前范围之外。请切到该文件后再追问。")
            return self._complete(turn, draft)
        if len(unique) > 1:
            labels = "、".join(
                f"{item['file_name']} 第 {item.get('locator', {}).get('page', '?')} 页的表"
                for item in unique.values()
            )
            return self._clarify(
                turn,
                f"上一轮涉及多张表格，请说明你要继续问哪一张：{labels}。",
                [],
                list(refs.values()),
                historical=True,
            )
        entry = next(iter(unique.values()))
        rev = entry["revision"]
        key = SourceKey.model_validate(rev["source"])
        ref = refs[(key.document_id, key.version)]
        snapshot = self.snapshots.read(
            SnapshotRequest(tuple(session.course_ids), (key.document_id,))
        )
        doc = next((d for d in snapshot.documents if d.document_id == key.document_id), None)
        source = next(
            (
                s
                for s in snapshot.sources
                if s.document_id == key.document_id
                and s.version == key.version
                and s.source_id == key.source_id
            ),
            None,
        )
        if (
            snapshot.resolution_status != "resolved"
            or doc is None
            or source is None
            or doc.version != key.version
            or doc.publication_epoch != rev["publication_epoch"]
            or doc.content_sha256 != rev["content_sha256"]
        ):
            draft.notice(
                "historical_only",
                "这是先前回答中的原表。可以回看旧表，或从目录另选当前版本。",
            )
            return self._complete(turn, draft, historical=[ref])
        docs = {doc.document_id: doc}
        refs = [_doc_ref(doc)]
        trace: dict[str, Any] = {
            "route": "followup_table",
            "effective_scope": _effective(docs),
            "topic_document_id": doc.document_id,
            "selected_source": _key(source).model_dump(),
            "external_call_count": 0,
        }
        if self.settings.workspace_answer_enabled and (
            _OBJECT_FACT.search(turn.request.text) or not _OBJECT_PREVIEW.search(turn.request.text)
        ):
            raw = self.library.store.read_source(_key(source), course_ids=[doc.course_id])
            if (
                not raw.is_current
                or raw.content_sha256 != doc.content_sha256
                or raw.publication_epoch != doc.publication_epoch
            ):
                raise WorkspaceError("stale_reference")
            if raw.status != "available" or raw.source is None or not raw.source.text.strip():
                draft.notice("object_unreadable", "已定位表格，未读到可核对的原生表格内容。")
            else:
                table_text = raw.source.text.strip()
                read = TurnRead(
                    actual_source=_key(source),
                    method="native_table",
                    status="succeeded",
                    text=table_text,
                )
                item = draft.add_evidence(doc, source, [read])
                item.locator["text_span"] = [0, len(table_text)]
                draft.text(table_text, item.evidence_id)
                self._generate_draft(turn, draft, trace, "ask", refs=refs)
        else:
            self._render_table_crop(turn, draft, docs, source)
        return self._complete(turn, draft, refs=refs, trace=trace)

    def _finish_invalid_read(self, turn: Turn, error: str, trace: dict[str, Any]) -> Turn:
        vision = trace["vision"]
        return self.store.finish(
            turn,
            "failed",
            {
                "error": error,
                "retryable": error == "stale_reference",
                "trace": {
                    "external_call_count": trace["external_call_count"],
                    "vision": {"elapsed_ms": vision["elapsed_ms"], "status": vision["status"]},
                },
            },
            None,
            (),
            (),
            self.allowed_courses,
        )

    def _read_selected_image(
        self,
        turn: Turn,
        draft: Draft,
        document: SnapshotDocument,
        source: SnapshotSource,
        question: str,
        trace: dict[str, Any],
    ) -> str | None:
        if not self.store.active(turn):
            raise WorkspaceError("interrupted")
        request = self._render_request(document, source)
        result = self.evidence.render_current(request)
        display = result.display
        if result.status in {"stale_reference", "not_accessible"}:
            raise WorkspaceError(result.status)
        if display is not None:
            trace["display"] = {
                "source": display.actual_source.model_dump(),
                "byte_sha256": display.byte_sha256,
                "rect": display.rect_visible_norm,
                "fallback_reason": display.fallback_reason,
            }
        if display is not None and (
            display.actual_source == request.source and display.fallback_reason is None
        ):
            started = perf_counter()
            result = self.vision.process_current(
                request.model_copy(
                    update={"read_intent": "directed_question", "question": question}
                ),
                expected_input_sha256=display.byte_sha256,
            )
            trace.update(_vision_trace(result, (perf_counter() - started) * 1000))
            if result.status in {"stale_reference", "not_accessible", "original_unavailable"}:
                return result.status
        trace["read_status"] = result.read_status
        reads = _visual_reads(result)
        item = draft.add_evidence(document, source, reads)
        draft.add_image(
            item, result, request, f"{document.file_name} · v{document.version} · {_label(source)}"
        )
        for read in reads:
            if (
                read.actual_source == request.source
                and read.status in {"succeeded", "partial"}
                and read.origin_execution == "real_provider"
                and read.text
            ):
                draft.text(read.text, item.evidence_id)
                return None
        draft.notice(
            "object_unreadable",
            "本轮没有读到可核对的指定图像。已显示的原件或回退预览可打开核对。",
        )
        return None

    def _answer_with_supplement(
        self,
        turn: Turn,
        session: Session,
        text: str,
        documents: dict[str, SnapshotDocument],
        file_names: list[str],
    ) -> Turn:
        draft = Draft(session.session_id)
        if file_names:
            selected = self.snapshots.read(
                SnapshotRequest(tuple(session.course_ids), tuple(documents), tuple(file_names))
            )
            expected = {key: _doc_ref(doc) for key, doc in documents.items()}
            if any(_doc_ref(doc) != expected.get(doc.document_id) for doc in selected.documents):
                raise WorkspaceError("stale_reference")
            narrowed = {doc.document_id: doc for doc in selected.documents}
            if selected.resolution_status == "ambiguous_scope":
                return self._clarify(
                    turn,
                    "有多份资料符合范围，请选择一份。",
                    self._document_choices(session.session_id, narrowed),
                    [_doc_ref(doc) for doc in narrowed.values()],
                )
            if selected.resolution_status != "resolved" or not narrowed:
                draft.notice("scope_not_found", "没有找到符合范围的资料。")
                return self._complete(turn, draft, refs=list(expected.values()))
            documents = narrowed
        if not documents:
            draft.notice("empty_library", "这个范围没有已发布资料。请导入文件或切换范围。")
            return self._complete(turn, draft)
        effective = _effective(documents)
        run = self.answerer.answer(
            StructureSearchRequest(
                query=text, scope=StructureScope(**effective), top_k=_SUPPLEMENT_BUDGET
            ),
            allowed_course_ids=self.allowed_courses,
            cancelled=lambda: not self.store.active(turn),
        )
        report = run.retrieval
        if report is None:
            code = {
                "cancelled": "interrupted",
                "not_accessible": "not_accessible",
                "stale_snapshot": "stale_reference",
            }.get(run.answer.stop_reason, "temporarily_unavailable")
            if code == "not_accessible":
                raise inaccessible()
            raise WorkspaceError(code)
        pins = {pin.document_id: _doc_ref(pin) for pin in report.snapshot}
        refs = {key: _doc_ref(doc) for key, doc in documents.items()}
        if pins != refs:
            raise WorkspaceError("stale_reference")
        evidence = {item.evidence_id: item for item in report.final_evidence}
        for section in run.answer.sections:
            if section.message:
                draft.notice(section.requirement, section.message)
            for statement in section.statements:
                key = statement.citation.evidence_id
                draft.sources[key] = evidence[key]
                draft.text(statement.text, key)
        trace = report.model_dump(
            mode="json",
            exclude={"question", "reads", "final_evidence", "supports", "scope_options"},
        )
        trace.update(
            route="evidence_answer",
            effective_scope=effective,
            external_call_count=0,
            answer_status=run.answer.status,
            answer_sections={
                section.requirement: section.status for section in run.answer.sections
            },
            final_evidence_ids=list(evidence),
            cited_evidence_ids=list(draft.sources),
        )
        return self._complete(turn, draft, refs=list(refs.values()), trace=trace)

    def _read_text(
        self,
        turn: Turn,
        draft: Draft,
        documents: dict[str, SnapshotDocument],
        ranked: list[SnapshotSource],
        limit: int,
    ) -> dict[tuple[str, str], str]:
        consumed = 0
        failures: dict[tuple[str, str], str] = {}
        for source in ranked:
            if not source.text.strip():
                continue
            if consumed == limit:
                break
            if not self.store.active(turn):
                raise WorkspaceError("interrupted")
            doc = documents[source.document_id]
            raw = self.library.store.read_source(_key(source), course_ids=[doc.course_id])
            if (
                not raw.is_current
                or raw.content_sha256 != doc.content_sha256
                or raw.publication_epoch != doc.publication_epoch
            ):
                raise WorkspaceError("stale_reference")
            if raw.status != "available" or raw.source is None or not raw.source.text.strip():
                failures[(source.document_id, source.source_id)] = (
                    raw.status if raw.status != "available" else "native_text_unavailable"
                )
                continue
            consumed += 1
            text = raw.source.text.strip()
            clipped = len(text) > _TEXT_LIMIT
            readable = text[:_TEXT_LIMIT] + ("…" if clipped else "")
            read = TurnRead(
                actual_source=_key(source),
                method=(
                    "native_table"
                    if raw.source.locator.get("object_type") == "table"
                    else "native_text"
                ),
                status="partial" if clipped else "succeeded",
                text=readable,
                warnings=["excerpt_truncated"] if clipped else [],
            )
            item = draft.add_evidence(doc, source, [read])
            draft.text(
                f"{doc.file_name} · v{doc.version} · {_label(source)}\n{readable}", item.evidence_id
            )
        return failures

    def _attach_images(
        self,
        turn: Turn,
        draft: Draft,
        documents: dict[str, SnapshotDocument],
        chosen: list[SnapshotSource],
    ) -> dict[str, int]:
        attempted = 0
        failed = 0
        for source in chosen[:_IMAGE_ATTEMPT_LIMIT]:
            if len(draft.images) >= _IMAGE_LIMIT:
                break
            if not self.store.active(turn):
                raise WorkspaceError("interrupted")
            doc = documents[source.document_id]
            req = self._render_request(doc, source)
            image = self.evidence.render_current(req)
            if image.status in {"stale_reference", "not_accessible"}:
                raise WorkspaceError(image.status)
            item = draft.add_evidence(doc, source, [])
            label = f"{doc.file_name} · v{doc.version} · {_label(source)}"
            previous = len(draft.images)
            draft.add_image(item, image, req, label)
            attempted += 1
            failed += int(len(draft.images) == previous)
        return {
            "eligible": len(chosen),
            "attempted": attempted,
            "failed": failed,
            "displayed": len(draft.images),
            "unattempted": len(chosen) - attempted,
            "attempt_limit": _IMAGE_ATTEMPT_LIMIT,
            "display_limit": _IMAGE_LIMIT,
        }

    def _render_request(
        self, document: SnapshotDocument, source: SnapshotSource
    ) -> EvidenceRequest:
        region_id = self.region_ids.get((source.document_id, source.version, source.source_id))
        rendition: RenditionKind
        if source.kind == "notebook_output":
            rendition = "notebook_output"
        elif source.kind == "page_object" or region_id:
            rendition = "object_crop"
        else:
            rendition = "original_page"
        return EvidenceRequest(
            source=_key(source),
            expected_content_sha256=document.content_sha256,
            expected_publication_epoch=document.publication_epoch,
            rendition=rendition,
            registered_region_id=region_id,
        )

    def _complete(
        self,
        turn: Turn,
        draft: Draft,
        *,
        refs: list[dict[str, Any]] | None = None,
        historical: list[dict[str, Any]] | None = None,
        resolution: dict[str, Any] | None = None,
        trace: dict[str, Any] | None = None,
        failed: bool = False,
    ) -> Turn:
        refs, historical = refs or [], historical or []
        outcome = {"dependencies": [*refs, *historical], "resolution": resolution, "trace": trace}
        if failed:
            outcome.update({"error": "vision_failed", "retryable": True})
        return self.store.finish(
            turn,
            "failed" if failed else "completed",
            outcome,
            draft.answer(),
            draft.images,
            refs,
            self.allowed_courses,
            historical_refs=historical,
        )

    def _clarify(
        self,
        turn: Turn,
        text: str,
        options: list[dict[str, Any]],
        refs: list[dict[str, Any]],
        *,
        historical: bool = False,
    ) -> Turn:
        draft = Draft(turn.session_id)
        draft.notice("clarification", text)
        return self.store.finish(
            turn,
            "clarification",
            {"dependencies": refs, "options": options},
            draft.answer(),
            (),
            () if historical else refs,
            self.allowed_courses,
            historical_refs=refs if historical else (),
        )

    def _document_choices(
        self, session_id: str, documents: dict[str, SnapshotDocument]
    ) -> list[dict[str, Any]]:
        return [
            {
                "option_id": uuid4().hex,
                "label": f"{doc.file_name} · {doc.course_id} · v{doc.version}",
                "kind": "document",
                "node_id": _node_id(session_id, doc, "document", []),
            }
            for doc in list(documents.values())[:8]
        ]

    def _historical_source(self, desc: dict[str, Any], key: SourceKey):
        if desc["course_id"] not in self.allowed_courses:
            raise inaccessible()
        snapshot = self.library.store.read_source(key, course_ids=[desc["course_id"]])
        if snapshot.status == "not_accessible":
            raise inaccessible()
        if snapshot.status != "available" or snapshot.content_sha256 != desc["content_sha256"]:
            raise WorkspaceError("original_unavailable", 409)
        return snapshot

    def _attachment(self, session_id: str, message_id: str, display_id: str) -> Attachment:
        item = self.store.attachment(session_id, message_id, display_id)
        if item is None:
            raise inaccessible()
        if not all(self.store.filter_accessible([_attachment_ref(item)], self.allowed_courses)):
            raise inaccessible()
        return item

    def _ordinal(self, session_id: str, attachment: Attachment, message_id: str) -> int:
        return self.store.display_ordinal(session_id, message_id, attachment.display_id)

    def _turn_access(self, turn: Turn) -> bool:
        dependencies = turn.outcome.get("dependencies", [])
        return not dependencies or all(
            self.store.filter_accessible(dependencies, self.allowed_courses)
        )

    @staticmethod
    def _project_message(message: dict[str, Any], access: bool) -> dict[str, Any]:
        if access or message["role"] == "user":
            return message
        return {
            **message,
            "blocks": [
                {
                    "type": "notice",
                    "code": "not_accessible",
                    "text": "此回答的来源已不可访问。",
                }
            ],
            "sources": {},
            "visible_display_ids": [],
        }

    @staticmethod
    def _turn_entry(turn: Turn, access: bool) -> dict[str, Any]:
        outcome = turn.outcome if access else {"error": "not_accessible"}
        options = [
            {key: item[key] for key in ("option_id", "label", "kind")}
            for item in outcome.get("options", [])
        ]
        return {
            "turn_id": turn.turn_id,
            "client_turn_id": turn.request.client_turn_id,
            "kind": turn.request.kind,
            "text": turn.request.text,
            "status": turn.status,
            "resolution": outcome.get("resolution"),
            "trace": outcome.get("trace"),
            "error": outcome.get("error"),
            "retryable": bool(outcome.get("retryable")),
            "options": options,
        }

    def _envelope(self, turn: Turn) -> dict[str, Any]:
        access = self._turn_access(turn)
        answer = self.store.answer(turn)
        entry = self._turn_entry(turn, access)
        return {
            "turn": entry,
            "message": self._project_message(answer, access) if answer else None,
            "options": entry["options"],
        }

    def _header(self, session: Session) -> dict[str, Any]:
        approved = [item for item in session.course_ids if item in self.allowed_courses]
        docs = (
            self.snapshots.read(
                SnapshotRequest(
                    tuple(approved),
                    tuple(session.document_ids),
                )
            ).documents
            if session.document_ids
            else ()
        )
        return {
            "session_id": session.session_id,
            "title": session.title,
            "scope": {"course_ids": approved, "document_ids": [doc.document_id for doc in docs]},
            "scope_revision": session.scope_revision,
        }

    def _session_revision(self, owner: str, session_id: str, revision: int) -> Session:
        session = self.store.session(owner, session_id)
        if session.scope_revision != revision:
            raise WorkspaceError("scope_changed")
        return session

    def _scope(self, courses: list[str], documents: list[str]) -> tuple[list[str], list[str]]:
        courses = list(dict.fromkeys(courses))
        documents = list(dict.fromkeys(documents))
        if not courses or not set(courses) <= self.allowed_courses:
            raise inaccessible()
        if documents:
            resolved = self.snapshots.read(SnapshotRequest(tuple(courses), tuple(documents)))
            if set(documents) != {doc.document_id for doc in resolved.documents}:
                raise inaccessible()
        return courses, documents

    def _node(self, session: Session, node_id: str, *, stale: bool = False) -> CatalogNode:
        node = self._catalog(session.session_id).nodes.get(node_id)
        if node is None:
            if stale:
                raise WorkspaceError("stale_reference")
            raise inaccessible()
        return node

    def _catalog(self, session_id: str) -> Catalog:
        course_ids = sorted(self.allowed_courses)
        snapshot = self.snapshots.read(SnapshotRequest(tuple(course_ids)))
        groups: dict[str, list[SnapshotSource]] = {}
        for source in snapshot.sources:
            groups.setdefault(source.document_id, []).append(source)
        courses: dict[str, list[dict[str, Any]]] = {course: [] for course in course_ids}
        nodes: dict[str, CatalogNode] = {}
        for doc in snapshot.documents:
            entry = _tree(session_id, doc, groups.get(doc.document_id, []), nodes)
            courses[doc.course_id].append(entry)
        return Catalog(
            [
                {"course_id": key, "label": key, "documents": value}
                for key, value in courses.items()
            ],
            nodes,
        )


def _tree(
    session_id: str,
    doc: SnapshotDocument,
    sources: list[SnapshotSource],
    nodes: dict[str, CatalogNode],
) -> dict[str, Any]:
    def entry(
        kind: str,
        index: list[str],
        label: str,
        parent: dict[str, Any] | None,
        source: SnapshotSource | None = None,
    ) -> dict[str, Any]:
        node_id = _node_id(session_id, doc, kind, index)
        value = {
            "node_id": node_id,
            "label": label,
            "kind": kind,
            "version": doc.version,
            "file_name": doc.file_name,
            "document_id": doc.document_id,
            "course_id": doc.course_id,
            "parent_id": parent["node_id"] if parent else None,
            "previous_id": None,
            "next_id": None,
            "children": [],
            "locator": _positions(source.locator) if source else {},
            "text": source.text.strip()[:800] if source else "",
            "has_media": _can_display(source) if source else False,
        }
        nodes[node_id] = CatalogNode(doc, value, source)
        if parent is not None:
            siblings = parent["children"]
            if siblings:
                value["previous_id"] = siblings[-1]["node_id"]
                siblings[-1]["next_id"] = node_id
            siblings.append(value)
        return value

    root = entry("document", [], doc.file_name, None)
    sections: dict[tuple[str, ...], dict[str, Any]] = {(): root}
    by_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        parent = by_source.get(source.parent_id) if source.parent_id else None
        if parent is None:
            raw = source.locator.get("section_path") or []
            path: tuple[str, ...] = (
                tuple(value for value in raw if isinstance(value, str))
                if isinstance(raw, list)
                else ()
            )
            for size, label in enumerate(path, 1):
                part = path[:size]
                if part not in sections:
                    sections[part] = entry("section", list(part), label, sections[part[:-1]])
            parent = sections[path]
        by_source[source.source_id] = entry(
            source.kind, [source.source_id], _label(source), parent, source
        )
    return {
        **root,
        "title": doc.title,
        "document_format": doc.document_format,
        "nodes": root["children"],
    }


def _summary_coverage(
    sources: Sequence[SnapshotSource],
    evidence: Mapping[str, TurnEvidence],
    failures: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    reads = {
        (item.revision.source.document_id, item.revision.source.source_id): item
        for item in evidence.values()
    }
    gaps: list[dict[str, Any]] = []
    boundary_count = 0
    for source in sources:
        if source.kind in _BASE_KINDS:
            boundary_count += 1
        elif not source.text.strip():
            continue
        key = (source.document_id, source.source_id)
        item = reads.get(key)
        if item and item.read_status == "succeeded":
            continue
        if item:
            reason = next(iter(item.reads[0].warnings), item.read_status)
            status = item.read_status
        elif not source.text.strip():
            status, reason = (
                "not_attempted",
                source.state if source.state != "available" else "no_native_text",
            )
        elif key in failures:
            status, reason = "failed", failures[key]
        else:
            status, reason = "not_attempted", "source_budget"
        gaps.append(
            {
                "source": _key(source).model_dump(),
                "kind": source.kind,
                "read_status": status,
                "reason": reason,
            }
        )
    return {
        "full_text_scope": bool(boundary_count) and not gaps,
        "base_source_count": boundary_count,
        "uncovered_source_count": len(gaps),
        "uncovered_sources": gaps[:12],
    }


def _location(query: str, node: CatalogNode | None) -> StructureLocation | None:
    if match := _PAGE.search(query) or _PAGE_PREFIX.search(query):
        return StructureLocation(page=int(match.group(1)))
    if match := _CELL.search(query) or _CELL_PREFIX.search(query):
        return StructureLocation(cell_index=int(match.group(1)) - 1)
    if node and node.source:
        if isinstance(page := node.source.locator.get("page"), int):
            return StructureLocation(page=page)
        if isinstance(index := node.source.locator.get("cell_index"), int):
            return StructureLocation(cell_index=index)
    return None


def _explicit_filenames(text: str) -> list[str]:
    files = []
    for first in _ONLY_FILE.finditer(text):
        files.append(first[1])
        offset = first.end()
        while following := _NEXT_FILE.match(text, offset):
            files.append(following[1])
            offset = following.end()
    return files


def _affirmed_switch(text: str) -> bool:
    for match in _SWITCH_FILE.finditer(text):
        clause = re.split(r"[，。！？；,;!?\n]", text[: match.start()])[-1]
        if not _SWITCH_DENIAL.search(clause):
            return True
    return False


def _object_file_names(text: str, documents: Iterable[SnapshotDocument]) -> list[str] | None:
    names = []
    resolved: list[tuple[int, int]] = []
    for doc in documents:
        pattern = rf"(?<![a-z0-9._-]){re.escape(doc.file_name)}(?=$|[^a-z0-9._-]|\.(?=$|\s))"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            preceding = re.split(r"[，。！？；,;!?\n]", text[: match.start()])[-1]
            if _FILE_EXCLUSION.search(preceding):
                return None
            names.append(doc.file_name)
            resolved.append(match.span())
    if any(
        not any(start <= mention.start() and mention.end() <= end for start, end in resolved)
        for mention in _FILE_MENTION.finditer(text)
    ):
        return None
    return _explicit_filenames(text) or list(dict.fromkeys(names))


def _media_candidates(
    ranked: list[SnapshotSource],
    available: Sequence[SnapshotSource],
    node: CatalogNode | None,
) -> list[SnapshotSource]:
    if not ranked:
        return []
    seed = node.source if node and node.source else ranked[0]
    children = [
        source
        for source in available
        if source.document_id == seed.document_id
        and source.parent_id == seed.source_id
        and _can_display(source)
    ]
    if seed.kind == "cell":
        return children
    if seed.kind == "page":
        return [seed, *children]
    if _can_display(seed):
        return [seed]
    return [source for source in ranked if _can_display(source)][:1]


def _can_display(source: SnapshotSource) -> bool:
    if source.kind in {"page", "page_object"}:
        return True
    if source.kind != "notebook_output":
        return False
    return any(
        isinstance(item, dict)
        and item.get("status") == "available"
        and item.get("mime_type")
        in {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
        for item in source.locator.get("representations", [])
    )


def _label(source: SnapshotSource) -> str:
    label = source.title.strip()
    if source.kind == "notebook_output":
        return f"输出 {source.locator.get('output_index', 0)}"
    if source.kind == "cell":
        return f"Cell {source.locator.get('cell_index', 0) + 1}" + (f" · {label}" if label else "")
    if source.kind == "page":
        return f"第 {source.locator.get('page', '?')} 页" + (f" · {label}" if label else "")
    if source.kind == "page_object":
        object_type = source.locator.get("object_type")
        if isinstance(object_type, str):
            kind = {
                "table": "表格",
                "chart": "图表",
                "picture": "图片",
                "image": "图片",
                "text": "文本框",
            }.get(object_type, "页内对象")
            return f"{kind} · {label}" if label else kind
        return label or "页内对象"
    return label or source.kind


def _key(source: SnapshotSource) -> SourceKey:
    return SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )


def _status(reads: list[TurnRead]):
    return reads[0].status if reads else "not_attempted"


def _visual_reads(response: EvidenceResult) -> list[TurnRead]:
    return [
        TurnRead(
            actual_source=record.actual_source,
            method="vision",
            status=record.status if record.origin_execution == "real_provider" else "not_attempted",
            text=record.text if record.origin_execution == "real_provider" else "",
            warnings=list(record.missing_or_error),
            execution=record.execution,
            origin_execution=record.origin_execution,
            actual_input_sha256=record.actual_input_sha256,
            actual_input_media_type=record.actual_input_media_type,
            actual_input_rect=record.actual_input_rect,
            model=record.model,
            prompt_version=record.prompt_version,
            question_sha256=record.question_sha256,
        )
        for record in response.reads
    ]


def _vision_trace(response: EvidenceResult, elapsed_ms: float) -> dict[str, Any]:
    return {
        "external_call_count": response.external_call_count,
        "vision": {
            "elapsed_ms": elapsed_ms,
            "status": response.read_status,
            "capability_reason": response.capability_reason,
            "reads": [
                {
                    "execution": record.execution,
                    "cache_hit": record.cache_hit,
                    "external_call": record.external_call,
                    "provider": record.provider,
                    "model": record.model,
                    "prompt_version": record.prompt_version,
                    "usage": record.usage if record.external_call else {},
                    "missing_or_error": record.missing_or_error,
                }
                for record in response.reads
            ],
        },
    }


def _image_response(result: EvidenceResult) -> tuple[bytes, str]:
    if result.status == "not_accessible":
        raise inaccessible()
    if result.status != "available" or result.display is None:
        raise WorkspaceError(result.status)
    image = base64.b64decode(result.display.data_base64, validate=True)
    if _sha(image) != result.display.byte_sha256:
        raise WorkspaceError("rendition_unavailable")
    return image, result.display.media_type


def _positions(locator: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in locator.items() if name in _POSITION_KEYS}


def _focus_entry(node: CatalogNode) -> dict[str, Any]:
    return {key: value for key, value in node.entry.items() if key != "children"}


def _in_scope(session: Session, doc: SnapshotDocument) -> bool:
    return doc.course_id in session.course_ids and (
        not session.document_ids or doc.document_id in session.document_ids
    )


def _focus_matches_display(
    focus: CatalogNode, message: dict[str, Any], block: dict[str, Any]
) -> bool:
    displayed = SourceKey.model_validate(block["display_source"])
    doc = focus.document
    if (displayed.document_id, displayed.version) != (doc.document_id, doc.version):
        return False
    if focus.source is None or displayed == focus.key():
        return True
    if focus.source.kind not in {"page", "cell"}:
        return False
    key = "page" if focus.source.kind == "page" else "cell_index"
    source = message["sources"].get(block["evidence_id"], {})
    return source.get("locator", {}).get(key) == focus.source.locator.get(key)


def _focus_ordinal_matches(
    catalog: Catalog, focus: CatalogNode, ordinal: int, block: dict[str, Any]
) -> bool:
    sources = [
        entry.source
        for entry in catalog.nodes.values()
        if entry.source and _doc_ref(entry.document) == _doc_ref(focus.document)
    ]
    choices = object_choices(
        ObjectIntent(kind="image", ordinal=ordinal, order="images"), sources, focus=focus.source
    )
    displayed = SourceKey.model_validate(block["display_source"])
    return len(choices) == 1 and _key(choices[0]) == displayed


def _doc_ref(doc) -> dict[str, Any]:
    return {
        "course_id": doc.course_id,
        "document_id": doc.document_id,
        "version": doc.version,
        "content_sha256": doc.content_sha256,
        "publication_epoch": doc.publication_epoch,
    }


def _attachment_ref(item: Attachment) -> dict[str, Any]:
    details = item.description
    return {
        "course_id": details["course_id"],
        "document_id": details["display_source"]["document_id"],
        "version": details["display_source"]["version"],
        "content_sha256": details["content_sha256"],
        "publication_epoch": details["publication_epoch"],
    }


def _effective(documents: dict[str, SnapshotDocument]) -> dict[str, Any]:
    return TurnScope(
        course_ids=sorted({doc.course_id for doc in documents.values()}),
        document_ids=sorted(documents),
    ).model_dump()


def _node_id(session_id: str, doc: SnapshotDocument, kind: str, index: list[str]) -> str:
    return "node-" + _sha(json_text([session_id, _doc_ref(doc), kind, index]).encode())[:32]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(word: str) -> int:
    numerals = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if word.isascii() and word.isdecimal() and len(word) <= 4:
        return int(word)
    if word.count("十") == 1:
        left, right = word.split("十")
        return 10 * numerals.get(left, 1) + numerals.get(right, 0)
    return numerals.get(word, 0)


_TABLE_STOPWORDS = re.compile(
    r"打开|查看|展示|显示|预览|核对|请|帮我|包含|含有|关于|里面|中的|里的|那张|这张|原表|表格|表|的"
)


def _is_table_evidence(item: Mapping[str, Any]) -> bool:
    locator = item.get("locator") or {}
    if locator.get("object_type") == "table":
        return True
    source_id = ((item.get("revision") or {}).get("source") or {}).get("source_id", "")
    if ":table:" in source_id:
        return True
    return any(read.get("method") == "native_table" for read in (item.get("reads") or ()))


def _disambiguate_table_choices(
    question: str,
    choices: list[SnapshotSource],
    all_sources: Sequence[SnapshotSource],
) -> list[SnapshotSource]:
    cleaned = _TABLE_STOPWORDS.sub(" ", question)
    latin_terms = [tok.casefold() for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,}", cleaned)]
    cjk_terms = [tok for tok in re.findall(r"[\u3400-\u9fff]{2,}", cleaned)]
    terms = [*latin_terms, *cjk_terms]
    if not terms:
        return choices
    parent_text = {
        (src.document_id, src.version, src.source_id): src.text
        for src in all_sources
        if src.kind in _BASE_KINDS
    }
    scored: list[tuple[int, SnapshotSource]] = []
    for choice in choices:
        haystack = choice.text.casefold()
        if choice.parent_id:
            haystack += "\n" + parent_text.get(
                (choice.document_id, choice.version, choice.parent_id), ""
            ).casefold()
        score = sum(1 for term in terms if term in haystack)
        scored.append((score, choice))
    best = max(score for score, _ in scored)
    if best <= 0:
        return choices
    winners = [choice for score, choice in scored if score == best]
    return winners if len(winners) == 1 else choices


def _format_table_ambiguity(
    choices: Sequence[SnapshotSource], docs: Mapping[str, SnapshotDocument]
) -> str:
    labels: list[str] = []
    for choice in choices[:4]:
        doc = docs.get(choice.document_id)
        name = doc.file_name if doc else choice.document_id
        page = choice.locator.get("page")
        loc = f"第 {page} 页" if page is not None else choice.source_id
        labels.append(f"{name} {loc}（{_label(choice)}）")
    suffix = " 等" if len(choices) > 4 else ""
    return f"找到多张可能的表格，请说明你要打开哪一张：{'、'.join(labels)}{suffix}。"
