"""Persistent local study sessions with versioned display and reading evidence."""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
from app.rag.workspace_objects import ObjectIntent, object_choices, object_intent
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
    StructureSnapshotStore,
)
from app.stores.workspace import Attachment, Session, Turn, WorkspaceStore, json_text

_ORDINAL = re.compile(r"第\s*([0-9]+|[一二两三四五六七八九十]+)\s*张(?:图|图片)")
_IMAGE_REFERENCE = re.compile(r"(?:刚才|上一条|上面|那|这|哪)\s*(?:的\s*)?(?:一?张\s*)?(?:图|图片)")
_HISTORY_IMAGE = re.compile(r"刚才|上一条|前一条")
_BARE_REFERENCE = re.compile(r"^(?:刚才\s*)?第\s*\S+\s*张(?:图|图片)[。！？?\s]*$")
_PAGE = re.compile(r"第?\s*([1-9][0-9]{0,3})\s*(?:页|张幻灯片)")
_CELL = re.compile(r"第?\s*([1-9][0-9]{0,3})\s*(?:个)?\s*cell", re.IGNORECASE)
_CELL_PREFIX = re.compile(r"\bcell\s+([1-9][0-9]{0,3})\b", re.IGNORECASE)
_FILE_MENTION = re.compile(r"\.(?:pdf|pptx|ipynb)(?=$|[^a-z0-9_])", re.IGNORECASE)
_FILE_EXCLUSION = re.compile(
    r"(?:不要|请勿|禁止|不用|不许|无需|不得|(?<!\w)别)(?:再|去)?\s*"
    r"(?:使用|用|查|看|读取|读|打开|引用|选择|选)"
    r"|\b(?:do not|don['’]t|must not|never)\s+(?:use|read|open|select|cite)\b",
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
                and item.method in {"native_text", "native_table"}
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
        excerpt = read.text
        if "excerpt_truncated" in read.warnings:
            excerpt = excerpt.removesuffix("…")
        if source is None or not (
            source.text.startswith(excerpt) or source.text.strip().startswith(excerpt)
        ):
            raise WorkspaceError("original_unavailable", 409)
        if not self._turn_access(turn):
            raise inaccessible()
        return {
            "source": key.model_dump(),
            "file_name": evidence.file_name,
            "locator": _positions(source.locator),
            "text": read.text,
            "is_current": current.is_current,
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
            ordinal = _ORDINAL.search(turn.request.text)
            located = _location(turn.request.text, None) is not None or _FILE_MENTION.search(
                turn.request.text
            )
            image_reference = ordinal or _IMAGE_REFERENCE.search(turn.request.text)
            if (
                turn.request.kind == "ask"
                and image_reference
                and (not located or _HISTORY_IMAGE.search(turn.request.text))
            ):
                return self._resolve_reference(owner, turn, session, ordinal)
            return self._retrieve(turn, session, turn.request.text)
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
            return self._clarify(turn, "找不到带图片的已完成回答，请从原件卡片选择。", [], [])
        ids = previous["visible_display_ids"]
        index = _number(match.group(1)) if match else (1 if len(ids) == 1 else 0)
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
    ) -> Turn:
        mode = kind or turn.request.kind
        node = forced
        if node is None and session.focus_id and mode == "ask":
            node = self._node(session, session.focus_id, stale=True)
        if not set(session.course_ids) <= self.allowed_courses:
            raise inaccessible()
        doc_ids = (node.document.document_id,) if node else tuple(session.document_ids)
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
        if mode == "summary" and len(documents) != 1:
            options = self._document_choices(session.session_id, documents)
            return self._clarify(turn, "请选择一份资料进行总结。", options, refs)
        sources = {(source.document_id, source.source_id): source for source in snapshot.sources}
        explicit_location = _location(text, None)
        location = explicit_location or _location("", node)
        file_names = [
            doc.file_name
            for doc in documents.values()
            if doc.file_name.casefold() in text.casefold()
        ]
        if mode == "ask" and location is None and asks_for_definition_and_experiment(text):
            file_names = _explicit_filenames(text) or file_names
            return self._answer_with_supplement(turn, session, text, documents, file_names)
        if mode == "ask" and (target := object_intent(text)):
            return self._answer_object(turn, session, text, target, documents, node, location)
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
            search = self.searcher.search(
                StructureSearchRequest(
                    query=text,
                    location=location,
                    top_k=12,
                    scope=StructureScope(
                        course_ids=session.course_ids,
                        document_ids=list(documents),
                        file_names=[] if node else file_names,
                    ),
                ),
                allowed_course_ids=self.allowed_courses,
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
            ranked = [
                sources[(hit.source.document_id, hit.source.source_id)]
                for hit in search.candidates
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
                    for hit in search.candidates
                ],
            }
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
        return self._complete(turn, draft, refs=refs, trace=trace)

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
        trace: dict[str, Any] = {
            "route": "target_object",
            "effective_scope": _effective(docs),
            "object_candidates": [_key(choice).model_dump() for choice in choices],
            "external_call_count": 0,
        }
        if len(choices) != 1:
            note = (
                "有多个可能的对象，请指定原件位置或图序。"
                if choices
                else "找不到能核对位置的指定对象，请补充文件、页或 Cell 及对象位置。"
            )
            draft.notice("object_ambiguous" if choices else "object_not_found", note)
            return self._complete(turn, draft, refs=refs, trace=trace)
        source = choices[0]
        trace["selected_source"] = _key(source).model_dump()
        if target.kind == "table":
            failures = self._read_text(turn, draft, docs, choices, 1)
            if failures or not draft.sources:
                draft.notice("object_unreadable", "已定位表格，未读到可核对的原生表格内容。")
        else:
            error = self._read_selected_image(
                turn, draft, docs[source.document_id], source, text, trace
            )
            if error:
                return self._finish_invalid_read(turn, error, trace)
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
    if match := _PAGE.search(query):
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
