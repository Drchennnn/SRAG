"""Consistent read-only snapshots for structure search."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.stores.local_visibility import promote_due
from app.stores.sqlite import MetadataStore


@dataclass(frozen=True)
class SnapshotRequest:
    course_ids: tuple[str, ...]
    document_ids: tuple[str, ...] = ()
    file_names: tuple[str, ...] = ()
    fts_match: str | None = None
    fts_limit: int = 0
    require_single_document: bool = False


@dataclass(frozen=True)
class SnapshotDocument:
    course_id: str
    document_id: str
    version: int
    publication_epoch: int
    content_sha256: str
    source_path: str
    file_name: str
    title: str
    document_format: str


@dataclass(frozen=True)
class SnapshotSource:
    course_id: str
    document_id: str
    version: int
    source_id: str
    parent_id: str | None
    ordinal: int
    kind: str
    title: str
    text: str
    locator: dict[str, Any]
    state: str


@dataclass(frozen=True)
class SnapshotFtsMatch:
    origin: str
    title: str
    body: str
    rank: float


@dataclass(frozen=True)
class SnapshotFtsSeed:
    document_id: str
    version: int
    source_id: str
    source_rowid: int
    rank: float
    matches: tuple[SnapshotFtsMatch, ...]


@dataclass(frozen=True)
class StructureSnapshot:
    resolution_status: Literal["resolved", "scope_not_found", "ambiguous_scope"]
    resolution_misses: tuple[str, ...]
    documents: tuple[SnapshotDocument, ...]
    sources: tuple[SnapshotSource, ...]
    fts_seeds: tuple[SnapshotFtsSeed, ...]
    name_matches: tuple[tuple[str, tuple[str, ...]], ...]
    searchable_source_count: int
    fts_scan_truncated: bool


class StructureSnapshotStore(MetadataStore):
    """Read a fixed B publication snapshot without changing B's store surface."""

    def read(self, request: SnapshotRequest) -> StructureSnapshot:
        if not request.course_ids:
            return StructureSnapshot("scope_not_found", ("empty_scope",), (), (), (), (), 0, False)
        with self._connect() as db:
            promote_due(db, _now())
            db.execute("BEGIN")
            course_marks = ",".join("?" for _ in request.course_ids)
            operands: list[object] = list(request.course_ids)
            document_sql = ""
            if request.document_ids:
                marks = ",".join("?" for _ in request.document_ids)
                document_sql = f" AND d.document_id IN ({marks})"
                operands.extend(request.document_ids)
            rows = db.execute(
                f"""
                SELECT d.course_id, d.document_id, d.current_version AS version,
                       d.publication_epoch, d.source_path, v.content_sha256,
                       v.title, v.document_format
                FROM local_documents d
                JOIN local_versions v ON v.document_id = d.document_id
                    AND v.version = d.current_version AND v.published_at IS NOT NULL
                WHERE d.deleted = 0 AND d.current_version IS NOT NULL
                    AND d.course_id IN ({course_marks}) {document_sql}
                ORDER BY d.course_id, d.document_id
                """,
                operands,
            ).fetchall()
            all_documents = tuple(_document(row) for row in rows)
            selected, name_matches = _select_names(all_documents, request.file_names)
            missing_documents = tuple(
                f"document_not_found:{document_id}"
                for document_id in request.document_ids
                if document_id not in {document.document_id for document in all_documents}
            )
            missing_names = tuple(
                f"file_not_found:{name}" for name, matches in name_matches if not matches
            )
            if missing_documents or missing_names or not selected:
                misses = (*missing_documents, *missing_names) or ("empty_scope",)
                return StructureSnapshot(
                    "scope_not_found", misses, selected, (), (), name_matches, 0, False
                )
            ambiguous = any(len(matches) > 1 for _, matches in name_matches) or (
                request.require_single_document and len(selected) > 1
            )
            if ambiguous:
                return StructureSnapshot(
                    "ambiguous_scope", (), selected, (), (), name_matches, 0, False
                )

            document_ids = tuple(document.document_id for document in selected)
            marks = ",".join("?" for _ in document_ids)
            source_rows = db.execute(
                f"""
                SELECT d.course_id, s.*
                FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                WHERE s.document_id IN ({marks})
                ORDER BY d.course_id, d.document_id, s.ordinal, s.source_id
                """,
                document_ids,
            ).fetchall()
            sources = tuple(_source(row) for row in source_rows)
            rowid_by_key = {
                (row["document_id"], row["version"], row["source_id"]): int(row["id"])
                for row in source_rows
            }
            searchable = db.execute(
                f"""
                SELECT count(DISTINCT f.source_rowid)
                FROM local_fts f
                JOIN local_sources s ON s.id = f.source_rowid
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                WHERE s.document_id IN ({marks})
                """,
                document_ids,
            ).fetchone()[0]
            seeds, scan_truncated = _fts_seeds(
                db,
                document_ids=document_ids,
                match=request.fts_match,
                limit=request.fts_limit,
                sources=sources,
                rowid_by_key=rowid_by_key,
            )
            return StructureSnapshot(
                "resolved",
                (),
                selected,
                sources,
                seeds,
                name_matches,
                searchable,
                scan_truncated,
            )

    def multi_version_document_ids(self, document_ids: Sequence[str]) -> set[str]:
        """Return document_ids that have at least two distinct published versions."""
        clean_ids = [doc_id for doc_id in dict.fromkeys(document_ids) if doc_id]
        if not clean_ids:
            return set()
        marks = ",".join("?" for _ in clean_ids)
        with self._connect() as db:
            promote_due(db, _now())
            rows = db.execute(
                f"""
                SELECT document_id
                FROM local_versions
                WHERE document_id IN ({marks}) AND published_at IS NOT NULL
                GROUP BY document_id
                HAVING max(version) >= 2 AND count(DISTINCT version) >= 2
                """,
                clean_ids,
            ).fetchall()
        return {str(row["document_id"]) for row in rows}


_QUOTED_TERM = re.compile(r'"([^"]+)"')
_CJK_PHRASE = re.compile(r"[\u3400-\u9fff]{2,}")
_CJK_MIXED_SECTION = re.compile(r"[\u3400-\u9fff]{1,2}[0-9]{1,4}[\u3400-\u9fff]{1,2}")
_CJK_GLUED_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}(?=[\u3400-\u9fff])")
_CJK_SPLIT_WORDS = re.compile(
    r"请问|请说明|请解释|请给出|请对比|请比较|有什么|怎么样|为什么|怎么|怎样|为何|"
    r"什么|哪些|哪个|多少|是否|哪里|在哪|以及|还有|并且|而且|或者|关于|针对|对于|"
    r"里面|之中|当中|的话|时候|分别|各自|主要|具体|之间|区别|差异|变化|如何|"
    r"处理|过程|机制|原理|方法|参数|步骤|性质|特点|特征|概念|作用|意义|关系|情况|效果|"
    r"[请和与或的了是在有中里内吗呢啊吧]"
)
_CJK_STOP_TERMS = frozenset(
    {
        "课件", "讲义", "论文", "文档", "资料", "笔记", "实验", "结果", "内容",
        "问题", "方法", "说明", "介绍", "解释", "定义", "公式", "例子", "示例",
    }
)


def _cjk_query_phrases(match: str | None) -> list[str]:
    if not match:
        return []
    phrases: list[str] = []
    seen: set[str] = set()
    for token in _QUOTED_TERM.findall(match):
        for mixed in _CJK_MIXED_SECTION.findall(token):
            if mixed not in seen:
                seen.add(mixed)
                phrases.append(mixed)
        for glued in _CJK_GLUED_ASCII.findall(token):
            low = glued.casefold()
            if low not in seen:
                seen.add(low)
                phrases.append(low)
        for run in _CJK_PHRASE.findall(token):
            chunks = [c.strip() for c in _CJK_SPLIT_WORDS.split(run) if len(c.strip()) >= 2]
            for chunk in chunks:
                if chunk in _CJK_STOP_TERMS:
                    continue
                if chunk not in seen:
                    seen.add(chunk)
                    phrases.append(chunk)
            if not chunks and run not in _CJK_STOP_TERMS and len(run) >= 2 and run not in seen:
                seen.add(run)
                phrases.append(run)
    return phrases


def _ascii_query_terms(match: str | None) -> list[str]:
    if not match:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in _QUOTED_TERM.findall(match):
        low = token.casefold().strip()
        if low.isascii() and len(low) >= 2 and low not in seen:
            seen.add(low)
            out.append(low)
    return out


def _cjk_substring_seeds(
    *,
    sources: Sequence[SnapshotSource],
    rowid_by_key: dict[tuple[str, int, str], int],
    match: str | None,
    limit: int,
    existing_seeds: tuple[SnapshotFtsSeed, ...],
) -> list[SnapshotFtsSeed]:
    phrases = _cjk_query_phrases(match)
    if not phrases or limit <= 0:
        return []
    ascii_terms = _ascii_query_terms(match)
    bigrams: dict[str, list[str]] = {}
    for phrase in phrases:
        if len(phrase) >= 4 and not phrase.isascii():
            grams = list(
                dict.fromkeys(
                    phrase[i : i + 2]
                    for i in range(len(phrase) - 1)
                    if phrase[i : i + 2] not in _CJK_STOP_TERMS
                )
            )
            if grams:
                bigrams[phrase] = grams

    corpus_texts = [
        (src, f"{src.title}\n{src.text}", f"{src.title}\n{src.text}".casefold())
        for src in sources
        if src.text.strip() or src.title.strip()
    ]
    n_docs = max(1, len(corpus_texts))
    ascii_pats = {
        p: re.compile(rf"(?<![0-9a-z]){re.escape(p)}(?![0-9a-z])")
        for p in phrases
        if p.isascii()
    }

    def _has_phrase(phrase: str, hay: str, hay_low: str) -> bool:
        if pat := ascii_pats.get(phrase):
            return pat.search(hay_low) is not None
        return phrase in hay

    phrase_idf = {
        p: (
            math.log(
                (n_docs + 1.0)
                / (sum(1 for _, hay, hay_low in corpus_texts if _has_phrase(p, hay, hay_low)) + 1.0)
            )
            + 0.5
        )
        for p in phrases
    }

    existing_by_rowid = {seed.source_rowid: seed for seed in existing_seeds}
    new_scored: list[tuple[float, SnapshotSource, int]] = []
    for src, hay, hay_low in corpus_texts:
        key = (src.document_id, src.version, src.source_id)
        rowid = rowid_by_key.get(key)
        if rowid is None or rowid in existing_by_rowid:
            continue
        score = 0.0
        exact_cjk_hits = 0
        for phrase in phrases:
            if _has_phrase(phrase, hay, hay_low):
                score += (3.0 + 0.4 * min(len(phrase), 8)) * phrase_idf[phrase]
                exact_cjk_hits += 1
            elif phrase in bigrams:
                bg_hits = sum(1 for g in bigrams[phrase] if g in hay)
                if bg_hits >= 2:
                    score += 1.5 * bg_hits
        if exact_cjk_hits == 0 and score < 3.0:
            continue
        if ascii_terms:
            score += 2.0 * sum(1 for tok in ascii_terms if tok in hay_low)
        new_scored.append((score, src, rowid))

    if not new_scored:
        return []
    new_scored.sort(
        key=lambda item: (-item[0], item[1].document_id, item[1].ordinal, item[1].source_id)
    )
    out: list[SnapshotFtsSeed] = []
    for score, src, rowid in new_scored[:limit]:
        origin = str(src.locator.get("text_origin") or "native_text")
        rank_val = -round(score, 6)
        out.append(
            SnapshotFtsSeed(
                document_id=src.document_id,
                version=src.version,
                source_id=src.source_id,
                source_rowid=rowid,
                rank=rank_val,
                matches=(SnapshotFtsMatch(origin, src.title, src.text, rank_val),),
            )
        )
    return out


def _fts_seeds(
    db,
    *,
    document_ids: tuple[str, ...],
    match: str | None,
    limit: int,
    sources: Sequence[SnapshotSource] = (),
    rowid_by_key: dict[tuple[str, int, str], int] | None = None,
):
    if not match or limit <= 0:
        return (), False
    marks = ",".join("?" for _ in document_ids)
    page_size = max(32, limit * 2)
    scan_limit = max(64, limit * 8)
    selected: dict[int, tuple[float, str, int, str]] = {}
    offset = 0
    exhausted = False
    while len(selected) < limit and offset < scan_limit:
        rows = db.execute(
            f"""
            SELECT f.source_rowid, s.document_id, s.version, s.source_id,
                   bm25(local_fts) AS rank
            FROM local_fts f
            JOIN local_sources s ON s.id = f.source_rowid
            JOIN local_documents d ON d.document_id = s.document_id
                AND d.current_version = s.version AND d.deleted = 0
            WHERE local_fts MATCH ? AND s.document_id IN ({marks})
            ORDER BY rank, s.document_id, s.ordinal, f.rowid
            LIMIT ? OFFSET ?
            """,
            [match, *document_ids, page_size, offset],
        ).fetchall()
        if not rows:
            exhausted = True
            break
        for row in rows:
            selected.setdefault(
                row["source_rowid"],
                (row["rank"], row["document_id"], row["version"], row["source_id"]),
            )
            if len(selected) == limit:
                break
        offset += len(rows)
        if len(rows) < page_size:
            exhausted = True
            break
    if not selected:
        cjk_seeds = (
            _cjk_substring_seeds(
                sources=sources,
                rowid_by_key=rowid_by_key or {},
                match=match,
                limit=limit,
                existing_seeds=(),
            )
            if sources and rowid_by_key
            else []
        )
        return tuple(cjk_seeds[:limit]), False

    rowids = tuple(selected)
    row_marks = ",".join("?" for _ in rowids)
    match_rows = db.execute(
        f"""
        SELECT f.source_rowid, f.origin, f.title, f.body, bm25(local_fts) AS rank
        FROM local_fts f
        WHERE local_fts MATCH ? AND f.source_rowid IN ({row_marks})
        ORDER BY rank, f.rowid
        """,
        [match, *rowids],
    ).fetchall()
    matches: dict[int, list[SnapshotFtsMatch]] = {rowid: [] for rowid in rowids}
    for row in match_rows:
        matches[row["source_rowid"]].append(
            SnapshotFtsMatch(row["origin"], row["title"], row["body"], row["rank"])
        )
    seeds = tuple(
        SnapshotFtsSeed(
            document_id=document_id,
            version=version,
            source_id=source_id,
            source_rowid=rowid,
            rank=rank,
            matches=tuple(matches[rowid]),
        )
        for rowid, (rank, document_id, version, source_id) in selected.items()
    )
    truncated = not exhausted and offset >= scan_limit and len(seeds) < limit
    if sources and rowid_by_key:
        cjk_seeds = _cjk_substring_seeds(
            sources=sources,
            rowid_by_key=rowid_by_key,
            match=match,
            limit=limit,
            existing_seeds=seeds,
        )
        if cjk_seeds:
            if len(seeds) < limit or limit < 8:
                combined = (*seeds, *cjk_seeds)[:limit]
            else:
                reserve = min(len(cjk_seeds), max(2, limit // 4))
                combined = (*seeds[: limit - reserve], *cjk_seeds[:reserve])
            return combined, truncated
    return seeds, truncated


def _select_names(
    documents: tuple[SnapshotDocument, ...], file_names: tuple[str, ...]
) -> tuple[tuple[SnapshotDocument, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    if not file_names:
        return documents, ()
    normalized = tuple(
        dict.fromkeys(_normalize_name(Path(name).name) for name in file_names if name.strip())
    )
    mapping = []
    selected_ids: set[str] = set()
    for name in normalized:
        matches = tuple(
            document.document_id
            for document in documents
            if _normalize_name(document.file_name) == name
        )
        mapping.append((name, matches))
        selected_ids.update(matches)
    return tuple(d for d in documents if d.document_id in selected_ids), tuple(mapping)


def _document(row) -> SnapshotDocument:
    source_path = row["source_path"]
    return SnapshotDocument(
        course_id=row["course_id"],
        document_id=row["document_id"],
        version=row["version"],
        publication_epoch=row["publication_epoch"],
        content_sha256=row["content_sha256"],
        source_path=source_path,
        file_name=Path(source_path).name,
        title=row["title"],
        document_format=row["document_format"],
    )


def _source(row) -> SnapshotSource:
    return SnapshotSource(
        course_id=row["course_id"],
        document_id=row["document_id"],
        version=row["version"],
        source_id=row["source_id"],
        parent_id=row["parent_id"],
        ordinal=row["ordinal"],
        kind=row["kind"],
        title=row["title"],
        text=row["text"],
        locator=json.loads(row["locator_json"]),
        state=row["state"],
    )


def _normalize_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _now() -> float:
    from datetime import UTC, datetime

    return datetime.now(UTC).timestamp()
