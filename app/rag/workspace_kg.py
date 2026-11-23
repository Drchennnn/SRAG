"""Workspace KG navigation DTOs, permission-filtered queries, and bounded 1-hop expansion."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.rag.structure_search import (
    StructureCandidate,
    _structural_candidate,
)
from app.rag.workspace_evidence import readable_source_text
from app.schemas.local import SourceKey
from app.schemas.workspace import inaccessible
from app.stores.local_kg import KgConceptSummary, KgEdgeRecord, LocalKgStore
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource, StructureSnapshot

if TYPE_CHECKING:
    from app.rag.workspace import Catalog, CatalogNode
    from app.stores.workspace import Session

_DISPLAYABLE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
)


@dataclass(frozen=True)
class KgExpansionResult:
    pool: list[StructureCandidate]
    visual_candidates: list[StructureCandidate]
    trace: dict[str, Any]


def _can_display_source(source: SnapshotSource) -> bool:
    if source.kind in {"page", "page_object"}:
        return True
    if source.kind != "notebook_output":
        return False
    return any(
        isinstance(item, dict)
        and item.get("status") == "available"
        and item.get("mime_type") in _DISPLAYABLE_MIMES
        for item in source.locator.get("representations", [])
    )


def _node_media_url(session_id: str, node: CatalogNode | None) -> str | None:
    if node is None or not node.entry.get("has_media"):
        return None
    node_id = node.entry["node_id"]
    return f"/api/workspace/sessions/{session_id}/catalog/{node_id}/media"


def _catalog_nodes_by_source(
    catalog: Catalog, allowed_courses: frozenset[str]
) -> dict[tuple[str, int, str], CatalogNode]:
    mapping: dict[tuple[str, int, str], CatalogNode] = {}
    for node in catalog.nodes.values():
        if node.source is None:
            continue
        if node.document.course_id not in allowed_courses:
            continue
        key = (node.source.document_id, node.source.version, node.source.source_id)
        mapping[key] = node
    return mapping


def _edge_matches_snapshot(
    edge: KgEdgeRecord,
    docs_by_id: Mapping[str, SnapshotDocument],
    sources_by_key: Mapping[tuple[str, int, str], SnapshotSource],
) -> bool:
    doc = docs_by_id.get(edge.document_id)
    if doc is None or doc.course_id != edge.course_id:
        return False
    if (
        doc.version != edge.version
        or doc.publication_epoch != edge.publication_epoch
        or doc.content_sha256 != edge.content_sha256
    ):
        return False
    return (edge.document_id, edge.version, edge.source_id) in sources_by_key


def list_workspace_concepts(
    kg_store: LocalKgStore,
    session: Session,
    snapshot: StructureSnapshot,
    *,
    effective_courses: tuple[str, ...],
    selected_course_id: str | None = None,
    query: str | None = None,
    respect_document_scope: bool = False,
) -> dict[str, Any]:
    """List verified course concepts strictly within session effective courses."""
    if not effective_courses:
        raise inaccessible()
    target_courses = effective_courses
    if selected_course_id is not None:
        if selected_course_id not in effective_courses:
            raise inaccessible()
        target_courses = (selected_course_id,)

    doc_filter = tuple(session.document_ids) if respect_document_scope else ()
    concepts, edges = kg_store.active_edges_for_courses(target_courses)
    docs_by_id = {
        doc.document_id: doc
        for doc in snapshot.documents
        if doc.course_id in target_courses
    }
    sources_by_key = {
        (src.document_id, src.version, src.source_id): src
        for src in snapshot.sources
        if src.document_id in docs_by_id
    }
    doc_filter_set = set(doc_filter)
    counts: dict[tuple[str, str], list[int]] = {}
    for edge in edges:
        if doc_filter_set and edge.document_id not in doc_filter_set:
            continue
        if not _edge_matches_snapshot(edge, docs_by_id, sources_by_key):
            continue
        pair = counts.setdefault((edge.course_id, edge.concept_key), [0, 0])
        pair[0 if edge.relation_type == "defined_in" else 1] += 1

    needle = query.strip().casefold() if query else ""
    items: list[dict[str, Any]] = []
    for concept in concepts:
        pair = counts.get((concept.course_id, concept.concept_key))
        if not pair or (pair[0] == 0 and pair[1] == 0):
            continue
        if needle:
            haystack = " ".join(
                (concept.canonical_name, *concept.aliases, concept.concept_key)
            ).casefold()
            if needle not in haystack:
                continue
        def_count, demo_count = pair
        items.append(
            {
                "course_id": concept.course_id,
                "concept_key": concept.concept_key,
                "canonical_name": concept.canonical_name,
                "aliases": list(concept.aliases),
                "defined_in_count": def_count,
                "demonstrated_by_count": demo_count,
                "completeness": "complete" if def_count > 0 and demo_count > 0 else "partial",
                "has_reliable_links": True,
            }
        )
    return {
        "scope_revision": session.scope_revision,
        "course_ids": list(target_courses),
        "concepts": items,
    }


def get_workspace_concept_detail(
    kg_store: LocalKgStore,
    session: Session,
    snapshot: StructureSnapshot,
    catalog: Catalog,
    *,
    effective_courses: tuple[str, ...],
    concept_key: str,
    selected_course_id: str | None = None,
) -> dict[str, Any]:
    """Return grouped lecture/experiment edges & child outputs for one authorized concept."""
    if not effective_courses:
        raise inaccessible()
    target_courses = effective_courses
    if selected_course_id is not None:
        if selected_course_id not in effective_courses:
            raise inaccessible()
        target_courses = (selected_course_id,)

    detail = kg_store.concept_detail(target_courses, concept_key)
    if detail is None:
        raise inaccessible()
    summary, edges = detail
    if summary.course_id not in target_courses:
        raise inaccessible()

    course_set = frozenset((summary.course_id,))
    docs_by_id = {
        doc.document_id: doc
        for doc in snapshot.documents
        if doc.course_id == summary.course_id
    }
    sources_by_key = {
        (src.document_id, src.version, src.source_id): src
        for src in snapshot.sources
        if src.document_id in docs_by_id
    }
    nodes_by_source = _catalog_nodes_by_source(catalog, course_set)
    outputs_by_cell: dict[tuple[str, int, str], list[SnapshotSource]] = {}
    for src in sources_by_key.values():
        if src.kind == "notebook_output" and src.parent_id:
            outputs_by_cell.setdefault(
                (src.document_id, src.version, src.parent_id), []
            ).append(src)
    for children in outputs_by_cell.values():
        children.sort(
            key=lambda item: (int(item.locator.get("output_index", item.ordinal)), item.ordinal)
        )

    selected_doc_set = set(session.document_ids)
    defined_in: list[dict[str, Any]] = []
    demonstrated_by: list[dict[str, Any]] = []

    for edge in edges:
        if not _edge_matches_snapshot(edge, docs_by_id, sources_by_key):
            continue
        doc = docs_by_id[edge.document_id]
        src = sources_by_key[(edge.document_id, edge.version, edge.source_id)]
        node = nodes_by_source.get((edge.document_id, edge.version, edge.source_id))
        in_selected = (not selected_doc_set) or (edge.document_id in selected_doc_set)
        entry: dict[str, Any] = {
            "edge_id": edge.edge_id,
            "document_id": edge.document_id,
            "file_name": doc.file_name,
            "version": edge.version,
            "source_id": edge.source_id,
            "kind": src.kind,
            "node_id": node.entry["node_id"] if node else None,
            "locator": node.entry["locator"] if node else src.locator,
            "evidence_span": [edge.evidence_span_start, edge.evidence_span_end],
            "evidence_excerpt": edge.evidence_excerpt,
            "has_media": bool(node and node.entry.get("has_media")),
            "media_url": _node_media_url(session.session_id, node),
            "in_selected_documents": in_selected,
        }
        if edge.relation_type == "defined_in":
            defined_in.append(entry)
        else:
            child_outputs: list[dict[str, Any]] = []
            if src.kind == "cell":
                for child in outputs_by_cell.get(
                    (edge.document_id, edge.version, edge.source_id), ()
                ):
                    child_node = nodes_by_source.get(
                        (child.document_id, child.version, child.source_id)
                    )
                    child_outputs.append(
                        {
                            "source_id": child.source_id,
                            "node_id": child_node.entry["node_id"] if child_node else None,
                            "output_index": int(child.locator.get("output_index", 0)),
                            "text_excerpt": readable_source_text(child).strip()[:300],
                            "has_media": bool(child_node and child_node.entry.get("has_media")),
                            "media_url": _node_media_url(session.session_id, child_node),
                        }
                    )
            entry["child_outputs"] = child_outputs
            demonstrated_by.append(entry)

    if not defined_in and not demonstrated_by:
        raise inaccessible()

    completeness = "complete" if (defined_in and demonstrated_by) else "partial"
    fallback_to_search = completeness != "complete"
    if not fallback_to_search:
        notice = ""
    elif defined_in:
        notice = "当前版本暂无可靠的实验演示关系，可使用常规检索查找相关实验。"
    else:
        notice = "当前版本暂无可靠的讲义定义关系，可使用常规检索查找相关定义。"

    return {
        "course_id": summary.course_id,
        "concept_key": summary.concept_key,
        "canonical_name": summary.canonical_name,
        "aliases": list(summary.aliases),
        "completeness": completeness,
        "has_reliable_relations": True,
        "fallback_to_search": fallback_to_search,
        "fallback_query": summary.canonical_name,
        "notice": notice,
        "defined_in": defined_in,
        "demonstrated_by": demonstrated_by,
    }


def project_answer_concepts(
    kg_store: LocalKgStore,
    snapshot: StructureSnapshot,
    *,
    effective_courses: tuple[str, ...],
    cited_sources: Sequence[SourceKey],
) -> list[dict[str, Any]]:
    """Find active accepted concepts anchored on cited sources (strictly current snapshot)."""
    if not effective_courses or not cited_sources:
        return []
    concepts, edges = kg_store.active_edges_for_courses(effective_courses)
    if not edges:
        return []
    docs_by_id = {
        doc.document_id: doc
        for doc in snapshot.documents
        if doc.course_id in effective_courses
    }
    sources_by_key = {
        (src.document_id, src.version, src.source_id): src
        for src in snapshot.sources
        if src.document_id in docs_by_id
    }
    valid_edges = [
        edge for edge in edges if _edge_matches_snapshot(edge, docs_by_id, sources_by_key)
    ]
    if not valid_edges:
        return []

    target_keys: set[tuple[str, int, str]] = set()
    for key in cited_sources:
        triple = (key.document_id, key.version, key.source_id)
        target_keys.add(triple)
        src = sources_by_key.get(triple)
        if src is not None and src.parent_id:
            target_keys.add((src.document_id, src.version, src.parent_id))

    concept_map = {(c.course_id, c.concept_key): c for c in concepts}
    edges_by_concept: dict[tuple[str, str], list[KgEdgeRecord]] = {}
    for edge in valid_edges:
        edges_by_concept.setdefault((edge.course_id, edge.concept_key), []).append(edge)

    matched: dict[tuple[str, str], set[str]] = {}
    order: list[tuple[str, str]] = []
    for edge in valid_edges:
        if (edge.document_id, edge.version, edge.source_id) not in target_keys:
            continue
        ck = (edge.course_id, edge.concept_key)
        if ck not in matched:
            matched[ck] = set()
            order.append(ck)
        matched[ck].add(edge.relation_type)

    result: list[dict[str, Any]] = []
    for ck in order[:6]:
        summary = concept_map.get(ck)
        if summary is None:
            continue
        all_rels = {e.relation_type for e in edges_by_concept.get(ck, ())}
        result.append(
            {
                "course_id": summary.course_id,
                "concept_key": summary.concept_key,
                "canonical_name": summary.canonical_name,
                "matched_relations": sorted(matched[ck]),
                "has_counterpart": bool(all_rels - matched[ck]) or len(all_rels) >= 2,
            }
        )
    return result


def _query_matches_concept(text: str, concept: KgConceptSummary) -> bool:
    terms = sorted({concept.canonical_name, *concept.aliases}, key=len, reverse=True)
    for term in terms:
        if len(term) < 2:
            continue
        if re.search(r"[\u3400-\u9fff]", term):
            if re.search(re.escape(term), text, re.IGNORECASE):
                return True
        else:
            if re.search(rf"(?<![\w.-]){re.escape(term)}(?![\w.-])", text, re.IGNORECASE):
                return True
    return False


def expand_kg_one_hop(
    kg_store: LocalKgStore,
    *,
    course_ids: Sequence[str],
    docs: Mapping[str, SnapshotDocument],
    sources: Mapping[tuple[str, int, str], SnapshotSource],
    question: str,
    alternate_query: str | None,
    pool: Sequence[StructureCandidate],
    budget: int,
    verify: Callable[[], None] = lambda: None,
) -> KgExpansionResult:
    """Bounded 1-hop KG candidate expansion with per-concept quota & child_output co-ranking."""
    trace: dict[str, Any] = {
        "status": "not_requested",
        "external_call_count": 0,
        "matched_concepts": [],
        "added_sources": [],
        "filtered_out_of_scope": 0,
    }
    if budget < 2 or not course_ids or not docs:
        trace["status"] = "insufficient_budget"
        return KgExpansionResult(list(pool), [], trace)

    verify()
    concepts, edges = kg_store.active_edges_for_courses(course_ids)
    verify()
    if not concepts or not edges:
        trace["status"] = "no_reliable_edges"
        return KgExpansionResult(list(pool), [], trace)

    # Filter edges strictly to current scope.docs & verified snapshot versions/epochs.
    in_scope_edges: list[KgEdgeRecord] = []
    filtered_out = 0
    for edge in edges:
        if edge.document_id not in docs:
            if edge.course_id in course_ids:
                filtered_out += 1
            continue
        if _edge_matches_snapshot(edge, docs, sources):
            in_scope_edges.append(edge)
        else:
            filtered_out += 1
    trace["filtered_out_of_scope"] = filtered_out

    if not in_scope_edges:
        trace["status"] = "scope_excluded" if filtered_out else "no_reliable_edges"
        return KgExpansionResult(list(pool), [], trace)

    edges_by_concept: dict[tuple[str, str], list[KgEdgeRecord]] = {}
    edges_by_source: dict[tuple[str, int, str], list[KgEdgeRecord]] = {}
    for edge in in_scope_edges:
        edges_by_concept.setdefault((edge.course_id, edge.concept_key), []).append(edge)
        edges_by_source.setdefault(
            (edge.document_id, edge.version, edge.source_id), []
        ).append(edge)

    # Step 1: Anchor concepts from (B) question/alternate match & (A) top-6 pool hits.
    query_blob = "\n".join(filter(None, (question, alternate_query)))
    concept_scores: dict[tuple[str, str], tuple[int, int, str]] = {}
    concept_map = {(c.course_id, c.concept_key): c for c in concepts}

    for ck in edges_by_concept:
        concept = concept_map.get(ck)
        if concept is None:
            continue
        if _query_matches_concept(query_blob, concept):
            # Priority 0: explicit question concept match.
            concept_scores[ck] = (0, 0, ck[1])

    for rank, cand in enumerate(pool[: min(6, len(pool))]):
        if cand.match_kind == "browse":
            continue
        key = (cand.source.document_id, cand.source.version, cand.source.source_id)
        src = sources.get(key)
        lookup_keys = [key]
        if src is not None and src.parent_id:
            lookup_keys.append((src.document_id, src.version, src.parent_id))
        for lk in lookup_keys:
            for edge in edges_by_source.get(lk, ()):
                ck = (edge.course_id, edge.concept_key)
                if ck not in concept_scores:
                    concept_scores[ck] = (1, rank, ck[1])

    if not concept_scores:
        trace["status"] = "no_concept_match"
        return KgExpansionResult(list(pool), [], trace)

    chosen_keys = sorted(concept_scores, key=lambda ck: concept_scores[ck])[:3]
    trace["matched_concepts"] = [
        {
            "course_id": ck[0],
            "concept_key": ck[1],
            "canonical_name": concept_map[ck].canonical_name,
        }
        for ck in chosen_keys
        if ck in concept_map
    ]

    # Build child_output lookup for notebook cells.
    outputs_by_cell: dict[tuple[str, int, str], list[SnapshotSource]] = {}
    for src in sources.values():
        if src.kind == "notebook_output" and src.parent_id:
            outputs_by_cell.setdefault(
                (src.document_id, src.version, src.parent_id), []
            ).append(src)
    for children in outputs_by_cell.values():
        children.sort(
            key=lambda item: (int(item.locator.get("output_index", item.ordinal)), item.ordinal)
        )

    # Determine protected S0 base anchors.
    non_browse = [item for item in pool if item.match_kind != "browse"]
    base_pool = non_browse or list(pool)
    protected_count = min(2, max(1, budget // 3)) if base_pool else 0
    protected = list(base_pool[:protected_count])
    protected_ids = {
        (item.source.document_id, item.source.version, item.source.source_id)
        for item in protected
    }
    in_budget_ids = {
        (item.source.document_id, item.source.version, item.source.source_id)
        for item in base_pool[:budget]
    }
    # Include parents of items already in budget so we know which side is already represented.
    in_budget_with_parents = set(in_budget_ids)
    for item_id in in_budget_ids:
        src = sources.get(item_id)
        if src is not None and src.parent_id:
            in_budget_with_parents.add((src.document_id, src.version, src.parent_id))

    kg_reserve_cap = min(max(2, budget // 2), max(1, budget - len(protected)))
    per_concept_cap = min(3, kg_reserve_cap) if len(chosen_keys) == 1 else 2

    # Build per-concept ordered expansion units (with Notebook child_output co-ranking).
    per_concept_candidates: list[list[StructureCandidate]] = []
    kg_visual_candidates: list[StructureCandidate] = []
    seen_visual: set[tuple[str, int, str]] = set()

    for ck in chosen_keys:
        c_edges = edges_by_concept.get(ck, ())
        present_rels = {
            edge.relation_type
            for edge in c_edges
            if (edge.document_id, edge.version, edge.source_id) in in_budget_with_parents
        }
        # Prefer missing counterpart edges first, then uncovered edges on present sides.
        ordered_edges = sorted(
            c_edges,
            key=lambda e: (
                e.relation_type in present_rels,
                e.relation_type != "demonstrated_by",
                e.edge_id,
            ),
        )
        items_for_concept: list[StructureCandidate] = []
        seen_for_concept: set[tuple[str, int, str]] = set()
        for edge in ordered_edges:
            t_key = (edge.document_id, edge.version, edge.source_id)
            target_src = sources.get(t_key)
            if target_src is None:
                continue
            if t_key not in protected_ids and t_key not in seen_for_concept:
                # Emit readable target candidate (for empty-code cells include if text exists).
                if target_src.text.strip():
                    cand = _structural_candidate(
                        target_src, target_src, "kg_one_hop", docs.values()
                    )
                    items_for_concept.append(cand)
                    seen_for_concept.add(t_key)
                elif _can_display_source(target_src) and t_key not in seen_visual:
                    seen_visual.add(t_key)
                    kg_visual_candidates.append(
                        _structural_candidate(target_src, target_src, "kg_one_hop", docs.values())
                    )
            # Pair Notebook code cell with its saved child_output(s).
            if target_src.kind == "cell":
                children = outputs_by_cell.get(t_key, ())
                added_text_child = False
                for child in children:
                    c_key = (child.document_id, child.version, child.source_id)
                    if (
                        child.text.strip()
                        and not added_text_child
                        and c_key not in seen_for_concept
                    ):
                        items_for_concept.append(
                            _structural_candidate(
                                child, target_src, "kg_one_hop_child_output", docs.values()
                            )
                        )
                        seen_for_concept.add(c_key)
                        added_text_child = True
                    if _can_display_source(child) and c_key not in seen_visual:
                        seen_visual.add(c_key)
                        kg_visual_candidates.append(
                            _structural_candidate(
                                child, target_src, "kg_one_hop_child_output", docs.values()
                            )
                        )
        if items_for_concept:
            per_concept_candidates.append(items_for_concept)

    if not per_concept_candidates and not kg_visual_candidates:
        trace["status"] = "already_covered"
        return KgExpansionResult(list(pool), [], trace)

    # Round-robin interleave across concepts respecting per_concept_cap and kg_reserve_cap.
    additions: list[StructureCandidate] = []
    added_ids: set[tuple[str, int, str]] = set()
    concept_used = [0 for _ in per_concept_candidates]
    concept_cursors = [0 for _ in per_concept_candidates]
    single_concept_mode = len(per_concept_candidates) == 1

    while len(additions) < kg_reserve_cap:
        progressed = False
        for idx, cand_list in enumerate(per_concept_candidates):
            if len(additions) >= kg_reserve_cap:
                break
            if concept_used[idx] >= per_concept_cap:
                continue
            while concept_cursors[idx] < len(cand_list):
                cand = cand_list[concept_cursors[idx]]
                concept_cursors[idx] += 1
                c_id = (cand.source.document_id, cand.source.version, cand.source.source_id)
                if c_id in protected_ids or c_id in added_ids:
                    continue
                is_child_output = any(
                    r.code == "kg_one_hop_child_output" for r in cand.reasons
                )
                # Single-concept expansion keeps cell + child_output paired together at once;
                # multi-concept interleaves 1st counterpart per concept before 2nd-round children.
                if (
                    single_concept_mode
                    and cand.kind == "cell"
                    and concept_cursors[idx] < len(cand_list)
                    and any(
                        r.code == "kg_one_hop_child_output"
                        for r in cand_list[concept_cursors[idx]].reasons
                    )
                    and concept_used[idx] + 2 <= per_concept_cap
                    and len(additions) + 2 <= kg_reserve_cap
                ):
                    child_cand = cand_list[concept_cursors[idx]]
                    child_id = (
                        child_cand.source.document_id,
                        child_cand.source.version,
                        child_cand.source.source_id,
                    )
                    additions.append(cand)
                    added_ids.add(c_id)
                    concept_used[idx] += 1
                    if child_id not in protected_ids and child_id not in added_ids:
                        additions.append(child_cand)
                        added_ids.add(child_id)
                        concept_used[idx] += 1
                        concept_cursors[idx] += 1
                elif is_child_output:
                    cand_src = sources.get(c_id)
                    parent_id = cand_src.parent_id if cand_src is not None else None
                    parent_pos = (
                        next(
                            (
                                pos
                                for pos, prev in enumerate(additions)
                                if prev.source.document_id == cand.source.document_id
                                and prev.source.version == cand.source.version
                                and prev.source.source_id == parent_id
                            ),
                            None,
                        )
                        if parent_id
                        else None
                    )
                    if parent_pos is not None:
                        additions.insert(parent_pos + 1, cand)
                    else:
                        additions.append(cand)
                    added_ids.add(c_id)
                    concept_used[idx] += 1
                else:
                    additions.append(cand)
                    added_ids.add(c_id)
                    concept_used[idx] += 1
                progressed = True
                break
        if not progressed:
            break

    # Skip rewriting if every selected KG candidate was already within pool[:budget].
    uncovered_new = [
        item
        for item in additions
        if (item.source.document_id, item.source.version, item.source.source_id)
        not in in_budget_ids
    ]
    if not uncovered_new and not kg_visual_candidates:
        trace["status"] = "already_covered"
        return KgExpansionResult(list(pool), [], trace)

    final_pool: list[StructureCandidate] = []
    seen_final: set[tuple[str, int, str]] = set()
    for item in (*protected, *additions, *base_pool, *pool):
        key = (item.source.document_id, item.source.version, item.source.source_id)
        if key in seen_final:
            continue
        seen_final.add(key)
        final_pool.append(item)
        if len(final_pool) >= max(budget, len(pool)):
            break

    trace.update(
        status="succeeded",
        added_sources=[cand.source.model_dump() for cand in additions],
        visual_added=[cand.source.model_dump() for cand in kg_visual_candidates],
        per_concept_cap=per_concept_cap,
        kg_reserve_cap=kg_reserve_cap,
    )
    return KgExpansionResult(final_pool, kg_visual_candidates, trace)
