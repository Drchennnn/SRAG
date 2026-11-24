"""G17 scope, bilingual query fidelity and identical-pool ranking regressions."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from app.clients.jev import JevCandidate, JevClient, JevResult
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureSearcher, StructureSearchRequest
from app.rag.workspace_retrieval import BilingualRewriter, RewriteResult, plan_retrieval
from app.schemas.local import ParsedSource, SourceDraft
from app.schemas.workspace import WorkspaceError
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

_QUESTION = "振动校准需要满足哪些限制？"
_ALTERNATE = "Which limits govern vibration calibration?"
_COURSES = frozenset({"main"})


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        sqlite_path=tmp_path / "library.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "courses",
        deepseek_api_key="public-mock-deepseek",
        jev_api_key="public-mock-jev",
    )


def _publish(library, course_id, name, title, content):
    path = library.course_root / course_id / name
    original = f"{title}\n{content}".encode()
    plan = library.store.prepare(course_id, path)
    return library.store.publish(
        plan,
        ParsedSource(
            title=title,
            sources=[
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title=title,
                    text=content,
                    locator={"format": "pdf", "page": 1},
                )
            ],
        ),
        original=original,
        content_sha256=hashlib.sha256(original).hexdigest(),
        document_format="pdf",
    )


@pytest.fixture
def papers(settings):
    library = LocalLibrary(settings)
    docs = {
        _publish(library, course, filename, title, text).document_id: filename
        for course, filename, title, text in (
            (
                "main",
                "Guide.pdf",
                "Service guide",
                "Vibration calibration requires record METER-K4.",
            ),
            (
                "main",
                "Worksheet.pdf",
                "Practice",
                "A vibration calibration worksheet is illustrative.",
            ),
            ("main", "Station.pdf", "Station annex", "METER-K4 allows 17 mm only in dry rooms."),
            (
                "elsewhere",
                "Hidden.pdf",
                "Hidden",
                "METER-K4 permits 999 mm during vibration calibration.",
            ),
        )
    }
    store = StructureSnapshotStore(settings)
    return library, docs, store


def _rewrite(query, *, verify):
    verify()
    assert query == _QUESTION
    return RewriteResult(_ALTERNATE, {"status": "succeeded", "external_call_count": 0})


def _request(*, file_names=(), location=None):
    return StructureSearchRequest(
        query=_QUESTION,
        top_k=8,
        location=location,
        scope={"course_ids": ["main", "elsewhere"], "file_names": list(file_names)},
    )


class _Ranker:
    def __init__(self, illustration_score=0.15):
        self.requests = []
        self.illustration_score = illustration_score

    def score(self, question, candidates, *, evidence, verify):
        verify()
        assert question == _QUESTION
        self.requests.append((list(candidates), list(evidence)))
        scores = {
            item.candidate_id: (self.illustration_score if "illustrative" in item.passage else 0.91)
            for item in candidates
        }
        return JevResult(scores, {"status": "succeeded", "external_call_count": 0})


def test_four_arms_compare_same_pool_and_expand_once_from_a_read(settings, papers):
    library, names, store = papers
    snapshot = store.read(SnapshotRequest(("main", "elsewhere")))
    ranker = _Ranker()
    arms = {
        strategy: plan_retrieval(
            settings,
            StructureSearcher(store),
            snapshot,
            _request(),
            _COURSES,
            strategy=strategy,
            budget=2,
            library=library.store,
            rewrite=_rewrite,
            scorer=ranker,
        )
        for strategy in ("fts", "bilingual", "jev", "expanded")
    }
    assert not arms["fts"].selected
    assert arms["bilingual"].trace["candidate_pool"] == arms["jev"].trace["candidate_pool"]
    assert arms["jev"].trace["candidate_pool"] == arms["expanded"].trace["candidate_pool"]
    assert [item.candidate_id for item in ranker.requests[0][0]] == [
        item["candidate_id"] for item in arms["bilingual"].trace["candidate_pool"]
    ]
    assert ranker.requests[0] == ranker.requests[1]
    assert len(ranker.requests) == 3
    assert [names[item.source.document_id] for item in arms["expanded"].selected] == [
        "Guide.pdf",
        "Station.pdf",
    ]
    expansion = arms["expanded"].trace["expansion"]
    assert expansion["term"] == "METER-K4"
    first_read = ranker.requests[2][1][0].passage
    assert first_read[slice(*expansion["anchor_span"])] == "METER-K4"
    assert hashlib.sha256(first_read.encode()).hexdigest() == expansion["anchor_text_sha256"]
    assert "Hidden.pdf" not in str(arms)
    assert len(expansion["new_candidates"]) == 1
    assert all(len(result.selected) <= 2 for result in arms.values())


def test_scoped_query_never_reopens_alternate_file_or_location(settings, papers):
    library, _, store = papers
    snapshot = store.read(SnapshotRequest(("main", "elsewhere")))
    ranker = _Ranker()
    result = plan_retrieval(
        settings,
        StructureSearcher(store),
        snapshot,
        _request(file_names=("Guide.pdf",)),
        _COURSES,
        strategy="expanded",
        budget=2,
        library=library.store,
        rewrite=_rewrite,
        scorer=ranker,
    )
    assert [item.file_name for item in result.selected] == ["Guide.pdf"]
    assert result.trace["expansion"]["new_candidates"] == []
    assert result.trace["expansion"]["status"] == "no_new_candidates"
    assert len(ranker.requests) == 1
    direct = plan_retrieval(
        settings,
        StructureSearcher(store),
        snapshot,
        _request(file_names=("Guide.pdf",), location={"page": 1}),
        _COURSES,
        strategy="jev",
        budget=2,
        rewrite=lambda *args, **kwargs: pytest.fail("explicit position was rewritten"),
        scorer=ranker,
    )
    assert [item.file_name for item in direct.selected] == ["Guide.pdf"]
    assert direct.trace["effective_strategy"] == "fts"
    assert len(ranker.requests) == 1


def test_expansion_preserves_the_anchor_and_respects_one_slot_budget(settings, papers):
    library, names, store = papers
    snapshot = store.read(SnapshotRequest(("main",)))
    ranker = _Ranker(illustration_score=0.98)
    common = {
        "strategy": "expanded",
        "library": library.store,
        "rewrite": _rewrite,
        "scorer": ranker,
    }
    two_slots = plan_retrieval(
        settings,
        StructureSearcher(store),
        snapshot,
        _request(),
        _COURSES,
        budget=2,
        **common,
    )
    assert [names[item.source.document_id] for item in two_slots.selected] == [
        "Guide.pdf",
        "Station.pdf",
    ]
    assert names[two_slots.trace["expansion"]["displaced_source"]["document_id"]] == "Worksheet.pdf"
    one_slot = plan_retrieval(
        settings,
        StructureSearcher(store),
        snapshot,
        _request(),
        _COURSES,
        budget=1,
        **common,
    )
    assert one_slot.trace["expansion"]["status"] == "insufficient_budget"


def test_scoring_falls_back_without_a_jev_key(settings, papers):
    _, _, store = papers
    settings.jev_api_key = ""
    snapshot = store.read(SnapshotRequest(("main",)))
    result = plan_retrieval(
        settings,
        StructureSearcher(store),
        snapshot,
        _request(),
        _COURSES,
        strategy="expanded",
        budget=2,
        rewrite=_rewrite,
    )
    assert result.selected
    assert result.trace["effective_strategy"] == "bilingual"
    assert result.trace["jev"]["status"] == "not_configured"
    assert result.trace["expansion"]["status"] == "unavailable_without_jev"
    assert result.trace["external_call_count"] == 0


def test_refuses_result_after_publication_changes_during_rewrite(settings, papers):
    library, _, store = papers
    snapshot = store.read(SnapshotRequest(("main",)))

    def republish(query, *, verify):
        verify()
        _publish(library, "main", "Guide.pdf", "New guide", "New calibration protocol.")
        return _rewrite(query, verify=lambda: None)

    with pytest.raises(WorkspaceError, match="stale_reference"):
        plan_retrieval(
            settings,
            StructureSearcher(store),
            snapshot,
            _request(),
            _COURSES,
            strategy="jev",
            budget=2,
            rewrite=republish,
            scorer=_Ranker(),
        )


def test_bilingual_copy_preserves_limited_units_numbers_files_and_negation(settings):
    query = "查 Guide.pdf v2 第 4 页，不使用 `sensor.read`，有没有 5 毫秒上限？"
    equivalent = "On Guide.pdf v2 page 4, without `sensor.read`, is there a 5 millisecond limit?"
    replies = iter((equivalent.replace("without", "using"), equivalent))
    calls = []

    def fake(request):
        payload = json.loads(request.content)
        assert json.loads(payload["messages"][1]["content"]) == {
            "query": query,
            "target": "English",
        }
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"query": next(replies)})}}],
                "usage": {"total_tokens": 100},
            },
        )

    rewriter = BilingualRewriter(settings, transport=httpx.MockTransport(fake))
    assert rewriter(query).trace["status"] == "constraints_unverified"
    accepted = rewriter(query)
    assert accepted.query == equivalent
    assert accepted.trace["total_tokens"] == 100
    repeat = rewriter(query)
    assert repeat.query == equivalent
    assert repeat.trace["cache_hit"] and repeat.trace["external_call_count"] == 0
    assert len(calls) == 2


def test_jev_http_uses_systemone_exact_ids_and_retries_with_cache(settings):
    calls = []

    def fake(request):
        assert request.url == httpx.URL("https://api.zleap.com/v1/systemone")
        assert request.headers["Authorization"] == "Bearer public-mock-jev"
        assert request.method == "POST"
        payload = json.loads(request.content)
        assert payload["model"] == "Zev-1"
        assert json.loads(payload["state"]) == {
            "original_question": "question",
            "previously_retrieved_evidence": [],
        }
        assert payload["questions"]["candidate_a"]["type"] == "noul"
        assert payload["questions"]["candidate_a"]["instructions"]["candidate_passage"] == "point"
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "answers": {"candidate_a": {"type": "noul", "noul": 0.8}},
                "usage": {"total_tokens": 36},
            },
        )

    client = JevClient(settings, transport=httpx.MockTransport(fake))
    items = [JevCandidate("candidate_a", "title", "point")]
    result = client.score("question", items)
    assert result.scores == {"candidate_a": 0.8}
    assert [item["status"] for item in result.trace["attempts"]] == [503, 200]
    assert result.trace["usage"] == {"total_tokens": 36}
    assert result.trace["external_call_count"] == 2
    result.scores["candidate_a"] = 0
    warm = client.score("question", items)
    assert warm.scores == {"candidate_a": 0.8}
    assert warm.trace["cache_hit"] and warm.trace["external_call_count"] == 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"unexpected": {"type": "noul", "noul": 0.5}},
        {"known": {"type": "choice", "noul": 0.1}},
        {"known": {"type": "noul", "noul": True}},
        {"known": {"type": "noul", "noul": 1.5}},
    ],
)
def test_jev_rejects_malformed_score_without_caching(settings, answers):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": answers}))
    client = JevClient(settings, transport=transport)
    for _ in range(2):
        result = client.score("question", [JevCandidate("known", "title", "point")])
        assert result.scores == {}
        assert result.trace["status"] == "invalid_response"
        assert result.trace["external_call_count"] == 1
        assert not result.trace["cache_hit"]


def test_jev_does_not_retry_after_snapshot_guard_fails(settings):
    attempts = []

    def fake(request):
        attempts.append(request)
        return httpx.Response(503)

    def verify():
        if attempts:
            raise WorkspaceError("stale_reference")

    client = JevClient(settings, transport=httpx.MockTransport(fake))
    with pytest.raises(WorkspaceError, match="stale_reference"):
        client.score("question", [JevCandidate("known", "title", "point")], verify=verify)
    assert len(attempts) == 1


def test_negation_symmetry_ignores_feichang_and_preserves_real_negation():
    from app.rag.workspace_retrieval import _valid_rewrite

    assert _valid_rewrite(
        "这个优化器非常稳定吗？",
        "Is this optimizer very stable?",
        "English",
    )
    assert _valid_rewrite(
        "不同参数规模模型的“随机失活”比例分别是多少？",
        'What are the "dropout" ratios for models of different parameter scales?',
        "English",
    )
    assert not _valid_rewrite(
        "使用“MMLU 基准”做了哪些消融对比？",
        "What ablations were run on benchmarks?",
        "English",
    )
    assert not _valid_rewrite(
        "不要使用非线性层",
        "Use linear layers",
        "English",
    )
    assert _valid_rewrite(
        "不要使用非线性层",
        "Do not use non-linear layers",
        "English",
    )


def test_plan_retrieval_separates_text_pool_from_empty_visual_candidates(settings, tmp_path):
    library = LocalLibrary(settings)
    pages = [
        SourceDraft(
            source_id=f"page:{p}",
            kind="page",
            ordinal=p - 1,
            title=f"Page {p}",
            text=(
                f"Intro section anchor term on page {p}."
                if p <= 8
                else (
                    "Appendix B uses RMSprop with learning rate 1e-6."
                    if p == 20
                    else f"Page {p}"
                )
            ),
            locator={"format": "pdf", "page": p},
        )
        for p in range(1, 22)
    ]
    shapes = [
        SourceDraft(
            source_id=f"page:{p}:image:1",
            kind="page_object",
            parent_id=f"page:{p}",
            ordinal=30 + p,
            title=f"Figure on page {p}",
            text="",
            locator={"format": "pdf", "page": p, "object_type": "image"},
        )
        for p in range(1, 6)
    ]
    path = library.course_root / "main" / "LongPaper.pdf"
    original = b"%PDF-1.4\nlong-paper\n"
    plan = library.store.prepare("main", path)
    doc = library.store.publish(
        plan,
        ParsedSource(title="LongPaper", sources=[*pages, *shapes]),
        original=original,
        content_sha256=hashlib.sha256(original).hexdigest(),
        document_format="pdf",
    )
    store = StructureSnapshotStore(settings)
    snap = store.read(SnapshotRequest(("main",)))
    searcher = StructureSearcher(store)
    req = StructureSearchRequest(
        query="请介绍 anchor 优化器",
        top_k=12,
        scope={"course_ids": ["main"], "document_ids": [doc.document_id]},
    )

    def _rewrite(_query, *, verify=lambda: None):
        return RewriteResult(
            "Please introduce anchor RMSprop optimizer",
            {"status": "succeeded", "external_call_count": 1, "cache_hit": False},
        )

    plan_res = plan_retrieval(
        settings,
        searcher,
        snap,
        req,
        _COURSES,
        strategy="bilingual",
        budget=12,
        library=library.store,
        rewrite=_rewrite,
    )
    selected_ids = [item.source.source_id for item in plan_res.selected]
    assert "page:20" in selected_ids
    assert all(
        item.text.strip() and "image" not in item.source.source_id for item in plan_res.selected
    )
    assert all(not item.text.strip() for item in plan_res.visual_candidates)


def test_bilingual_places_anchor_next_position_before_tail_noise():
    from app.rag.structure_search import CandidateReason, StructureCandidate
    from app.schemas.local import SourceKey

    def _cand(page_no: int, *, seed_page: int | None = None) -> StructureCandidate:
        src = SourceKey(document_id="doc-1", version=1, source_id=f"page:{page_no}")
        reasons = (
            [
                CandidateReason(
                    code="next_position",
                    seed=SourceKey(document_id="doc-1", version=1, source_id=f"page:{seed_page}"),
                )
            ]
            if seed_page is not None
            else [CandidateReason(code="fts_native", seed=src)]
        )
        return StructureCandidate(
            source=src,
            course_id="c1",
            file_name="paper.pdf",
            kind="page",
            title=f"Page {page_no}",
            locator={"format": "pdf", "page": page_no},
            text=f"Page {page_no} body",
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=reasons,
        )

    from app.rag.workspace_retrieval import _prioritize_continuations

    pool = [
        _cand(5),
        _cand(23),
        _cand(14),
        _cand(15),
        _cand(2),
        _cand(24, seed_page=23),
    ]
    reordered = _prioritize_continuations(pool)
    assert [c.source.source_id for c in reordered] == [
        "page:5",
        "page:23",
        "page:24",
        "page:14",
        "page:15",
        "page:2",
    ]


@pytest.mark.parametrize(
    ("original", "candidate", "target", "expected", "expected_reason"),
    [
        # Chinese numerals + billion/million/thousand & hyphenation
        (
            "五千四百亿参数模型在推理任务上的表现如何？",
            "How does the 540-billion-parameter model perform on reasoning tasks?",
            "English",
            True,
            None,
        ),
        (
            "五千四百亿参数模型在推理任务上的表现如何？",
            "How does the 540-million-parameter model perform on reasoning tasks?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        (
            "一点五亿参数模型在两点五万亿词元上训练的困惑度是多少？",
            "What is the perplexity of a 150 million parameter model "
            "trained on 2.5 trillion tokens?",
            "English",
            True,
            None,
        ),
        (
            "在两千四百零五万条样本上微调零点二五比例的收敛效果如何？",
            "How is the convergence when fine-tuning on 24,050,000 samples with a ratio of 0.25?",
            "English",
            True,
            None,
        ),
        (
            "在两千四百零五万条样本上微调零点二五比例的收敛效果如何？",
            "How is the convergence when fine-tuning on 24,500,000 samples with a ratio of 0.25?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        # Units & percentages
        (
            "每日八百毫克剂量下，响应延迟三十毫秒时的误差是多少？",
            "What is the error at a daily dose of 800 mg and a response latency of 30 ms?",
            "English",
            True,
            None,
        ),
        (
            "每日八百毫克剂量下，响应延迟三十毫秒时的误差是多少？",
            "What is the error at a daily dose of 800 g and a response latency of 30 us?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        (
            "当单步延迟上限设为 500 毫秒时系统吞吐量如何？",
            "How is system throughput when the per-step latency cap is set to 500 us?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        (
            "按百分之五的显著性阈值计算有多少神经元不服从正态分布，超出预期假阳性率多少？",
            "At a 5% significance threshold, how many neurons are non-normal, "
            "exceeding the expected false positive rate by how much?",
            "English",
            True,
            None,
        ),
        # Comparison directions & fractions
        (
            "在至少 5 个任务上准确率大于 10% 且全秩学习率不超过千分之一的配置有哪些？",
            "Which configurations achieve accuracy greater than 10% on at least 5 tasks "
            "with full-rank learning rate at most 0.001?",
            "English",
            True,
            None,
        ),
        (
            "与全秩基线通常采用不超过千分之一的学习率相比是多少？",
            "What is it compared with the full-rank baseline using "
            "no more than one-thousandth learning rate?",
            "English",
            True,
            None,
        ),
        (
            "与全秩基线通常采用不超过千分之一的学习率相比是多少？",
            "What is it compared with the full-rank baseline using "
            "no more than one-hundredth learning rate?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        (
            "与全秩基线通常采用不超过千分之一的学习率相比是多少？",
            "What is it compared with the full-rank baseline using "
            "more than one-thousandth learning rate?",
            "English",
            False,
            "comparison_mismatch",
        ),
        (
            "在至少 5 个任务上准确率大于 10% 的配置有哪些？",
            "Which configurations achieve accuracy less than 10% on at least 5 tasks?",
            "English",
            False,
            "comparison_mismatch",
        ),
        (
            "在至少 5 个任务上准确率大于 10% 的配置有哪些？",
            "Which configurations achieve accuracy greater than 10% on at most 5 tasks?",
            "English",
            False,
            "comparison_mismatch",
        ),
        # Versions & code identifiers + two-object swap + add/delete
        (
            "在 v2.1 中设置 top_k=5 和 batch_size=64 对比 v1.0 在 llama-7b 与 gpt-4 上表现如何？",
            "How does v2.1 with top_k=5 and batch_size=64 compare against v1.0 "
            "on llama-7b and gpt-4?",
            "English",
            True,
            None,
        ),
        (
            "在 v2.1 中设置 top_k=5 和 batch_size=64 对比 v1.0 在 llama-7b 与 gpt-4 上表现如何？",
            "How does v2.0 with top_k=5 and batch_size=64 compare against v1.0 "
            "on llama-7b and gpt-4?",
            "English",
            False,
            "literal_missing",
        ),
        (
            "7B 用 16 批大小、65B 用 64 批大小时显存分别是多少？",
            "What is the memory usage when 7B uses batch size 16 and 65B uses batch size 64?",
            "English",
            True,
            None,
        ),
        (
            "7B 用 16 批大小、65B 用 64 批大小时显存分别是多少？",
            "What is the memory usage when 7B uses batch size 64 and 65B uses batch size 16?",
            "English",
            False,
            "binding_mismatch",
        ),
        (
            "五千四百亿参数模型在推理任务上的表现如何？",
            "How does the 540-billion-parameter model perform on 3 reasoning tasks?",
            "English",
            False,
            "numeric_or_unit_mismatch",
        ),
        # Vague Chinese numerals must fail safely + non-numeric idioms must pass
        (
            "在几亿参数规模下该优化器的内存开销如何？",
            "What is the memory cost of this optimizer at several hundred million parameters?",
            "English",
            False,
            "vague_numeral",
        ),
        (
            "每隔上百次迭代切换一次子空间会带来多少额外开销？",
            "How much extra overhead is incurred by switching subspaces every 100 iterations?",
            "English",
            False,
            "vague_numeral",
        ),
        (
            "在十分有限的显存下，该方法在一般训练条件下与全秩训练的收敛是否一致？",
            "Under very limited VRAM, is the convergence consistent with general conditions?",
            "English",
            True,
            None,
        ),
        # English -> Chinese symmetry
        (
            "How does the 540-billion-parameter model compare "
            "at least 5 benchmarks under 10% budget?",
            "五千四百亿参数模型在百分之十预算下对比至少 5 个基准表现如何？",
            "Chinese",
            True,
            None,
        ),
        (
            "How does the 540-billion-parameter model compare "
            "at least 5 benchmarks under 10% budget?",
            "五百四十万参数模型在百分之十预算下对比至少 5 个基准表现如何？",
            "Chinese",
            False,
            "numeric_or_unit_mismatch",
        ),
    ],
)
def test_g23_paired_numeric_unit_comparison_and_binding_constraints(
    original, candidate, target, expected, expected_reason
):
    from app.rag.workspace_retrieval import _validate_rewrite_constraints

    reason = _validate_rewrite_constraints(original, candidate, target)
    assert (reason is None) is expected
    assert reason == expected_reason


def test_g23_rewriter_records_constraint_reason_and_falls_back_cleanly(settings):
    def fake(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"query": "What about 540 million parameter model?"}
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    rewriter = BilingualRewriter(settings, transport=httpx.MockTransport(fake))
    result = rewriter("五千四百亿参数模型的性能如何？")
    assert result.query is None
    assert result.trace["status"] == "constraints_unverified"
    assert result.trace["constraint_reason"] == "numeric_or_unit_mismatch"
    assert result.trace["total_tokens"] == 42


def test_rejected_rewrite_reserves_scoped_saved_output_within_candidate_cap(settings, papers):
    library, _, store = papers
    outputs = {}
    for course in ("main", "elsewhere"):
        name = "Run.ipynb"
        path = library.course_root / course / name
        raw = f"notebook for {course}".encode()
        plan = library.store.prepare(course, path)
        outputs[course] = library.store.publish(
            plan,
            ParsedSource(
                title=name,
                sources=[
                    SourceDraft(
                        source_id="cell:0", kind="cell", ordinal=0, title="Run",
                        text="print(measured)", locator={"format": "ipynb", "cell_index": 0},
                    ),
                    SourceDraft(
                        source_id="cell:0:output:0", kind="notebook_output", ordinal=1,
                        parent_id="cell:0", title="Saved output", text="Twelve completions.",
                        locator={"format": "ipynb", "cell_index": 0, "output_index": 0},
                    ),
                ],
            ),
            original=raw,
            content_sha256=hashlib.sha256(raw).hexdigest(),
            document_format="ipynb",
        )

    def reject(query, *, verify):
        verify()
        return RewriteResult(
            None, {"status": "constraints_unverified", "constraint_reason": "vague_numeral",
                   "external_call_count": 1},
        )

    snapshot = store.read(SnapshotRequest(("main", "elsewhere")))
    for file_names, expected in (([], True), (["Guide.pdf"], False)):
        request = StructureSearchRequest(
            query="vibration calibration 怎么定义？实验处理十一个输入时完成了几次？",
            top_k=2,
            scope={"course_ids": ["main", "elsewhere"], "file_names": file_names},
        )
        plan = plan_retrieval(
            settings, StructureSearcher(store), snapshot, request, _COURSES,
            strategy="bilingual", budget=2, rewrite=reject, supplement_saved_outputs=True,
        )
        assert plan.trace["rewrite"]["constraint_reason"] == "vague_numeral"
        assert plan.trace["external_call_count"] == 1
        assert plan.trace["alternate_query"] is None
        assert len(plan.trace["candidate_pool"]) <= request.top_k
        assert len(plan.selected) <= request.top_k
        result = [item for item in plan.selected if item.kind == "notebook_output"]
        assert bool(result) is expected
        if expected:
            assert plan.trace["saved_output_fallback"]["status"] == "reserved"
            assert len(plan.trace["saved_output_fallback"]["displaced"]) == 1
            assert result[0].source.document_id == outputs["main"].document_id
        else:
            assert plan.trace["saved_output_fallback"]["added"] == []
