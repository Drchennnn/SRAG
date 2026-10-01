"""消融实验必须真实执行所标记的策略。"""

from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.config import Settings
from app.evaluation.retrieval import ExpectedSource, RetrievalCase
from app.schemas.embedding import EmbeddingVector
from evals.run_retrieval_ablation import evaluate_strategy, index_corpus


async def test_rerank_failure_does_not_become_a_labeled_result() -> None:
    class FailingDashScope:
        async def embed_query(self, query: str) -> EmbeddingVector:
            return EmbeddingVector(index=0, dense=[0.1])

        async def rerank(self, query: str, documents: list[str], *, top_n: int):
            raise RuntimeError("rerank API unavailable")

    class FakeQdrant:
        async def search_text_dense(self, vector, **kwargs):
            return [
                models.ScoredPoint(id=1, version=1, score=0.9, payload={"content": "one"}),
                models.ScoredPoint(id=2, version=1, score=0.8, payload={"content": "two"}),
            ]

        async def search_document_outline(self, vector, **kwargs):
            return []

    case = RetrievalCase(
        case_id="one",
        split="dev",
        course_id="CS5489",
        query="测试查询",
        reference_answer="测试答案",
        expected_sources=[
            ExpectedSource(source_path="lesson.ipynb", section_path=["一"], cell_ids=["cell-1"])
        ],
    )
    metadata = SimpleNamespace(
        visible_versions=lambda **kwargs: [SimpleNamespace(document_id="doc-1", version=1)]
    )

    with pytest.raises(RuntimeError, match="rerank API unavailable"):
        await evaluate_strategy(
            strategy="rrf+outline+rerank",
            cases=[case],
            top_k=1,
            dashscope=FailingDashScope(),
            qdrant=FakeQdrant(),
            metadata_store=metadata,
            settings=Settings(_env_file=None),
        )


async def test_index_corpus_groups_paths_and_rejects_course_root_escape(
    tmp_path, monkeypatch
) -> None:
    notebook = tmp_path / "lesson.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    indexed = []

    class FakeIndexer:
        text_store = object()

        async def index(self, request):
            indexed.append(request)
            return SimpleNamespace(failed_files=0, errors={}, indexed_files=1, skipped_files=0)

    monkeypatch.setattr(
        "app.rag.indexer.DocumentIndexer.from_settings", lambda settings: FakeIndexer()
    )
    case = SimpleNamespace(
        course_id="CS5489",
        expected_sources=[
            SimpleNamespace(source_path="lesson.ipynb"),
            SimpleNamespace(source_path="lesson.ipynb"),
        ],
    )

    await index_corpus(Settings(_env_file=None, course_root=tmp_path), [case])

    assert len(indexed) == 1
    assert indexed[0].course_id == "CS5489"
    assert indexed[0].paths == [notebook]

    escape = SimpleNamespace(
        course_id="CS5489", expected_sources=[SimpleNamespace(source_path="../outside.ipynb")]
    )
    with pytest.raises(ValueError, match="越出 COURSE_ROOT"):
        await index_corpus(Settings(_env_file=None, course_root=tmp_path), [escape])
