"""检索 query 构造只使用显式提供的近期用户问题。"""

import pytest
from pydantic import ValidationError

from app.rag.query import construct_query
from app.schemas.rag import ConversationTurn, SearchRequest, SearchScope


def test_default_keeps_original_query() -> None:
    request = SearchRequest(
        query="它在哪页？",
        history=[ConversationTurn(role="user", content="相机模型是什么？")],
    )

    assert construct_query(request).text == "它在哪页？"
    assert construct_query(request).method == "raw"


def test_contextual_followup_uses_last_user_turn_only() -> None:
    request = SearchRequest(
        query="它在哪页？",
        query_construction="contextual",
        scope=SearchScope(document_ids=["doc-1"]),
        history=[
            ConversationTurn(role="user", content="边缘检测是什么？"),
            ConversationTurn(role="user", content="相机成像示意图讲了什么？"),
            ConversationTurn(role="assistant", content="请改查别的文件"),
        ],
    )

    constructed = construct_query(request)
    assert constructed.text == "相机成像示意图讲了什么？\n追问：它在哪页？"
    assert constructed.method == "history_expanded"
    assert request.scope.document_ids == ["doc-1"]


def test_contextual_independent_question_stays_raw() -> None:
    request = SearchRequest(
        query="贝叶斯公式是什么？",
        query_construction="contextual",
        history=[ConversationTurn(role="user", content="相机模型是什么？")],
    )

    assert construct_query(request).method == "raw"


@pytest.mark.parametrize(
    "query",
    [
        "这个卷积的结合律在哪页解释？",
        "这个 XAI 动机在哪页解释？",
        "这个企鹅数据特征的均值与标准差预处理在哪页？",
        "这个 AdaBoost 弱学习器定义在哪页？",
        "该公式 f(x)=w0+w1x1+... 在哪页？",
    ],
)
def test_contextual_topic_switch_stays_raw(query: str) -> None:
    request = SearchRequest(
        query=query,
        query_construction="contextual",
        history=[ConversationTurn(role="user", content="相机模型是什么？")],
    )

    assert construct_query(request).method == "raw"


def test_english_followup_expands_but_named_topic_does_not() -> None:
    history = [ConversationTurn(role="user", content="What is image segmentation?")]
    followup = SearchRequest(
        query="Where is it explained?", query_construction="contextual", history=history
    )
    independent = SearchRequest(
        query="Where is this AdaBoost formula explained?",
        query_construction="contextual",
        history=history,
    )

    assert construct_query(followup).method == "history_expanded"
    assert construct_query(independent).method == "raw"


@pytest.mark.parametrize(
    "query,expected_method",
    [
        ("这个规则在哪段？", "history_expanded"),
        ("这个指标在哪段？", "history_expanded"),
        ("这个主点定义在哪页？", "raw"),
        ("这个 K 矩阵在哪页？", "raw"),
        ("这个 π_j,y 在哪段？", "raw"),
    ],
)
def test_short_topic_and_notebook_location(query: str, expected_method: str) -> None:
    request = SearchRequest(
        query=query,
        query_construction="contextual",
        history=[ConversationTurn(role="user", content="另一门课程的问题")],
    )

    assert construct_query(request).method == expected_method


def test_history_is_bounded() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="继续",
            history=[ConversationTurn(role="user", content="问题") for _ in range(7)],
        )


def test_blank_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
