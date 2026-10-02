"""任务路由、追问与话题切换的行为验收。"""

import pytest

from app.rag.dialogue import (
    DialogueContext,
    construct_dialogue_query,
    extract_page_numbers,
    route_question,
)


@pytest.mark.parametrize(
    "query,route",
    [
        ("我第一个问题是什么", "history"),
        ("我之前都问了什么？", "history"),
        ("你刚才的回答是什么？", "history"),
        ("有哪些已入库的课件？", "catalog"),
        ("列出当前课程的文件", "catalog"),
        ("有哪些课件讲 Sobel？", "knowledge"),
        ("这两个方向的核有什么用？课件如果没有你自己回答", "knowledge"),
        ("概括本章知识点", "document_summary"),
        ("概括 Sobel 在课件里的定义", "knowledge"),
        ("总结这份课件", "document_summary"),
        ("给我简要总结一下 lectrue 2-1", "document_summary"),
        ("总一下 lecture2-1，顺便说一下有哪些重点", "document_summary"),
        ("lecture2a 讲了什么", "document_summary"),
        ("Lecture 8-2 主要讲什么？", "document_summary"),
        ("这份课件有哪些知识点？", "document_summary"),
        ("Lecture3a.pdf 的内容是什么？", "document_summary"),
        ("Sobel 讲了什么？", "knowledge"),
        ("linear regression 的主要内容是什么？", "knowledge"),
        ("第九页的 temperature 图是什么意思？", "document_page"),
        ("总结 Lecture2a.pdf 第十四页", "document_page"),
        ("总结 Sobel 的作用", "knowledge"),
        ("你刚才回答的 Sobel 公式对吗？", "knowledge"),
        ("你好", "greeting"),
    ],
)
def test_route_question(query: str, route: str) -> None:
    assert route_question(query) == route


@pytest.mark.parametrize(
    "query",
    [
        "这两个方向的核有什么用？课件如果没有你自己回答",
        "它有什么用？",
        "这个机制怎么理解？",
        "这个公式在哪页？",
        "继续",
        "Gx 是什么？",
        "它的矩阵公式是什么？",
        "这个矩阵怎么理解？",
        "这个超参数应该怎么选？",
        "这个参数怎么设置？",
    ],
)
def test_followup_uses_known_topic(query: str) -> None:
    context = DialogueContext("Sobel operator 是什么？", entities=("Sobel", "Gx", "Gy"))
    result = construct_dialogue_query(query, context, enabled=True)
    assert result.method == "history_expanded"
    assert result.text.startswith("Sobel operator 是什么？\n追问：")


@pytest.mark.parametrize(
    "query",
    [
        "the sobel operator 是啥",
        "extrinsic parameters 又是什么？",
        "这个主点定义在哪页？",
        "这个 K 矩阵在哪页？",
        "这个 AdaBoost 弱学习器定义在哪页？",
        "贝叶斯公式是什么？",
    ],
)
def test_named_topic_switch_does_not_expand(query: str) -> None:
    context = DialogueContext("Sobel operator 是什么？", entities=("Sobel", "Gx", "Gy"))
    result = construct_dialogue_query(query, context, enabled=True)
    assert result.method == "raw"
    assert result.text == query


def test_raw_and_missing_context_do_not_guess_topic() -> None:
    query = "这两个方向的核有什么用？"
    context = DialogueContext("Sobel 是什么？")
    assert construct_dialogue_query(query, context, enabled=False).text == query
    assert construct_dialogue_query(query, None, enabled=True).text == query


@pytest.mark.parametrize(
    "query,pages",
    [
        ("解释第九页的图", (9,)),
        ("总结第十四页", (14,)),
        ("对比第二十页和第 21 页", (20, 21)),
        ("第一百零二页讲什么？", (102,)),
        ("看二〇二页，再看第九页和第九页", (202, 9)),
        ("这个公式在哪一页？", ()),
        ("第零页", ()),
    ],
)
def test_extract_page_numbers(query: str, pages: tuple[int, ...]) -> None:
    assert extract_page_numbers(query) == pages
