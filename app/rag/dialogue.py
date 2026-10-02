"""终端对话的任务路由与短期主题；不推断检索权限。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.rag.query import ConstructedQuery, construct_query
from app.schemas.rag import ConversationTurn, SearchRequest

ChatRoute = Literal[
    "knowledge", "follow_up", "history", "catalog", "document_summary", "document_page", "greeting"
]

_HISTORY = re.compile(
    r"(?:我|我们|用户)(?:的)?(?:第[一二三四五六七八九十\d]+个|第一|最早|上一个|上一|刚才|之前|前面|"
    r"所有|全部|曾经|都).*?(?:问|提问)|"
    r"(?:你|助手)(?:的)?(?:上一个|上一|刚才|之前|前面).*?(?:回答|回复)"
)
_CATALOG = re.compile(
    r"(?:有哪些|有什么|列出|列一下|看看|查看|浏览|显示|多少|几份|几个)"
    r"(?:(?:已入库|当前课程|当前|本课程|这门课|课程|可见|全部|所有)的?|\s)*"
    r"(?:文件|课件)(?:[？?。！!\s]|$)|(?:文件|课件)(?:清单|列表)"
)
_SUMMARY = re.compile(r"总结|总一下|概括|归纳|复习重点|summari[sz]e", re.IGNORECASE)
_DOCUMENT = re.compile(
    r"(?:总结|概括|归纳)\s*(?:一下|下)?\s*(?:(?:这份|该|本|整份|整个|当前)\s*)?(?:课件|课程)|"
    r"这份课件|章节|本章|本节|这一讲|本讲|整门课|这门课|lecture|lectrue|\.(?:pdf|pptx|ipynb)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(r"(?:你好|您好|嗨|hello|hi|谢谢|再见)[！!。.\s]*", re.IGNORECASE)
_GENERIC_FOLLOW_UP = re.compile(
    r"^(?:(?:这|那)(?:两|几|\d+)?(?:个|些|种)?"
    r"(?:方向的核|卷积核|核|机制|流程|过程|总结|公式|定义|步骤|结果|方法|图|例子|参数|部分)|它们?)"
    r"(?:的)?(?:有什么用|有什么作用|有什么区别|有什么关系|是怎么来的|怎么理解|怎么用|为什么|"
    r"能解释|能举例|能再|可以解释|可以举例|可以再|在哪里|在哪|是什么|什么意思|如何|讲一下|解释一下)"
)
_EXPLICIT_CONTINUATION = re.compile(r"^(?:继续|再讲讲|再解释一下|举个例子)[。！!？?\s]*$")
_DETAIL_FOLLOW_UP = re.compile(
    r"^(?:它们?|这个|该)(?:的)?(?:矩阵公式|计算公式|公式|矩阵|表达式|物理含义|推导过程)"
    r"(?:是什么|是啥|怎么写|怎么推导|如何推导|怎么理解|能解释|什么意思)"
)
_SUPPLEMENT = re.compile(
    r"[，,。；;？?]?\s*课件(?:里)?如果没有(?:你)?(?:就)?(?:自己回答|自己解释)[。！!？?\s]*$"
)


@dataclass(frozen=True)
class ContextSource:
    document_id: str
    version: int
    chunk_id: str
    page: int | None


@dataclass(frozen=True)
class DialogueContext:
    """仅作为追问线索；旧来源不直接作为本轮回答证据。"""

    topic_query: str
    entities: tuple[str, ...] = ()
    sources: tuple[ContextSource, ...] = ()
    focus_document_id: str | None = None
    focus_version: int | None = None


def route_question(question: str) -> ChatRoute:
    """只路由高确定性表达；其他问题继续走知识检索。"""

    if _GREETING.fullmatch(question):
        return "greeting"
    if _HISTORY.search(question) and not re.search(
        r"对吗|正确|为什么|如何|怎么|公式.*(?:用|推导)", question
    ):
        return "history"
    if _SUMMARY.search(question) and _DOCUMENT.search(question):
        return "document_summary"
    if _CATALOG.search(question) and not re.search(r"(?:讲|关于|涉及|包含).*\S", question):
        return "catalog"
    return "knowledge"


def construct_dialogue_query(
    question: str, context: DialogueContext | None, *, enabled: bool
) -> ConstructedQuery:
    """复用正式追问规则，并覆盖对话中明确的泛称对象；原问题另行保留。"""

    if not enabled:
        return ConstructedQuery(question, "raw")
    current = _SUPPLEMENT.sub("", question).strip() or question
    if context is None:
        return ConstructedQuery(current, "raw")
    constructed = construct_query(
        SearchRequest(
            query=current,
            query_construction="contextual",
            history=[ConversationTurn(role="user", content=context.topic_query[:2000])],
        )
    )
    if constructed.method == "history_expanded":
        return constructed
    if (
        _GENERIC_FOLLOW_UP.search(current)
        or _DETAIL_FOLLOW_UP.search(current)
        or _EXPLICIT_CONTINUATION.fullmatch(current)
    ):
        return ConstructedQuery(f"{context.topic_query}\n追问：{current}", "history_expanded")
    # 回答中新引入的短数学符号也可能需要上轮主题，例如 Gx/Gy；不同的命名知识点不补入旧题。
    symbols = [
        entity for entity in context.entities if re.fullmatch(r"[A-Z](?:[a-z]|\d{1,2})", entity)
    ]
    if any(re.search(rf"\b{re.escape(symbol)}\b", current) for symbol in symbols):
        remainder = current
        for symbol in symbols:
            remainder = re.sub(rf"\b{re.escape(symbol)}\b", "", remainder)
        if re.fullmatch(
            r"[\s，,？?。]*?(?:是什么|是啥|什么意思|有什么用|有什么作用|在哪页)[\s？?。]*",
            remainder,
        ):
            return ConstructedQuery(f"{context.topic_query}\n追问：{current}", "history_expanded")
    return constructed


def answer_entities(answer: str) -> tuple[str, ...]:
    """提取回答中的术语/符号作为指代线索，不作为事实证据。"""

    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", answer)
    return tuple(dict.fromkeys(candidates))[:8]


def history_answer(question: str, turns: list[tuple[str, str]]) -> str:
    """从本次范围内保存的完整历史查问题；不依赖模型的两轮窗口。"""

    if not turns:
        return "本次范围内还没有之前的问答记录。切换课程或文件会清空对话历史。"
    if re.search(r"(?:你|助手).*?(?:回答|回复)", question):
        return "上一轮回答是：\n" + turns[-1][1]
    if re.search(r"所有|全部|都.*问|问过.*哪些", question):
        return "本次范围内的问题记录：\n" + "\n".join(
            f"{index}. {query}" for index, (query, _) in enumerate(turns, start=1)
        )
    ordinal = re.search(r"第([一二三四五六七八九十\d]+)个", question)
    number = None
    if ordinal:
        token = ordinal[1]
        number = (
            int(token)
            if token.isdecimal()
            else {
                "一": 1,
                "二": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
            }.get(token)
        )
        if number is None or not 1 <= number <= len(turns):
            return "没有找到这个序号的问题。可询问：我之前都问了什么？"
    elif re.search(r"第一|最早", question):
        number = 1
    if number is not None:
        return f"你的第 {number} 个问题是：\n“{turns[number - 1][0]}”"
    if re.search(r"上一个|上一|刚才", question):
        return f"你的上一个问题是：\n“{turns[-1][0]}”"
    return "本次范围内的问题记录：\n" + "\n".join(
        f"{index}. {query}" for index, (query, _) in enumerate(turns, start=1)
    )
