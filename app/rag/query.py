"""从当前问题构造检索文本，不推断或修改课程与文件范围。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.schemas.rag import SearchRequest

_REFERENT_PATTERN = re.compile(
    r"它们?|这个|这种|上述|前面|刚才|上一张|那张|该图|该公式|"
    r"\b(?:it|this|that|these|those)\b",
    re.IGNORECASE,
)
_QUESTION_FRAME_PATTERN = re.compile(
    r"在哪(?:一|几)?(?:页|段|节|处|里|儿)|第几(?:页|段|节)|哪里|出现|介绍|列出|解释|讲述|讲|说|展示|提到|"
    r"例子|应用|公式|动机|结果|理由|总览|定义|步骤|问题|方法|内容|信息|概念|过程|原理|"
    r"原因|条件|变换|练习|规则|区别|参数|指标|"
    r"课件|课程|图片|图表|该|的|是|有|和|与|吗|呢|"
    r"\b(?:where|what|which|how|is|are|was|were|does|do|did|can|could|would|"
    r"please|find|explain|explained|discussed|shown|show|appear|appears|page|slide|"
    r"example|formula|result|the|a|an|of|in|on|for|about)\b|"
    r"？|\?|[，,。：:、.\s]",
    re.IGNORECASE,
)


def _has_specific_topic(query: str) -> bool:
    """保守识别当前问题已提供的主题词，避免把话题切换误判为追问。"""

    remaining = _QUESTION_FRAME_PATTERN.sub("", _REFERENT_PATTERN.sub("", query))
    return (
        bool(re.search(r"[A-Za-z]{2,}|[A-Z]|[α-ωΑ-Ω]|[a-zA-Z]\s*\(|\d", remaining))
        or sum("\u4e00" <= char <= "\u9fff" for char in remaining) >= 2
    )


@dataclass(frozen=True, slots=True)
class ConstructedQuery:
    text: str
    method: Literal["raw", "history_expanded"]


def construct_query(request: SearchRequest) -> ConstructedQuery:
    """仅将最近的用户问题补入显式上下文追问；不信任历史中的范围信息。"""

    current = request.query.strip()
    if (
        request.query_construction == "raw"
        or not _REFERENT_PATTERN.search(current)
        or _has_specific_topic(current)
    ):
        return ConstructedQuery(current, "raw")

    previous_user = next(
        (turn.content.strip() for turn in reversed(request.history) if turn.role == "user"),
        "",
    )
    if not previous_user:
        return ConstructedQuery(current, "raw")

    return ConstructedQuery(f"{previous_user}\n追问：{current}", "history_expanded")
