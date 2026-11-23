"""Course-scoped provenance knowledge graph storage, deterministic builder, and invalidation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from sqlite3 import Connection, Row
from typing import Any, Literal

from app.schemas.local import SourceKey
from app.stores.local_visibility import promote_due
from app.stores.sqlite import MetadataStore

KG_SCHEMA = """
CREATE TABLE IF NOT EXISTS local_kg_concepts (
    course_id TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    anchor_terms_json TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    updated_at REAL NOT NULL,
    PRIMARY KEY(course_id, concept_key)
);
CREATE TABLE IF NOT EXISTS local_kg_edges (
    edge_id INTEGER PRIMARY KEY,
    course_id TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK(relation_type IN ('defined_in', 'demonstrated_by')),
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    source_id TEXT NOT NULL,
    publication_epoch INTEGER NOT NULL CHECK(publication_epoch >= 0),
    content_sha256 TEXT NOT NULL,
    evidence_span_start INTEGER NOT NULL CHECK(evidence_span_start >= 0),
    evidence_span_end INTEGER NOT NULL CHECK(evidence_span_end > evidence_span_start),
    evidence_excerpt TEXT NOT NULL CHECK(length(trim(evidence_excerpt)) > 0),
    status TEXT NOT NULL CHECK(status IN ('candidate', 'accepted', 'rejected', 'superseded')),
    evidence_rule TEXT NOT NULL DEFAULT '',
    rejection_reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    UNIQUE(course_id, concept_key, relation_type, document_id, version, source_id),
    FOREIGN KEY(course_id, concept_key)
        REFERENCES local_kg_concepts(course_id, concept_key),
    FOREIGN KEY(document_id, version)
        REFERENCES local_versions(document_id, version),
    FOREIGN KEY(document_id, version, source_id)
        REFERENCES local_sources(document_id, version, source_id)
);
CREATE TABLE IF NOT EXISTS local_kg_doc_state (
    document_id TEXT PRIMARY KEY REFERENCES local_documents(document_id),
    course_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    publication_epoch INTEGER NOT NULL CHECK(publication_epoch >= 0),
    content_sha256 TEXT NOT NULL,
    synced_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_course_concept_status
ON local_kg_edges(course_id, concept_key, status, relation_type);
CREATE INDEX IF NOT EXISTS idx_kg_edges_doc_ver_status
ON local_kg_edges(document_id, version, status);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source_lookup
ON local_kg_edges(document_id, version, source_id, status);
"""

_MAX_EXCERPT_CHARS = 600
_MIN_DEF_BODY_CHARS = 12
_DISPLAYABLE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
)

_SEC_NUM_PREFIX = (
    r"(?:(?:§[ \t]*)?\d+(?:\.\d+)*[.)、]?[ \t]+"
    r"|[（(]\d+[)）][ \t]+|（[一二三四五六七八九十]+）[ \t]+)*"
)
_D1_HEADER_LINE = re.compile(
    r"^[ \t]*(?:#+[ \t]+)?" + _SEC_NUM_PREFIX +
    r"(?P<label>Formal[ \t]+Definition|Core[ \t]+Concept|Definition|Def\.|Rule|Theorem|Concept|"
    r"核心概念定义|概念定义|核心概念|正式定义|基本定义|定义与触发条件|判定规则|判定条件|核心原理|"
    r"定义|概念)"
    r"[ \t]*[:：][ \t]*(?P<rest>[^\r\n]+)",
    re.IGNORECASE,
)
_D1_INLINE_RULE_LINE = re.compile(
    r"^[ \t]*(?:#+[ \t]+)?" + _SEC_NUM_PREFIX +
    r"(?P<name>[\u3400-\u9fffA-Za-z0-9_.\-（）() ]{2,60}?)"
    r"[ \t]*(?:定义与触发条件|判定规则|判定条件|分发规则|分配规则|核心原理|定义)"
    r"[ \t]*[:：][ \t]*(?P<body>[^\r\n]+)",
    re.IGNORECASE,
)
_SECTION_BOUNDARY_LINE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+\S|(?:§[ \t]*)?[1-9]\d*(?:\.\d+)+[.)]?[ \t]+(?P<after>\S[^\r\n]*))",
    re.MULTILINE,
)
_NUMERIC_UNIT_OR_CONT = re.compile(
    r"^(?:(?:s|ms|us|ns|min|hr|h|hz|khz|mhz|ghz|db|k|m|b|g|mb|gb|tb|kb|px|em|cm|mm|rad|deg|"
    r"v|mv|a|ma|w|mw|kv|ohm|mol|fps|x)(?:[\s,;，。、.)/]|$)"
    r"|(?:秒|毫秒|微秒|赫兹|千赫|分贝|倍|次|个|条|项|页|行|的|时|处|中|上|下|内|外|后|前|"
    r"之间|以上|以下|以内|左右|或|和|与|及|为|即|则|并|而|但|，|。|；|：|、)"
    r"|[×*/=<>≤≥≈~±+%％])",
    re.IGNORECASE,
)

_D2_ZH = re.compile(
    r"(?P<name>[\u3400-\u9fffA-Za-z0-9_.\- ]{2,56}?(?:[ \t\r\n]*[（(][^()（）]{2,72}[)）])?)"
    r"[ \t]*(?:\r?\n[ \t]*)?"
    r"(?:定义为|是指|指的是|的判定规则为|的定义为|即是指)"
    r"[ \t]*(?:\r?\n[ \t]*)?"
    r"(?P<body>(?:[^。！？；]|\r?\n(?!\s*\r?\n)){10,}[。！？；]?)"
)
_D2_EN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_\- \t\r\n]{2,64}?(?:[ \t\r\n]*[（(][^()（）]{2,72}[)）])?)"
    r"[ \t]*(?:\r?\n[ \t]*)?"
    r"\b(?:is[ \t]+defined[ \t]+as|refers[ \t]+to|denotes|is[ \t]+formulated[ \t]+as)\b"
    r"[ \t]*(?:\r?\n[ \t]*)?"
    r"(?P<body>(?:[^!?]|\.(?=\d|[A-Za-z_])|\r?\n(?!\s*\r?\n)){12,}\.?)"
)

_D1_CONDITION_START = re.compile(
    r"^(?:当|如果|若|在|对于|按|通过|由|根据|when|if|where|given|for|a\s+|an\s+|the\s+)",
    re.IGNORECASE,
)
_D1_BODY_START = re.compile(
    r"^(?:是指|定义为|指的是|用于|表示|即|采用|使用|利用|设为|令|假设|已知|若设|"
    r"将|把|把输入|对输入|计算|度量|衡量|抑制|衰减|实现|满足|"
    r"is\s+|are\s+|refers\s+to|denotes|uses|measures|computes|suppresses|"
    r"attenuates|routes|controls|limits|triggers|scales|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*=|"
    r"[A-Za-zα-ωΑ-Ω_\\]+\s*=|[$\\])",
    re.IGNORECASE,
)
_D1_BODY_VERB_SPLIT = re.compile(
    r"^(?P<name>[\u3400-\u9fffA-Za-z0-9_.\-（）() ]{2,64}?)"
    r"(?:\s+(?:is\s+defined\s+as|is\s+a|is\s+an|refers\s+to|denotes|"
    r"attenuates|routes|scales|controls|limits|triggers|computes|measures|uses)\b"
    r"|[ \t]*(?:定义为|是指|指的是|使用|通过|采用|用于|控制|表示))",
    re.IGNORECASE,
)
_D1_INLINE_SEP = re.compile(r"[：:]|\s+[—–-]{1,2}\s+")
_MATH_EXPR_HINT = re.compile(r"[=+*/<>≤≥≈~±∑∫→\\^]")
_SUBTRACT_LEFT_END = re.compile(r"(?:\d|[A-Za-z]\([^)]*\)|[A-Za-z])$")
_SUBTRACT_RIGHT_START = re.compile(r"^(?:\d|[A-Za-z]\([^)]*\)|[A-Za-z](?:[\s*/+^()\-=]|$))")
_LEADING_FILLER_ZH = re.compile(
    r"^(?:其中|所谓|这里|因此|此外|具体地|本课程中|在本课程中|通常|一般地)[，,、\s]*"
)
_LEADING_FILLER_EN = re.compile(
    r"^(?:in\s+this\s+\w+|specifically|moreover|here|thus|note\s+that|formally)[,\s]+",
    re.IGNORECASE,
)

_PARENS_ALIAS = re.compile(r"^(?P<main>.+?)\s*[（(](?P<alias>[^()（）]{2,72})[)）]\s*$")
_GENERIC_CONCEPT_NAMES = frozenset(
    {
        "它",
        "它们",
        "这",
        "那",
        "该",
        "此",
        "方法",
        "该方法",
        "本方法",
        "该机制",
        "模型",
        "算法",
        "概念",
        "定义",
        "核心概念",
        "概念定义",
        "核心概念定义",
        "正式定义",
        "基本定义",
        "定义与触发条件",
        "判定规则",
        "判定条件",
        "分发规则",
        "分配规则",
        "核心原理",
        "规则",
        "系统",
        "实验",
        "结果",
        "数据",
        "it",
        "this",
        "that",
        "these",
        "those",
        "they",
        "which",
        "method",
        "the method",
        "our method",
        "this method",
        "model",
        "the model",
        "algorithm",
        "concept",
        "core concept",
        "definition",
        "formal definition",
        "def.",
        "theorem",
        "rule",
        "system",
        "experiment",
        "result",
        "agenda",
        "outline",
        "contents",
        "table of contents",
        "summary",
        "references",
    }
)
_PAGE_OR_LOC_TITLE = re.compile(
    r"^(?:page|slide|cell)\s+\d+$|^第\s*\d+\s*页$",
    re.IGNORECASE,
)
_D1_BODY_LABEL = re.compile(
    r"^(?:Formal[ \t]+Definition|Core[ \t]+Concept|Definition|Def\.|Rule|Theorem|Concept|"
    r"核心概念定义|概念定义|核心概念|正式定义|基本定义|定义与触发条件|判定规则|判定条件|核心原理|"
    r"定义|概念)$",
    re.IGNORECASE,
)
_OUTLINE_TITLE = re.compile(
    r"^(?:agenda|outline|table\s+of\s+contents|contents|topics|overview|references|"
    r"目录|提纲|课程大纲|议程|参考文献|致谢)\b",
    re.IGNORECASE,
)

_R2_SECTION = re.compile(
    r"^(?:references?|bibliography|related\s+work|acknowledg?ments?|参考文献|致谢|相关工作)\b",
    re.IGNORECASE,
)
_R2_MENTION = re.compile(
    r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]|\bet\s+al\.|"
    r"\b(?:prior\s+work|previous\s+work|unlike|compared\s+(?:with|to)|"
    r"in\s+contrast\s+to|as\s+opposed\s+to)\b|"
    r"(?:不同于|区别于|与[^，。]{1,24}不同|相比(?:之下|于)?|参见参考文献|详见文献|参考文献)",
    re.IGNORECASE,
)
_R3_UNOBSERVED = re.compile(
    r"\b(?:future\s+work|experiment(?:al)?\s+plan|planned|plan\s+to|propose(?:d|s)?\s+to|"
    r"hypothetical|expected\s+(?:to|result)|anticipated|unobserved|unmeasured|"
    r"not\s+(?:observed|measured|performed|run)|never\s+(?:performed|measured|run))\b|"
    r"(?:未来工作|后续工作|实验计划|实验提案|实验设想|拟开展|拟采用|拟测试|"
    r"假设(?:的)?(?:结果|数值|数据)|预期(?:结果)?|预计|尚?未(?:运行|实测|测量|观测|执行)|"
    r"未有实际结果)",
    re.IGNORECASE,
)
_R4_ANCHOR_NEGATION = re.compile(
    r"\b(?:do\s+not\s+use|don't\s+use|never\s+use|without|exclude|excluding|not\s+use)\b|"
    r"(?:不使用|不采用|禁用|禁止使用|排除|切勿使用|不要使用)",
    re.IGNORECASE,
)
_R4_DEF_NEGATION = re.compile(
    r"\b(?:not\s+defined\s+as|is\s+not\s+a\s+definition)\b|"
    r"(?:并非[^，。！？；\n]{0,16}定义|不是[^，。！？；\n]{0,16}定义|不构成[^，。！？；\n]{0,16}定义)",
    re.IGNORECASE,
)
_R4_DEMO_NEGATION = re.compile(
    r"\b(?:do\s+not|does\s+not|did\s+not|don't|doesn't|didn't|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|"
    r"cannot|can't|never|unrelated\s+to|independent\s+of|"
    r"not\s+(?:a|an|the)?\s*(?:demo|demonstration|experiment|verification|"
    r"implementation|definition|part\s+of)|"
    r"not\s+(?:treat|implement|demonstrate|verify|validate|define|use|for)|"
    r"disclaimer|caveat)\b|"
    r"(?:并非|不是|不作为|不要把|切勿把|勿将|不得视作|不代表|不构成|"
    r"未实现|未演示|未验证|未采用|未使用|不用作|不用于|不涉及|"
    r"非[^，。！？；\n]{0,64}(?:实验|演示|验证|实现|方法|定义)|"
    r"无关|排除|禁用|禁止使用)",
    re.IGNORECASE,
)

_EXP_CUE = re.compile(
    r"实验|验证|演示|实现|测试|复现|函数|接口|代码|脚本|数据|算子|审计|遥测|日志|仿真|模拟|评估|参见|见|"
    r"notebook|ipynb|experiment|exp|demo|verif|validat|\blab\b|implement|code|script|"
    r"benchmark|dataset|trace|metric|run|probe|audit|kernel|simulat|evaluat|function|routine|helper",
    re.IGNORECASE,
)
_DEMO_EXPLICIT_CUE = re.compile(
    r"(?:^|[#\s/*])(?:demo|demonstrat(?:e|es|ing|ion)|experiment|verify|verification|"
    r"validation|lab|test\s+case)\b|(?:演示|实验验证|验证|实验演示|实验记录|实测)",
    re.IGNORECASE,
)
_STRUCTURED_EXP_CUE = re.compile(
    r"(?:\bExperiment[- ]ID\s*[:：]|\bMetric\s*[:：]|\bSplit\s*[:：]|实验(?:编号|代号)\s*[:：])",
    re.IGNORECASE,
)
_TODO_ONLY = re.compile(r"\bTODO\b|\bFIXME\b", re.IGNORECASE)

_ANCHOR_BACKTICK = re.compile(r"`([A-Za-z0-9_.-]{3,64})`")
_ANCHOR_SNAKE = re.compile(r"(?<![\w.-])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![\w.-])")
_ANCHOR_DOTTED = re.compile(r"(?<![\w.-])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w.-])")
_ANCHOR_IDENT = re.compile(
    r"(?<![\w.-])([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+"
    r"|[A-Z]{2,}[A-Za-z0-9]*\d+[A-Za-z0-9_-]*)(?![\w.-])"
)
_GENERIC_SYMBOL_STOPS = frozenset(
    {
        "accuracy",
        "loss",
        "plot",
        "test",
        "eval",
        "train",
        "val",
        "result",
        "results",
        "output",
        "outputs",
        "data",
        "value",
        "values",
        "score",
        "scores",
        "metric",
        "metrics",
        "baseline",
        "model",
        "table-1",
        "table-2",
        "table-3",
        "table-4",
        "figure-1",
        "figure-2",
        "figure-3",
        "fig-1",
        "fig-2",
        "page-1",
        "page-2",
        "slide-1",
        "slide-2",
        "section-1",
        "section-2",
        "e-mail",
        "co-author",
        "pre-trained",
        "fine-tuning",
        "fine-tuned",
        "open-source",
        "high-level",
        "low-rank",
        "real-time",
        "end-to-end",
        "self-attention",
        "cross-attention",
        "zero-shot",
        "few-shot",
        "top-1",
        "top-2",
        "top-k",
        "v1",
        "v2",
        "v3",
    }
)


@dataclass(frozen=True)
class KgConceptSummary:
    course_id: str
    concept_key: str
    canonical_name: str
    aliases: tuple[str, ...]
    anchor_terms: tuple[str, ...]
    defined_in_count: int
    demonstrated_by_count: int


@dataclass(frozen=True)
class KgEdgeRecord:
    edge_id: int
    course_id: str
    concept_key: str
    relation_type: Literal["defined_in", "demonstrated_by"]
    document_id: str
    version: int
    source_id: str
    publication_epoch: int
    content_sha256: str
    evidence_span_start: int
    evidence_span_end: int
    evidence_excerpt: str
    status: str
    evidence_rule: str
    rejection_reason: str


@dataclass(frozen=True)
class _ExtractedDef:
    concept_key: str
    canonical_name: str
    aliases: tuple[str, ...]
    anchor_terms: tuple[str, ...]
    source_id: str
    version: int
    publication_epoch: int
    content_sha256: str
    span_start: int
    span_end: int
    excerpt: str
    status: Literal["accepted", "rejected"]
    evidence_rule: str
    rejection_reason: str


@dataclass(frozen=True)
class _ActiveConcept:
    course_id: str
    concept_key: str
    canonical_name: str
    aliases: tuple[str, ...]
    anchor_terms: tuple[str, ...]


def source_text_for_kg(text: str, kind: str, locator: dict[str, Any]) -> str:
    """Return authoritative native text matching workspace_evidence.readable_source_text."""
    if text.strip():
        return text
    if (
        kind == "page_object"
        and locator.get("object_type") in {"caption", "formula"}
        and isinstance(object_text := locator.get("object_text"), str)
    ):
        return object_text
    return ""


def _can_display_locator(kind: str, locator: dict[str, Any]) -> bool:
    if kind in {"page", "page_object"}:
        return True
    if kind != "notebook_output":
        return False
    return any(
        isinstance(item, dict)
        and item.get("status") == "available"
        and item.get("mime_type") in _DISPLAYABLE_MIMES
        for item in locator.get("representations", [])
    )


def _now() -> float:
    return datetime.now(UTC).timestamp()


def _collapse_pdf_breaks(text: str) -> str:
    collapsed = re.sub(r"(?<=[\u3400-\u9fff])[ \t]*\r?\n[ \t]*(?=[\u3400-\u9fff])", "", text)
    collapsed = re.sub(r"[ \t]*\r?\n[ \t]*", " ", collapsed)
    return re.sub(r"[ \t]{2,}", " ", collapsed).strip()


_LEADING_SECTION_NUM = re.compile(
    r"^(?:§[ \t]*\d+(?:\.\d+)*[.)、]?[ \t]*|\d+(?:\.\d+)+[.)、]?[ \t]*"
    r"|\d+[.)、][ \t]+|\(\d+\)[ \t]*|（[一二三四五六七八九十]+）[ \t]*)"
)
_LEADING_ARTICLE_EN = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _clean_term_and_aliases(raw_name: str) -> tuple[str, tuple[str, ...]] | None:
    cleaned = _collapse_pdf_breaks(raw_name)
    cleaned = _LEADING_SECTION_NUM.sub("", cleaned)
    cleaned = _LEADING_FILLER_ZH.sub("", cleaned)
    cleaned = _LEADING_FILLER_EN.sub("", cleaned)
    cleaned = _LEADING_ARTICLE_EN.sub("", cleaned)
    cleaned = re.sub(r"([（(])[ \t]+", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([)）])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t:：—–-·•\"'“”‘’「」『』,，。.")
    if not cleaned:
        return None
    aliases: list[str] = []
    primary = cleaned
    if parens := _PARENS_ALIAS.match(cleaned):
        main = parens.group("main").strip(" \t:：—–-·•\"'“”‘’")
        inner = parens.group("alias").strip()
        if len(main) >= 2:
            primary = main
            aliases.append(cleaned)
            for part in re.split(r"[,;/，；]", inner):
                candidate = part.strip()
                if len(candidate) >= 2:
                    aliases.append(candidate)
    if len(primary) < 2 or len(primary) > 72:
        return None
    if primary.casefold() in _GENERIC_CONCEPT_NAMES or _OUTLINE_TITLE.match(primary):
        return None
    if _PAGE_OR_LOC_TITLE.match(primary) or _D1_BODY_LABEL.match(primary):
        return None
    if re.fullmatch(r"[\d\W_]+", primary):
        return None
    deduped = tuple(dict.fromkeys([primary, *aliases]))
    return primary, deduped


def _make_concept_key(primary_name: str) -> str:
    norm = unicodedata.normalize("NFKC", primary_name).strip().casefold()
    if re.fullmatch(r"[a-z0-9 _-]+", norm):
        slug = re.sub(r"[ _-]+", "_", norm).strip("_")
        if slug:
            return slug[:64]
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
    ascii_bits = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")[:24]
    return f"{ascii_bits}_{digest}" if ascii_bits else f"c_{digest}"


def _has_unclosed_parens(text: str) -> bool:
    return text.count("(") > text.count(")") or text.count("（") > text.count("）")


def _find_section_boundary_pos(raw_text: str, pos: int) -> int:
    """Return index of next genuine section boundary >= pos, skipping wrapped decimals."""
    scan = pos
    length = len(raw_text)
    while scan < length:
        match = _SECTION_BOUNDARY_LINE.search(raw_text, pos=scan)
        if match is None:
            return length
        after = match.groupdict().get("after")
        if after is None or not _NUMERIC_UNIT_OR_CONT.match(after.strip()):
            return match.start()
        scan = match.end()
    return length


def _is_section_boundary_line(line: str) -> bool:
    match = _SECTION_BOUNDARY_LINE.match(line)
    if match is None:
        return False
    after = match.groupdict().get("after")
    if after is None:
        return True
    return not _NUMERIC_UNIT_OR_CONT.match(after.strip())


def _strip_prev_title_prefix(raw_prev: str) -> str | None:
    collapsed = _collapse_pdf_breaks(raw_prev)
    if not collapsed or _OUTLINE_TITLE.match(collapsed) or _PAGE_OR_LOC_TITLE.match(collapsed):
        return None
    if m := _D1_HEADER_LINE.match(collapsed):
        candidate = m.group("rest").strip()
    else:
        candidate = _LEADING_SECTION_NUM.sub("", collapsed).strip()
    if len(candidate) < 2 or len(candidate) > 96:
        return None
    if _clean_term_and_aliases(candidate) is None:
        return None
    return candidate


def _split_logical_blocks(text: str) -> list[tuple[int, int]]:
    """Split text into logical paragraphs preserving single hard line breaks."""
    blocks: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    pos = 0
    cur_start: int | None = None
    cur_end = 0
    cur_lines: list[str] = []
    for line in lines:
        line_len = len(line)
        stripped = line.strip()
        if not stripped:
            if cur_start is not None:
                blocks.append((cur_start, cur_end))
                cur_start = None
                cur_lines = []
            pos += line_len
            continue
        is_new_d1 = bool(_D1_HEADER_LINE.match(line) or _D1_INLINE_RULE_LINE.match(line))
        is_sec = _is_section_boundary_line(line)
        should_split = cur_start is not None and (
            is_sec or (is_new_d1 and _should_split_before_d1(cur_lines, line))
        )
        if should_split:
            blocks.append((cur_start, cur_end))
            cur_start = pos
            cur_lines = [line]
        elif cur_start is None:
            cur_start = pos
            cur_lines = [line]
        else:
            cur_lines.append(line)
        cur_end = pos + line_len
        pos += line_len
    if cur_start is not None:
        blocks.append((cur_start, cur_end))
    return blocks


def _should_split_before_d1(cur_lines: Sequence[str], next_line: str) -> bool:
    if not cur_lines:
        return True
    prev_collapsed = _collapse_pdf_breaks("".join(cur_lines))
    if not prev_collapsed or len(cur_lines) > 3 or len(prev_collapsed) > 120:
        return True
    if _has_unclosed_parens(prev_collapsed):
        return False
    # If next line carries its own section numbering, treat it as a separate definition block.
    if _LEADING_SECTION_NUM.match(next_line.lstrip()):
        return True
    if re.search(r"[。！？；!?]$|(?<!Def)\.$", prev_collapsed):
        return True
    if _D1_INLINE_RULE_LINE.match(prev_collapsed) and not _D1_HEADER_LINE.match(prev_collapsed):
        return True
    if prev_m := _D1_HEADER_LINE.match(prev_collapsed):
        # Standalone header-only line (e.g. "1. 核心概念定义：<Concept>") binds to "正式定义：..."
        split_prev = _split_d1_rest(prev_m.group("rest"), None)
        if split_prev is None or split_prev[1] != "":
            return True
        if next_m := _D1_HEADER_LINE.match(next_line):
            next_rest = next_m.group("rest").strip()
            if re.search(r"\s+[—–]\s+|\s+--\s+", next_rest):
                return True
            return False
        return True
    return False


def _split_sentences_in_block(raw_text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split block into sentences at terminators without breaking at PDF single newlines."""
    segment = raw_text[start:end]
    spans: list[tuple[int, int]] = []
    sent_start = 0
    for match in re.finditer(r"[。！？；!?]|(?<!\d)\.(?!\d|[A-Za-z_])(?=\s|$)", segment):
        sent_end = match.end()
        if segment[sent_start:sent_end].strip():
            spans.append((start + sent_start, start + sent_end))
        sent_start = sent_end
    if segment[sent_start:].strip():
        spans.append((start + sent_start, end))
    return spans


def _extract_anchor_symbols(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_ANCHOR_BACKTICK, _ANCHOR_SNAKE, _ANCHOR_DOTTED, _ANCHOR_IDENT):
        for match in pattern.finditer(text):
            sym = match.group(1).strip()
            low = sym.casefold()
            if len(sym) < 3 or low in _GENERIC_SYMBOL_STOPS:
                continue
            if low.endswith((".pdf", ".pptx", ".ipynb", ".png", ".jpg")):
                continue
            if low not in seen:
                seen.add(low)
                found.append(sym)
    return found


def _extract_companion_anchors(
    raw_text: str, def_start: int, boundary_end: int
) -> tuple[str, ...]:
    """Scan the entire concept scope [def_start, boundary_end) for experimental anchors."""
    anchors: list[str] = []
    seen: set[str] = set()
    for b_start, b_end in _split_logical_blocks(raw_text[def_start:boundary_end]):
        abs_start = def_start + b_start
        abs_end = def_start + b_end
        for s_start, s_end in _split_sentences_in_block(raw_text, abs_start, abs_end):
            sentence = _collapse_pdf_breaks(raw_text[s_start:s_end])
            if not sentence:
                continue
            is_first = s_start == def_start
            if not is_first and not _EXP_CUE.search(sentence):
                continue
            if _R2_MENTION.search(sentence) or _R4_ANCHOR_NEGATION.search(sentence):
                continue
            for sym in _extract_anchor_symbols(sentence):
                low = sym.casefold()
                if low not in seen:
                    seen.add(low)
                    anchors.append(sym)
    return tuple(anchors)


def _find_valid_inline_sep(stripped: str) -> tuple[str, str] | None:
    for sep in _D1_INLINE_SEP.finditer(stripped):
        left = stripped[: sep.start()].strip()
        right = stripped[sep.end() :].strip()
        if len(left) < 2 or len(right) < 6:
            continue
        if left.count("(") != left.count(")") or left.count("（") != left.count("）"):
            continue
        if _MATH_EXPR_HINT.search(left):
            continue
        if sep.group(0).strip() == "-" and (
            _SUBTRACT_LEFT_END.search(left) or _SUBTRACT_RIGHT_START.match(right)
        ):
            continue
        if _clean_term_and_aliases(left) is None:
            continue
        return left, right
    return None


def _split_d1_rest(
    rest: str,
    prev_title: str | None,
    *,
    allow_body_prev_bind: bool = False,
    prev_header_locked: bool = False,
) -> tuple[str, str] | None:
    stripped = rest.strip()
    if not stripped:
        return None
    if prev_header_locked and prev_title and _has_min_def_body(stripped):
        return prev_title, stripped
    if prev_title and (_D1_CONDITION_START.match(stripped) or _D1_BODY_START.match(stripped)):
        return prev_title, stripped
    if inline_pair := _find_valid_inline_sep(stripped):
        return inline_pair
    if verb_match := _D1_BODY_VERB_SPLIT.match(stripped):
        left = verb_match.group("name").strip()
        right = stripped[len(left) :].strip()
        if (
            len(left) >= 2
            and len(right) >= 8
            and not _MATH_EXPR_HINT.search(left)
            and _clean_term_and_aliases(left) is not None
        ):
            return left, right
    if allow_body_prev_bind and prev_title and _has_min_def_body(stripped):
        return prev_title, stripped
    if (
        len(stripped) <= 96
        and not _has_unclosed_parens(stripped)
        and not _MATH_EXPR_HINT.search(stripped)
        and not re.search(r"[。！？.!?]$", stripped)
        and _clean_term_and_aliases(stripped) is not None
    ):
        return stripped, ""
    return None


def _trim_span(raw_text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1
    if end - start > _MAX_EXCERPT_CHARS:
        end = start + _MAX_EXCERPT_CHARS
        while end > start and raw_text[end - 1].isspace():
            end -= 1
    return start, end, raw_text[start:end]


def _has_min_def_body(text: str) -> bool:
    collapsed = _collapse_pdf_breaks(text)
    threshold = 8 if re.search(r"[\u3400-\u9fff]", collapsed) else _MIN_DEF_BODY_CHARS
    return len(collapsed) >= threshold


def _is_outline_source(raw_text: str, title: str) -> bool:
    first = next((line.strip() for line in raw_text.splitlines() if line.strip()), "")
    if _OUTLINE_TITLE.match(title.strip()) or _OUTLINE_TITLE.match(first):
        if not (
            _D1_HEADER_LINE.search(raw_text)
            or _D2_ZH.search(raw_text)
            or _D2_EN.search(raw_text)
        ):
            return True
    return False


def _extract_definitions_from_source(
    *,
    source_id: str,
    version: int,
    publication_epoch: int,
    content_sha256: str,
    title: str,
    raw_text: str,
    section_path: Sequence[str],
) -> list[_ExtractedDef]:
    if not raw_text.strip():
        return []
    if any(_R2_SECTION.match(str(part).strip()) for part in section_path):
        return []
    if _is_outline_source(raw_text, title):
        return []

    raw_candidates: list[tuple[int, int, str, str, str]] = []
    blocks = _split_logical_blocks(raw_text)
    pending_header: tuple[int, str, bool] | None = None
    for b_start, b_end in blocks:
        block = raw_text[b_start:b_end]
        lines = block.splitlines(keepends=True)
        rel = 0
        matched_d1 = False
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            line_abs = b_start + rel
            line_rel_end = rel + len(line)
            prev_title = None
            prev_abs = line_abs
            prev_from_block = False
            prev_header_locked = False
            if idx >= 1:
                prev_raw = "".join(lines[:idx])
                if candidate_prev := _strip_prev_title_prefix(prev_raw):
                    prev_title = candidate_prev
                    prev_abs = b_start
                    prev_from_block = True
                    prev_header_locked = bool(
                        _D1_HEADER_LINE.match(_collapse_pdf_breaks(prev_raw))
                    )
            elif idx == 0 and pending_header is not None:
                prev_abs, prev_title, prev_header_locked = pending_header
                prev_from_block = True
            elif idx == 0:
                if candidate_doc_title := _strip_prev_title_prefix(title):
                    prev_title = candidate_doc_title

            # Merge wrapped parenthesized alias lines (e.g., "(Adaptive Notch\nFilter, ANF)")
            header_lines = [line]
            end_idx = idx + 1
            header_rel_end = line_rel_end
            header_line_text = _collapse_pdf_breaks(line)
            while (
                _D1_HEADER_LINE.match(header_line_text)
                and _has_unclosed_parens(header_line_text)
                and end_idx < len(lines)
                and (end_idx - idx) < 4
            ):
                nxt = lines[end_idx]
                header_lines.append(nxt)
                header_rel_end += len(nxt)
                end_idx += 1
                header_line_text = _collapse_pdf_breaks("".join(header_lines))
                if not _has_unclosed_parens(header_line_text):
                    break

            if m := _D1_HEADER_LINE.match(header_line_text):
                split_res = _split_d1_rest(
                    m.group("rest"),
                    prev_title,
                    allow_body_prev_bind=prev_from_block,
                    prev_header_locked=prev_header_locked,
                )
                if split_res is not None:
                    term_raw, body_inline = split_res
                    following = (
                        block[header_rel_end:].strip()
                        if not body_inline
                        else block[header_rel_end:]
                    )
                    body_full = (
                        f"{body_inline} {following}".strip() if following else body_inline
                    )
                    if _has_min_def_body(body_full):
                        span_start = (
                            prev_abs if (term_raw == prev_title and prev_from_block) else line_abs
                        )
                        ctx = raw_text[span_start:b_end]
                        raw_candidates.append(
                            (span_start, b_end, term_raw, "explicit_definition_header", ctx)
                        )
                        matched_d1 = True
                        pending_header = None
                        break
            elif m := _D1_INLINE_RULE_LINE.match(line):
                if _clean_term_and_aliases(m.group("name")) is not None:
                    body_full = f"{m.group('body')} {block[line_rel_end:]}".strip()
                    if _has_min_def_body(body_full):
                        raw_candidates.append(
                            (line_abs, b_end, m.group("name"), "explicit_rule_header", block)
                        )
                        matched_d1 = True
                        pending_header = None
                        break
            rel = line_rel_end
            idx += 1

        if matched_d1:
            continue

        found_d2 = False
        for s_start, s_end in _split_sentences_in_block(raw_text, b_start, b_end):
            sentence = raw_text[s_start:s_end]
            for pattern, rule_name in ((_D2_ZH, "normative_zh"), (_D2_EN, "normative_en")):
                if m := pattern.search(sentence):
                    raw_candidates.append(
                        (
                            s_start + m.start(),
                            s_start + m.end(),
                            m.group("name"),
                            rule_name,
                            sentence,
                        )
                    )
                    found_d2 = True
                    break
        if found_d2:
            pending_header = None
        else:
            collapsed_block = _collapse_pdf_breaks(block)
            if (
                len(collapsed_block) <= 120
                and not re.search(r"[。！？；!?]$|(?<!Def)\.$", collapsed_block)
                and (cand := _strip_prev_title_prefix(block)) is not None
            ):
                pending_header = (
                    b_start,
                    cand,
                    bool(_D1_HEADER_LINE.match(collapsed_block)),
                )
            else:
                pending_header = None

    if not raw_candidates:
        return []

    raw_candidates.sort(key=lambda item: (item[0], item[1]))
    results: list[_ExtractedDef] = []
    for idx, (c_start, c_end, term_raw, rule_name, ctx_text) in enumerate(raw_candidates):
        cleaned = _clean_term_and_aliases(term_raw)
        if cleaned is None:
            continue
        primary, aliases = cleaned
        concept_key = _make_concept_key(primary)
        s_start, s_end, excerpt = _trim_span(raw_text, c_start, c_end)
        if not excerpt:
            continue
        next_def_start = (
            raw_candidates[idx + 1][0] if idx + 1 < len(raw_candidates) else len(raw_text)
        )
        sec_pos = _find_section_boundary_pos(raw_text, min(s_end, len(raw_text)))
        boundary_end = max(min(next_def_start, sec_pos), s_end)

        ctx_norm = _collapse_pdf_breaks(ctx_text)
        status: Literal["accepted", "rejected"] = "accepted"
        rejection_reason = ""
        if _R2_MENTION.search(ctx_norm):
            status = "rejected"
            rejection_reason = "citation_or_comparative_mention"
        elif _R3_UNOBSERVED.search(ctx_norm):
            status = "rejected"
            rejection_reason = "hypothetical_or_unexecuted_plan"
        elif _R4_DEF_NEGATION.search(ctx_norm):
            status = "rejected"
            rejection_reason = "negated_or_mismatched_condition"

        anchors = (
            _extract_companion_anchors(raw_text, s_start, boundary_end)
            if status == "accepted"
            else ()
        )
        results.append(
            _ExtractedDef(
                concept_key=concept_key,
                canonical_name=primary,
                aliases=aliases,
                anchor_terms=anchors,
                source_id=source_id,
                version=version,
                publication_epoch=publication_epoch,
                content_sha256=content_sha256,
                span_start=s_start,
                span_end=s_end,
                excerpt=excerpt,
                status=status,
                evidence_rule=rule_name,
                rejection_reason=rejection_reason,
            )
        )
    return results


def _contains_concept_term(text: str, concept: _ActiveConcept) -> re.Match[str] | None:
    candidates = sorted(
        {concept.canonical_name, *concept.aliases}, key=len, reverse=True
    )
    for term in candidates:
        if len(term) < 2:
            continue
        if re.search(r"[\u3400-\u9fff]", term):
            m = re.search(re.escape(term), text, re.IGNORECASE)
        else:
            m = re.search(rf"(?<![\w.-]){re.escape(term)}(?![\w.-])", text, re.IGNORECASE)
        if m is not None:
            return m
    return None


def _contains_anchor_symbol(text: str, symbol: str) -> re.Match[str] | None:
    if len(symbol) < 3 or symbol.casefold() in _GENERIC_SYMBOL_STOPS:
        return None
    pattern = rf"(?<![\w.-]){re.escape(symbol)}(?![\w.-])"
    return re.search(pattern, text, re.IGNORECASE)


def _strip_python_comments(code: str) -> str:
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        if "#" in line:
            # Remove comment tail only when outside obvious quotes (simple conservative strip).
            idx = _find_comment_hash(line)
            if idx >= 0:
                cleaned_lines.append(line[:idx])
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _find_comment_hash(line: str) -> int:
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote is not None:
            escaped = True
            continue
        if ch in {"'", '"'}:
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
        elif ch == "#" and quote is None:
            return idx
    return -1


def _extract_comment_lines(code: str) -> list[str]:
    comments: list[str] = []
    for line in code.splitlines():
        idx = _find_comment_hash(line)
        if idx >= 0:
            comments.append(line[idx + 1 :].strip())
    return comments


def _has_negation_for_concept_or_symbol(
    lines: Sequence[str], concept: _ActiveConcept
) -> bool:
    for raw_line in lines:
        for sentence in re.split(r"[。！？；.!?;]+", raw_line):
            sent = sentence.strip()
            if not sent:
                continue
            mentions_concept = _contains_concept_term(sent, concept) is not None
            mentions_anchor = any(
                _contains_anchor_symbol(sent, sym) is not None for sym in concept.anchor_terms
            )
            if (mentions_concept or mentions_anchor) and _R4_DEMO_NEGATION.search(sent):
                return True
    return False


def _line_span_around_match(raw_text: str, match: re.Match[str]) -> tuple[int, int, str]:
    line_start = raw_text.rfind("\n", 0, match.start())
    start = 0 if line_start < 0 else line_start + 1
    end = min(len(raw_text), max(match.end(), start + _MAX_EXCERPT_CHARS))
    # Prefer ending at a clean line boundary within _MAX_EXCERPT_CHARS.
    line_end = raw_text.find("\n", match.end(), start + _MAX_EXCERPT_CHARS)
    if line_end > match.end():
        end = line_end
    return _trim_span(raw_text, start, end)


def _upsert_concept_row(
    db: Connection,
    *,
    course_id: str,
    concept_key: str,
    canonical_name: str,
    aliases: Sequence[str],
    anchor_terms: Sequence[str],
    active: int,
    now: float,
) -> None:
    db.execute(
        """
        INSERT INTO local_kg_concepts(
            course_id, concept_key, canonical_name, aliases_json,
            anchor_terms_json, active, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(course_id, concept_key) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            aliases_json = excluded.aliases_json,
            anchor_terms_json = excluded.anchor_terms_json,
            active = excluded.active,
            updated_at = excluded.updated_at
        """,
        (
            course_id,
            concept_key,
            canonical_name,
            json.dumps(list(aliases), ensure_ascii=False),
            json.dumps(list(anchor_terms), ensure_ascii=False),
            active,
            now,
        ),
    )


def _upsert_edge_row(
    db: Connection,
    *,
    course_id: str,
    concept_key: str,
    relation_type: Literal["defined_in", "demonstrated_by"],
    document_id: str,
    version: int,
    source_id: str,
    publication_epoch: int,
    content_sha256: str,
    span_start: int,
    span_end: int,
    excerpt: str,
    status: Literal["candidate", "accepted", "rejected", "superseded"],
    evidence_rule: str,
    rejection_reason: str,
    now: float,
) -> None:
    db.execute(
        """
        INSERT INTO local_kg_edges(
            course_id, concept_key, relation_type, document_id, version, source_id,
            publication_epoch, content_sha256, evidence_span_start, evidence_span_end,
            evidence_excerpt, status, evidence_rule, rejection_reason, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(course_id, concept_key, relation_type, document_id, version, source_id)
        DO UPDATE SET
            publication_epoch = excluded.publication_epoch,
            content_sha256 = excluded.content_sha256,
            evidence_span_start = excluded.evidence_span_start,
            evidence_span_end = excluded.evidence_span_end,
            evidence_excerpt = excluded.evidence_excerpt,
            status = excluded.status,
            evidence_rule = excluded.evidence_rule,
            rejection_reason = excluded.rejection_reason
        """,
        (
            course_id,
            concept_key,
            relation_type,
            document_id,
            version,
            source_id,
            publication_epoch,
            content_sha256,
            span_start,
            span_end,
            excerpt,
            status,
            evidence_rule,
            rejection_reason,
            now,
        ),
    )


def _has_kg_tables(db: Connection) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'local_kg_edges'"
        ).fetchone()
        is not None
    )


def invalidate_document_kg_in_tx(
    db: Connection, document_id: str, *, reason: str = "document_deleted"
) -> None:
    """Mark all live/candidate KG edges for a deleted or superseded document as superseded."""
    if not _has_kg_tables(db):
        return
    db.execute(
        """
        UPDATE local_kg_edges
        SET status = 'superseded',
            rejection_reason = CASE
                WHEN rejection_reason = '' THEN ?
                ELSE rejection_reason
            END
        WHERE document_id = ? AND status IN ('candidate', 'accepted')
        """,
        (reason, document_id),
    )
    db.execute("DELETE FROM local_kg_doc_state WHERE document_id = ?", (document_id,))


def invalidate_stale_edges_on_promote_in_tx(db: Connection, document_ids: Sequence[str]) -> None:
    """Immediately mark old version edges superseded when promote_due flips current_version."""
    if not document_ids or not _has_kg_tables(db):
        return
    marks = ", ".join("?" for _ in document_ids)
    db.execute(
        f"""
        UPDATE local_kg_edges
        SET status = 'superseded',
            rejection_reason = CASE
                WHEN rejection_reason = '' THEN 'version_superseded'
                ELSE rejection_reason
            END
        WHERE document_id IN ({marks})
          AND status IN ('candidate', 'accepted')
          AND NOT EXISTS (
              SELECT 1 FROM local_documents d
              WHERE d.document_id = local_kg_edges.document_id
                AND d.deleted = 0
                AND d.current_version = local_kg_edges.version
          )
        """,
        list(document_ids),
    )


def sync_course_kg_in_tx(db: Connection, course_id: str, *, now: float | None = None) -> None:
    """Recompute all active concepts, anchors, and edges for one course inside an active TX."""
    if not _has_kg_tables(db):
        return
    ts = _now() if now is None else now
    # Step 0: Supersede any candidate/accepted edges not matching current visible version/epoch.
    db.execute(
        """
        UPDATE local_kg_edges
        SET status = 'superseded',
            rejection_reason = CASE
                WHEN rejection_reason = '' THEN 'version_superseded'
                ELSE rejection_reason
            END
        WHERE course_id = ?
          AND status IN ('candidate', 'accepted')
          AND NOT EXISTS (
              SELECT 1 FROM local_documents d
              JOIN local_versions v
                ON v.document_id = d.document_id AND v.version = d.current_version
              WHERE d.document_id = local_kg_edges.document_id
                AND d.course_id = local_kg_edges.course_id
                AND d.deleted = 0
                AND d.current_version = local_kg_edges.version
                AND d.publication_epoch = local_kg_edges.publication_epoch
                AND v.content_sha256 = local_kg_edges.content_sha256
                AND v.published_at IS NOT NULL
          )
        """,
        (course_id,),
    )

    active_docs = db.execute(
        """
        SELECT d.document_id, d.course_id, d.current_version AS version,
               d.publication_epoch, v.content_sha256, v.document_format
        FROM local_documents d
        JOIN local_versions v ON v.document_id = d.document_id AND v.version = d.current_version
        WHERE d.course_id = ? AND d.deleted = 0 AND d.current_version IS NOT NULL
          AND v.published_at IS NOT NULL
        ORDER BY d.document_id
        """,
        (course_id,),
    ).fetchall()

    # Clean doc_state for deleted/unpublished docs in this course.
    active_doc_ids = {row["document_id"] for row in active_docs}
    existing_states = db.execute(
        "SELECT document_id FROM local_kg_doc_state WHERE course_id = ?", (course_id,)
    ).fetchall()
    for state_row in existing_states:
        if state_row["document_id"] not in active_doc_ids:
            db.execute(
                "DELETE FROM local_kg_doc_state WHERE document_id = ?", (state_row["document_id"],)
            )

    if not active_docs:
        _retire_inactive_concepts(db, course_id, active_concepts={}, now=ts)
        return

    doc_map = {row["document_id"]: row for row in active_docs}
    doc_marks = ", ".join("?" for _ in active_docs)
    source_rows = db.execute(
        f"""
        SELECT s.*, d.publication_epoch, v.content_sha256, v.document_format
        FROM local_sources s
        JOIN local_documents d ON d.document_id = s.document_id
            AND d.current_version = s.version AND d.deleted = 0
        JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
        WHERE s.document_id IN ({doc_marks})
        ORDER BY s.document_id, s.ordinal, s.source_id
        """,
        list(doc_map),
    ).fetchall()

    by_doc_source: dict[tuple[str, int, str], dict[str, Any]] = {}
    outputs_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    cells_by_doc: dict[str, list[dict[str, Any]]] = {}
    parsed_sources: list[dict[str, Any]] = []
    for row in source_rows:
        locator = json.loads(row["locator_json"])
        raw_text = source_text_for_kg(row["text"], row["kind"], locator)
        entry = {
            "document_id": row["document_id"],
            "version": row["version"],
            "source_id": row["source_id"],
            "parent_id": row["parent_id"],
            "ordinal": row["ordinal"],
            "kind": row["kind"],
            "title": row["title"],
            "text": raw_text,
            "locator": locator,
            "state": row["state"],
            "publication_epoch": row["publication_epoch"],
            "content_sha256": row["content_sha256"],
            "document_format": row["document_format"],
        }
        parsed_sources.append(entry)
        by_doc_source[(row["document_id"], row["version"], row["source_id"])] = entry
        if row["kind"] == "notebook_output" and row["parent_id"]:
            outputs_by_cell.setdefault(
                (row["document_id"], row["version"], row["parent_id"]), []
            ).append(entry)
        if row["kind"] == "cell":
            cells_by_doc.setdefault(row["document_id"], []).append(entry)

    # Pass 1: Extract defined_in edges from active non-code sources & merge concept definitions.
    concept_defs: dict[str, list[tuple[str, _ExtractedDef]]] = {}
    alias_to_key: dict[str, str] = {}
    for entry in parsed_sources:
        kind = entry["kind"]
        locator = entry["locator"]
        if kind not in {"page", "page_object", "cell"}:
            continue
        if kind == "cell" and locator.get("cell_type") != "markdown":
            continue
        if (
            kind == "page_object"
            and locator.get("format") == "pptx"
            and locator.get("object_type") in {"title", "text"}
            and entry["parent_id"]
        ):
            parent = by_doc_source.get((entry["document_id"], entry["version"], entry["parent_id"]))
            if parent is not None and entry["text"].strip() in parent["text"]:
                continue

        extracted = _extract_definitions_from_source(
            source_id=entry["source_id"],
            version=entry["version"],
            publication_epoch=entry["publication_epoch"],
            content_sha256=entry["content_sha256"],
            title=entry["title"],
            raw_text=entry["text"],
            section_path=locator.get("section_path") or (),
        )
        for item in extracted:
            target_key = item.concept_key
            if item.status == "accepted":
                for al in (item.canonical_name, *item.aliases):
                    mapped = alias_to_key.get(al.casefold())
                    if mapped is not None:
                        target_key = mapped
                        break
                for al in (item.canonical_name, *item.aliases):
                    alias_to_key.setdefault(al.casefold(), target_key)
            concept_defs.setdefault(target_key, []).append((entry["document_id"], item))

    # Full recomputation of local_kg_concepts for active defined_in concepts.
    active_concepts: dict[str, _ActiveConcept] = {}
    for concept_key, entries in concept_defs.items():
        accepted_items = [item for _, item in entries if item.status == "accepted"]
        if accepted_items:
            canonical = accepted_items[0].canonical_name
            aliases = list(dict.fromkeys(al for item in accepted_items for al in item.aliases))
            anchors = list(
                dict.fromkeys(sym for item in accepted_items for sym in item.anchor_terms)
            )
            active_flag = 1
            active_concepts[concept_key] = _ActiveConcept(
                course_id=course_id,
                concept_key=concept_key,
                canonical_name=canonical,
                aliases=tuple(aliases),
                anchor_terms=tuple(anchors),
            )
        else:
            canonical = entries[0][1].canonical_name
            aliases = []
            anchors = []
            active_flag = 0
        _upsert_concept_row(
            db,
            course_id=course_id,
            concept_key=concept_key,
            canonical_name=canonical,
            aliases=aliases,
            anchor_terms=anchors,
            active=active_flag,
            now=ts,
        )
        for doc_id, item in entries:
            _upsert_edge_row(
                db,
                course_id=course_id,
                concept_key=concept_key,
                relation_type="defined_in",
                document_id=doc_id,
                version=item.version,
                source_id=item.source_id,
                publication_epoch=item.publication_epoch,
                content_sha256=item.content_sha256,
                span_start=item.span_start,
                span_end=item.span_end,
                excerpt=item.excerpt,
                status=item.status,
                evidence_rule=item.evidence_rule,
                rejection_reason=item.rejection_reason,
                now=ts,
            )

    # Mark stale defined_in edges on current docs that were removed/changed inside same version.
    current_def_keys = {
        (concept_key, doc_id, item.version, item.source_id)
        for concept_key, entries in concept_defs.items()
        for doc_id, item in entries
    }
    live_def_rows = db.execute(
        """
        SELECT edge_id, concept_key, document_id, version, source_id
        FROM local_kg_edges
        WHERE course_id = ? AND relation_type = 'defined_in' AND status IN ('accepted', 'candidate')
        """,
        (course_id,),
    ).fetchall()
    for row in live_def_rows:
        def_key = (row["concept_key"], row["document_id"], row["version"], row["source_id"])
        if def_key not in current_def_keys:
            db.execute(
                "UPDATE local_kg_edges "
                "SET status = 'superseded', rejection_reason = 'definition_removed' "
                "WHERE edge_id = ?",
                (row["edge_id"],),
            )

    _retire_inactive_concepts(db, course_id, active_concepts=active_concepts, now=ts)

    # Pass 2: Evaluate demonstrated_by edges & anti-examples over active concepts.
    def_edge_set = {
        (ck, d_id, it.version, it.source_id)
        for ck, ents in concept_defs.items()
        for d_id, it in ents
    }
    current_demo_keys: set[tuple[str, str, int, str]] = set()
    if active_concepts:
        # Build preceding markdown cell map for notebook cells.
        prev_md_by_cell: dict[tuple[str, int, str], str] = {}
        for doc_id, cells in cells_by_doc.items():
            last_md = ""
            for cell in cells:
                ctype = cell["locator"].get("cell_type")
                if ctype == "markdown":
                    last_md = cell["text"]
                elif ctype == "code":
                    prev_md_by_cell[(doc_id, cell["version"], cell["source_id"])] = last_md

        for entry in parsed_sources:
            doc_id = entry["document_id"]
            version = entry["version"]
            source_id = entry["source_id"]
            kind = entry["kind"]
            locator = entry["locator"]
            raw_text = entry["text"]

            if kind == "cell" and locator.get("cell_type") == "code":
                outputs = outputs_by_cell.get((doc_id, version, source_id), [])
                has_valid_output = locator.get("outputs_state") == "available" and any(
                    bool(o["text"].strip()) or _can_display_locator(o["kind"], o["locator"])
                    for o in outputs
                )
                prev_md = prev_md_by_cell.get((doc_id, version, source_id), "")
                comments = _extract_comment_lines(raw_text)
                comment_and_md_lines = [*comments, *prev_md.splitlines()]
                executable_code = _strip_python_comments(raw_text)
                output_texts = [o["text"] for o in outputs if o["text"].strip()]
                combined_output = "\n".join(output_texts)
                combined_all = "\n".join(filter(None, [prev_md, raw_text, combined_output]))

                for concept in active_concepts.values():
                    # R4 full-line / comment disclaimer guard blocks BOTH E1 and E2.
                    if _has_negation_for_concept_or_symbol(comment_and_md_lines, concept):
                        if m := (
                            _contains_concept_term(raw_text, concept)
                            or next(
                                (
                                    _contains_anchor_symbol(raw_text, sym)
                                    for sym in concept.anchor_terms
                                    if _contains_anchor_symbol(raw_text, sym) is not None
                                ),
                                None,
                            )
                        ):
                            s_start, s_end, excerpt = _line_span_around_match(raw_text, m)
                            _upsert_edge_row(
                                db,
                                course_id=course_id,
                                concept_key=concept.concept_key,
                                relation_type="demonstrated_by",
                                document_id=doc_id,
                                version=version,
                                source_id=source_id,
                                publication_epoch=entry["publication_epoch"],
                                content_sha256=entry["content_sha256"],
                                span_start=s_start,
                                span_end=s_end,
                                excerpt=excerpt,
                                status="rejected",
                                evidence_rule="negation_blocked",
                                rejection_reason="negated_or_mismatched_condition",
                                now=ts,
                            )
                            current_demo_keys.add(
                                (concept.concept_key, doc_id, version, source_id)
                            )
                        continue

                    # R3 hypothetical/unexecuted guard when outputs missing or explicitly planned.
                    concept_hit_in_code = _contains_concept_term(raw_text, concept)
                    concept_hit_in_md = _contains_concept_term(prev_md, concept)
                    anchor_in_code = next(
                        (
                            (sym, m)
                            for sym in concept.anchor_terms
                            if (m := _contains_anchor_symbol(executable_code, sym)) is not None
                        ),
                        None,
                    )
                    anchor_in_output = next(
                        (
                            (sym, o, m)
                            for o in outputs
                            for sym in concept.anchor_terms
                            if o["text"].strip()
                            and (m := _contains_anchor_symbol(o["text"], sym)) is not None
                        ),
                        None,
                    )
                    if (
                        (concept_hit_in_code or concept_hit_in_md or anchor_in_code)
                        and (not has_valid_output or _R3_UNOBSERVED.search(combined_all))
                    ):
                        target_m = concept_hit_in_code or (
                            anchor_in_code[1] if anchor_in_code else None
                        )
                        if target_m is not None and raw_text.strip():
                            s_start, s_end, excerpt = _line_span_around_match(raw_text, target_m)
                        elif raw_text.strip():
                            s_start, s_end, excerpt = _trim_span(raw_text, 0, len(raw_text))
                        else:
                            continue
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status="rejected",
                            evidence_rule="unexecuted_guard",
                            rejection_reason="hypothetical_or_unexecuted_plan",
                            now=ts,
                        )
                        current_demo_keys.add((concept.concept_key, doc_id, version, source_id))
                        continue

                    if not has_valid_output:
                        continue

                    # E2: Cross-file anchor symbol verified in executable code or output.
                    if anchor_in_code is not None:
                        _, sym_m = anchor_in_code
                        s_start, s_end, excerpt = _line_span_around_match(raw_text, sym_m)
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status="accepted",
                            evidence_rule="cross_file_symbol_verified",
                            rejection_reason="",
                            now=ts,
                        )
                        current_demo_keys.add((concept.concept_key, doc_id, version, source_id))
                        continue

                    if anchor_in_output is not None:
                        _, out_entry, out_m = anchor_in_output
                        s_start, s_end, excerpt = _line_span_around_match(out_entry["text"], out_m)
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=out_entry["source_id"],
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status="accepted",
                            evidence_rule="cross_file_symbol_output_verified",
                            rejection_reason="",
                            now=ts,
                        )
                        current_demo_keys.add(
                            (concept.concept_key, doc_id, version, out_entry["source_id"])
                        )
                        continue

                    # E1: Explicit demonstration cue in comments or immediately preceding markdown.
                    has_demo_cue_in_cell = any(
                        _DEMO_EXPLICIT_CUE.search(c)
                        and not _TODO_ONLY.search(c)
                        and _contains_concept_term(c, concept) is not None
                        for c in comments
                    )
                    has_demo_cue_in_md = bool(
                        concept_hit_in_md
                        and _DEMO_EXPLICIT_CUE.search(prev_md)
                        and not _TODO_ONLY.search(prev_md)
                    )
                    if (has_demo_cue_in_cell or has_demo_cue_in_md) and raw_text.strip():
                        if concept_hit_in_code is not None:
                            s_start, s_end, excerpt = _line_span_around_match(
                                raw_text, concept_hit_in_code
                            )
                        else:
                            s_start, s_end, excerpt = _trim_span(raw_text, 0, len(raw_text))
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status="accepted",
                            evidence_rule="explicit_demo_cell",
                            rejection_reason="",
                            now=ts,
                        )
                        current_demo_keys.add((concept.concept_key, doc_id, version, source_id))
                    elif concept_hit_in_code is not None:
                        # R1: Mere comment/TODO mention without demo cue or verified code logic.
                        s_start, s_end, excerpt = _line_span_around_match(
                            raw_text, concept_hit_in_code
                        )
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status="candidate",
                            evidence_rule="cooccurrence_only",
                            rejection_reason="cooccurrence_only",
                            now=ts,
                        )
                        current_demo_keys.add((concept.concept_key, doc_id, version, source_id))
            elif kind in {"page", "page_object", "cell"} and raw_text.strip():
                if (
                    kind == "page_object"
                    and locator.get("format") == "pptx"
                    and locator.get("object_type") in {"title", "text"}
                    and entry["parent_id"]
                ):
                    parent = by_doc_source.get((doc_id, version, entry["parent_id"]))
                    if parent is not None and raw_text.strip() in parent["text"]:
                        continue
                # Structured non-code experiment records or R1/R2/R3/R4 audit tracking.
                is_structured_exp = bool(_STRUCTURED_EXP_CUE.search(raw_text))
                for concept in active_concepts.values():
                    if (concept.concept_key, doc_id, version, source_id) in def_edge_set:
                        continue
                    m = _contains_concept_term(raw_text, concept)
                    if m is None:
                        continue
                    s_start, s_end, excerpt = _line_span_around_match(raw_text, m)
                    if is_structured_exp:
                        if _R3_UNOBSERVED.search(raw_text):
                            status_val: Literal["accepted", "rejected", "candidate"] = "rejected"
                            rej = "hypothetical_or_unexecuted_plan"
                            rule = "unexecuted_guard"
                        elif _has_negation_for_concept_or_symbol(raw_text.splitlines(), concept):
                            status_val = "rejected"
                            rej = "negated_or_mismatched_condition"
                            rule = "negation_blocked"
                        else:
                            status_val = "accepted"
                            rej = ""
                            rule = "explicit_experiment_record"
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="demonstrated_by",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status=status_val,
                            evidence_rule=rule,
                            rejection_reason=rej,
                            now=ts,
                        )
                        current_demo_keys.add((concept.concept_key, doc_id, version, source_id))
                    else:
                        # Record R2/R3/R4 or pure co-occurrence (never accepted).
                        if _R2_MENTION.search(excerpt) or any(
                            _R2_SECTION.match(str(p).strip())
                            for p in (locator.get("section_path") or ())
                        ):
                            co_status: Literal["rejected", "candidate"] = "rejected"
                            co_rej = "citation_or_comparative_mention"
                            co_rule = "citation_blocked"
                        elif _R3_UNOBSERVED.search(excerpt):
                            co_status = "rejected"
                            co_rej = "hypothetical_or_unexecuted_plan"
                            co_rule = "unexecuted_guard"
                        else:
                            co_status = "candidate"
                            co_rej = "cooccurrence_only"
                            co_rule = "cooccurrence_only"
                        _upsert_edge_row(
                            db,
                            course_id=course_id,
                            concept_key=concept.concept_key,
                            relation_type="defined_in",
                            document_id=doc_id,
                            version=version,
                            source_id=source_id,
                            publication_epoch=entry["publication_epoch"],
                            content_sha256=entry["content_sha256"],
                            span_start=s_start,
                            span_end=s_end,
                            excerpt=excerpt,
                            status=co_status,
                            evidence_rule=co_rule,
                            rejection_reason=co_rej,
                            now=ts,
                        )

    # Retire active/candidate demonstrated_by edges that lost active anchor or support.
    live_demo_rows = db.execute(
        """
        SELECT edge_id, concept_key, document_id, version, source_id
        FROM local_kg_edges
        WHERE course_id = ? AND relation_type = 'demonstrated_by'
          AND status IN ('accepted', 'candidate')
        """,
        (course_id,),
    ).fetchall()
    for row in live_demo_rows:
        demo_key = (row["concept_key"], row["document_id"], row["version"], row["source_id"])
        if demo_key not in current_demo_keys:
            db.execute(
                """
                UPDATE local_kg_edges
                SET status = 'superseded',
                    rejection_reason = 'definition_anchor_removed'
                WHERE edge_id = ?
                """,
                (row["edge_id"],),
            )

    # Update sync watermark for all active documents in course_id.
    for row in active_docs:
        db.execute(
            """
            INSERT INTO local_kg_doc_state(
                document_id, course_id, version, publication_epoch, content_sha256, synced_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                course_id = excluded.course_id,
                version = excluded.version,
                publication_epoch = excluded.publication_epoch,
                content_sha256 = excluded.content_sha256,
                synced_at = excluded.synced_at
            """,
            (
                row["document_id"],
                row["course_id"],
                row["version"],
                row["publication_epoch"],
                row["content_sha256"],
                ts,
            ),
        )


def _retire_inactive_concepts(
    db: Connection,
    course_id: str,
    *,
    active_concepts: dict[str, _ActiveConcept],
    now: float,
) -> None:
    rows = db.execute(
        "SELECT concept_key FROM local_kg_concepts WHERE course_id = ?", (course_id,)
    ).fetchall()
    for row in rows:
        key = row["concept_key"]
        if key in active_concepts:
            continue
        db.execute(
            """
            UPDATE local_kg_edges
            SET status = 'superseded',
                rejection_reason = CASE
                    WHEN rejection_reason = '' THEN 'definition_anchor_removed'
                    ELSE rejection_reason
                END
            WHERE course_id = ? AND concept_key = ? AND status IN ('candidate', 'accepted')
            """,
            (course_id, key),
        )
        has_edges = db.execute(
            "SELECT 1 FROM local_kg_edges WHERE course_id = ? AND concept_key = ? LIMIT 1",
            (course_id, key),
        ).fetchone()
        if has_edges is None:
            db.execute(
                "DELETE FROM local_kg_concepts WHERE course_id = ? AND concept_key = ?",
                (course_id, key),
            )
        else:
            db.execute(
                """
                UPDATE local_kg_concepts
                SET active = 0, aliases_json = '[]', anchor_terms_json = '[]', updated_at = ?
                WHERE course_id = ? AND concept_key = ?
                """,
                (now, course_id, key),
            )


class LocalKgStore(MetadataStore):
    """Transactional and snapshot-filtered SQLite KG store sharing LocalLibraryStore's DB."""

    def initialize(self) -> None:
        super().initialize()
        with self._connect() as db:
            db.executescript(KG_SCHEMA)

    def ensure_course_kg_synced(self, course_ids: Sequence[str]) -> list[str]:
        """Synchronize KG tables lazily & idempotently after promote_due or legacy migration."""
        courses = tuple(dict.fromkeys(c.strip() for c in course_ids if c and c.strip()))
        if not courses:
            return []
        cutoff = _now()
        with self._connect() as db:
            if not _has_kg_tables(db):
                return []
            promote_due(db, cutoff)
            dirty = _dirty_courses(db, courses)
            if not dirty:
                return []
            db.execute("BEGIN IMMEDIATE")
            promote_due(db, cutoff)
            dirty = _dirty_courses(db, courses)
            for course_id in dirty:
                sync_course_kg_in_tx(db, course_id, now=cutoff)
            return dirty

    def write_kg_edges(
        self,
        key: SourceKey,
        *,
        course_id: str,
        content_sha256: str,
        publication_epoch: int,
        concept_key: str,
        canonical_name: str,
        relation_type: Literal["defined_in", "demonstrated_by"],
        span_start: int,
        span_end: int,
        excerpt: str,
        status: Literal["candidate", "accepted", "rejected"] = "accepted",
        evidence_rule: str = "cas_write",
        rejection_reason: str = "",
    ) -> bool:
        """CAS-guarded late-writer entry symmetric to LocalLibraryStore.write_enhancement."""
        if not course_id or not excerpt.strip() or span_start < 0 or span_end <= span_start:
            return False
        cutoff = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            promote_due(db, cutoff)
            row = db.execute(
                """
                SELECT s.text, s.kind, s.locator_json
                FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                WHERE d.course_id = ? AND s.document_id = ? AND s.version = ? AND s.source_id = ?
                  AND d.publication_epoch = ? AND v.content_sha256 = ?
                  AND v.published_at IS NOT NULL
                """,
                (
                    course_id,
                    key.document_id,
                    key.version,
                    key.source_id,
                    publication_epoch,
                    content_sha256,
                ),
            ).fetchone()
            if row is None:
                return False
            src_text = source_text_for_kg(row["text"], row["kind"], json.loads(row["locator_json"]))
            if span_end > len(src_text) or src_text[span_start:span_end] != excerpt:
                return False
            existing_concept = db.execute(
                "SELECT aliases_json, anchor_terms_json FROM local_kg_concepts "
                "WHERE course_id = ? AND concept_key = ?",
                (course_id, concept_key),
            ).fetchone()
            aliases = (
                json.loads(existing_concept["aliases_json"])
                if existing_concept
                else [canonical_name]
            )
            anchors = (
                json.loads(existing_concept["anchor_terms_json"])
                if existing_concept
                else []
            )
            _upsert_concept_row(
                db,
                course_id=course_id,
                concept_key=concept_key,
                canonical_name=canonical_name,
                aliases=aliases,
                anchor_terms=anchors,
                active=1,
                now=cutoff,
            )
            _upsert_edge_row(
                db,
                course_id=course_id,
                concept_key=concept_key,
                relation_type=relation_type,
                document_id=key.document_id,
                version=key.version,
                source_id=key.source_id,
                publication_epoch=publication_epoch,
                content_sha256=content_sha256,
                span_start=span_start,
                span_end=span_end,
                excerpt=excerpt,
                status=status,
                evidence_rule=evidence_rule,
                rejection_reason=rejection_reason,
                now=cutoff,
            )
            return True

    def list_concepts(
        self,
        course_ids: Sequence[str],
        *,
        document_ids: Sequence[str] = (),
        query: str | None = None,
    ) -> list[KgConceptSummary]:
        courses = tuple(dict.fromkeys(c.strip() for c in course_ids if c and c.strip()))
        if not courses:
            return []
        self.ensure_course_kg_synced(courses)
        marks = ", ".join("?" for _ in courses)
        doc_filter = ""
        params: list[Any] = list(courses)
        if document_ids:
            d_marks = ", ".join("?" for _ in document_ids)
            doc_filter = f" AND e.document_id IN ({d_marks})"
            params.extend(document_ids)
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT c.course_id, c.concept_key, c.canonical_name, c.aliases_json,
                       c.anchor_terms_json,
                       sum(CASE WHEN e.relation_type = 'defined_in' THEN 1 ELSE 0 END) AS def_cnt,
                       sum(
                           CASE WHEN e.relation_type = 'demonstrated_by' THEN 1 ELSE 0 END
                       ) AS demo_cnt
                FROM local_kg_edges e
                JOIN local_kg_concepts c
                  ON c.course_id = e.course_id AND c.concept_key = e.concept_key AND c.active = 1
                JOIN local_documents d ON d.document_id = e.document_id
                JOIN local_versions v ON v.document_id = e.document_id AND v.version = e.version
                WHERE e.course_id IN ({marks})
                  AND e.status = 'accepted'
                  AND d.course_id = e.course_id
                  AND d.deleted = 0
                  AND d.current_version = e.version
                  AND d.publication_epoch = e.publication_epoch
                  AND v.published_at IS NOT NULL
                  AND v.content_sha256 = e.content_sha256
                  {doc_filter}
                GROUP BY c.course_id, c.concept_key
                ORDER BY c.course_id, c.canonical_name, c.concept_key
                """,
                params,
            ).fetchall()
        needle = query.strip().casefold() if query else ""
        items: list[KgConceptSummary] = []
        for row in rows:
            aliases = tuple(json.loads(row["aliases_json"]))
            anchors = tuple(json.loads(row["anchor_terms_json"]))
            if needle:
                haystack = " ".join(
                    (row["canonical_name"], *aliases, row["concept_key"])
                ).casefold()
                if needle not in haystack:
                    continue
            items.append(
                KgConceptSummary(
                    course_id=row["course_id"],
                    concept_key=row["concept_key"],
                    canonical_name=row["canonical_name"],
                    aliases=aliases,
                    anchor_terms=anchors,
                    defined_in_count=int(row["def_cnt"] or 0),
                    demonstrated_by_count=int(row["demo_cnt"] or 0),
                )
            )
        return items

    def concept_detail(
        self,
        course_ids: Sequence[str],
        concept_key: str,
    ) -> tuple[KgConceptSummary, list[KgEdgeRecord]] | None:
        courses = tuple(dict.fromkeys(c.strip() for c in course_ids if c and c.strip()))
        if not courses:
            return None
        self.ensure_course_kg_synced(courses)
        marks = ", ".join("?" for _ in courses)
        with self._connect() as db:
            concept_row = db.execute(
                f"""
                SELECT * FROM local_kg_concepts
                WHERE course_id IN ({marks}) AND concept_key = ? AND active = 1
                ORDER BY course_id
                LIMIT 1
                """,
                [*courses, concept_key],
            ).fetchone()
            if concept_row is None:
                return None
            target_course = concept_row["course_id"]
            edge_rows = db.execute(
                """
                SELECT e.*
                FROM local_kg_edges e
                JOIN local_documents d ON d.document_id = e.document_id
                JOIN local_versions v ON v.document_id = e.document_id AND v.version = e.version
                WHERE e.course_id = ?
                  AND e.concept_key = ?
                  AND e.status = 'accepted'
                  AND d.course_id = e.course_id
                  AND d.deleted = 0
                  AND d.current_version = e.version
                  AND d.publication_epoch = e.publication_epoch
                  AND v.published_at IS NOT NULL
                  AND v.content_sha256 = e.content_sha256
                ORDER BY e.relation_type, e.document_id, e.edge_id
                """,
                (target_course, concept_key),
            ).fetchall()
        if not edge_rows:
            return None
        edges = [_edge(row) for row in edge_rows]
        summary = KgConceptSummary(
            course_id=target_course,
            concept_key=concept_row["concept_key"],
            canonical_name=concept_row["canonical_name"],
            aliases=tuple(json.loads(concept_row["aliases_json"])),
            anchor_terms=tuple(json.loads(concept_row["anchor_terms_json"])),
            defined_in_count=sum(1 for e in edges if e.relation_type == "defined_in"),
            demonstrated_by_count=sum(1 for e in edges if e.relation_type == "demonstrated_by"),
        )
        return summary, edges

    def active_edges_for_courses(
        self, course_ids: Sequence[str]
    ) -> tuple[list[KgConceptSummary], list[KgEdgeRecord]]:
        """Read all current-version accepted concepts & edges across authorized courses."""
        courses = tuple(dict.fromkeys(c.strip() for c in course_ids if c and c.strip()))
        if not courses:
            return [], []
        self.ensure_course_kg_synced(courses)
        marks = ", ".join("?" for _ in courses)
        with self._connect() as db:
            concept_rows = db.execute(
                f"""
                SELECT * FROM local_kg_concepts
                WHERE course_id IN ({marks}) AND active = 1
                ORDER BY course_id, canonical_name, concept_key
                """,
                list(courses),
            ).fetchall()
            edge_rows = db.execute(
                f"""
                SELECT e.*
                FROM local_kg_edges e
                JOIN local_kg_concepts c
                  ON c.course_id = e.course_id AND c.concept_key = e.concept_key AND c.active = 1
                JOIN local_documents d ON d.document_id = e.document_id
                JOIN local_versions v ON v.document_id = e.document_id AND v.version = e.version
                WHERE e.course_id IN ({marks})
                  AND e.status = 'accepted'
                  AND d.course_id = e.course_id
                  AND d.deleted = 0
                  AND d.current_version = e.version
                  AND d.publication_epoch = e.publication_epoch
                  AND v.published_at IS NOT NULL
                  AND v.content_sha256 = e.content_sha256
                ORDER BY e.course_id, e.concept_key, e.relation_type, e.edge_id
                """,
                list(courses),
            ).fetchall()
        edges = [_edge(row) for row in edge_rows]
        counts: dict[tuple[str, str], list[int]] = {}
        for edge in edges:
            pair = counts.setdefault((edge.course_id, edge.concept_key), [0, 0])
            pair[0 if edge.relation_type == "defined_in" else 1] += 1
        concepts = [
            KgConceptSummary(
                course_id=row["course_id"],
                concept_key=row["concept_key"],
                canonical_name=row["canonical_name"],
                aliases=tuple(json.loads(row["aliases_json"])),
                anchor_terms=tuple(json.loads(row["anchor_terms_json"])),
                defined_in_count=counts.get((row["course_id"], row["concept_key"]), [0, 0])[0],
                demonstrated_by_count=counts.get((row["course_id"], row["concept_key"]), [0, 0])[1],
            )
            for row in concept_rows
            if (row["course_id"], row["concept_key"]) in counts
        ]
        return concepts, edges


def _dirty_courses(db: Connection, courses: tuple[str, ...]) -> list[str]:
    marks = ", ".join("?" for _ in courses)
    dirty_active = db.execute(
        f"""
        SELECT DISTINCT d.course_id
        FROM local_documents d
        JOIN local_versions v ON v.document_id = d.document_id AND v.version = d.current_version
        LEFT JOIN local_kg_doc_state s ON s.document_id = d.document_id
        WHERE d.course_id IN ({marks})
          AND d.deleted = 0
          AND d.current_version IS NOT NULL
          AND v.published_at IS NOT NULL
          AND (
              s.document_id IS NULL
              OR s.course_id <> d.course_id
              OR s.version <> d.current_version
              OR s.publication_epoch <> d.publication_epoch
              OR s.content_sha256 <> v.content_sha256
          )
        """,
        list(courses),
    ).fetchall()
    dirty_orphans = db.execute(
        f"""
        SELECT DISTINCT s.course_id
        FROM local_kg_doc_state s
        LEFT JOIN local_documents d ON d.document_id = s.document_id
        WHERE s.course_id IN ({marks})
          AND (d.document_id IS NULL OR d.deleted = 1 OR d.current_version IS NULL)
        """,
        list(courses),
    ).fetchall()
    return list(dict.fromkeys(row["course_id"] for row in (*dirty_active, *dirty_orphans)))


def _edge(row: Row) -> KgEdgeRecord:
    return KgEdgeRecord(
        edge_id=int(row["edge_id"]),
        course_id=row["course_id"],
        concept_key=row["concept_key"],
        relation_type=row["relation_type"],
        document_id=row["document_id"],
        version=int(row["version"]),
        source_id=row["source_id"],
        publication_epoch=int(row["publication_epoch"]),
        content_sha256=row["content_sha256"],
        evidence_span_start=int(row["evidence_span_start"]),
        evidence_span_end=int(row["evidence_span_end"]),
        evidence_excerpt=row["evidence_excerpt"],
        status=row["status"],
        evidence_rule=row["evidence_rule"],
        rejection_reason=row["rejection_reason"],
    )
