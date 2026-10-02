"""真实连续对话评估入口的输入与指标测试。"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from evals.run_real_dialogues import evaluate, load_sessions, summarize


def test_load_sessions_rejects_missing_turn_label(tmp_path) -> None:
    path = tmp_path / "dialogues.jsonl"
    path.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "turns": [{"query": "什么是 LIME？"}, {"query": "这个机制在哪段？"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必须标注 kind"):
        load_sessions(path)


@pytest.mark.asyncio
async def test_evaluate_rule_uses_previous_turn_and_excludes_unclear(tmp_path) -> None:
    cases = tmp_path / "dialogues.jsonl"
    output = tmp_path / "report.json"
    cases.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "turns": [
                    {"query": "什么是 LIME？", "assistant_reply": "LIME 用局部模型解释预测。"},
                    {"query": "这个机制在哪段？", "kind": "followup"},
                    {"query": "SHAP 是什么？", "kind": "independent", "tags": ["topic_switch"]},
                    {"query": "可能是追问", "kind": "unclear"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = await evaluate(cases, output)

    assert report["session_count"] == 1
    assert report["rule"]["missed_followups"] == 1
    assert report["rule"]["false_expansions"] == 0
    assert report["rule"]["unclear_count"] == 1
    assert report["rule"]["answer_observed_count"] == 1
    assert report["by_tag"]["topic_switch"]["rule"]["false_expansions"] == 0
    assert report["cases"][1]["previous_user"] == "这个机制在哪段？"
    assert report["cases"][0]["previous_assistant_reply_present"] is True
    assert report["cases"][1]["previous_assistant_reply_present"] is False
    assert report["cases"][1]["tags"] == ["topic_switch"]
    assert "model" not in report
    assert json.loads(output.read_text(encoding="utf-8"))["rule"] == report["rule"]

    client = Mock()
    client.classify_context_need = AsyncMock(
        side_effect=[
            {"needs_context": True, "reason": "指代", "attempts": 1, "usage": {"total_tokens": 12}},
            {
                "needs_context": False,
                "reason": "新主题",
                "attempts": 1,
                "usage": {"total_tokens": 13},
            },
            {
                "needs_context": False,
                "reason": "不确定",
                "attempts": 1,
                "usage": {"total_tokens": 14},
            },
        ]
    )
    model_report = await evaluate(cases, tmp_path / "model.json", compare_model=True, client=client)
    assert model_report["model"]["missed_followups"] == 0
    assert model_report["model"]["false_expansions"] == 0
    assert model_report["model_tokens"] == 39
    assert model_report["by_tag"]["topic_switch"]["model"]["false_expansions"] == 0
    assert client.classify_context_need.await_count == 3


def test_summarize_reports_missing_denominators() -> None:
    metrics = summarize([{"kind": "followup", "rule_method": "raw"}], "rule_method")

    assert metrics["missed_followups"] == 1
    assert metrics["missed_followup_rate"] == 1.0
    assert metrics["false_expansion_rate"] is None
