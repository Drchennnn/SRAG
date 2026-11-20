"""Offline wire and evidence checks for G16 workspace answer generation."""

from __future__ import annotations

import json

import httpx
import pytest

from app.clients import deepseek
from app.config import Settings
from app.rag import workspace_generation
from app.rag.workspace_generation import OUTPUT_RESERVE, generate_response
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import SourceRevision, TurnEvidence, TurnRead

_QUESTION = "这份课件的读数与限制是什么？"
_SOURCE = "Mean [Hz] | 47. Only at steady state. The protocol excludes cold-start readings."


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        workspace_answer_enabled=True,
        deepseek_api_key="local-mock-secret",
        deepseek_max_retries=1,
    )


def _evidence(
    identifier: str, *, page: int = 8, version: int = 2, reads: list[TurnRead] | None = None
):
    source = SourceKey(document_id="lecture", version=version, source_id=f"page:{page}")
    return TurnEvidence(
        evidence_id=identifier,
        revision=SourceRevision(
            source=source,
            course_id="public-g16",
            publication_epoch=3,
            content_sha256="0" * 64,
        ),
        file_name="calibration.pdf",
        locator={"page": page, "storage_path": "/synthetic/not-for-provider.pdf"},
        reads=reads
        if reads is not None
        else [
            TurnRead(actual_source=source, method="native_text", status="succeeded", text=_SOURCE)
        ],
    )


def _completion(content, *, total_tokens=62, finish_reason="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"total_tokens": total_tokens},
        },
    )


def _wire(monkeypatch, *outcomes):
    requests = []

    def handle_http(request):
        payload = json.loads(request.content)
        requests.append(payload)
        outcome = outcomes[len(requests) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def mock_client(settings, **kwargs):
        return deepseek.DeepSeekClient(
            settings, transport=httpx.MockTransport(handle_http), **kwargs
        )

    async def no_delay(_seconds):
        pass

    monkeypatch.setattr(workspace_generation, "DeepSeekClient", mock_client)
    monkeypatch.setattr(deepseek.asyncio, "sleep", no_delay)
    return requests


def test_generation_only_sends_current_textual_evidence_and_uses_valid_id(settings, monkeypatch):
    answer = {
        "statements": [{"text": "稳态均值为 47 Hz，不适用于冷启动。", "citations": ["ev-1"]}],
        "missing": "",
    }
    requests = _wire(monkeypatch, _completion(json.dumps(answer)))
    display = _evidence("ev-2", page=2, reads=[])
    other_source = SourceKey(document_id="lecture", version=2, source_id="page:7")
    incorrect_source = TurnRead(
        actual_source=other_source, method="native_text", status="succeeded", text="out of scope"
    )
    unverified = _evidence("ev-3", page=3, reads=[incorrect_source])
    visual = TurnRead(
        actual_source=display.revision.source,
        method="vision",
        status="succeeded",
        text="model-only image description",
    )
    display.reads.append(visual)
    evidence = {"ev-1": _evidence("ev-1"), "ev-2": display, "ev-3": unverified}
    result = generate_response(
        settings,
        _QUESTION,
        evidence,
        mode="ask",
        context={
            "previous_question": "如何校准？",
            "previous_answer": "前一轮原文",
            "token": "ignore-me",
        },
        coverage={
            "mode": "complete",
            "full_text_scope": True,
            "unread_visual_total": 2,
            "scope": {"path": "/never-send"},
        },
    )
    (request,) = requests
    assert request["model"] == settings.deepseek_chat_model
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] <= OUTPUT_RESERVE
    sent = json.loads(request["messages"][1]["content"])
    assert sent["question"] == _QUESTION
    assert sent["previous_context"] == {
        "previous_question": "如何校准？",
        "previous_answer": "前一轮原文",
    }
    assert sent["coverage"] == {
        "mode": "complete",
        "full_text_scope": True,
        "unread_visual_total": 2,
    }
    (item,) = sent["evidence"]
    assert item["source"] == evidence["ev-1"].revision.source.model_dump()
    assert item["locator"] == {"page": 8}
    assert item["reads"][0]["text"] == _SOURCE
    assert "model-only image" not in str(request) and "out of scope" not in str(request)
    assert result.status == "succeeded"
    assert result.statements[0].citations == ("ev-1",)
    assert result.trace["input_evidence_ids"] == ["ev-1"]
    assert result.trace["external_call_count"] == 1 and result.trace["total_tokens"] == 62
    assert _QUESTION not in str(result.trace) and "local-mock-secret" not in str(result.trace)


@pytest.mark.parametrize(
    ("statements", "missing", "status"),
    [
        ([{"text": "稳态为 47 Hz。", "citations": ["ev-1"]}], "课件没有冷启动读数。", "partial"),
        ([], "无法核对冷启动读数。", "insufficient"),
    ],
)
def test_generation_reports_supported_part_and_missing_part(
    settings, monkeypatch, statements, missing, status
):
    _wire(monkeypatch, _completion(json.dumps({"statements": statements, "missing": missing})))
    result = generate_response(settings, _QUESTION, {"ev-1": _evidence("ev-1")}, mode="ask")
    assert result.status == status and result.missing == missing
    assert len(result.statements) == len(statements)


def test_generation_keeps_publication_roles_separate_from_version_numbers(settings, monkeypatch):
    answer = {
        "statements": [{"text": "当前发布为 v1，历史引用为 v2。", "citations": ["ev-1", "ev-2"]}],
        "missing": "",
    }
    requests = _wire(monkeypatch, _completion(json.dumps(answer)))
    old = _evidence("ev-1", version=2)
    old.locator["version_role"] = "historical"
    current = _evidence("ev-2", version=1)
    current.locator["version_role"] = "current"
    result = generate_response(
        settings, "和旧版相比呢？", {"ev-1": old, "ev-2": current}, mode="compare_versions"
    )
    assert result.status == "succeeded"
    sent = json.loads(requests[0]["messages"][1]["content"])
    assert sent["mode"] == "compare_versions"
    assert [
        (item["source"]["version"], item["locator"]["version_role"]) for item in sent["evidence"]
    ] == [(2, "historical"), (1, "current")]
    assert "请勿按版号大小推断角色" in requests[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    "content",
    [
        "unfinished not json",
        '{"missing":"", "missing":"重复字段", "statements":[]}',
        '{"statements":[{"text":"读数。", "citations": []}], "missing":""}',
        '{"statements":[{"text":"正确。", "citations":["ev-1"]},'
        '{"text":"未知来源。", "citations":["ev-9"]}], "missing":""}',
        '{"statements":[{"text":"伪造引用 [ev-9]。", "citations":["ev-1"]}], "missing":""}',
        '{"statements":[{"text":"错引候选。", "citations":["ev-2"]}], "missing":""}',
    ],
)
def test_invalid_generated_content_is_atomically_discarded(settings, monkeypatch, content):
    _wire(monkeypatch, _completion(content))
    result = generate_response(
        settings,
        _QUESTION,
        {"ev-1": _evidence("ev-1"), "ev-2": _evidence("ev-2", reads=[])},
        mode="ask",
    )
    assert result.status == "invalid_response" and not result.statements and result.missing == ""
    assert result.trace["external_call_count"] == 1


@pytest.mark.parametrize(
    ("updates", "status"),
    [
        ({"workspace_answer_enabled": False}, "disabled"),
        ({"deepseek_api_key": " "}, "not_configured"),
        ({"workspace_input_token_limit": 10}, "budget_exceeded"),
        ({"deepseek_context_tokens": OUTPUT_RESERVE + 10}, "budget_exceeded"),
    ],
)
def test_disabled_or_oversized_inputs_are_not_sent(settings, monkeypatch, updates, status):
    requests = _wire(monkeypatch)
    result = generate_response(
        settings.model_copy(update=updates), _QUESTION, {"ev-1": _evidence("ev-1")}, mode="ask"
    )
    assert result.status == status and result.trace["external_call_count"] == 0
    assert not requests


def test_contingent_request_counts_include_retry_and_unused_tokens(settings, monkeypatch):
    requests = _wire(
        monkeypatch,
        httpx.Response(503),
        _completion("截断", total_tokens=4096, finish_reason="length"),
        _completion('{"statements":[{"text":"稳态 47 Hz", "citations":["ev-1"]}],"missing":""}'),
    )
    result = generate_response(settings, _QUESTION, {"ev-1": _evidence("ev-1")}, mode="ask")
    assert result.status == "succeeded"
    assert result.trace["external_call_count"] == len(requests) == 3
    assert [attempt["status"] for attempt in result.trace["attempts"]] == [503, 200, 200]
    assert requests[-1]["max_tokens"] == OUTPUT_RESERVE
    assert result.trace["total_tokens"] == 4158


def test_timeout_is_sanitized_and_counted(settings, monkeypatch):
    _wire(monkeypatch, httpx.ReadTimeout("local-mock-secret"), httpx.Response(401, text="secret"))
    result = generate_response(settings, _QUESTION, {"ev-1": _evidence("ev-1")}, mode="ask")
    assert result.status == "provider_error" and not result.statements
    assert result.trace["external_call_count"] == 2
    assert result.trace["attempts"][0]["outcome"] == "timeout"
    assert result.trace["attempts"][1]["status"] == 401
    assert "secret" not in str(result)
