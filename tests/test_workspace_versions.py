"""Real workspace routes with synthetic versioned public sources."""

import base64
import json

import nbformat
import pytest
from test_workspace import (
    _ask,
    _create,
    _sha,
    _upload,
)
from test_workspace import (
    public_object_files as _public_object_files,
)
from test_workspace import (
    system as _system,
)

from app.rag.workspace_versions import is_version_comparison

public_object_files = _public_object_files
system = _system


def _notebook(path, result: str, *, cell_id="stability"):
    cell = nbformat.v4.new_code_cell(
        "summarize()",
        id=cell_id,
        execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text=result)],
    )
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)


def _version_turn(client, session, path, publication, *args, **kwargs):
    second = _upload(client, session, path, document_id=publication["document_id"])
    assert second["version"] == 2
    return _ask(client, session, *args, **kwargs)


def _sides(message):
    return {
        item["revision"]["source"]["version"]: (evidence_id, item)
        for evidence_id, item in message["sources"].items()
    }


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("比较上一版和当前版的速率", True),
        ("比较第1版和第2版的速率", True),
        ("Compare v1 with v2 in the saved output", True),
        ("Compare historical v1 against the current version", True),
        ("What was the value in the old version?", False),
        ("Compare this image against the table", False),
        ("Compare sampling-v1.ipynb against another-v2.ipynb", False),
        ("Compare 'sampling notes v1.ipynb' against “another notes v2.ipynb”", False),
        ("请比较 旧版实验.ipynb 和 新版实验.ipynb 的保存输出", False),
        ("请比较 旧版-v1.ipynb的输出和新版-v2.ipynb的输出", False),
        ("请比较Harbor-v1.ipynb 上一版和当前版", True),
        ("Compare chapters 1 and 2", False),
    ],
)
def test_comparison_route_requires_two_version_indicators(question, expected) -> None:
    assert is_version_comparison(question) is expected


def test_same_turn_rereads_saved_output_from_two_versioned_sources(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    path = tmp_path / "VersionDeck.ipynb"
    _notebook(path, "Pilot amplitude frequency: 2.5 kHz")
    first = _upload(client, session, path)
    earlier = _ask(client, session, "Pilot amplitude frequency", kind="search")
    assert any(
        item["revision"]["source"]["source_id"] == "cell:0:output:0"
        for item in earlier["message"]["sources"].values()
    )
    _notebook(path, "Pilot amplitude frequency: 4.5 kHz")
    compared = _version_turn(
        client, session, path, first, "比较 VersionDeck.ipynb 上一版和当前版 Cell 1 保存输出。"
    )
    assert compared["turn"]["trace"]["route"] == "version_comparison"
    assert compared["turn"]["trace"]["external_call_count"] == 0
    message = compared["message"]
    citations = {
        cited
        for block in message["blocks"]
        if block["type"] == "text"
        for cited in block["citations"]
    }
    sides = _sides(message)
    assert set(sides) == {1, 2}
    assert citations == {value[0] for value in sides.values()}
    for version, expected in ((1, "2.5 kHz"), (2, "4.5 kHz")):
        evidence_id, source = sides[version]
        key = source["revision"]["source"]
        assert source["reads"][0]["actual_source"] == key
        assert key["source_id"] == "cell:0:output:0"
        assert source["reads"][0]["status"] == "succeeded"
        response = client.get(f"{session}/messages/{message['message_id']}/evidence/{evidence_id}")
        assert response.status_code == 200, response.text
        assert response.json()["source"] == key
        assert response.json()["is_current"] == (version == 2)
        assert expected in response.json()["text"]


def test_old_reference_must_belong_to_session_and_live_document(system, tmp_path):
    _, service, _, client = system
    private_session = _create(client)
    other_session = _create(client)
    path = tmp_path / "VersionDeck.ipynb"
    _notebook(path, "Pilot amplitude frequency: 2.5 kHz")
    first = _upload(client, private_session, path)
    _ask(client, private_session, "Pilot amplitude frequency", kind="search")
    _notebook(path, "Pilot amplitude frequency: 4.5 kHz")
    question = "比较 VersionDeck.ipynb 的旧版与当前版 Cell 1 保存输出。"
    no_history = _version_turn(client, other_session, path, first, question)
    assert no_history["message"]["blocks"][0]["code"] == "comparison_missing_history"
    assert not no_history["message"]["sources"]
    assert service.library.store.delete_local(first["document_id"])
    no_access = _ask(client, private_session, question)
    assert no_access["message"]["blocks"][0]["code"] == "comparison_scope"
    assert not no_access["message"]["sources"]


def test_different_cell_identifier_does_not_match_old_object(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    path = tmp_path / "VersionDeck.ipynb"
    _notebook(path, "Pilot amplitude frequency: 2.5 kHz")
    first = _upload(client, session, path)
    _ask(client, session, "Pilot amplitude frequency", kind="search")
    _notebook(path, "Unrelated amplitude frequency: 4.5 kHz", cell_id="other")
    compared = _version_turn(
        client, session, path, first, "比较 VersionDeck.ipynb 上一版和当前版 Cell 1 保存输出。"
    )
    assert compared["message"]["blocks"][0]["code"] == "comparison_ambiguous"
    assert not compared["message"]["sources"]


def test_compare_two_images_reopens_exact_bytes_and_never_reads_old_image(
    system, public_object_files, monkeypatch
):
    _, service, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    first_path = originals / "v1" / "HarborTrace.ipynb"
    second_path = originals / "v2" / "HarborTrace.ipynb"
    first = _upload(client, session, first_path)
    asked = _ask(client, session, "HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？")
    assert len(asked["message"]["visible_display_ids"]) == 1
    real = service.vision.process_current
    requested = []

    def spy(request, **kwargs):
        requested.append(request.source.version)
        return real(request, **kwargs)

    monkeypatch.setattr(service.vision, "process_current", spy)
    compared = _version_turn(
        client,
        session,
        second_path,
        first,
        "比较 HarborTrace.ipynb 上一版和当前版 Cell 2 第 2 张图的横轴单位。",
    )
    assert requested == [2]
    trace = compared["turn"]["trace"]
    assert trace["route"] == "version_comparison"
    assert trace["historical_read_status"] == "not_attempted"
    assert trace["current_read_status"] == "not_attempted"
    images = [item for item in compared["message"]["blocks"] if item["type"] == "source_image"]
    assert [image["display_source"]["version"] for image in images] == [1, 2]
    assert all(image["display_source"]["source_id"] == "cell:1:output:1" for image in images)
    for path, image in zip((first_path, second_path), images, strict=True):
        data = json.loads(path.read_text())["cells"][1]["outputs"][1]["data"]["image/png"]
        expected = base64.b64decode(data if isinstance(data, str) else "".join(data))
        assert _sha(client.get(image["media_url"]).content) == _sha(expected)


@pytest.mark.parametrize("attribute", ["横轴单位", "横轴标签"])
def test_historical_visual_result_is_reused_only_for_verified_matching_attribute(
    system, public_object_files, monkeypatch, attribute
):
    from app.rag.source_evidence import ReadRecord

    _, service, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    first_path = originals / "v1" / "HarborTrace.ipynb"
    second_path = originals / "v2" / "HarborTrace.ipynb"
    first = _upload(client, session, first_path)
    requested = []

    def recorded_provider_result(request, *, expected_input_sha256):
        """Simulate verified provider metadata locally; no API request is made."""
        base = service.evidence.render_current(request)
        display = base.display
        assert display is not None and display.byte_sha256 == expected_input_sha256
        assert request.question
        requested.append(request.source.version)
        reading = ReadRecord(
            requested_source=request.source,
            actual_source=display.actual_source,
            actual_input_sha256=display.byte_sha256,
            actual_input_media_type=display.media_type,
            actual_input_rect=display.rect_visible_norm,
            purpose="directed_question",
            execution="real_provider",
            origin_execution="real_provider",
            provider="simulated-fixture",
            model="simulated-fixture",
            prompt_version="test-v1",
            question_sha256=_sha(request.question.encode()),
            status="succeeded",
            text="seconds" if request.source.version == 1 else "minutes",
        )
        return base.model_copy(update={"read_status": "succeeded", "reads": [reading]})

    monkeypatch.setattr(service.vision, "process_current", recorded_provider_result)
    _ask(client, session, "HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？")
    compared = _version_turn(
        client,
        session,
        second_path,
        first,
        f"比较 HarborTrace.ipynb 上一版与当前版 Cell 2 第 2 张图的{attribute}。",
    )
    assert requested == [1, 2]  # The historical image was never sent during the comparison.
    sides = _sides(compared["message"])
    previous = sides[1][1]
    assert sides[2][1]["reads"][0]["text"] == "minutes"
    if attribute == "横轴单位":
        assert previous["read_status"] == "succeeded"
        assert previous["reads"][0]["text"] == "seconds"
    else:
        assert previous["read_status"] == "not_attempted"
        assert previous["reads"] == []


def test_two_different_named_files_are_not_silently_compared_as_one(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    path = tmp_path / "VersionDeck.ipynb"
    another = tmp_path / "DifferentDeck.ipynb"
    _notebook(path, "Pilot amplitude frequency: 2.5 kHz")
    first = _upload(client, session, path)
    _ask(client, session, "Pilot amplitude frequency", kind="search")
    _notebook(another, "Second project amplitude frequency: 3 kHz")
    _upload(client, session, another)
    _notebook(path, "Pilot amplitude frequency: 4.5 kHz")
    compared = _version_turn(
        client,
        session,
        path,
        first,
        "比较 VersionDeck.ipynb 的上一版与 DifferentDeck.ipynb 的当前版保存输出。",
    )
    assert compared["message"]["blocks"][0]["code"] == "comparison_scope"
    assert not compared["message"]["sources"]


def test_filename_edition_is_not_a_version_or_a_historical_role(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    path = tmp_path / "Harbor-v1.ipynb"
    _notebook(path, "Pilot amplitude frequency: 2.5 kHz")
    first = _upload(client, session, path)
    _ask(client, session, "Pilot amplitude frequency", kind="search")
    _notebook(path, "Pilot amplitude frequency: 4.5 kHz")
    _upload(client, session, path, document_id=first["document_id"])
    _ask(client, session, "Pilot amplitude frequency v2", kind="search")
    _notebook(path, "Pilot amplitude frequency: 9.0 kHz")
    third = _upload(client, session, path, document_id=first["document_id"])
    assert third["version"] == 3

    implicit = _ask(client, session, "比较 Harbor-v1.ipynb 上一版与当前版的 Cell 1 保存输出。")
    assert set(_sides(implicit["message"])) == {2, 3}
    for question in (
        "比较 Harbor-v1.ipynb 的旧版 v3 和当前版 v1 的 Cell 1 保存输出。",
        "比较 Harbor-v1.ipynb v3（历史版）和 v1（当前版）的 Cell 1 保存输出。",
        "Compare the historical v3 and current v1 in the saved output of Harbor-v1.ipynb.",
    ):
        rejected = _ask(client, session, question)
        assert rejected["message"]["blocks"][0]["code"] == "comparison_version_mismatch"
        assert not rejected["message"]["sources"]
        assert rejected["turn"]["trace"]["external_call_count"] == 0
    for question in (
        "比较 Harbor-v1.ipynb 的旧版 v1 和当前版 v3 的 Cell 1 保存输出。",
        "Compare v1 with v3 in the saved output of Harbor-v1.ipynb.",
    ):
        explicit = _ask(client, session, question)
        assert set(_sides(explicit["message"])) == {1, 3}
