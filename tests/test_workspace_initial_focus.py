from test_workspace import (
    _ask,
    _create,
    _focus,
    _node,
    _upload,
    public_object_files,
    system,
)

__all__ = ["public_object_files", "system"]


def test_first_ordinal_reads_current_focused_cell_without_history(system, public_object_files):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    imported = _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    cell = _node(client, session, imported["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, cell)
    result = _ask(client, session, "第二张图", revision)
    resolution = result["turn"]["resolution"]
    assert resolution["method"] == "current_focus"
    assert resolution["actual_source"]["source_id"] == "cell:1:output:1"
    assert not result["turn"]["trace"]["external_call_count"]


def test_ambiguous_current_image_choice_remains_available_after_session_reload(
    system, public_object_files
):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    imported = _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    result = _ask(client, session, "那张图是哪一张？")
    turn = result["turn"]
    assert turn["status"] == "clarification"
    assert len(turn["options"]) == 2
    persisted = client.get(session).json()["turns"][-1]
    assert persisted["options"] == turn["options"]
    anchor = {
        "type": "clarification",
        "turn_id": turn["turn_id"],
        "option_id": persisted["options"][1]["option_id"],
    }
    chosen = _ask(client, session, "用这张图", anchor=anchor)
    resolution = chosen["turn"]["resolution"]
    assert resolution["method"] == "current_choice"
    assert resolution["actual_source"] == {
        "document_id": imported["document_id"],
        "version": 1,
        "source_id": "cell:1:output:1",
    }


def test_explicit_history_reference_does_not_borrow_current_objects(system, public_object_files):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    imported = _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    cell = _node(client, session, imported["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, cell)
    for question in ("刚才第二张图", "上面的第二张图的横轴单位是什么？", "前一条的那张图"):
        result = _ask(client, session, question, revision)
        assert result["turn"]["status"] == "clarification"
        assert not result["turn"]["options"]
        assert not result["message"]["sources"]


def test_bare_ordinal_with_different_history_and_focus_prompts_which_scope(
    system, public_object_files
):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    path = originals / "v1" / "HarborTrace.ipynb"
    first = _upload(client, session, path)
    first_cell = _node(client, session, first["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, first_cell)
    shown = _ask(client, session, "展示这个 Cell 保存的两张图", revision)
    assert len(shown["message"]["visible_display_ids"]) == 2
    other = _upload(client, session, path, filename="OtherHarbor.ipynb")
    other_cell = _node(client, session, other["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, other_cell, revision)
    ambiguous = _ask(client, session, "第二张图", revision)
    assert ambiguous["turn"]["status"] == "clarification"
    assert not ambiguous["message"]["sources"]
    local = _ask(client, session, "当前第二张图", revision)
    resolution = local["turn"]["resolution"]
    assert resolution["method"] == "current_focus"
    assert resolution["actual_source"]["document_id"] == other["document_id"]


def test_parent_cell_does_not_hide_conflict_between_display_and_internal_order(
    system, public_object_files
):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    imported = _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    revision = _focus(
        client, session, _node(client, session, imported["document_id"], "cell", cell_index=1)
    )
    shown = _ask(client, session, "HarborTrace.ipynb Cell 2 第 2 张图的横轴单位是什么？", revision)
    assert len(shown["message"]["visible_display_ids"]) == 1
    ambiguous = _ask(client, session, "第一张图的横轴单位是什么？", revision)
    assert ambiguous["turn"]["status"] == "clarification"
    old = _ask(client, session, "刚才第一张图的横轴单位是什么？", revision)
    assert old["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:1"
    current = _ask(client, session, "当前第一张图的横轴单位是什么？", revision)
    assert current["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:0"
