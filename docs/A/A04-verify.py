"""Independent A04 cross-layer contract verifier for the fixed A fixtures."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from pathlib import Path

from app.rag.parsers.notebook import NotebookParser

ROOT = Path(__file__).parent
VECTOR_PATH = ROOT / "A01-契约样例.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_key(source: dict) -> tuple[str, int, str]:
    return source["document_id"], source["version"], source["source_id"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser_consumer(vector: dict, asset_root: Path) -> tuple[dict, bytes]:
    """Resolve a source vector through the production NotebookParser."""

    target = next(
        source
        for source in vector["sources"]
        if (
            source["document_id"],
            source["version"],
            source["source_id"],
        )
        == ("signal-lab", 1, "signal-plots-output-2")
    )
    document_record = next(
        item
        for item in vector["documents"]
        if (item["document_id"], item["version"]) == (target["document_id"], target["version"])
    )
    parent = next(
        item
        for item in vector["sources"]
        if (item["document_id"], item["version"], item["source_id"])
        == (target["document_id"], target["version"], target["parent_id"])
    )
    document = NotebookParser(asset_root / "v1").parse(
        ROOT / document_record["fixture_path"],
        document_id=target["document_id"],
    )
    block = next(
        item
        for item in document.blocks
        if item.cell_id == parent["cell_id"]
        and item.cell_index == parent["cell_index"]
        and item.block_id.endswith(f":output:{target['output_index']}")
    )
    assert block.cell_index == 1
    assert len(block.assets) == 1
    asset_bytes = Path(block.assets[0].path).read_bytes()
    assert _sha256(asset_bytes) == target["asset_sha256"]
    return target, asset_bytes


def _message_consumer(vector: dict) -> tuple[dict, bytes]:
    """Resolve the second visible image directly from message order and raw JSON."""

    image_blocks = [
        block for block in vector["message"]["blocks"] if block["type"] == "source_image"
    ]
    selected = image_blocks[1]
    assert selected["display_id"] == "img-2"
    assert selected["display_source"] == vector["expected"]["second_display_source"]

    selected_key = (
        selected["display_source"]["document_id"],
        selected["display_source"]["version"],
        selected["display_source"]["source_id"],
    )
    source_record = next(
        item
        for item in vector["sources"]
        if (item["document_id"], item["version"], item["source_id"]) == selected_key
    )
    parent_record = next(
        item
        for item in vector["sources"]
        if (item["document_id"], item["version"], item["source_id"])
        == (
            source_record["document_id"],
            source_record["version"],
            source_record["parent_id"],
        )
    )
    document_record = next(
        item
        for item in vector["documents"]
        if (item["document_id"], item["version"])
        == (source_record["document_id"], source_record["version"])
    )
    notebook = _load_json(ROOT / document_record["fixture_path"])
    cell = notebook["cells"][parent_record["cell_index"]]
    assert cell["id"] == parent_record["cell_id"]
    output = cell["outputs"][source_record["output_index"]]
    asset_bytes = base64.b64decode(output["data"][source_record["mime_type"]])
    assert _sha256(asset_bytes) == vector["expected"]["second_display_sha256"]
    return selected["display_source"], asset_bytes


def _verify_versions_and_failures(vector: dict, asset_root: Path) -> None:
    v1 = _load_json(ROOT / "样例" / "Signal-lab.ipynb")
    v2 = _load_json(ROOT / "样例" / "revisions" / "v2" / "Signal-lab.ipynb")
    v1_bytes = base64.b64decode(v1["cells"][1]["outputs"][2]["data"]["image/png"])
    v2_bytes = base64.b64decode(v2["cells"][1]["outputs"][2]["data"]["image/png"])
    assert _sha256(v1_bytes) == "ce741007281a939680e7a21020fc80c34abfcb10d6d76d13e7984bc9d5743aa5"
    assert _sha256(v2_bytes) == "7ec65dab6628a30590421e4c1a150c52a6ec22933d67efce47eda279097fbd68"
    assert v1_bytes != v2_bytes

    parsed_v2 = NotebookParser(asset_root / "v2").parse(
        ROOT / "样例" / "revisions" / "v2" / "Signal-lab.ipynb",
        document_id="signal-lab",
    )
    v2_block = next(item for item in parsed_v2.blocks if item.block_id == "signal-plots:output:2")
    assert _sha256(Path(v2_block.assets[0].path).read_bytes()) == _sha256(v2_bytes)

    image_blocks = [
        block for block in vector["message"]["blocks"] if block["type"] == "source_image"
    ]
    assert image_blocks[1]["display_id"] != image_blocks[2]["display_id"]
    assert image_blocks[1]["display_source"] == image_blocks[2]["display_source"]

    failures = {item["case"]: item for item in vector["negative_vectors"]}
    assert failures["historical_source_removed"]["status"] == "original_unavailable"
    assert failures["same_cell_output_has_no_image"]["status"] == "image_unavailable"
    assert failures["unreliable_table_bbox"]["display_source"]["source_id"] == "slide-1"
    stream_block = next(
        item for item in parsed_v2.blocks if item.block_id == "signal-plots:output:0"
    )
    assert stream_block.assets == []


def main() -> None:
    vector = _load_json(VECTOR_PATH)
    with tempfile.TemporaryDirectory(prefix="srag-a04-assets-") as directory:
        asset_root = Path(directory)
        source_reference, parser_bytes = _parser_consumer(vector, asset_root)
        display_reference, message_bytes = _message_consumer(vector)
        assert _source_key(source_reference) == _source_key(display_reference)
        assert parser_bytes == message_bytes
        _verify_versions_and_failures(vector, asset_root)

    print(
        "A04 contract verified: two consumers -> "
        "signal-lab/v1/signal-plots-output-2 -> ce741007281a; "
        "v2=7ec65dab6628; three fixture fallbacks present"
    )


if __name__ == "__main__":
    main()
