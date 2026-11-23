"""Isolated real-Chromium dual-entry verification script for H02 course KG navigation & QA.

Run with:
  SRAG_SOFFICE=/opt/homebrew/bin/soffice PYTHONDONTWRITEBYTECODE=1 \
    /Users/Link/code/SRAG/.venv/bin/python docs/H/verify_h02_web_kg.py \
    --work-dir /tmp/srag-h-web-verify
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import nbformat
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = Path(sys.executable)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_type1_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> None:
    """Write a deterministic searchable PDF using built-in Type1 Helvetica."""

    def _pdf_esc(text: str) -> bytes:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return safe.encode("ascii")

    kids = " ".join(f"{4 + idx * 2} 0 R" for idx in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_num, (heading, lines) in enumerate(pages, start=1):
        stream_id = 2 * page_num + 3
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 650 460] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        stream = b"BT /F1 18 Tf 42 405 Td (" + _pdf_esc(heading) + b") Tj ET\n"
        for idx, line in enumerate(lines):
            stream += (
                b"BT /F1 11 Tf 42 %d Td (" % (366 - idx * 20) + _pdf_esc(line) + b") Tj ET\n"
            )
        stream += b"BT /F1 10 Tf 42 32 Td (Page %d) Tj ET\n" % page_num
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(b"%d 0 obj\n" % idx + obj + b"\nendobj\n")
    xref_pos = len(out)
    out.extend(b"xref\n0 %d\n0000000000 65535 f \n" % len(offsets))
    for off in offsets[1:]:
        out.extend(b"%010d 00000 n \n" % off)
    out.extend(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(offsets), xref_pos)
    )
    path.write_bytes(bytes(out))


def _make_pptx(path: Path, slides: list[tuple[str, list[str]]]) -> None:
    prs = Presentation()
    for title, bullets in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        tbox = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(11.8), Inches(0.9))
        tf = tbox.text_frame
        tf.text = title
        tf.paragraphs[0].font.size = Pt(24)
        bbox = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(11.8), Inches(5.0))
        btf = bbox.text_frame
        btf.word_wrap = True
        for i, line in enumerate(bullets):
            p = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
            p.text = line
            p.font.size = Pt(16)
    prs.save(path)


def _make_png_b64(label: str, accent: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (520, 260), (248, 251, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((16, 16, 504, 244), outline=accent, width=4)
    draw.line([(50, 200), (180, 120), (300, 185), (460, 75)], fill=accent, width=5)
    draw.text((40, 32), label, fill=(20, 35, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_ipynb(path: Path, *, version: int) -> None:
    db = "-34.2" if version == 1 else "-48.5"
    img_b64 = _make_png_b64(
        f"Attenuation Curve v{version}: {db} dB",
        (28, 103, 77) if version == 1 else (39, 107, 151),
    )
    metric_text = f"attenuation={db} dB\n"
    cells = [
        nbformat.v4.new_code_cell(
            f"# Measured step response run v{version}\n"
            f"res = notch_filter_step(raw_sig, f0=60.0)\n"
            f"print('{metric_text.strip()}')",
            outputs=[
                nbformat.v4.new_output(output_type="stream", text=metric_text),
                nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_b64, "text/plain": ["<NotchCurve>"]},
                    metadata={},
                ),
            ],
        ),
        nbformat.v4.new_code_cell(
            "# Do NOT treat this cell as Adaptive Notch Filter;\n"
            "# it does NOT implement or demonstrate Adaptive Notch Filter.\n"
            "res_base = notch_filter_step(zero_sig, f0=0.0)\nprint('baseline=-1.0 dB')",
            outputs=[nbformat.v4.new_output(output_type="stream", text="baseline=-1.0 dB\n")],
        ),
        nbformat.v4.new_code_cell(
            "selected = sparse_route_top2(tokens)\nprint('throughput=1280 tok/s')",
            outputs=[
                nbformat.v4.new_output(output_type="stream", text="throughput=1280 tok/s\n")
            ],
        ),
        nbformat.v4.new_code_cell(
            "# TODO: review Adaptive Notch Filter timing in future work\nval = 0",
            outputs=[],
        ),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def _prepare_fixtures(fixtures_dir: Path) -> dict[str, Path]:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    lec1 = fixtures_dir / "Lecture01-Spectral.pdf"
    _make_type1_pdf(
        lec1,
        [
            (
                "2.1 Formal Definition: Adaptive Notch Filter",
                [
                    "Adaptive Notch Filter attenuates narrowband 60Hz periodic interference.",
                    "Math condition: pole radius r lies in (0, 1) to control notch bandwidth.",
                    "In the lab script, call notch_filter_step to verify attenuation.",
                ],
            ),
            (
                "2.2 Definition: Spectral Gatekeeper",
                [
                    "Spectral Gatekeeper suppresses sudden transient spikes above threshold tau.",
                    "This concept currently only appears in lecture notes without lab code.",
                ],
            ),
        ],
    )
    lec2 = fixtures_dir / "Lecture02-Routing.pptx"
    _make_pptx(
        lec2,
        [
            ("Agenda", ["1. Adaptive Notch Filter", "2. 门控路由", "3. Experimental Checklist"]),
            (
                "1.1 门控路由 (Gated Routing)",
                [
                    "核心概念定义：门控路由 — 按打分向量将请求路由至 Top-2 专家子网。",
                    "配套实验调用 `sparse_route_top2` 记录每秒吞吐。",
                ],
            ),
        ],
    )
    lec3_v1 = fixtures_dir / "Lecture03-Transient.pdf"
    _make_type1_pdf(
        lec3_v1,
        [
            (
                "Definition: Transient Probe",
                [
                    "Transient Probe measures short-window impulse overshoot during startup.",
                    "Check impulse duration using benchmark notes.",
                ],
            )
        ],
    )
    lec3_v2 = fixtures_dir / "Lecture03-Transient-v2.pdf"
    _make_type1_pdf(
        lec3_v2,
        [
            (
                "Course Revision Notice",
                [
                    "This updated handout replaces earlier transient terminology with notes.",
                    "No formal concept definitions remain on this page.",
                ],
            )
        ],
    )
    anti = fixtures_dir / "AntiExamples-Refs.pdf"
    _make_type1_pdf(
        anti,
        [
            (
                "Related Work and Future Plan",
                [
                    "Unlike Adaptive Notch Filter [2], Smith et al. rely on static smoothing.",
                    "Quantum Shuffler is not defined as an adaptive filter method.",
                    "Future work: planned Quantum Shuffler experiments remain unexecuted.",
                ],
            )
        ],
    )
    lab_v1 = fixtures_dir / "Lab01-SignalLab.ipynb"
    _make_ipynb(lab_v1, version=1)
    lab_v2 = fixtures_dir / "Lab01-SignalLab-v2.ipynb"
    _make_ipynb(lab_v2, version=2)
    other_b = fixtures_dir / "CourseB-Systems.pdf"
    _make_type1_pdf(
        other_b,
        [
            (
                "Definition: Bus Arbitrator",
                [
                    "Bus Arbitrator serializes shared memory controller bursts across cores.",
                ],
            )
        ],
    )
    return {
        "lec1": lec1,
        "lec2": lec2,
        "lec3_v1": lec3_v1,
        "lec3_v2": lec3_v2,
        "anti": anti,
        "lab_v1": lab_v1,
        "lab_v2": lab_v2,
        "other_b": other_b,
    }


def _ab(session: str, *args: str) -> str:
    cmd = ["agent-browser", "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"agent-browser failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
    return res.stdout.strip()


def _ab_eval(session: str, js_expr: str) -> Any:
    return json.loads(_ab(session, "eval", js_expr))


def _wait_for_js(
    session: str, predicate_js: str, *, timeout: float = 25.0, desc: str = ""
) -> Any:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            val = _ab_eval(session, predicate_js)
            if val:
                return val
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for: {desc or predicate_js} (last_err={last_err})")


def _wait_health(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    raise RuntimeError(f"Server at {base_url} failed healthcheck")


def _http_status(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", errors="replace")


def run_verification(work_dir: Path) -> dict[str, Any]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shots = work_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    fixtures = _prepare_fixtures(work_dir / "fixtures")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "APP_HOST": "127.0.0.1",
        "APP_PORT": str(port),
        "SQLITE_PATH": str(work_dir / "srag.sqlite3"),
        "DATA_DIR": str(work_dir / "data"),
        "COURSE_ROOT": str(work_dir / "courses"),
        "WORKSPACE_COURSE_IDS": "cs-signal-kg,cs-isolated-b",
        "WORKSPACE_ANSWER_ENABLED": "false",
        "WORKSPACE_KG_RETRIEVAL_ENABLED": "true",
        "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
        "SOURCE_VISION_ENABLED": "false",
        "DEEPSEEK_API_KEY": "",
        "DASHSCOPE_API_KEY": "",
        "JEV_API_KEY": "",
        "SRAG_SOFFICE": os.environ.get("SRAG_SOFFICE", "/opt/homebrew/bin/soffice"),
    }

    server = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    browser_a = f"h02-verify-a-{port}"
    browser_b = f"h02-verify-b-{port}"
    report: dict[str, Any] = {"work_dir": str(work_dir), "base_url": base_url, "checks": {}}

    try:
        _wait_health(base_url)

        _ab(browser_a, "set", "viewport", "1380", "900")
        _ab(browser_a, "open", f"{base_url}/workspace/")
        session_a_id = _wait_for_js(
            browser_a,
            "(() => { const s = document.getElementById('sessions'); "
            "return s && !s.disabled && s.value ? s.value : null; })()",
            desc="initial workspace session",
        )
        # Switch scope to cs-signal-kg specifically.
        _ab_eval(
            browser_a,
            "(() => { const r = [...document.querySelectorAll('#catalog .catalog-row')]"
            ".find(el => el.textContent.includes('cs-signal-kg')); "
            "if (r) r.click(); return !!r; })()",
        )
        _wait_for_js(
            browser_a,
            "document.getElementById('scope-summary')?.textContent?.includes('cs-signal-kg')",
            desc="course cs-signal-kg scoped",
        )

        # 1. Import 5 files through real UI #import-form (PDF + PPTX + IPYNB + Anti + Transient).
        _ab(browser_a, "eval", "document.getElementById('import-panel').open = true")
        _ab(browser_a, "select", "#import-course", "cs-signal-kg")
        _ab(
            browser_a,
            "upload",
            "#import-files",
            str(fixtures["lec1"]),
            str(fixtures["lec2"]),
            str(fixtures["lec3_v1"]),
            str(fixtures["anti"]),
            str(fixtures["lab_v1"]),
        )
        _ab(browser_a, "click", "#import-button")
        _wait_for_js(
            browser_a,
            "(() => { const items = [...document.querySelectorAll('#import-results li')]; "
            "return items.length === 5 && "
            "items.every(li => li.textContent.includes('导入完成')); })()",
            timeout=45.0,
            desc="all 5 imports published via UI",
        )
        report["checks"]["import_5_files"] = True

        # 2. Entry 1: Open #kg-panel, verify concept list & anti-example rejection.
        _ab(browser_a, "eval", "document.getElementById('kg-panel').open = true")
        concept_names = _wait_for_js(
            browser_a,
            "(() => { const els = [...document.querySelectorAll('#kg-list .kg-item-name')]; "
            "return els.length >= 4 ? els.map(e => e.textContent) : null; })()",
            desc="sidebar KG concepts populated",
        )
        joined = " ".join(concept_names)
        assert "Quantum Shuffler" not in joined
        assert "Agenda" not in joined
        assert any("Adaptive Notch Filter" in n for n in concept_names)
        assert any("门控路由" in n for n in concept_names)
        assert any("Spectral Gatekeeper" in n for n in concept_names)
        assert any("Transient Probe" in n for n in concept_names)
        report["checks"]["entry1_concept_list"] = concept_names

        # Click Adaptive Notch Filter -> open lecture page in #reader, then lab child_output!
        _ab(browser_a, "click", "#kg-list .kg-item[data-concept-key='adaptive_notch_filter']")
        _wait_for_js(
            browser_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "return d && !d.hidden && d.textContent.includes('Lecture01-Spectral.pdf') "
            "&& d.textContent.includes('Lab01-SignalLab.ipynb'); })()",
            desc="Adaptive Notch Filter detail loaded",
        )
        _ab(browser_a, "screenshot", str(shots / "01-entry1-kg-detail.png"))

        _ab_eval(
            browser_a,
            "(() => { const b = [...document.querySelectorAll('#kg-detail button')]"
            ".find(x => x.textContent.trim() === '打开讲义原页'); b.click(); return true; })()",
        )
        _wait_for_js(
            browser_a,
            "(() => { const r = document.getElementById('reader'); "
            "const img = r?.querySelector('img'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "return r && r.open && t.includes('Lecture01-Spectral.pdf') "
            "&& img && img.complete && img.naturalWidth > 0; })()",
            desc="reader opened lecture page 1 with rendered PNG",
        )
        _ab(browser_a, "screenshot", str(shots / "02-entry1-lecture-reader.png"))

        _ab_eval(
            browser_a,
            "(() => { const b = [...document.querySelectorAll('#kg-detail button')]"
            ".find(x => x.textContent.trim() === '查看输出图'); b.click(); return true; })()",
        )
        _wait_for_js(
            browser_a,
            "(() => { const r = document.getElementById('reader'); "
            "const img = r?.querySelector('img'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "return r && r.open && t.includes('Lab01-SignalLab.ipynb') "
            "&& img && img.complete && img.naturalWidth > 0; })()",
            desc="reader opened lab child_output plot PNG",
        )
        _ab(browser_a, "screenshot", str(shots / "03-entry1-lab-child-output.png"))
        report["checks"]["entry1_lecture_and_lab_navigation"] = True

        # Reset file scope back to whole course so fallback search sees full course.
        _ab_eval(
            browser_a,
            "(() => { const b = document.getElementById('clear-file'); "
            "if (b && !b.hidden) b.click(); return true; })()",
        )
        _wait_for_js(
            browser_a,
            "document.getElementById('clear-file').hidden === true",
            desc="course scope cleared before partial fallback",
        )

        # 3. Partial concept fallback -> Spectral Gatekeeper -> click "用常规检索查找"!
        _ab(browser_a, "click", "#kg-list .kg-item[data-concept-key='spectral_gatekeeper']")
        _wait_for_js(
            browser_a,
            "document.querySelector('#kg-detail .kg-fallback-search') !== null",
            desc="partial concept fallback button visible",
        )
        msg_count_before = _ab_eval(
            browser_a, "document.querySelectorAll('#messages article').length"
        )
        _ab(browser_a, "click", "#kg-detail .kg-fallback-search")
        _wait_for_js(
            browser_a,
            f"document.querySelectorAll('#messages article').length >= {msg_count_before + 2} "
            "&& (document.getElementById('composer-status')?.textContent || '') === '' "
            "&& !document.querySelector('#turn-actions button')",
            desc="conventional search turn completed via fallback button",
        )
        report["checks"]["partial_concept_fallback_to_search"] = True

        # 4. Entry 2: Submit cross-file question & check concept chip -> click navigates KG detail!
        _ab(browser_a, "fill", "#question", "Adaptive Notch Filter attenuation")
        _ab(browser_a, "click", "#send")
        _wait_for_js(
            browser_a,
            "(() => { const c = document.querySelector("
            "'#messages .kg-concept-chip[data-concept-key=\"adaptive_notch_filter\"]'); "
            "const text = document.getElementById('messages')?.textContent || ''; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "return c && text.includes('-34.2 dB') && st === '' "
            "&& !document.querySelector('#turn-actions button'); })()",
            desc="cross-file answer + Adaptive Notch Filter chip",
        )
        _ab(browser_a, "fill", "#question", "Transient Probe")
        _ab(browser_a, "click", "#send")
        _wait_for_js(
            browser_a,
            "(() => document.querySelector("
            "'#messages .kg-concept-chip[data-concept-key=\"transient_probe\"]') !== null "
            "&& (document.getElementById('composer-status')?.textContent || '') === '' "
            "&& !document.querySelector('#turn-actions button'))()",
            desc="Transient Probe chip rendered in message",
        )
        _ab(browser_a, "eval", "document.getElementById('kg-detail').hidden = true")
        _ab(
            browser_a,
            "click",
            "#messages .kg-concept-chip[data-concept-key='adaptive_notch_filter']",
        )
        _wait_for_js(
            browser_a,
            "(() => { const txt = document.getElementById('kg-detail')?.textContent || ''; "
            "return txt.includes('attenuation=-34.2 dB') "
            "&& txt.includes('Lab01-SignalLab.ipynb'); })()",
            desc="Entry 2 chip opened concept detail",
        )
        _ab(browser_a, "screenshot", str(shots / "04-entry2-message-chip.png"))
        report["checks"]["entry2_message_chip_navigation"] = True

        # 5. Version update (Lab01 -> v2, Lecture03 -> v2 removing Transient Probe),
        #    verify stale historical chip degrades gracefully WITHOUT global banner!
        docs_info = _ab_eval(
            browser_a,
            "(async () => { "
            f"const r = await fetch('/api/workspace/sessions/{session_a_id}/catalog'); "
            "const j = await r.json(); "
            "const c = j.courses.find(x => x.course_id === 'cs-signal-kg'); "
            "return Object.fromEntries(c.documents.map(d => [d.file_name, d])); })()",
        )
        lab_doc = docs_info["Lab01-SignalLab.ipynb"]
        lec3_doc = docs_info["Lecture03-Transient.pdf"]

        _ab_eval(
            browser_a,
            f"(() => {{ const r = document.querySelector("
            f"'#catalog [data-node-id=\"{lab_doc['node_id']}\"]'); "
            "if (r) r.click(); return !!r; })()",
        )
        _wait_for_js(
            browser_a,
            "!document.getElementById('replace-label').hidden",
            desc="replace checkbox visible for selected Lab01",
        )
        _ab(browser_a, "check", "#replace-document")
        _ab(browser_a, "upload", "#import-files", str(fixtures["lab_v2"]))
        _ab(browser_a, "click", "#import-button")
        _wait_for_js(
            browser_a,
            "document.getElementById('import-results')?.textContent?.includes('v2')",
            desc="Lab01 v2 uploaded via UI replace flow",
        )

        _ab_eval(
            browser_a,
            f"(() => {{ const r = document.querySelector("
            f"'#catalog [data-node-id=\"{lec3_doc['node_id']}\"]'); "
            "if (r) r.click(); return !!r; })()",
        )
        _wait_for_js(
            browser_a,
            "document.getElementById('location-title')?.textContent"
            "?.includes('Lecture03-Transient.pdf')",
            desc="Lecture03 selected",
        )
        _ab(browser_a, "check", "#replace-document")
        _ab(browser_a, "upload", "#import-files", str(fixtures["lec3_v2"]))
        _ab(browser_a, "click", "#import-button")
        _wait_for_js(
            browser_a,
            "document.getElementById('import-results')?.textContent?.includes('v2')",
            desc="Lecture03 v2 uploaded via UI replace flow",
        )

        _ab(browser_a, "reload")
        _wait_for_js(
            browser_a,
            "document.querySelector('#messages .kg-concept-chip"
            "[data-concept-key=\"transient_probe\"][data-stale=\"true\"]') !== null",
            desc="historical Transient Probe chip flagged stale after reload",
        )

        _ab(
            browser_a,
            "click",
            "#messages .kg-concept-chip[data-concept-key='transient_probe']",
        )
        _wait_for_js(
            browser_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "const st = document.getElementById('workspace-status'); "
            "return d && !d.hidden && d.querySelector('.kg-fallback-search') !== null "
            "&& st && st.hidden === true; })()",
            desc="stale chip graceful in-panel fallback without global error banner",
        )
        _ab(browser_a, "screenshot", str(shots / "05-stale-chip-graceful-fallback.png"))

        _ab(browser_a, "click", "#kg-list .kg-item[data-concept-key='adaptive_notch_filter']")
        _wait_for_js(
            browser_a,
            "document.getElementById('kg-detail')?.textContent?.includes('-48.5 dB') "
            "&& document.getElementById('kg-detail')?.textContent?.includes('v2')",
            desc="KG detail switched to Lab v2 (-48.5 dB)",
        )
        _ab_eval(
            browser_a,
            "(() => { const b = document.getElementById('clear-file'); "
            "if (b && !b.hidden) b.click(); return true; })()",
        )
        _wait_for_js(
            browser_a,
            "document.getElementById('clear-file').hidden === true",
            desc="full course scope restored after v2 upload",
        )
        _ab(browser_a, "fill", "#question", "Adaptive Notch Filter v2 measurement")
        _ab(browser_a, "click", "#send")
        _wait_for_js(
            browser_a,
            "document.querySelector('#messages article:last-child')?.textContent"
            "?.includes('-48.5 dB')",
            desc="new QA turn uses v2 (-48.5 dB)",
        )
        _ab_eval(
            browser_a,
            "(() => { const b = [...document.querySelectorAll('#messages .citations button')]"
            ".find(x => x.textContent.includes('Lab01-SignalLab.ipynb') "
            "&& x.textContent.includes('v1')); if (!b) throw new Error('missing Lab v1 citation'); "
            "b.click(); return true; })()",
        )
        _wait_for_js(
            browser_a,
            "document.getElementById('reader').open "
            "&& document.getElementById('reader-title')?.textContent?.includes('v1') "
            "&& document.getElementById('reader-location')?.textContent?.includes('历史版本')",
            desc="historical v1 message evidence still viewable",
        )
        _ab(browser_a, "screenshot", str(shots / "06-v2-new-and-v1-history.png"))
        report["checks"]["version_update_and_stale_chip_fallback"] = True

        # 6. Cross-session / cross-owner 404 isolation using independent browser session B.
        _ab(browser_b, "open", f"{base_url}/workspace/")
        session_b_id = _wait_for_js(
            browser_b,
            "(() => { const s = document.getElementById('sessions'); "
            "return s && !s.disabled && s.value ? s.value : null; })()",
            desc="second isolated browser session created",
        )
        cross_status = _ab_eval(
            browser_b,
            f"(async () => {{ "
            f"const r1 = await fetch('/api/workspace/sessions/{session_a_id}/kg/concepts'); "
            "const r2 = await fetch("
            f"'/api/workspace/sessions/{session_a_id}/kg/concepts/adaptive_notch_filter'); "
            "return [r1.status, r2.status]; })()",
        )
        assert cross_status == [404, 404], f"Expected [404, 404], got {cross_status}"
        unauth_code, _ = _http_status(
            f"{base_url}/api/workspace/sessions/{session_a_id}/kg/concepts"
        )
        assert unauth_code == 404, f"Expected unauth 404, got {unauth_code}"
        report["checks"]["cross_owner_404"] = {
            "session_b_id": session_b_id,
            "statuses": cross_status,
        }
        report["checks"]["document_ids"] = {
            "lab_v1": lab_doc["document_id"],
            "lec3_v1": lec3_doc["document_id"],
        }
        report["status"] = "PASSED"
        summary_path = work_dir / "verification_summary.json"
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        for bs in (browser_a, browser_b):
            subprocess.run(
                ["agent-browser", "--session", bs, "close"],
                check=False,
                capture_output=True,
            )
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="H02 real-Chromium dual-entry KG verification")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/srag-h-web-verify"))
    args = parser.parse_args()
    run_verification(args.work_dir.resolve())


if __name__ == "__main__":
    main()
