"""Generate the public fictional Northpoint lecture for G14 and later checks."""

from __future__ import annotations

import argparse
from pathlib import Path

COURSE = "northpoint-completion"
FILENAME = "Northpoint-Completion-Briefing.pdf"
QUESTIONS = (
    "这份课件说结业有哪些时间可以选？项目签字赶不上春季截止，但七月十五日的暑期实习要正式结业证明，该怎么选？",
    "用中文把刚才的回答总结成两句话。",
)
EXPECTED = {
    3: ("May 31", "June 18"),
    5: ("August 23", "September 12"),
    7: ("December 8", "January 14"),
    8: ("July 15", "eligibility letter", "not an award certificate"),
}
PAGES = (
    (
        "Northpoint program completion briefing",
        "Northpoint School of Instrument Design",
        "This fictional briefing introduces administrative planning for the lab program.",
        "Completion dates and the rules for choosing a round appear in later sections.",
        "Keep a working copy of the program handbook and check all signed forms.",
    ),
    (
        "Directory and administrative contacts",
        "The lab office retains copies of project submissions and assessment forms.",
        "The Registry publishes the authorized record once an award is recorded.",
        "Use the general help desk for account and building access questions.",
    ),
    (
        "Spring award round",
        "The last accepted project sign-off is May 31.",
        "The spring award is recorded on June 18.",
        "Only after recording may the Registry issue an official award certificate.",
    ),
    (
        "Project sign-off workflow",
        "Submit the completed report to the project assessor and wait for approval.",
        "Uploading a draft does not satisfy the signed project requirement.",
        "Collect the approved form before selecting an administrative round.",
    ),
    (
        "Summer award round",
        "The last accepted project sign-off is August 23.",
        "The summer award is recorded on September 12.",
        "Select this round when spring has passed and sign-off arrives by August 23.",
    ),
    (
        "Ceremonies and records",
        "A ceremony invitation is separate from the official award record.",
        "A public celebration can be scheduled independently of the award date.",
        "Use the Registry record date for employment paperwork.",
    ),
    (
        "Winter award round",
        "The last accepted project sign-off is December 8.",
        "The winter award is recorded on January 14 of the following year.",
        "Use this round when the summer sign-off deadline has passed.",
    ),
    (
        "Choosing a round with an internship deadline",
        "For a July 15 internship that requires the official award certificate,",
        "a student who misses spring cannot supply that certificate by July 15.",
        "Ask the employer whether an eligibility letter from the Registry is acceptable.",
        "The eligibility letter is not an award certificate and must not be presented as one.",
        "If alternative proof is accepted, use the earliest round whose sign-off you can meet.",
        "If the official certificate is mandatory, ask to move the internship start date.",
    ),
)


def make_pdf(path: Path) -> None:
    """Write a deterministic eight-page PDF without additional dependencies."""

    def pdf_text(text: str) -> bytes:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return escaped.encode("ascii")

    children = " ".join(f"{4 + index * 2} 0 R" for index in range(len(PAGES)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{children}] /Count {len(PAGES)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for number, (heading, *lines) in enumerate(PAGES, start=1):
        stream_id = 2 * number + 3
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 650 440] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode()
        )
        stream = b"BT /F1 22 Tf 42 378 Td (" + pdf_text(heading) + b") Tj ET\n"
        for index, line in enumerate(lines):
            stream += (
                b"BT /F1 12 Tf 42 %d Td (" % (336 - index * 22) + pdf_text(line) + b") Tj ET\n"
            )
        stream += b"BT /F1 10 Tf 42 40 Td (Page %d) Tj ET\n" % number
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, item in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(b"%d 0 obj\n" % number + item + b"\nendobj\n")
    xref = len(output)
    output.extend(b"xref\n0 %d\n0000000000 65535 f \n" % len(offsets))
    for offset in offsets[1:]:
        output.extend(b"%010d 00000 n \n" % offset)
    output.extend(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(offsets), xref)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    make_pdf(parser.parse_args().directory / FILENAME)
