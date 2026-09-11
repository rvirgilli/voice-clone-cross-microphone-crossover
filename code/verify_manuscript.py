#!/usr/bin/env python3
"""Clean-build the manuscript and compare its rendered text with the released PDF."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "F2"
REQUIRED_TOOLS = ("pdflatex", "bibtex", "pdfinfo", "pdftotext")
FATAL_LOG_MARKERS = (
    "Undefined control sequence",
    "LaTeX Warning: Citation",
    "LaTeX Warning: Reference",
    "There were undefined references",
    "Overfull \\hbox",
    "Overfull \\vbox",
)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


def pdf_text(path: Path) -> str:
    return subprocess.run(
        ["pdftotext", str(path), "-"], text=True, capture_output=True, check=True
    ).stdout


def main() -> int:
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"manuscript verification needs: {', '.join(missing)}")
    with tempfile.TemporaryDirectory(prefix="f2-manuscript-") as raw:
        scratch = Path(raw) / "F2"
        shutil.copytree(PAPER, scratch, ignore=shutil.ignore_patterns(
            "*.aux", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.log"
        ))
        run(["./build.sh"], scratch)
        log = (scratch / "main.log").read_text(encoding="utf-8", errors="replace")
        found = [marker for marker in FATAL_LOG_MARKERS if marker in log]
        if found:
            raise AssertionError(f"manuscript log contains: {found}")
        info = run(["pdfinfo", "main.pdf"], scratch).stdout
        pages = [line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("Pages:")]
        # ICASSP allows four pages of technical content plus an optional fifth page
        # holding references only.
        if pages not in (["4"], ["5"]):
            raise AssertionError(f"expected a four- or five-page manuscript, observed {pages}")
        if pages == ["5"]:
            page4 = subprocess.run(["pdftotext", "-f", "4", "-l", "4", "main.pdf", "-"],
                                   cwd=scratch, text=True, capture_output=True, check=True).stdout
            page5 = subprocess.run(["pdftotext", "-f", "5", "-l", "5", "main.pdf", "-"],
                                   cwd=scratch, text=True, capture_output=True, check=True).stdout
            # ICASSP 2027 allows references, funding acknowledgements and the compliance
            # statement on page 5. Everything from the acknowledgement onward must be the
            # tail of the document, so no technical content can appear there.
            full = " ".join(pdf_text(scratch / "main.pdf").split())
            # The reference list may start on page 4 or at the top of page 5; either way the
            # whole of page 5 must be its tail, so no section text can appear there.
            marker = next((m for m in ("ACKNOWLEDGMENT", "COMPLIANCE WITH ETHICAL STANDARDS", "REFERENCES")
                           if m in full), None)
            references = full[full.index(marker):] if marker else ""
            tail = " ".join(page5.split())
            for heading_name in ("ACKNOWLEDGMENT", "COMPLIANCE WITH ETHICAL STANDARDS", "REFERENCES"):
                if heading_name in tail:
                    heading, rest = tail.split(heading_name, 1)
                    if heading.strip(" .0123456789"):
                        raise AssertionError("page 5 carries content the venue does not allow there")
                    tail = heading_name + rest
                    break
            if not references.endswith(tail):
                raise AssertionError("page 5 must contain references only")
        if pdf_text(scratch / "main.pdf") != pdf_text(PAPER / "main.pdf"):
            raise AssertionError("released PDF text does not match a clean build of released source")
    print(f"PASS — clean {pages[0]}-page build; no unresolved/overfull log markers; PDF text matches source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
