#!/usr/bin/env python3
"""
Convert TECHNICAL_REPORT_PROSIT3.md (or another .md) to .docx using markdown + python-docx.

Dependencies: pip install markdown python-docx

Usage (from repo root):
  python scripts/md_to_docx.py
  python scripts/md_to_docx.py path/to/report.md path/to/out.docx
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

import markdown
from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt
from docx.table import Table


def _set_cell_shading(cell, fill: str) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _add_horizontal_line(paragraph) -> None:
    p = paragraph._p
    p_pr = p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "auto")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


class DocxHTMLParser(HTMLParser):
    """Minimal HTML → python-docx mapping for reports (headings, p, lists, tables, code, img)."""

    def __init__(self, document: Document, base_dir: Path):
        super().__init__()
        self.doc = document
        self.base_dir = base_dir
        self._para_stack: list = []
        self._run_bold = 0
        self._run_italic = 0
        self._in_pre = False
        self._in_code = False
        self._in_li = False
        self._list_num = False
        self._table: Table | None = None
        self._row_cells: list | None = None
        self._pending_hr = False

    def _current_para(self):
        if self._para_stack:
            return self._para_stack[-1]
        p = self.doc.add_paragraph()
        self._para_stack.append(p)
        return p

    def _ensure_para(self):
        if not self._para_stack:
            p = self.doc.add_paragraph()
            self._para_stack.append(p)

    def _add_run(self, text: str):
        if not text:
            return
        self._ensure_para()
        p = self._current_para()
        run = p.add_run(text)
        if self._run_bold:
            run.bold = True
        if self._run_italic:
            run.italic = True
        if self._in_pre or self._in_code:
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        return run

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1":
            self._flush_para()
            p = self.doc.add_heading(level=1)
            self._para_stack.append(p)
        elif tag == "h2":
            self._flush_para()
            p = self.doc.add_heading(level=2)
            self._para_stack.append(p)
        elif tag == "h3":
            self._flush_para()
            p = self.doc.add_heading(level=3)
            self._para_stack.append(p)
        elif tag == "h4":
            self._flush_para()
            p = self.doc.add_heading(level=4)
            self._para_stack.append(p)
        elif tag == "p":
            self._flush_para()
            p = self.doc.add_paragraph()
            self._para_stack.append(p)
        elif tag == "br":
            self._add_run("\n")
        elif tag == "strong" or tag == "b":
            self._run_bold += 1
        elif tag == "em" or tag == "i":
            self._run_italic += 1
        elif tag == "code":
            self._in_code = True
        elif tag == "pre":
            self._in_pre = True
            self._flush_para()
            p = self.doc.add_paragraph()
            self._para_stack.append(p)
        elif tag == "ul":
            self._list_num = False
        elif tag == "ol":
            self._list_num = True
        elif tag == "li":
            self._flush_para()
            p = self.doc.add_paragraph(style="List Number" if self._list_num else "List Bullet")
            self._para_stack.append(p)
            self._in_li = True
        elif tag == "a":
            self._link_href = attrs.get("href", "")
        elif tag == "table":
            self._flush_para()
            self._table = None  # created on first row
        elif tag == "tr":
            self._row_cells = []
        elif tag in ("th", "td"):
            self._cell_tag = tag
            self._cell_text = ""
        elif tag in ("thead", "tbody"):
            pass
        elif tag == "hr":
            self._pending_hr = True
        elif tag == "img":
            src = attrs.get("src", "")
            alt = attrs.get("alt", "")
            self._flush_para()
            path = (self.base_dir / src).resolve()
            if path.is_file():
                try:
                    p = self.doc.add_paragraph()
                    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    run = p.add_run()
                    run.add_picture(str(path), width=Inches(5.5))
                    if alt:
                        cap = self.doc.add_paragraph(alt)
                        cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                        for r in cap.runs:
                            r.italic = True
                            r.font.size = Pt(9)
                except Exception as e:
                    self.doc.add_paragraph(f"[Image missing or error: {src} — {e}]")
            else:
                self.doc.add_paragraph(f"[Image not found: {path}]")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "p", "li", "pre"):
            if tag == "pre":
                self._in_pre = False
            self._flush_para()
        elif tag in ("strong", "b"):
            self._run_bold = max(0, self._run_bold - 1)
        elif tag in ("em", "i"):
            self._run_italic = max(0, self._run_italic - 1)
        elif tag == "code":
            self._in_code = False
        elif tag == "a":
            if getattr(self, "_link_href", None):
                del self._link_href
        elif tag in ("th", "td"):
            text = getattr(self, "_cell_text", "")
            self._row_cells.append((tag, text))
            self._cell_tag = None
            if hasattr(self, "_cell_text"):
                del self._cell_text
        elif tag == "tr":
            if self._row_cells:
                ncols = len(self._row_cells)
                if self._table is None:
                    self._table = self.doc.add_table(rows=0, cols=ncols)
                    self._table.style = "Table Grid"
                row = self._table.add_row()
                for j, (ct, celltext) in enumerate(self._row_cells):
                    row.cells[j].text = celltext.strip()
                    if ct == "th":
                        for r in row.cells[j].paragraphs[0].runs:
                            r.bold = True
                        _set_cell_shading(row.cells[j], "E7E6E6")
            self._row_cells = None
        elif tag == "table":
            self._table = None
        elif tag == "hr":
            p = self.doc.add_paragraph()
            _add_horizontal_line(p)

    def handle_data(self, data):
        if getattr(self, "_cell_tag", None) in ("th", "td"):
            self._cell_text = getattr(self, "_cell_text", "") + data
            return
        if self._pending_hr:
            self._pending_hr = False
        self._add_run(data)

    def _flush_para(self):
        self._para_stack.clear()

    def close(self):
        self._flush_para()
        super().close()


def md_to_docx(md_path: Path, out_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "nl2br",
            "sane_lists",
        ],
    )
    # Wrap for parser (fragment)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    parser = DocxHTMLParser(doc, md_path.parent)
    # Feed wrapped HTML — use regex to fix unclosed br if any
    parser.feed(html)
    parser.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"Wrote {out_path}")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    default_md = repo / "TECHNICAL_REPORT_PROSIT3.md"
    default_out = repo / "TECHNICAL_REPORT_PROSIT3.docx"
    md_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_md
    out_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else default_out
    if not md_path.is_file():
        print(f"Not found: {md_path}", file=sys.stderr)
        sys.exit(1)
    md_to_docx(md_path, out_path)


if __name__ == "__main__":
    main()
