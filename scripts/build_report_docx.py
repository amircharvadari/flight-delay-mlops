"""Build the Word report from the Markdown source (report/report_draft.md).

Run (needs pandoc and python-docx; ooxml_order.py next to this file):
    python scripts/build_report_docx.py report/report_draft.md report/figures report/report_DRAFT.docx \
        --template report/template/AIN_4311_Research_Project_Template.docx

    python build_report_docx.py <report_draft.md> <figures_dir> <out.docx> [--template T.docx]

The Markdown file stays the single source of truth; this script only converts.
- pandoc converts the body (headings, tables, equations, figures); with
  --template the university template is used as pandoc's reference document,
  so its styles (fonts, headings, body text, page set-up) apply.
- python-docx then adds the template's front matter: title page, TABLE OF
  CONTENTS, LIST OF TABLES and LIST OF FIGURES (Word fields, updated when the
  document is opened), page numbers, and a DRAFT marking on every page.
- If a template is given, its title-page paragraphs are copied and their text
  replaced, so the title page keeps the template's exact formatting.
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

DRAFT_TEXT = "DRAFT - pending the author's review - not for submission"
TITLE_LINES = [
    ("T.C.", True),
    ("BAHÇEŞEHİR UNIVERSITY", True),
    ("AIN4312 Special Topics in Artificial Intelligence Engineering 26/27", True),
    None,  # title from the Markdown file
    None,  # author line from the Markdown file
    None,  # place and date line from the Markdown file
]


def split_markdown(md: str) -> tuple[str, str, str, str]:
    """Return (title, author line, place/date line, body starting at # ABSTRACT)."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL).strip()
    head, body = md.split("\n# ABSTRACT", 1)
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    title = lines[0].lstrip("# ").strip()
    return title, lines[1], lines[2], "# ABSTRACT" + body


def add_field(paragraph, instr: str) -> None:
    """Insert a Word field (e.g. TOC, PAGE) into a paragraph."""
    run = paragraph.add_run()
    for kind, text in (("begin", None), ("instr", instr), ("separate", None),
                       ("text", "Right-click and choose Update Field."), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = f" {text} "
            run._r.append(el)
        elif kind == "text":
            t = OxmlElement("w:t")
            t.text = text
            run._r.append(t)
        else:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            run._r.append(fc)


def insert_paragraph_before(par, text: str = "", style: str | None = None):
    new = par.insert_paragraph_before(text)
    if style:
        new.style = style
    return new


def ensure_caption_styles(doc) -> None:
    styles = doc.styles
    base = styles["Caption"] if "Caption" in [s.name for s in styles] else styles["Normal"]
    for name in ("Table Caption", "Figure Caption"):
        if name not in [s.name for s in styles]:
            st = styles.add_style(name, 1)
            st.base_style = base
            st.font.bold = True
            st.font.size = Pt(10)


def mark_captions(doc) -> tuple[int, int]:
    """Paragraphs that start with 'Table n.' / 'Figure n.' get caption styles."""
    ensure_caption_styles(doc)
    nt = nf = 0
    for p in doc.paragraphs:
        txt = p.text.strip()
        if re.match(r"^Table \d+\. ", txt):
            p.style = doc.styles["Table Caption"]
            nt += 1
        elif re.match(r"^Figure \d+\. ", txt):
            p.style = doc.styles["Figure Caption"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            nf += 1
    return nt, nf


def page_break_before(par) -> None:
    br = insert_paragraph_before(par)
    br.add_run().add_break(WD_BREAK.PAGE)


def build_front_matter(doc, title, author, place, template: Path | None) -> None:
    body_first = doc.paragraphs[0]  # "ABSTRACT" heading produced by pandoc
    texts = ["T.C.", "BAHÇEŞEHİR UNIVERSITY",
             "AIN4312 Special Topics in Artificial Intelligence Engineering 26/27",
             title, author, place]
    if template is not None:
        tdoc = Document(str(template))
        tpars = []
        for p in tdoc.paragraphs:
            if p.text.strip().upper().startswith("ABSTRACT"):
                break
            tpars.append(p)
        # template title block: T.C., university, course, title, three name lines, place/date
        mapping = [texts[0], texts[1], texts[2], texts[3], texts[4], "", "", texts[5]]
        for i, tp in enumerate(tpars):
            el = copy.deepcopy(tp._p)
            body_first._p.addprevious(el)
        # replace text of the copied non-empty paragraphs in order
        copied = []
        for p in doc.paragraphs:
            if p._p is body_first._p:
                break
            copied.append(p)
        k = 0
        for p in copied:
            if p.text.strip():
                new = mapping[k] if k < len(mapping) else ""
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
                if p.runs:
                    p.runs[0].text = new
                k += 1
        draft = insert_paragraph_before(body_first, DRAFT_TEXT)
    else:
        for i, t in enumerate(texts):
            p = insert_paragraph_before(body_first)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(t)
            r.bold = True
            r.font.size = Pt(18 if i == 3 else 13)
            p.paragraph_format.space_after = Pt(36 if i in (2, 3, 4) else 6)
        draft = insert_paragraph_before(body_first, DRAFT_TEXT)
    draft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in draft.runs:
        r.bold = True
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    page_break_before(body_first)


def build_lists(doc) -> None:
    """TABLE OF CONTENTS, LIST OF TABLES, LIST OF FIGURES before LIST OF ABBREVIATIONS."""
    target = next(p for p in doc.paragraphs if p.text.strip() == "LIST OF ABBREVIATIONS")
    heading_style = target.style
    for title, instr in (("TABLE OF CONTENTS", 'TOC \\o "1-3" \\h \\z \\u'),
                         ("LIST OF TABLES", 'TOC \\h \\z \\t "Table Caption,1"'),
                         ("LIST OF FIGURES", 'TOC \\h \\z \\t "Figure Caption,1"')):
        page_break_before(target)
        h = insert_paragraph_before(target, title)
        h.style = heading_style
        add_field(insert_paragraph_before(target), instr)
    page_break_before(target)
    # each numbered main section starts on a new page, as in the template
    for p in doc.paragraphs:
        if p.style.style_id == heading_style.style_id and re.match(r"^(\d+\. |ACKNOWLEDGEMENTS|REFERENCES|APPENDIX)", p.text.strip()):
            page_break_before(p)


def format_tables(doc) -> None:
    """Readable tables: 9 pt text, header row repeated, widths in proportion to content."""
    from docx.shared import Emu, Inches
    sec = doc.sections[0]
    if sec.page_width and sec.left_margin is not None and sec.right_margin is not None:
        usable = sec.page_width - sec.left_margin - sec.right_margin
    else:
        usable = Inches(6.5)
    for t in doc.tables:
        rows = t.rows
        ncol = len(t.columns)
        longest = [1] * ncol
        for r in rows:
            for j, c in enumerate(r.cells[:ncol]):
                words = c.text.split() or [""]
                longest[j] = max(longest[j], min(len(c.text), 40), max(len(w) for w in words))
        total = sum(longest)
        for r in rows:
            for j, c in enumerate(r.cells[:ncol]):
                c.width = Emu(int(usable * longest[j] / total))
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(9)
        tr_pr = rows[0]._tr.get_or_add_trPr()
        h = OxmlElement("w:tblHeader")
        h.set(qn("w:val"), "true")
        tr_pr.append(h)


def header_footer(doc) -> None:
    for section in doc.sections:
        hp = section.header.paragraphs[0] if section.header.paragraphs else section.header.add_paragraph()
        hp.text = ""
        r = hp.add_run(DRAFT_TEXT)
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        fp = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(fp, "PAGE")


SETTINGS_ORDER = ['writeProtection', 'view', 'zoom', 'removePersonalInformation', 'removeDateAndTime', 'doNotDisplayPageBoundaries', 'displayBackgroundShape', 'printPostScriptOverText', 'printFractionalCharacterWidth', 'printFormsData', 'embedTrueTypeFonts', 'embedSystemFonts', 'saveSubsetFonts', 'saveFormsData', 'mirrorMargins', 'alignBordersAndEdges', 'bordersDoNotSurroundHeader', 'bordersDoNotSurroundFooter', 'gutterAtTop', 'hideSpellingErrors', 'hideGrammaticalErrors', 'activeWritingStyle', 'proofState', 'formsDesign', 'attachedTemplate', 'linkStyles', 'stylePaneFormatFilter', 'stylePaneSortMethod', 'documentType', 'mailMerge', 'revisionView', 'trackRevisions', 'doNotTrackMoves', 'doNotTrackFormatting', 'documentProtection', 'autoFormatOverride', 'styleLockTheme', 'styleLockQFSet', 'defaultTabStop', 'autoHyphenation', 'consecutiveHyphenLimit', 'hyphenationZone', 'doNotHyphenateCaps', 'showEnvelope', 'summaryLength', 'clickAndTypeStyle', 'defaultTableStyle', 'evenAndOddHeaders', 'bookFoldRevPrinting', 'bookFoldPrinting', 'bookFoldPrintingSheets', 'drawingGridHorizontalSpacing', 'drawingGridVerticalSpacing', 'displayHorizontalDrawingGridEvery', 'displayVerticalDrawingGridEvery', 'doNotUseMarginsForDrawingGridOrigin', 'drawingGridHorizontalOrigin', 'drawingGridVerticalOrigin', 'doNotShadeFormData', 'noPunctuationKerning', 'characterSpacingControl', 'printTwoOnOne', 'strictFirstAndLastChars', 'noLineBreaksAfter', 'noLineBreaksBefore', 'savePreviewPicture', 'doNotValidateAgainstSchema', 'saveInvalidXml', 'ignoreMixedContent', 'alwaysShowPlaceholderText', 'doNotDemarcateInvalidXml', 'saveXmlDataOnly', 'useXSLTWhenSaving', 'saveThroughXslt', 'showXMLTags', 'alwaysMergeEmptyNamespace', 'updateFields', 'hdrShapeDefaults', 'footnotePr', 'endnotePr', 'compat', 'docVars', 'rsids', 'mathPr', 'attachedSchema', 'themeFontLang', 'clrSchemeMapping', 'doNotIncludeSubdocsInStats', 'doNotAutoCompressPictures', 'forceUpgrade', 'captions', 'readModeInkLockDown', 'smartTagType', 'schemaLibrary', 'shapeDefaults', 'doNotEmbedSmartTags', 'decimalSymbol', 'listSeparator']


def fix_settings(xml: str) -> str:
    """Add updateFields and put the children of w:settings in schema order."""
    from lxml import etree
    root = etree.fromstring(xml.encode("utf-8"))
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    if root.find(f"{{{W}}}updateFields") is None:
        el = etree.SubElement(root, f"{{{W}}}updateFields")
        el.set(f"{{{W}}}val", "true")
    def rank(child):
        name = etree.QName(child).localname
        return SETTINGS_ORDER.index(name) if name in SETTINGS_ORDER else len(SETTINGS_ORDER)
    children = sorted(root, key=rank)
    for c in list(root):
        root.remove(c)
    for c in children:
        root.append(c)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")


def update_fields_on_open(path: Path) -> None:
    """Ask Word to update the TOC and list fields when the file is opened."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.unpack_archive(str(path), tmp, "zip")
        settings = tmp / "word" / "settings.xml"
        xml = settings.read_text(encoding="utf-8")
        settings.write_text(fix_settings(xml), encoding="utf-8")
        from ooxml_order import normalise
        for part in ("document.xml", "styles.xml", "numbering.xml"):
            f = tmp / "word" / part
            if f.exists():
                f.write_bytes(normalise(f.read_bytes()))
        path.unlink()
        subprocess.run(["zip", "-qXr", str(path), "."], cwd=tmp, check=True)


# --- template mode (official AIN4312 template) ----------------------------------------

CODE_FONT = "Courier New"

TITLE_PLACEHOLDERS = {
    "The TITLE of the project goes here": "title",
    "The Names of the Students and their Departments go here,": "author",
    "In Alphabetical Order by Surname,": "",
    "Capitalize The First Letters Only": "",
    "ISTANBUL, Date": "place",
}


def _set_text(par, text: str) -> None:
    """Replace a paragraph's text but keep the formatting of its first run."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def template_title_block(doc, template: Path, values: dict[str, str]) -> None:
    """Copy the template's title page (logo included) in front of ABSTRACT."""
    import io

    tdoc = Document(str(template))
    anchor = next(p for p in doc.paragraphs if p.text.strip() == "ABSTRACT")
    block = []
    for p in tdoc.paragraphs:
        if p.text.strip().upper().startswith("ABSTRACT"):
            break
        block.append(p)
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    for tp in block:
        el = copy.deepcopy(tp._p)
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            old = blip.get(R + "embed")
            if old:
                blob = tdoc.part.related_parts[old].blob
                new_rid, _ = doc.part.get_or_add_image(io.BytesIO(blob))
                blip.set(R + "embed", new_rid)
        anchor._p.addprevious(el)
    for p in list(doc.paragraphs):
        if p._p is anchor._p:
            break
        key = p.text.strip()
        if key in TITLE_PLACEHOLDERS:
            role = TITLE_PLACEHOLDERS[key]
            _set_text(p, values.get(role, "") if role else "")
            if role == "place":
                d = insert_paragraph_before(anchor, DRAFT_TEXT)
                d.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in d.runs:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    page_break_before(anchor)


def seq_captions(doc) -> tuple[int, int]:
    """'Table n. ...' / 'Figure n. ...' -> template Caption style with SEQ fields."""
    counts = {"Table": 0, "Figure": 0}
    for p in doc.paragraphs:
        m = re.match(r"^(Table|Figure) (\d+)\. (.*)$", p.text.strip(), re.DOTALL)
        if not m:
            continue
        kind, num, rest = m.group(1), m.group(2), m.group(3)
        counts[kind] += 1
        if int(num) != counts[kind]:
            raise ValueError(f"{kind} {num} is out of order (expected {counts[kind]})")
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        p.style = doc.styles["Caption"]
        p.add_run(f"{kind} ")
        run = p.add_run()
        for kind_, val in (("begin", None), ("instr", f"SEQ {kind} \\* ARABIC"), ("separate", None),
                           ("text", num), ("end", None)):
            if kind_ == "instr":
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = f" {val} "
            elif kind_ == "text":
                el = OxmlElement("w:t")
                el.text = val
            else:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind_)
            run._r.append(el)
        p.add_run(f". {rest}")
        if kind == "Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return counts["Table"], counts["Figure"]


def template_lists(doc) -> None:
    """TOC and the two lists as in the template, before LIST OF ABBREVIATIONS."""
    target = next(p for p in doc.paragraphs if p.text.strip() == "LIST OF ABBREVIATIONS")
    h1 = target.style
    for title, instr in (("TABLE OF CONTENTS", 'TOC \\o "1-3" \\h \\z \\u'),
                         ("LIST OF TABLES", 'TOC \\h \\z \\c "Table"'),
                         ("LIST OF FIGURES", 'TOC \\h \\z \\c "Figure"')):
        page_break_before(target)
        h = insert_paragraph_before(target, title)
        h.style = h1
        add_field(insert_paragraph_before(target), instr)
    page_break_before(target)
    for p in doc.paragraphs:
        if p.style.style_id == h1.style_id and re.match(
                r"^([2-9]|1[0-3])\. |^(ACKNOWLEDGEMENTS|REFERENCES|APPENDIX)", p.text.strip()):
            page_break_before(p)


def _plain_style(doc, name: str, size: float | None):
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Cm
    names = [s.name for s in doc.styles]
    st = doc.styles[name] if name in names else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = doc.styles["Normal"]
    pf = st.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.line_spacing = 1.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(2)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if size:
        st.font.size = Pt(size)
    return st


def _code_styles(doc) -> None:
    """Monospace, single-spaced, left-aligned code (the template defines no code styles)."""
    from docx.enum.style import WD_STYLE_TYPE
    code = _plain_style(doc, "Source Code", 9)
    code.font.name = CODE_FONT
    names = [s.name for s in doc.styles]
    if "Verbatim Char" not in names:
        doc.styles.add_style("Verbatim Char", WD_STYLE_TYPE.CHARACTER)
    doc.styles["Verbatim Char"].font.name = CODE_FONT


def template_body_styles(doc) -> None:
    """Body text in the template's Normal style; tables with grid borders."""
    table_text = _plain_style(doc, "Table Text", 9)
    _plain_style(doc, "Compact", None)
    _code_styles(doc)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for cp in c.paragraphs:
                    cp.style = table_text
    normal = doc.styles["Normal"]
    for p in doc.paragraphs:
        if p.style.name in ("Body Text", "First Paragraph"):
            p.style = normal
    names = [s.name for s in doc.styles]
    if "Table Grid" in names:
        for t in doc.tables:
            t.style = doc.styles["Table Grid"]


def draft_header_only(doc) -> None:
    """Add the DRAFT line to the header; keep the template's footer (page number)."""
    for section in doc.sections:
        hp = section.header.add_paragraph()
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = hp.add_run(DRAFT_TEXT)
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        footer_xml = section.footer._element.xml   # includes content controls (w:sdt)
        if "PAGE" not in footer_xml:
            fp = section.footer.add_paragraph()
            fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_field(fp, "PAGE")


def english_title_language(doc) -> None:
    """Title-page runs in English: the template's capitalised title is tagged
    Turkish, which turns i into the dotted capital I (e.g. PIPELINE)."""
    stop = next(p for p in doc.paragraphs if p.text.strip() == "ABSTRACT")
    for p in doc.paragraphs:
        if p._p is stop._p:
            break
        for r in p.runs:
            rpr = r._r.get_or_add_rPr()
            for el in rpr.findall(qn("w:lang")):
                rpr.remove(el)
            lang = OxmlElement("w:lang")
            lang.set(qn("w:val"), "en-US")
            rpr.append(lang)


def template_sections(doc) -> None:
    """Front matter: lower-roman page numbers, none on the title page.
    Body from section 1: arabic numbers starting at 1 (as in the template)."""
    body_sect = doc.sections[-1]._sectPr
    first_chapter = next(p for p in doc.paragraphs if re.match(r"^1\. OVERVIEW", p.text.strip()))
    start = first_chapter._p
    while start.getprevious() is not None and start.getprevious().tag == qn("w:bookmarkStart"):
        start = start.getprevious()
    prev = OxmlElement("w:p")              # empty paragraph that ends the front section
    start.addprevious(prev)
    # the page-break paragraph before chapter 1 becomes the end of the front section
    front = copy.deepcopy(body_sect)
    for el in front.findall(qn("w:pgNumType")):
        front.remove(el)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), "lowerRoman")
    pg.set(qn("w:start"), "1")
    front.append(pg)
    if front.find(qn("w:titlePg")) is None:
        front.append(OxmlElement("w:titlePg"))
    ppr = prev.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        prev.insert(0, ppr)
    ppr.append(front)
    for el in body_sect.findall(qn("w:pgNumType")):
        body_sect.remove(el)
    pg2 = OxmlElement("w:pgNumType")
    pg2.set(qn("w:fmt"), "decimal")
    pg2.set(qn("w:start"), "1")
    body_sect.append(pg2)
    for el in body_sect.findall(qn("w:titlePg")):
        body_sect.remove(el)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("figures")
    ap.add_argument("out")
    ap.add_argument("--template")
    a = ap.parse_args()
    md = Path(a.markdown).read_text(encoding="utf-8")
    title, author, place, body = split_markdown(md)
    work = Path(tempfile.mkdtemp())
    (work / "body.md").write_text(body, encoding="utf-8")
    shutil.copytree(a.figures, work / "figures")
    cmd = ["pandoc", "body.md", "-f", "markdown+tex_math_dollars+pipe_tables-implicit_figures",
           "-o", "body.docx"]
    if a.template:
        cmd += ["--reference-doc", str(Path(a.template).resolve())]
    subprocess.run(cmd, cwd=work, check=True)
    doc = Document(str(work / "body.docx"))
    if a.template:
        nt, nf = seq_captions(doc)
        template_body_styles(doc)
        template_title_block(doc, Path(a.template), {"title": title, "author": author, "place": place})
        template_lists(doc)
        format_tables(doc)
        draft_header_only(doc)
        template_sections(doc)
        english_title_language(doc)
    else:
        nt, nf = mark_captions(doc)
        build_front_matter(doc, title, author, place, None)
        build_lists(doc)
        format_tables(doc)
        header_footer(doc)
    out = Path(a.out).resolve()
    doc.save(str(out))
    update_fields_on_open(out)
    print(f"written {out}: {nt} table captions, {nf} figure captions")


if __name__ == "__main__":
    main()
