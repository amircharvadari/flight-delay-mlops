"""Put OOXML property elements in schema order (content unchanged)."""
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
ORDERS = {
    (W, "pPr"): "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
    (W, "rPr"): "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath",
    (W, "tcPr"): "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark",
    (W, "tblPr"): "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription",
    (W, "trPr"): "cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden",
    (W, "abstractNum"): "nsid multiLevelType tmpl name styleLink numStyleLink lvl",
    (W, "style"): "name aliases basedOn next link autoRedefine hidden uiPriority semiHidden unhideWhenUsed qFormat locked personal personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr",
    (M, "dPr"): "begChr sepChr endChr grow shp ctrlPr",
    (W, "sectPr"): "headerReference footerReference footnotePr endnotePr type pgSz pgMar paperSrc pgBorders lnNumType pgNumType cols formProt vAlign noEndnote titlePg textDirection bidi rtlGutter docGrid printerSettings sectPrChange",
    (M, "rPr"): "lit nor scr sty brk aln",
}
ORDERS = {k: v.split() for k, v in ORDERS.items()}


def normalise(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    for (ns, tag), order in ORDERS.items():
        for el in root.iter(f"{{{ns}}}{tag}"):
            kids = list(el)
            def rank(c, order=order, ns=ns):
                q = etree.QName(c)
                return order.index(q.localname) if q.localname in order else len(order)
            kids_sorted = sorted(kids, key=rank)
            if kids_sorted != kids:
                for c in kids:
                    el.remove(c)
                for c in kids_sorted:
                    el.append(c)
    # math run properties: <m:nor/> and <m:sty>/<m:scr> are alternatives
    for rpr in root.iter(f"{{{M}}}rPr"):
        if rpr.find(f"{{{M}}}nor") is not None:
            for t in ("sty", "scr"):
                for c in rpr.findall(f"{{{M}}}{t}"):
                    rpr.remove(c)
    # property containers hold elements only; drop stray text (e.g. a lone '>')
    for (ns, tag) in list(ORDERS):
        for el in root.iter(f"{{{ns}}}{tag}"):
            if el.text and el.text.strip():
                el.text = None
            for c in el:
                if c.tail and c.tail.strip():
                    c.tail = None
    # list ids must be 8 hex digits; pad shorter values (same number)
    for tag in ("nsid", "tmpl"):
        for el in root.iter(f"{{{W}}}{tag}"):
            v = el.get(f"{{{W}}}val")
            if v and len(v) < 8:
                el.set(f"{{{W}}}val", v.rjust(8, "0"))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
