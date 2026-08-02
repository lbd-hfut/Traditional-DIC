"""Render the bilingual Markdown manuals as operation-manual PDFs."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent


def build_styles(chinese: bool):
    if chinese:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font = "STSong-Light"
    else:
        font = "Helvetica"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ManualTitle", parent=styles["Title"], fontName=font, fontSize=25, leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=18))
    styles.add(ParagraphStyle("ManualSub", parent=styles["Normal"], fontName=font, fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#52616B")))
    styles.add(ParagraphStyle("ManualH1", parent=styles["Heading1"], fontName=font, fontSize=17, leading=23, textColor=colors.HexColor("#17324D"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle("ManualH2", parent=styles["Heading2"], fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#245C78"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("ManualBody", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle("ManualBullet", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=14, leftIndent=14, firstLineIndent=-9, spaceAfter=3))
    styles.add(ParagraphStyle("ManualTOC", parent=styles["BodyText"], fontName=font, fontSize=10, leading=16, leftIndent=8))
    return styles, font


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B6C4CC"))
    canvas.line(doc.leftMargin, 1.2 * cm, A4[0] - doc.rightMargin, 1.2 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(doc.leftMargin, 0.78 * cm, "Traditional-DIC Operation Manual")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.78 * cm, f"Page {doc.page}")
    canvas.restoreState()


def paragraph(text: str, style):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    return Paragraph(text, style)


def make_table(lines, styles):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
            continue
        rows.append([paragraph(cell, styles["ManualBody"]) for cell in cells])
    widths = [(A4[0] - 3.6 * cm) / max(1, len(rows[0]))] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF2")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AABBC5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def render(markdown: Path, output: Path, chinese: bool):
    styles, font = build_styles(chinese)
    text = markdown.read_text(encoding="utf-8").splitlines()
    headings = [(len(line) - len(line.lstrip("#")), line.lstrip("#").strip()) for line in text if line.startswith("#")]
    story = [Spacer(1, 4.0 * cm), Paragraph("Traditional-DIC", styles["ManualTitle"]), Paragraph("Operation Manual" if not chinese else "操作使用手册", styles["ManualSub"]), Spacer(1, 0.8 * cm), Paragraph("English edition" if not chinese else "中文版", styles["ManualSub"]), Spacer(1, 6.0 * cm), Paragraph("Generated from the project documentation", styles["ManualSub"]), PageBreak(), Paragraph("Contents" if not chinese else "目录", styles["ManualH1"])]
    for level, title in headings:
        if level <= 2:
            story.append(Paragraph(("&nbsp;" * max(0, level - 1) * 4) + html.escape(title), styles["ManualTOC"]))
    story.append(PageBreak())
    index = 0
    while index < len(text):
        line = text[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            block = []
            index += 1
            while index < len(text) and not text[index].startswith("```"):
                block.append(text[index])
                index += 1
            story.append(Preformatted("\n".join(block), ParagraphStyle("Code", fontName="Courier", fontSize=7.6, leading=10, backColor=colors.HexColor("#F3F6F8"), borderColor=colors.HexColor("#CFD9DE"), borderWidth=0.5, borderPadding=7)))
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(html.escape(line[level:].strip()), styles["ManualH1"] if level == 1 else styles["ManualH2"]))
        elif line.startswith("|"):
            rows = []
            while index < len(text) and text[index].startswith("|"):
                rows.append(text[index])
                index += 1
            story.append(make_table(rows, styles))
            index -= 1
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            story.append(paragraph("• " + re.sub(r"^\d+\. ", "", line[2:] if line.startswith("- ") else line), styles["ManualBullet"]))
        else:
            story.append(paragraph(line, styles["ManualBody"]))
        index += 1
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm, title="Traditional-DIC Operation Manual", author="Traditional-DIC")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    render(ROOT / "user_manual.md", ROOT / "Traditional-DIC_User_Manual_EN.pdf", False)
    render(ROOT / "user_manual_zh.md", ROOT / "Traditional-DIC_User_Manual_ZH.pdf", True)
