"""Read, edit, and export PPTX / DOCX / XLSX / PDF files.

Two capabilities:
  - extract_text(): pull readable text out of an uploaded file so the
    chat agent can answer questions about it.
  - apply_edit(): perform a small, explicit edit on an uploaded file
    (find/replace, or append content) and return the modified bytes.
  - build_report(): render an incident analysis into a downloadable
    PPTX / DOCX / XLSX / PDF report.
"""

import io
import logging
import re

from docx import Document as DocxDocument
from openpyxl import Workbook, load_workbook
from pptx import Presentation
from pptx.util import Inches, Pt
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".docx", ".pptx", ".xlsx", ".xlsm", ".pdf", ".txt", ".csv",
} | IMAGE_EXTENSIONS

REPORT_MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf": "application/pdf",
}


# =========================================================
# TEXT EXTRACTION (for reading / summarizing an upload)
# =========================================================

def extract_text(filename: str, data: bytes) -> str:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    buffer = io.BytesIO(data)

    if suffix == ".docx":
        document = DocxDocument(buffer)
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if suffix == ".pptx":
        presentation = Presentation(buffer)
        parts = []
        for index, slide in enumerate(presentation.slides, start=1):
            texts = [
                shape.text_frame.text
                for shape in slide.shapes
                if shape.has_text_frame and shape.text_frame.text.strip()
            ]
            if texts:
                parts.append(f"Slide {index}: " + " | ".join(texts))
        return "\n".join(parts)

    if suffix in {".xlsx", ".xlsm"}:
        workbook = load_workbook(filename=buffer, read_only=True, data_only=True)
        try:
            parts = []
            for worksheet in workbook.worksheets:
                parts.append(f"Sheet: {worksheet.title}")
                for row in worksheet.iter_rows(values_only=True, max_row=200):
                    if any(cell not in (None, "") for cell in row):
                        parts.append(" | ".join("" if cell is None else str(cell) for cell in row))
            return "\n".join(parts)
        finally:
            workbook.close()

    if suffix == ".pdf":
        reader = PdfReader(buffer)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".csv":
        return data.decode("utf-8", errors="ignore")

    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")

    if suffix in IMAGE_EXTENSIONS:
        import pytesseract
        from PIL import Image

        try:
            with Image.open(buffer) as image:
                return pytesseract.image_to_string(image).strip()
        except pytesseract.TesseractNotFoundError:
            raise ValueError(
                "The OCR engine (Tesseract) is not installed on this server, "
                "so images can't be scanned for text yet."
            )

    raise ValueError(f"Unsupported file type: {suffix or filename}")


# =========================================================
# EDITING AN UPLOADED FILE
# =========================================================

REPLACE_PATTERN = re.compile(
    r"""replace\s+["'](?P<old>.+?)["']\s+with\s+["'](?P<new>.+?)["']""",
    re.IGNORECASE,
)

APPEND_LEAD_WORDS = re.compile(
    r"^(?:add|append|insert)\b[:\s]*",
    re.IGNORECASE,
)


def _parse_edit_instruction(instruction: str) -> tuple[str, str, str]:
    """Return (mode, a, b). mode is 'replace' or 'append'."""
    match = REPLACE_PATTERN.search(instruction)
    if match:
        return "replace", match.group("old"), match.group("new")
    content = APPEND_LEAD_WORDS.sub("", instruction).strip()
    return "append", content, ""


def apply_edit(filename: str, data: bytes, instruction: str) -> bytes:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mode, first, second = _parse_edit_instruction(instruction)
    buffer = io.BytesIO(data)

    if suffix == ".docx":
        document = DocxDocument(buffer)
        if mode == "replace":
            for paragraph in document.paragraphs:
                if first in paragraph.text:
                    for run in paragraph.runs:
                        if first in run.text:
                            run.text = run.text.replace(first, second)
        else:
            document.add_paragraph(first)
        output = io.BytesIO()
        document.save(output)
        return output.getvalue()

    if suffix == ".pptx":
        presentation = Presentation(buffer)
        if mode == "replace":
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame and first in shape.text_frame.text:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                if first in run.text:
                                    run.text = run.text.replace(first, second)
        else:
            layout = presentation.slide_layouts[1]
            slide = presentation.slides.add_slide(layout)
            slide.shapes.title.text = "Added Note"
            body = slide.placeholders[1]
            body.text_frame.text = first
        output = io.BytesIO()
        presentation.save(output)
        return output.getvalue()

    if suffix in {".xlsx", ".xlsm"}:
        workbook = load_workbook(filename=buffer)
        worksheet = workbook.active
        if mode == "replace":
            for row in worksheet.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and first in cell.value:
                        cell.value = cell.value.replace(first, second)
        else:
            worksheet.append([first])
        output = io.BytesIO()
        workbook.save(output)
        return output.getvalue()

    if suffix == ".pdf":
        reader = PdfReader(buffer)
        from pypdf import PdfWriter

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        note_buffer = io.BytesIO()
        note_canvas = canvas.Canvas(note_buffer, pagesize=letter)
        note_canvas.setFont("Helvetica", 11)
        text_object = note_canvas.beginText(1 * inch, 10 * inch)
        text_object.setLeading(15)
        for line in _wrap_text(first, 90):
            text_object.textLine(line)
        note_canvas.drawText(text_object)
        note_canvas.save()
        note_buffer.seek(0)
        writer.append(PdfReader(note_buffer))

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    raise ValueError(f"Editing is not supported for {suffix or filename} files.")


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


# =========================================================
# BUILDING A DOWNLOADABLE REPORT FROM AN INCIDENT ANALYSIS
# =========================================================

def _report_sections(analysis: dict) -> list[tuple[str, list[str]]]:
    return [
        ("Summary", [analysis.get("summary", "")]),
        ("Severity", [
            f"Severity: {analysis.get('severity', 'N/A')}",
            f"Confidence: {round((analysis.get('confidence') or 0) * 100)}%",
            f"Escalation required: {'Yes' if analysis.get('escalation_required') else 'No'}",
        ]),
        ("Likely Causes", analysis.get("likely_causes") or []),
        ("Immediate Actions", analysis.get("immediate_actions") or []),
        ("Verification Checks", analysis.get("verification_checks") or []),
        ("Equipment Specifications", [analysis.get("equipment_specifications", "")]),
        ("Supplier Availability", analysis.get("supplier_availability") or []),
    ]


def build_docx(title: str, sections: list[tuple[str, list[str]]]) -> bytes:
    document = DocxDocument()
    document.add_heading(title, level=0)
    for section_title, lines in sections:
        if not any(lines):
            continue
        document.add_heading(section_title, level=1)
        for line in lines:
            if line:
                document.add_paragraph(line, style="List Bullet" if len(lines) > 1 else None)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def build_pptx(title: str, sections: list[tuple[str, list[str]]], subtitle: str = "") -> bytes:
    presentation = Presentation()
    title_layout = presentation.slide_layouts[0]
    title_slide = presentation.slides.add_slide(title_layout)
    title_slide.shapes.title.text = title
    title_slide.placeholders[1].text = subtitle

    body_layout = presentation.slide_layouts[1]
    for section_title, lines in sections:
        lines = [line for line in lines if line]
        if not lines:
            continue
        slide = presentation.slides.add_slide(body_layout)
        slide.shapes.title.text = section_title
        text_frame = slide.placeholders[1].text_frame
        text_frame.text = lines[0]
        for line in lines[1:]:
            paragraph = text_frame.add_paragraph()
            paragraph.text = line
            paragraph.level = 0
    output = io.BytesIO()
    presentation.save(output)
    return output.getvalue()


def build_xlsx(
    title: str,
    sections: list[tuple[str, list[str]]],
    extra_sheets: list[tuple[str, list[str], list[list]]] | None = None,
) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = title[:31] or "Sheet1"
    worksheet.append(["Section", "Detail"])
    for section_title, lines in sections:
        for line in lines:
            if line:
                worksheet.append([section_title, line])

    for sheet_title, headers, rows in extra_sheets or []:
        extra_sheet = workbook.create_sheet(sheet_title[:31])
        if headers:
            extra_sheet.append(headers)
        for row in rows:
            extra_sheet.append(row)

    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def build_docx_report(analysis: dict) -> bytes:
    return build_docx("Manufacturing Incident Report", _report_sections(analysis))


def build_pptx_report(analysis: dict) -> bytes:
    return build_pptx(
        "Manufacturing Incident Report",
        _report_sections(analysis),
        subtitle=f"Severity: {analysis.get('severity', 'N/A')}",
    )


def build_xlsx_report(analysis: dict) -> bytes:
    extra_sheets = []
    pareto = analysis.get("pareto_analysis")
    if pareto and pareto.get("items"):
        extra_sheets.append((
            "Pareto Analysis",
            ["Rank", "Cause", "Count", "% of Total", "Cumulative %", "Vital Few"],
            [
                [
                    item.get("rank"),
                    item.get("cause") or item.get("category"),
                    item.get("count"),
                    item.get("percentage"),
                    item.get("cumulative_percentage"),
                    "Yes" if item.get("is_vital_few") else "No",
                ]
                for item in pareto["items"]
            ],
        ))
    return build_xlsx("Incident Report", _report_sections(analysis), extra_sheets)


def build_pdf(title: str, sections: list[tuple[str, list[str]]]) -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 1 * inch

    def write_line(text: str, size: int = 11, leading: int = 16):
        nonlocal y
        if y < 1 * inch:
            pdf.showPage()
            y = height - 1 * inch
        pdf.setFont("Helvetica", size)
        pdf.drawString(1 * inch, y, text)
        y -= leading

    write_line(title, size=16, leading=26)
    for section_title, lines in sections:
        lines = [line for line in lines if line]
        if not lines:
            continue
        write_line(section_title, size=13, leading=20)
        for line in lines:
            for wrapped in _wrap_text(line, 95):
                write_line(f"- {wrapped}" if len(lines) > 1 else wrapped)
        y -= 6

    pdf.save()
    return output.getvalue()


def build_pdf_report(analysis: dict) -> bytes:
    return build_pdf("Manufacturing Incident Report", _report_sections(analysis))


# =========================================================
# BUILDING A DOWNLOADABLE DOCUMENT FROM A FREEFORM CHAT
# REQUEST (e.g. "make a slide deck about gear pumps", or
# "build an excel template for pump specifications")
# =========================================================

def build_document(
    fmt: str,
    title: str,
    section_headings: list[str],
    section_bullets: list[list[str]],
) -> bytes:
    sections = list(zip(section_headings, section_bullets))
    builders = {
        "docx": lambda: build_docx(title, sections),
        "pptx": lambda: build_pptx(title, sections),
        "xlsx": lambda: build_xlsx(title, sections),
        "pdf": lambda: build_pdf(title, sections),
    }
    if fmt not in builders:
        raise ValueError(f"Unsupported document format: {fmt}")
    return builders[fmt]()


def build_report(analysis: dict, fmt: str) -> bytes:
    builders = {
        "docx": build_docx_report,
        "pptx": build_pptx_report,
        "xlsx": build_xlsx_report,
        "pdf": build_pdf_report,
    }
    if fmt not in builders:
        raise ValueError(f"Unsupported report format: {fmt}")
    return builders[fmt](analysis)
