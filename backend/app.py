from __future__ import annotations

import json
import shutil
import textwrap
from io import BytesIO
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

try:
    from .analysis import analyze_uploaded_image, make_report_payload, report_to_json
except ImportError:  # pragma: no cover - allows running from backend/ directly
    from analysis import analyze_uploaded_image, make_report_payload, report_to_json


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = ROOT_DIR / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tomato Grading API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def load_report(report_id: str) -> dict:
    report_path = REPORT_DIR / f"{report_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return json.loads(report_path.read_text(encoding="utf-8"))


def format_datetime(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def build_report_sections(report: dict) -> list[str]:
    summary_lines = [
        f"Report ID: {report.get('reportId', 'Unknown')}",
        f"Generated At: {format_datetime(report.get('generatedAt'))}",
        f"Source File: {report.get('fileName', 'Unknown')}",
        "",
        "Executive Summary",
        report.get("summary", ""),
        "",
        "Quality Metrics",
        f"Regions detected: {report.get('tomatoCount', 0)}",
        f"Ripe: {report.get('ripeCount', 0)} ({report.get('ripePct', 0):.1f}%)",
        f"Unripe: {report.get('unripeCount', 0)} ({report.get('unripePct', 0):.1f}%)",
        f"Defective: {report.get('defectiveCount', 0)} ({report.get('defectPct', 0):.1f}%)",
        "",
        "Detected Regions",
    ]

    for item in report.get("assessments", []):
        summary_lines.append(
            f"{item.get('number', 0):>2}. {item.get('label', 'Unknown')} | "
            f"Status: {item.get('status', 'Unknown')} | "
            f"Confidence: {float(item.get('confidence', 0)):0.3f} | "
            f"Defect: {float(item.get('defectPercent', 0)):0.1f}%"
        )

    if not report.get("assessments"):
        summary_lines.append("No regions were detected for this image.")

    return summary_lines


def build_pdf_bytes(report: dict) -> bytes:
    page_width = 612
    page_height = 792
    margin = 54
    font_size_map = {"title": 20, "heading": 13, "body": 10, "blank": 10}
    leading_map = {"title": 28, "heading": 18, "body": 14, "blank": 10}

    def wrap_line(text: str, kind: str) -> list[tuple[str, str]]:
        if kind == "title":
            return [(text, kind)]
        if kind == "heading":
            return [(text, kind)]
        if not text:
            return [("", "blank")]
        return [(line, kind) for line in textwrap.wrap(text, width=92) or [""]]

    content: list[tuple[str, str]] = []
    content.extend(wrap_line("Tomato Grading Report", "title"))
    content.append(("", "blank"))
    for line in build_report_sections(report):
        if line in {"Executive Summary", "Quality Metrics", "Detected Regions"}:
            content.append((line, "heading"))
            continue
        if line == "":
            content.append(("", "blank"))
            continue
        content.extend(wrap_line(line, "body"))

    pages: list[list[tuple[str, str]]] = []
    current_page: list[tuple[str, str]] = []
    used_height = 0
    max_height = page_height - margin * 2
    for line, kind in content:
        height = leading_map[kind]
        if current_page and used_height + height > max_height:
            pages.append(current_page)
            current_page = []
            used_height = 0
        current_page.append((line, kind))
        used_height += height
    if current_page:
        pages.append(current_page)

    def escape_pdf_text(text: str) -> str:
        safe = text.encode("latin-1", "replace").decode("latin-1")
        return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    font_obj_num = 3 + len(pages) * 2
    bold_font_obj_num = font_obj_num + 1
    pages_obj_num = 2
    catalog_obj_num = 1

    page_object_numbers: list[int] = []
    content_object_numbers: list[int] = []
    for index in range(len(pages)):
        page_object_numbers.append(3 + index * 2)
        content_object_numbers.append(4 + index * 2)

    def add_object(body: str) -> None:
        objects.append(body.encode("latin-1"))

    add_object(f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>")

    kids = " ".join(f"{obj_num} 0 R" for obj_num in page_object_numbers)
    add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")

    for index, page in enumerate(pages):
        stream_lines: list[str] = []
        y = page_height - margin
        for text, kind in page:
            if kind == "blank":
                y -= leading_map[kind]
                continue
            font_obj = bold_font_obj_num if kind in {"title", "heading"} else font_obj_num
            size = font_size_map[kind]
            color = "0.10 0.13 0.20" if kind == "heading" else "0 0 0"
            if kind == "title":
                color = "0.08 0.15 0.25"
            stream_lines.append(
                f"BT /F{1 if font_obj == font_obj_num else 2} {size} Tf {color} rg {margin} {y} Td ({escape_pdf_text(text)}) Tj ET"
            )
            y -= leading_map[kind]

        stream = "\n".join(stream_lines).encode("latin-1")
        add_object(f"<< /Type /Page /Parent {pages_obj_num} 0 R /MediaBox [0 0 {page_width} {page_height}] /Resources << /Font << /F1 {font_obj_num} 0 R /F2 {bold_font_obj_num} 0 R >> >> /Contents {content_object_numbers[index]} 0 R >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")

    add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_number} 0 obj\n".encode("latin-1"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root {catalog_obj_num} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


def _docx_run(text: str, *, bold: bool = False, size: int = 22) -> str:
    props = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
    if bold:
        props.insert(0, "<w:b/>")
    return f"<w:r><w:rPr>{''.join(props)}</w:rPr><w:t xml:space='preserve'>{xml_escape(text)}</w:t></w:r>"


def _docx_paragraph(text: str, *, bold: bool = False, size: int = 22, align: str | None = None) -> str:
    ppr = []
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{_docx_run(text, bold=bold, size=size)}</w:p>"


def build_docx_bytes(report: dict) -> bytes:
    paragraphs = [
        _docx_paragraph("Tomato Grading Report", bold=True, size=36, align="center"),
        _docx_paragraph(f"Report ID: {report.get('reportId', 'Unknown')}", size=22),
        _docx_paragraph(f"Generated At: {format_datetime(report.get('generatedAt'))}", size=22),
        _docx_paragraph(f"Source File: {report.get('fileName', 'Unknown')}", size=22),
        _docx_paragraph("Executive Summary", bold=True, size=28),
        _docx_paragraph(report.get("summary", ""), size=22),
        _docx_paragraph("Quality Metrics", bold=True, size=28),
        _docx_paragraph(f"Regions detected: {report.get('tomatoCount', 0)}", size=22),
        _docx_paragraph(f"Ripe: {report.get('ripeCount', 0)} ({report.get('ripePct', 0):.1f}%)", size=22),
        _docx_paragraph(f"Unripe: {report.get('unripeCount', 0)} ({report.get('unripePct', 0):.1f}%)", size=22),
        _docx_paragraph(f"Defective: {report.get('defectiveCount', 0)} ({report.get('defectPct', 0):.1f}%)", size=22),
        _docx_paragraph("Detected Regions", bold=True, size=28),
    ]

    if report.get("assessments"):
        for item in report["assessments"]:
            paragraphs.append(
                _docx_paragraph(
                    (
                        f"{item.get('number', 0)}. {item.get('label', 'Unknown')} "
                        f"- Status: {item.get('status', 'Unknown')} "
                        f"- Confidence: {float(item.get('confidence', 0)):0.3f} "
                        f"- Defect: {float(item.get('defectPercent', 0)):0.1f}%"
                    ),
                    size=22,
                )
            )
    else:
        paragraphs.append(_docx_paragraph("No regions were detected for this image.", size=22))

    body_xml = "".join(paragraphs)
    document_xml = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
        f"<w:body>{body_xml}<w:sectPr><w:pgSz w:w='12240' w:h='15840'/><w:pgMar w:top='1440' w:right='1440' w:bottom='1440' w:left='1440' w:header='720' w:footer='720' w:gutter='0'/></w:sectPr></w:body>"
        "</w:document>"
    )

    content_types = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
        "</Types>"
    )

    rels = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>"
        "</Relationships>"
    )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), confidence: float = 0.25) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a JPG, PNG, BMP, or WEBP image.")

    report_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stored_name = f"{report_id}_{Path(file.filename).name}"
    stored_path = UPLOAD_DIR / stored_name

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        report = analyze_uploaded_image(MODEL_PATH, stored_path, confidence=confidence)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    payload = make_report_payload(report)
    payload["reportId"] = report_id
    payload["storedImage"] = stored_name

    report_path = REPORT_DIR / f"{report_id}.json"
    report_path.write_text(json.dumps(report_to_json(payload), indent=2), encoding="utf-8")

    return payload


@app.get("/api/reports/{report_id}")
def get_report(report_id: str) -> dict:
    return load_report(report_id)


@app.get("/api/reports/{report_id}/download")
def download_report(report_id: str, format: str = "json") -> Response:
    report = load_report(report_id)
    download_format = format.lower()

    if download_format == "json":
        payload = json.dumps(report, indent=2).encode("utf-8")
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="tomato-report-{report_id}.json"'},
        )

    if download_format == "pdf":
        payload = build_pdf_bytes(report)
        return Response(
            content=payload,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="tomato-report-{report_id}.pdf"'},
        )

    if download_format in {"doc", "docx"}:
        payload = build_docx_bytes(report)
        return Response(
            content=payload,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="tomato-report-{report_id}.docx"'},
        )

    raise HTTPException(status_code=400, detail="Unsupported export format. Use json, pdf, or docx.")
