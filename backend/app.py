from __future__ import annotations

import json
import math
import shutil
import textwrap
from io import BytesIO
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from matplotlib import font_manager as mpl_font_manager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont

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


def shorten_text(value: str | None, max_length: int = 40) -> str:
    if not value:
        return "Unknown"
    if len(value) <= max_length:
        return value
    head = max(8, math.floor(max_length * 0.6))
    tail = max(6, max_length - head - 1)
    return f"{value[:head]}…{value[-tail:]}"


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


PDF_PAGE_SIZE = (1240, 1754)
PDF_MARGIN = 70
PDF_BG = (247, 243, 236)
PDF_PANEL = (255, 255, 255)
PDF_TEXT = (17, 30, 44)
PDF_MUTED = (102, 112, 133)
PDF_LINE = (215, 221, 230)
PDF_ACCENT = (217, 119, 6)
PDF_GREEN = (46, 125, 50)
PDF_AMBER = (185, 119, 0)
PDF_RED = (198, 40, 40)


@lru_cache(maxsize=16)
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = mpl_font_manager.findfont("DejaVu Sans Bold" if bold else "DejaVu Sans")
    return ImageFont.truetype(font_path, size=size)


def decode_data_uri(data_uri: str | None) -> Image.Image | None:
    if not data_uri:
        return None
    try:
        _, payload = data_uri.split(",", 1)
    except ValueError:
        payload = data_uri
    try:
        return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def image_with_padding(image: Image.Image, size: tuple[int, int], fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    resized = image.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        if line:
            draw.text((x, y), line, font=font, fill=fill)
            y += text_height(draw, line, font) + line_gap
        else:
            y += text_height(draw, "Ag", font) // 2
    return y


def draw_rounded_image(
    base: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 28,
) -> None:
    x1, y1, x2, y2 = box
    target = image_with_padding(image, (x2 - x1, y2 - y1), fill=PDF_PANEL)
    mask = Image.new("L", target.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, target.width - 1, target.height - 1), radius=radius, fill=255)
    base.paste(target, (x1, y1), mask)


def draw_header(draw: ImageDraw.ImageDraw, report: dict, page_width: int) -> int:
    title_font = load_font(34, bold=True)
    body_font = load_font(18)
    label_font = load_font(13, bold=True)
    x = PDF_MARGIN
    y = PDF_MARGIN
    draw.text((x, y), "Tomato Grading Report", font=title_font, fill=PDF_TEXT)
    y += text_height(draw, "Ag", title_font) + 8
    draw.text((x, y), shorten_text(report.get("fileName"), 56), font=body_font, fill=PDF_MUTED)
    y += text_height(draw, "Ag", body_font) + 16
    meta = f"Report ID: {report.get('reportId', 'Unknown')}   |   Generated: {format_datetime(report.get('generatedAt'))}"
    draw.text((x, y), meta, font=label_font, fill=PDF_ACCENT)
    pill_text = "Ready"
    pill_font = load_font(16, bold=True)
    pill_w = text_width(draw, pill_text, pill_font) + 36
    pill_h = 42
    pill_x2 = page_width - PDF_MARGIN
    pill_x1 = pill_x2 - pill_w
    pill_y1 = PDF_MARGIN + 2
    pill_y2 = pill_y1 + pill_h
    draw.rounded_rectangle((pill_x1, pill_y1, pill_x2, pill_y2), radius=21, fill=(228, 238, 224), outline=None)
    draw.text((pill_x1 + (pill_w - text_width(draw, pill_text, pill_font)) / 2, pill_y1 + 11), pill_text, font=pill_font, fill=PDF_GREEN)
    return y + 34


def draw_section_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, title_font, body_font) -> None:
    draw.rounded_rectangle(box, radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
    x1, y1, _, _ = box
    draw.text((x1 + 22, y1 + 18), title, font=title_font, fill=PDF_TEXT)


def create_overview_page(report: dict) -> Image.Image:
    page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
    draw = ImageDraw.Draw(page)
    title_font = load_font(24, bold=True)
    small_font = load_font(15)
    body_font = load_font(18)
    heading_font = load_font(18, bold=True)
    number_font = load_font(32, bold=True)

    content_top = draw_header(draw, report, PDF_PAGE_SIZE[0])

    # Summary card
    summary_box = (PDF_MARGIN, content_top, PDF_PAGE_SIZE[0] - PDF_MARGIN, content_top + 148)
    draw_section_card(draw, summary_box, "Summary", heading_font, body_font)
    summary_text = report.get("summary", "No summary available.")
    y = draw_text_block(draw, summary_box[0] + 22, summary_box[1] + 56, summary_text, body_font, PDF_TEXT, summary_box[2] - summary_box[0] - 44, 7)

    # Metric cards
    metric_top = summary_box[3] + 18
    metric_h = 120
    metric_gap = 16
    metric_w = (PDF_PAGE_SIZE[0] - PDF_MARGIN * 2 - metric_gap * 3) // 4
    metrics = [
        ("Regions", report.get("tomatoCount", 0), PDF_TEXT),
        ("Ripe", report.get("ripeCount", 0), PDF_GREEN),
        ("Unripe", report.get("unripeCount", 0), PDF_AMBER),
        ("Defective", report.get("defectiveCount", 0), PDF_RED),
    ]
    for index, (label, value, color) in enumerate(metrics):
        x1 = PDF_MARGIN + index * (metric_w + metric_gap)
        box = (x1, metric_top, x1 + metric_w, metric_top + metric_h)
        draw.rounded_rectangle(box, radius=24, fill=PDF_PANEL, outline=PDF_LINE, width=1)
        draw.text((box[0] + 18, box[1] + 18), label.upper(), font=small_font, fill=PDF_MUTED)
        draw.text((box[0] + 18, box[1] + 54), str(value), font=number_font, fill=color)

    # Images section
    images_top = metric_top + metric_h + 18
    image_box_h = 430
    image_box_w = (PDF_PAGE_SIZE[0] - PDF_MARGIN * 2 - 18) // 2
    image_titles = [("Original", decode_data_uri(report.get("image"))), ("Annotated", decode_data_uri(report.get("annotatedImage")))]
    for index, (title, image) in enumerate(image_titles):
        x1 = PDF_MARGIN + index * (image_box_w + 18)
        box = (x1, images_top, x1 + image_box_w, images_top + image_box_h)
        draw.rounded_rectangle(box, radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
        draw.text((box[0] + 22, box[1] + 18), title, font=heading_font, fill=PDF_TEXT)
        if image is not None:
            draw_rounded_image(page, image, (box[0] + 12, box[1] + 56, box[2] - 12, box[3] - 12), radius=22)

    # Chart below, if present
    chart = decode_data_uri(report.get("summaryChart"))
    chart_top = images_top + image_box_h + 18
    chart_box = (PDF_MARGIN, chart_top, PDF_PAGE_SIZE[0] - PDF_MARGIN, chart_top + 260)
    draw.rounded_rectangle(chart_box, radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
    draw.text((chart_box[0] + 22, chart_box[1] + 18), "Chart", font=heading_font, fill=PDF_TEXT)
    if chart is not None:
        draw_rounded_image(page, chart, (chart_box[0] + 12, chart_box[1] + 56, chart_box[2] - 12, chart_box[3] - 12), radius=22)

    # Footer note
    footer_font = load_font(13)
    footer_text = "Generated from the selected image or webcam frame."
    draw.text((PDF_MARGIN, PDF_PAGE_SIZE[1] - PDF_MARGIN - 18), footer_text, font=footer_font, fill=PDF_MUTED)
    return page


def create_assessment_pages(report: dict) -> list[Image.Image]:
    pages: list[Image.Image] = []
    rows = report.get("assessments", [])
    if not rows:
        page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
        draw = ImageDraw.Draw(page)
        draw_header(draw, report, PDF_PAGE_SIZE[0])
        draw.rounded_rectangle((PDF_MARGIN, 260, PDF_PAGE_SIZE[0] - PDF_MARGIN, 520), radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
        draw.text((PDF_MARGIN + 24, 290), "Detected Regions", font=load_font(22, bold=True), fill=PDF_TEXT)
        draw.text((PDF_MARGIN + 24, 340), "No regions were detected for this image.", font=load_font(18), fill=PDF_MUTED)
        pages.append(page)
        return pages

    row_font = load_font(16)
    head_font = load_font(16, bold=True)
    title_font = load_font(24, bold=True)
    for start in range(0, len(rows), 18):
        page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
        draw = ImageDraw.Draw(page)
        draw_header(draw, report, PDF_PAGE_SIZE[0])
        box = (PDF_MARGIN, 220, PDF_PAGE_SIZE[0] - PDF_MARGIN, PDF_PAGE_SIZE[1] - PDF_MARGIN)
        draw.rounded_rectangle(box, radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
        draw.text((box[0] + 22, box[1] + 18), "Detected Regions", font=title_font, fill=PDF_TEXT)

        table_x = box[0] + 20
        table_y = box[1] + 66
        cols = [64, 180, 200, 160, 120]
        headers = ["#", "Status", "Label", "Confidence", "Defect"]
        row_h = 34
        header_h = 34
        draw.rounded_rectangle((table_x, table_y, table_x + sum(cols), table_y + header_h), radius=12, fill=(241, 245, 249), outline=PDF_LINE, width=1)
        cx = table_x + 12
        for idx, header in enumerate(headers):
            draw.text((cx, table_y + 9), header, font=head_font, fill=PDF_MUTED)
            cx += cols[idx]

        y = table_y + header_h + 8
        for row in rows[start:start + 18]:
            if y + row_h > box[3] - 20:
                break
            draw.rounded_rectangle((table_x, y, table_x + sum(cols), y + row_h), radius=10, fill=PDF_PANEL if (row["number"] % 2) else PDF_PANEL, outline=PDF_LINE, width=1)
            cells = [
                str(row.get("number", "")),
                row.get("status", ""),
                shorten_text(row.get("label"), 26),
                f"{float(row.get('confidence', 0)):0.3f}",
                f"{float(row.get('defectPercent', 0)):0.1f}%",
            ]
            cx = table_x + 12
            for idx, cell in enumerate(cells):
                draw.text((cx, y + 8), cell, font=row_font, fill=PDF_TEXT)
                cx += cols[idx]
            y += row_h + 6
        pages.append(page)
    return pages


def create_detail_pages(report: dict) -> list[Image.Image]:
    detail_images = report.get("detailImages", [])
    if not detail_images:
        return []

    pages: list[Image.Image] = []
    title_font = load_font(24, bold=True)
    heading_font = load_font(18, bold=True)
    small_font = load_font(14)
    for start in range(0, len(detail_images), 4):
        page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
        draw = ImageDraw.Draw(page)
        draw_header(draw, report, PDF_PAGE_SIZE[0])
        draw.rounded_rectangle((PDF_MARGIN, 220, PDF_PAGE_SIZE[0] - PDF_MARGIN, PDF_PAGE_SIZE[1] - PDF_MARGIN), radius=28, fill=PDF_PANEL, outline=PDF_LINE, width=1)
        draw.text((PDF_MARGIN + 22, 238), "Region Details", font=title_font, fill=PDF_TEXT)
        card_w = (PDF_PAGE_SIZE[0] - PDF_MARGIN * 2 - 18) // 2
        card_h = 300
        for idx, item in enumerate(detail_images[start:start + 4]):
            col = idx % 2
            row = idx // 2
            x1 = PDF_MARGIN + col * (card_w + 18)
            y1 = 286 + row * (card_h + 18)
            box = (x1, y1, x1 + card_w, y1 + card_h)
            draw.rounded_rectangle(box, radius=22, fill=PDF_PANEL, outline=PDF_LINE, width=1)
            draw.text((box[0] + 18, box[1] + 16), item.get("title", "Detail"), font=heading_font, fill=PDF_TEXT)
            draw.text((box[0] + 18, box[1] + 42), item.get("caption", ""), font=small_font, fill=PDF_MUTED)
            img = decode_data_uri(item.get("image"))
            if img is not None:
                draw_rounded_image(page, img, (box[0] + 12, box[1] + 70, box[2] - 12, box[3] - 12), radius=18)
        pages.append(page)
    return pages


def build_pdf_bytes(report: dict) -> bytes:
    pages = [create_overview_page(report), *create_assessment_pages(report), *create_detail_pages(report)]
    if not pages:
        pages = [Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)]
    buffer = BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


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

    if report.get("detailImages"):
        paragraphs.append(_docx_paragraph("Region Details", bold=True, size=28))
        for item in report["detailImages"]:
            paragraphs.append(_docx_paragraph(f"{item.get('title', 'Detail')}: {item.get('caption', '')}", size=22))

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
