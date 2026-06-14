from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_DEFECTIVE = 0
CLASS_RIPE = 1
CLASS_UNRIPE = 2

DEFECT_THRESHOLD_PERCENT = 2.5
MIN_OVERLAP_PIXELS = 5
MERGE_IOU_THRESHOLD = 0.35

COLORS = {
    "navy": "#152238",
    "blue": "#2563EB",
    "muted": "#667085",
    "line": "#D0D5DD",
    "ripe": "#2E7D32",
    "unripe": "#F9A825",
    "defective": "#C62828",
    "background": "#F8FAFC",
    "panel": "#EEF4FF",
    "dark_panel": "#101828",
    "white": "#FFFFFF",
}


@dataclass
class Detection:
    mask: np.ndarray
    class_id: int
    confidence: float
    box: np.ndarray


@dataclass
class TomatoAssessment:
    number: int
    label: str
    status: str
    color: str
    defect_percent: float
    confidence: float
    center_x: int
    center_y: int
    mask: np.ndarray
    defect_mask: np.ndarray
    box: np.ndarray


@lru_cache(maxsize=1)
def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def encode_pil_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_figure(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    plt.close(figure)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def split_detections(result) -> tuple[list[Detection], list[Detection], tuple[int, int]]:
    tomatoes: list[Detection] = []
    defects: list[Detection] = []

    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        image_shape = getattr(result, "orig_shape", (10, 10))
        return tomatoes, defects, tuple(image_shape[:2])

    masks = result.masks.data.cpu().numpy().astype(bool)
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    mask_shape = tuple(masks[0].shape)

    for mask, class_id, confidence, box in zip(masks, classes, confidences, boxes):
        detection = Detection(mask=mask, class_id=int(class_id), confidence=float(confidence), box=box)
        if detection.class_id in {CLASS_RIPE, CLASS_UNRIPE}:
            tomatoes.append(detection)
        elif detection.class_id == CLASS_DEFECTIVE:
            defects.append(detection)

    return merge_similar_detections(tomatoes), merge_similar_detections(defects), mask_shape


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.sum(left & right))
    if intersection == 0:
        return 0.0
    union = int(np.sum(left | right))
    return intersection / union if union else 0.0


def merge_similar_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) <= 1:
        return detections

    clusters: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        matched_cluster: list[Detection] | None = None
        for cluster in clusters:
            if cluster[0].class_id != detection.class_id:
                continue
            cluster_mask = np.zeros_like(cluster[0].mask, dtype=bool)
            for member in cluster:
                cluster_mask |= member.mask
            if mask_iou(cluster_mask, detection.mask) >= MERGE_IOU_THRESHOLD:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append([detection])
        else:
            matched_cluster.append(detection)

    merged: list[Detection] = []
    for cluster in clusters:
        mask = np.zeros_like(cluster[0].mask, dtype=bool)
        x1 = y1 = float("inf")
        x2 = y2 = float("-inf")
        confidence = 0.0
        for member in cluster:
            mask |= member.mask
            confidence = max(confidence, member.confidence)
            x1 = min(x1, float(member.box[0]))
            y1 = min(y1, float(member.box[1]))
            x2 = max(x2, float(member.box[2]))
            y2 = max(y2, float(member.box[3]))

        merged.append(
            Detection(
                mask=mask,
                class_id=cluster[0].class_id,
                confidence=confidence,
                box=np.array([x1, y1, x2, y2], dtype=float),
            )
        )

    return merged


def calculate_quality_percentages(
    tomatoes: list[Detection], defects: list[Detection], mask_shape: tuple[int, int]
) -> tuple[float, float, float]:
    ripe_mask = np.zeros(mask_shape, dtype=bool)
    unripe_mask = np.zeros(mask_shape, dtype=bool)
    defect_mask = np.zeros(mask_shape, dtype=bool)

    for tomato in tomatoes:
        if tomato.class_id == CLASS_RIPE:
            ripe_mask |= tomato.mask
        else:
            unripe_mask |= tomato.mask

    for defect in defects:
        defect_mask |= defect.mask

    ripe_area = int(np.sum(ripe_mask & ~defect_mask))
    unripe_area = int(np.sum(unripe_mask & ~defect_mask))
    defect_area = int(np.sum(defect_mask))
    total_area = ripe_area + unripe_area + defect_area

    if total_area == 0:
        return 0.0, 0.0, 0.0

    return (
        ripe_area / total_area * 100,
        unripe_area / total_area * 100,
        defect_area / total_area * 100,
    )


def assess_tomatoes(tomatoes: list[Detection], defects: list[Detection]) -> list[TomatoAssessment]:
    assessments: list[TomatoAssessment] = []

    for index, tomato in enumerate(tomatoes, start=1):
        tomato_area = int(np.sum(tomato.mask))
        defect_union = np.zeros_like(tomato.mask, dtype=bool)

        for defect in defects:
            overlap = tomato.mask & defect.mask
            overlap_pixels = int(np.sum(overlap))
            defect_pixels = int(np.sum(defect.mask))
            required_overlap = max(MIN_OVERLAP_PIXELS, int(min(tomato_area, defect_pixels) * 0.01))
            if overlap_pixels >= required_overlap:
                defect_union |= overlap

        defect_area = int(np.sum(defect_union))
        defect_percent = (defect_area / tomato_area * 100) if tomato_area else 0.0

        if defect_percent > DEFECT_THRESHOLD_PERCENT:
            status = "Defective"
            label = f"Defective ({defect_percent:.1f}%)"
            color = COLORS["defective"]
        elif tomato.class_id == CLASS_RIPE:
            status = "Ripe"
            label = "Ripe"
            color = COLORS["ripe"]
        else:
            status = "Unripe"
            label = "Unripe"
            color = COLORS["unripe"]

        y_indices, x_indices = np.where(tomato.mask)
        if len(x_indices) == 0:
            continue

        assessments.append(
            TomatoAssessment(
                number=index,
                label=label,
                status=status,
                color=color,
                defect_percent=defect_percent,
                confidence=tomato.confidence,
                center_x=int(np.mean(x_indices)),
                center_y=int(np.mean(y_indices)),
                mask=tomato.mask,
                defect_mask=defect_union,
                box=tomato.box,
            )
        )

    return assessments


def assess_standalone_defects(
    defects: list[Detection], tomatoes: list[Detection], start_number: int
) -> list[TomatoAssessment]:
    assessments: list[TomatoAssessment] = []
    number = start_number

    for defect in defects:
        has_parent_tomato = any(int(np.sum(tomato.mask & defect.mask)) > MIN_OVERLAP_PIXELS for tomato in tomatoes)
        if has_parent_tomato:
            continue

        y_indices, x_indices = np.where(defect.mask)
        if len(x_indices) == 0:
            continue

        number += 1
        assessments.append(
            TomatoAssessment(
                number=number,
                label="Defective (100%)",
                status="Defective",
                color=COLORS["defective"],
                defect_percent=100.0,
                confidence=defect.confidence,
                center_x=int(np.mean(x_indices)),
                center_y=int(np.mean(y_indices)),
                mask=defect.mask,
                defect_mask=defect.mask,
                box=defect.box,
            )
        )

    return assessments


def safe_crop_bounds(box: np.ndarray, image_shape: tuple[int, ...], padding: int = 20) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def mask_crop_bounds(mask: np.ndarray, padding: int = 20) -> tuple[int, int, int, int] | None:
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0:
        return None
    return (
        max(0, int(x_indices.min()) - padding),
        max(0, int(y_indices.min()) - padding),
        min(mask.shape[1], int(x_indices.max()) + padding),
        min(mask.shape[0], int(y_indices.max()) + padding),
    )


def create_detail_figure(assessment: TomatoAssessment, image_array: np.ndarray) -> plt.Figure | None:
    image_x1, image_y1, image_x2, image_y2 = safe_crop_bounds(assessment.box, image_array.shape)
    crop_image = image_array[image_y1:image_y2, image_x1:image_x2]

    mask_bounds = mask_crop_bounds(assessment.mask)
    if mask_bounds is None or crop_image.size == 0:
        return None

    mask_x1, mask_y1, mask_x2, mask_y2 = mask_bounds
    crop_mask = assessment.mask[mask_y1:mask_y2, mask_x1:mask_x2]
    crop_defect = assessment.defect_mask[mask_y1:mask_y2, mask_x1:mask_x2]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"Tomato {assessment.number}: {assessment.label}",
        fontsize=15,
        fontweight="bold",
        color=assessment.color,
    )

    axes[0].imshow(crop_image)
    axes[0].set_title("Cropped Image")
    axes[0].axis("off")

    axes[1].imshow(crop_mask, cmap="gray")
    axes[1].set_title("Tomato Mask")
    axes[1].axis("off")

    axes[2].imshow(crop_mask, cmap="gray")
    overlay = np.zeros((crop_mask.shape[0], crop_mask.shape[1], 4))
    overlay[crop_defect] = [0.78, 0.16, 0.16, 0.85]
    axes[2].imshow(overlay)
    axes[2].set_title(f"Defect Area: {assessment.defect_percent:.1f}%")
    axes[2].axis("off")

    for axis in axes:
        axis.set_facecolor(COLORS["background"])

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def add_stacked_quality_bar(axis, ripe_pct: float, unripe_pct: float, defect_pct: float) -> None:
    axis.barh([""], [ripe_pct], color=COLORS["ripe"], edgecolor="white", height=0.45)
    axis.barh([""], [unripe_pct], left=ripe_pct, color=COLORS["unripe"], edgecolor="white", height=0.45)
    axis.barh([""], [defect_pct], left=ripe_pct + unripe_pct, color=COLORS["defective"], edgecolor="white", height=0.45)

    segments = [
        ("Ripe", ripe_pct, 0, "white"),
        ("Unripe", unripe_pct, ripe_pct, "black"),
        ("Defective", defect_pct, ripe_pct + unripe_pct, "white"),
    ]
    for label, value, left, text_color in segments:
        if value >= 5:
            axis.text(
                left + value / 2,
                0,
                f"{label} {value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=11,
            )

    axis.set_xlim(0, 100)
    axis.set_title("Overall Quality Composition", pad=8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def create_summary_figure(report: dict[str, Any]) -> plt.Figure:
    figure = plt.figure(figsize=(14, 5.8), constrained_layout=True)
    figure.patch.set_facecolor(COLORS["background"])
    grid = figure.add_gridspec(2, 4, height_ratios=[1, 1.25])

    cards = [
        ("Regions", report["tomatoCount"], COLORS["navy"]),
        ("Ripe", report["ripeCount"], COLORS["ripe"]),
        ("Unripe", report["unripeCount"], COLORS["unripe"]),
        ("Defective", report["defectiveCount"], COLORS["defective"]),
    ]

    for index, (label, value, color) in enumerate(cards):
        axis = figure.add_subplot(grid[0, index])
        axis.set_facecolor("white")
        axis.text(0.06, 0.72, label.upper(), transform=axis.transAxes, fontsize=9, fontweight="bold", color=COLORS["muted"])
        axis.text(0.06, 0.26, str(value), transform=axis.transAxes, fontsize=28, fontweight="bold", color=color)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(COLORS["line"])

    ratio_axis = figure.add_subplot(grid[1, :2])
    add_stacked_quality_bar(ratio_axis, report["ripePct"], report["unripePct"], report["defectPct"])

    count_axis = figure.add_subplot(grid[1, 2:])
    count_labels = ["Ripe", "Unripe", "Defective"]
    count_values = [report["ripeCount"], report["unripeCount"], report["defectiveCount"]]
    count_colors = [COLORS["ripe"], COLORS["unripe"], COLORS["defective"]]
    count_axis.bar(count_labels, count_values, color=count_colors, width=0.55)
    count_axis.set_title("Classification Count", fontsize=12, fontweight="bold", color=COLORS["navy"])
    count_axis.grid(axis="y", alpha=0.18)
    count_axis.spines["top"].set_visible(False)
    count_axis.spines["right"].set_visible(False)
    count_axis.spines["left"].set_color(COLORS["line"])
    count_axis.spines["bottom"].set_color(COLORS["line"])
    for index, value in enumerate(count_values):
        count_axis.text(index, value + max(count_values + [1]) * 0.03, str(value), ha="center", fontweight="bold")

    return figure


def summarize_assessment(report: dict[str, Any]) -> str:
    if report["defectiveCount"] > 0 or report["defectPct"] >= DEFECT_THRESHOLD_PERCENT:
        return "Defects detected. Review the annotated regions before accepting the file."
    if report["unripeCount"] > report["ripeCount"]:
        return "The sample is mostly unripe. It may need more maturation time."
    return "The sample looks acceptable based on the configured detection rule."


def analyze_image(model: YOLO, image_path: Path, confidence: float) -> dict[str, Any]:
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    result = results[0]

    tomatoes, defects, mask_shape = split_detections(result)
    quality = calculate_quality_percentages(tomatoes, defects, mask_shape)

    original_image = Image.open(image_path).convert("RGB")
    image_array = np.array(original_image)

    assessments = assess_tomatoes(tomatoes, defects)
    assessments.extend(assess_standalone_defects(defects, tomatoes, len(assessments)))

    ripe_count = sum(1 for item in assessments if item.status == "Ripe")
    unripe_count = sum(1 for item in assessments if item.status == "Unripe")
    defective_count = sum(1 for item in assessments if item.status == "Defective")

    plotted_prediction = result.plot()[..., ::-1]
    annotated_image = Image.fromarray(plotted_prediction)

    detail_images: list[dict[str, str]] = []
    for assessment in assessments:
        detail = create_detail_figure(assessment, image_array)
        if detail is None:
            continue
        detail_images.append(
            {
                "title": f"Tomato {assessment.number}",
                "caption": assessment.label,
                "image": encode_figure(detail),
            }
        )

    report = {
        "reportId": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "generatedAt": datetime.now().isoformat(),
        "fileName": image_path.name,
        "image": encode_pil_image(original_image),
        "annotatedImage": encode_pil_image(annotated_image),
        "tomatoCount": len(assessments),
        "ripeCount": ripe_count,
        "unripeCount": unripe_count,
        "defectiveCount": defective_count,
        "ripePct": round(float(quality[0]), 2),
        "unripePct": round(float(quality[1]), 2),
        "defectPct": round(float(quality[2]), 2),
        "summary": summarize_assessment(
            {
                "ripeCount": ripe_count,
                "unripeCount": unripe_count,
                "defectiveCount": defective_count,
                "ripePct": quality[0],
                "unripePct": quality[1],
                "defectPct": quality[2],
            }
        ),
        "assessments": [
            {
                "number": item.number,
                "label": item.label,
                "status": item.status,
                "color": item.color,
                "defectPercent": round(float(item.defect_percent), 2),
                "confidence": round(float(item.confidence), 3),
            }
            for item in assessments
        ],
        "detailImages": detail_images,
    }

    summary_figure = create_summary_figure(report)
    report["summaryChart"] = encode_figure(summary_figure)
    return report


def analyze_uploaded_image(model_path: Path, image_path: Path, confidence: float = 0.25) -> dict[str, Any]:
    model = load_model(model_path)
    return analyze_image(model, image_path, confidence)


def make_report_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        **report,
        "totals": {
            "regions": report["tomatoCount"],
            "ripe": report["ripeCount"],
            "unripe": report["unripeCount"],
            "defective": report["defectiveCount"],
        },
    }


def report_to_json(report: dict[str, Any]) -> dict[str, Any]:
    safe = dict(report)
    safe["assessments"] = list(report["assessments"])
    safe["detailImages"] = list(report["detailImages"])
    return safe
