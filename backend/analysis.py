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
    classifier_label: str | None
    classifier_confidence: float
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


def load_optional_model(model_path: Path | None) -> YOLO | None:
    if model_path is None:
        return None
    if not model_path.exists():
        return None
    return load_model(model_path)


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


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_x1, left_y1, left_x2, left_y2 = [float(value) for value in left]
    right_x1, right_y1, right_x2, right_y2 = [float(value) for value in right]

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    left_area = max(0.0, (left_x2 - left_x1) * (left_y2 - left_y1))
    right_area = max(0.0, (right_x2 - right_x1) * (right_y2 - right_y1))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def mask_overlap_pixels(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(left & right))


def normalize_classifier_label(raw_label: str | None) -> str | None:
    if not raw_label:
        return None

    label = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in label for token in ("defect", "rot", "rotten", "bad", "damaged", "bruise")):
        return "Defective"
    if any(token in label for token in ("unripe", "green", "immature", "raw")):
        return "Unripe"
    if any(token in label for token in ("ripe", "mature", "red")):
        return "Ripe"
    return None


def should_merge_same_class(left: Detection, right: Detection) -> bool:
    overlap_pixels = mask_overlap_pixels(left.mask, right.mask)
    if overlap_pixels < MIN_OVERLAP_PIXELS:
        return False

    left_area = int(np.sum(left.mask))
    right_area = int(np.sum(right.mask))
    if left_area == 0 or right_area == 0:
        return False

    smaller_area = min(left_area, right_area)
    coverage = overlap_pixels / smaller_area
    return coverage >= 0.25 or box_iou(left.box, right.box) >= 0.2


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
            cluster_box = np.array([np.inf, np.inf, -np.inf, -np.inf], dtype=float)
            for member in cluster:
                cluster_mask |= member.mask
                cluster_box[0] = min(cluster_box[0], float(member.box[0]))
                cluster_box[1] = min(cluster_box[1], float(member.box[1]))
                cluster_box[2] = max(cluster_box[2], float(member.box[2]))
                cluster_box[3] = max(cluster_box[3], float(member.box[3]))
            cluster_detection = Detection(
                mask=cluster_mask,
                class_id=cluster[0].class_id,
                confidence=max(member.confidence for member in cluster),
                box=cluster_box,
            )
            if should_merge_same_class(cluster_detection, detection):
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


def should_attach_defect_to_tomato(tomato: Detection, defect: Detection) -> bool:
    overlap_pixels = mask_overlap_pixels(tomato.mask, defect.mask)
    if overlap_pixels < MIN_OVERLAP_PIXELS:
        return False

    tomato_area = int(np.sum(tomato.mask))
    defect_area = int(np.sum(defect.mask))
    if tomato_area == 0 or defect_area == 0:
        return False

    defect_coverage = overlap_pixels / defect_area
    tomato_coverage = overlap_pixels / tomato_area
    return defect_coverage >= 0.45 or tomato_coverage >= 0.02 or box_iou(tomato.box, defect.box) >= 0.12


def classify_tomato_crop(
    classifier_model: YOLO | None, image_array: np.ndarray, tomato: Detection
) -> tuple[str | None, float]:
    if classifier_model is None:
        return None, 0.0

    y_indices, x_indices = np.where(tomato.mask)
    if len(x_indices) == 0:
        return None, 0.0

    x1 = max(0, int(x_indices.min()) - 12)
    y1 = max(0, int(y_indices.min()) - 12)
    x2 = min(image_array.shape[1], int(x_indices.max()) + 13)
    y2 = min(image_array.shape[0], int(y_indices.max()) + 13)
    crop = image_array[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    try:
        result = classifier_model.predict(source=crop, verbose=False)[0]
    except Exception:  # noqa: BLE001
        return None, 0.0

    probs = getattr(result, "probs", None)
    if probs is None:
        return None, 0.0

    top_index = int(getattr(probs, "top1", -1))
    top_confidence = float(getattr(probs, "top1conf", 0.0))
    if top_index < 0:
        return None, 0.0

    names = getattr(classifier_model, "names", None) or getattr(result, "names", None)
    raw_label: str | None = None
    if isinstance(names, dict):
        raw_label = names.get(top_index)
    elif isinstance(names, (list, tuple)) and top_index < len(names):
        raw_label = names[top_index]

    return normalize_classifier_label(raw_label), top_confidence


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(result, dtype=bool)
        height, width = result.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        result = grown
    return result


def expand_defect_mask(tomato: Detection, defect_union: np.ndarray, image_array: np.ndarray) -> np.ndarray:
    if not np.any(defect_union):
        return defect_union

    y_indices, x_indices = np.where(tomato.mask)
    if len(x_indices) == 0:
        return defect_union

    x1 = max(0, int(x_indices.min()) - 6)
    y1 = max(0, int(y_indices.min()) - 6)
    x2 = min(image_array.shape[1], int(x_indices.max()) + 7)
    y2 = min(image_array.shape[0], int(y_indices.max()) + 7)

    crop = image_array[y1:y2, x1:x2].astype(np.float32) / 255.0
    crop_mask = tomato.mask[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(crop_mask):
        return defect_union

    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = np.where(max_channel == 0, 0.0, (max_channel - min_channel) / np.maximum(max_channel, 1e-6))
    brightness = max_channel

    tomato_pixels = brightness[crop_mask]
    if tomato_pixels.size == 0:
        return defect_union

    dark_threshold = min(0.88, float(np.percentile(tomato_pixels, 65)))
    dark_candidates = crop_mask & (brightness <= dark_threshold) & (saturation <= 0.9)

    local_defect = defect_union[y1:y2, x1:x2]
    expanded_seed = dilate_mask(local_defect, iterations=12)
    refined_local = local_defect | (expanded_seed & dark_candidates)

    refined = defect_union.copy()
    refined[y1:y2, x1:x2] |= refined_local
    return refined


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


def assess_tomatoes(
    tomatoes: list[Detection], defects: list[Detection], image_array: np.ndarray, classifier_model: YOLO | None = None
) -> list[TomatoAssessment]:
    assessments: list[TomatoAssessment] = []

    for index, tomato in enumerate(tomatoes, start=1):
        tomato_area = int(np.sum(tomato.mask))
        defect_union = np.zeros_like(tomato.mask, dtype=bool)

        for defect in defects:
            if should_attach_defect_to_tomato(tomato, defect):
                defect_union |= defect.mask

        defect_area = int(np.sum(defect_union))
        effective_mask = tomato.mask | defect_union
        effective_area = int(np.sum(effective_mask))
        defect_percent = (defect_area / effective_area * 100) if effective_area else 0.0

        classifier_label, classifier_confidence = classify_tomato_crop(classifier_model, image_array, tomato)

        if defect_percent > DEFECT_THRESHOLD_PERCENT:
            status = "Defective"
            label = f"Defective ({defect_percent:.1f}%)"
            color = COLORS["defective"]
        elif classifier_label == "Defective" and classifier_confidence >= 0.55:
            status = "Defective"
            label = "Defective" if defect_percent == 0 else f"Defective ({defect_percent:.1f}%)"
            color = COLORS["defective"]
        elif classifier_label in {"Ripe", "Unripe"} and classifier_confidence >= 0.60:
            status = classifier_label
            label = classifier_label
            color = COLORS["ripe"] if classifier_label == "Ripe" else COLORS["unripe"]
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
                classifier_label=classifier_label,
                classifier_confidence=classifier_confidence,
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
        has_parent_tomato = any(should_attach_defect_to_tomato(tomato, defect) for tomato in tomatoes)
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
                classifier_label=None,
                classifier_confidence=0.0,
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


def analyze_image(
    model: YOLO, image_path: Path, confidence: float, classifier_model: YOLO | None = None
) -> dict[str, Any]:
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    result = results[0]

    tomatoes, defects, mask_shape = split_detections(result)
    quality = calculate_quality_percentages(tomatoes, defects, mask_shape)

    original_image = Image.open(image_path).convert("RGB")
    image_array = np.array(original_image)

    assessments = assess_tomatoes(tomatoes, defects, image_array, classifier_model=classifier_model)
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
                "classifierLabel": item.classifier_label,
                "classifierConfidence": round(float(item.classifier_confidence), 3),
            }
            for item in assessments
        ],
        "detailImages": detail_images,
    }

    summary_figure = create_summary_figure(report)
    report["summaryChart"] = encode_figure(summary_figure)
    return report


def analyze_uploaded_image(
    model_path: Path,
    image_path: Path,
    confidence: float = 0.25,
    classifier_path: Path | None = None,
) -> dict[str, Any]:
    model = load_model(model_path)
    classifier_model = load_optional_model(classifier_path)
    return analyze_image(model, image_path, confidence, classifier_model=classifier_model)


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
