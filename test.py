# =====================================================
# TOMATO GRADING SYSTEM - TEST 5 IMAGES
# =====================================================

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os
import random

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO("best.pt")

# =====================================================
# OUTPUT SETUP
# =====================================================

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "Tomato_Grading_Report.pdf")
pdf = PdfPages(pdf_path)
print(f"Report will be saved to: {pdf_path}")

# =====================================================
# GET 5 RANDOM TEST IMAGES
# =====================================================

image_folder = r"C:\Users\prita\Downloads\Tomato Final\test\images"

all_images = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

sample_images = random.sample(all_images, min(5, len(all_images)))

# =====================================================
# PROCESS EACH IMAGE
# =====================================================

for image_path in sample_images:

    print("\n" + "="*100)
    print("IMAGE:", os.path.basename(image_path))
    print("="*100)

    # =================================================
    # PREDICT
    # =================================================

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    r = results[0]

    # =================================================
    # SEPARATE TOMATOES AND DEFECTS
    # =================================================

    tomatoes = []
    defects = []

    # Check if masks exist
    if r.masks is not None and len(r.masks.data) > 0:
        mask_shape = r.masks.data[0].shape
        for mask, cls, conf in zip(
            r.masks.data.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):
            cls = int(cls)
            # 0: defective, 1: ripe, 2: unripe
            if cls in [1, 2]:
                tomatoes.append({
                    "mask": mask.astype(bool),
                    "class": cls,
                    "conf": float(conf)
                })
            elif cls == 0:
                defects.append({
                    "mask": mask.astype(bool),
                    "conf": float(conf)
                })
    else:
        print("No tomatoes detected")
        mask_shape = (10, 10)  # Dummy shape to avoid crash

    # =================================================
    # CALCULATE OVERALL IMAGE PERCENTAGES
    # =================================================

    ripe_mask_union = np.zeros(mask_shape, dtype=bool)
    unripe_mask_union = np.zeros(mask_shape, dtype=bool)
    defect_mask_union = np.zeros(mask_shape, dtype=bool)

    # Create union of all instances per class
    for t in tomatoes:
        if t["class"] == 1:
            ripe_mask_union |= t["mask"]
        else:
            unripe_mask_union |= t["mask"]

    for d in defects:
        defect_mask_union |= d["mask"]

    # Subtract defect areas from healthy areas so we don't double count
    true_ripe_mask = ripe_mask_union & ~defect_mask_union
    true_unripe_mask = unripe_mask_union & ~defect_mask_union

    ripe_area = np.sum(true_ripe_mask)
    unripe_area = np.sum(true_unripe_mask)
    defect_area = np.sum(defect_mask_union)

    total_area = ripe_area + unripe_area + defect_area

    if total_area > 0:
        ripe_pct = (ripe_area / total_area) * 100
        unripe_pct = (unripe_area / total_area) * 100
        defect_pct = (defect_area / total_area) * 100
    else:
        ripe_pct = unripe_pct = defect_pct = 0.0

    print(f"Overall Pixels -> Ripe: {ripe_pct:.1f}%, Unripe: {unripe_pct:.1f}%, Defective: {defect_pct:.1f}%")

    # =================================================
    # INDIVIDUAL TOMATO ANALYSIS
    # =================================================
    
    annotations = []
    mask_figures = []

    tomato_count = 0
    for tomato in tomatoes:
        tomato_count += 1
        tomato_mask = tomato["mask"]
        t_area = np.sum(tomato_mask)

        defect_union = np.zeros_like(tomato_mask, dtype=bool)
        for defect in defects:
            overlap = tomato_mask & defect["mask"]
            if np.sum(overlap) > 20:
                defect_union |= defect["mask"]

        d_area = np.sum(defect_union)
        d_percent = (d_area / t_area * 100) if t_area > 0 else 0

        if d_percent > 5:
            result = f"Defective ({d_percent:.1f}%)"
        else:
            result = "Ripe" if tomato["class"] == 1 else "Unripe"

        print(f"  -> Tomato {tomato_count}: {result}")

        # Compute center for text annotation
        y_idx, x_idx = np.where(tomato_mask)
        if len(x_idx) > 0:
            cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
            annotations.append({'x': cx, 'y': cy, 'text': result})

        # Generate individual mask figure
        fig_mask = plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(tomato_mask, cmap="gray")
        plt.title(f"Tomato {tomato_count} Mask")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(defect_union, cmap="gray")
        plt.title("Overlapping Defect Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(tomato_mask, cmap="gray")
        # Use an overlay that highlights the defect in red
        overlay = np.zeros((mask_shape[0], mask_shape[1], 4))
        overlay[defect_union] = [1, 0, 0, 0.7] # Red with alpha
        plt.imshow(overlay, alpha=1)
        plt.title(f"Result: {result}")
        plt.axis("off")

        plt.tight_layout()
        mask_figures.append(fig_mask)

    # Identify standalone fully defective regions
    for defect in defects:
        is_inside_tomato = False
        for tomato in tomatoes:
            overlap = tomato["mask"] & defect["mask"]
            if np.sum(overlap) > 20:
                is_inside_tomato = True
                break
        
        if not is_inside_tomato:
            tomato_count += 1
            result = "Defective (100%)"
            print(f"  -> Tomato {tomato_count}: {result}")

            y_idx, x_idx = np.where(defect["mask"])
            if len(x_idx) > 0:
                cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
                annotations.append({'x': cx, 'y': cy, 'text': result})
            
            fig_defect = plt.figure(figsize=(6, 6))
            plt.imshow(defect["mask"], cmap="gray")
            plt.title(f"Tomato {tomato_count}: Fully Defective (Standalone)")
            plt.axis("off")
            plt.tight_layout()
            mask_figures.append(fig_defect)

    # =================================================
    # SHOW ORIGINAL + PREDICTION WITH LEGEND & ANNOTATIONS
    # =================================================

    original_img = Image.open(image_path)

    fig_main = plt.figure(figsize=(14, 6))

    # Plot Original
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # Plot Prediction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(r.plot()[..., ::-1])
    plt.title("YOLO Prediction & Grading Result")
    plt.axis("off")

    # Add text annotations directly on the YOLO prediction
    for ann in annotations:
        plt.text(ann['x'], ann['y'], ann['text'], color='white', fontsize=11, fontweight='bold',
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    # Create Legend Elements
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='white', label=f'Overall Ripe: {ripe_pct:.1f}%'),
        Patch(facecolor='#FFC107', edgecolor='white', label=f'Overall Unripe: {unripe_pct:.1f}%'),
        Patch(facecolor='#F44336', edgecolor='white', label=f'Overall Defective: {defect_pct:.1f}%')
    ]
    
    # Place Legend at the bottom center
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig_main, bbox_inches="tight")
    plt.close(fig_main)

    # Save individual mask figures to PDF
    for fig in mask_figures:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    
# =====================================================
# SAVE AND CLOSE PDF
# =====================================================

pdf.close()
print("\n" + "="*100)
print(f"PDF Report successfully saved to: {pdf_path}")
print("="*100)
