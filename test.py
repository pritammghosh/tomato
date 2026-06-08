# from ultralytics import YOLO

# # Load model
# model = YOLO("best.pt")

# # Predict on all images inside images folder
# results = model.predict(
#     source="images",
#     conf=0.25,
#     save=True,
#     show=True
# )

# print("Prediction Complete!")




from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import os

# Load model
model = YOLO("best.pt")

# Get first 15 images
image_paths = glob.glob("images/*")[:15]

pdf = PdfPages("Tomato_Report.pdf")

for img_path in image_paths:

    results = model(img_path)

    annotated = results[0].plot()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(annotated[:, :, ::-1])
    ax.axis("off")

    title = os.path.basename(img_path)

    preds = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0]) * 100

        preds.append(
            f"{model.names[cls]} ({conf:.1f}%)"
        )

    ax.set_title(
        title + "\n" + ", ".join(preds),
        fontsize=10
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

pdf.close()

print("PDF saved as Tomato_Report.pdf")