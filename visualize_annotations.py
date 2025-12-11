import cv2
import numpy as np
from pathlib import Path

# --- Config ---
IMAGES_DIR = Path("auto_annotated_data/images")
LABELS_DET_DIR = Path("auto_annotated_data/labels_det")   # detection: class cx cy w h
LABELS_SEG_DIR = Path("auto_annotated_data/labels_seg")   # segmentation: class x1 y1 x2 y2 ...
OUTPUT_VIS_DIR = Path("auto_annotated_data/visual")
OUTPUT_VIS_DIR.mkdir(exist_ok=True)

# Toggle mode: "detection" or "segmentation"
VIS_MODE = "detection"  # ← change to "detection" for bbox-only view

# Class mapping (update based on your TARGET_CLASSES order in annotation)
CLASS_NAMES = {
    0: "deer",
    1: "elk",
    2: "stag",
    3: "buck",
    4: "doe",
}
CLASS_COLORS = {
    0: (0, 255, 0),    # green
    1: (255, 165, 0),  # orange
    2: (0, 191, 255),  # deep sky blue
    3: (255, 105, 180),# hot pink
    4: (147, 112, 219),# medium purple
    -1: (255, 0, 0),   # fallback red
}

MASK_ALPHA = 0.4
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Get frames
image_paths = sorted(IMAGES_DIR.glob("frame_*.jpg"))
print(f"🔍 Found {len(image_paths)} frames. Visualizing in '{VIS_MODE}' mode...")

for img_path in image_paths:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️ Skip {img_path.name}: failed to load")
        continue

    h, w = image.shape[:2]
    visual = image.copy()
    overlay = image.copy()

    # Choose label dir & parse
    if VIS_MODE == "segmentation":
        label_path = LABELS_SEG_DIR / f"{img_path.stem}.txt"
        label_type = "seg"
    else:  # "detection"
        label_path = LABELS_DET_DIR / f"{img_path.stem}.txt"
        label_type = "det"

    if not label_path.exists() or not label_path.stat().st_size:
        continue

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        color = CLASS_COLORS.get(cls_id, CLASS_COLORS[-1])
        class_name = CLASS_NAMES.get(cls_id, "unknown")

        if label_type == "seg":
            # Parse polygon: class_id x1 y1 x2 y2 ...
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            points = (coords * np.array([w, h])).astype(np.int32)

            # Draw mask
            cv2.fillPoly(overlay, [points], color=color)

            # Get tight bbox from mask
            x, y, bw, bh = cv2.boundingRect(points)
            x2, y2 = x + bw, y + bh

        else:  # detection: class_id cx cy w h
            cx, cy, nw, nh = map(float, parts[1:5])
            x = int((cx - nw / 2) * w)
            y = int((cy - nh / 2) * h)
            x2 = int((cx + nw / 2) * w)
            y2 = int((cy + nh / 2) * h)

            # Draw bbox directly
            cv2.rectangle(visual, (x, y), (x2, y2), color, 2)

        # Always draw bbox + label (consistency)
        cv2.rectangle(visual, (x, y), (x2, y2), color, 2)
        label_text = f"{class_name} ??"  # ?? = unknown confidence
        cv2.putText(visual, label_text, (x, y - 10), FONT, 0.6, color, 2, cv2.LINE_AA)

    # Blend mask if segmentation
    if VIS_MODE == "segmentation":
        visual = cv2.addWeighted(overlay, MASK_ALPHA, visual, 1 - MASK_ALPHA, 0)

    # Add frame ID
    cv2.putText(visual, img_path.stem, (10, 30), FONT, 0.7, (255, 255, 255), 2)

    # Save
    out_path = OUTPUT_VIS_DIR / f"{img_path.stem}_vis_{VIS_MODE}.jpg"
    cv2.imwrite(str(out_path), visual)

print(f"✅ Done! Visuals saved to: {OUTPUT_VIS_DIR}")