import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

# --- Config ---
VIDEO_PATH = "./1.mp4"
DET_MODEL_PATH = "./yolov8s-worldv2.pt"   
SAM_MODEL_PATH = "./mobile_sam.pt"       
OUTPUT_DIR = Path("auto_annotated_data")

# ✅ Create BOTH label types
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels_seg").mkdir(parents=True, exist_ok=True)  # segmentation (polygons)
(OUTPUT_DIR / "labels_det").mkdir(parents=True, exist_ok=True)  # detection (boxes)

TARGET_CLASSES = {"deer", "elk", "stag", "buck", "doe"}

# --- Load Models ---
print("📦 Loading YOLOv8s-Worldv2...")
det_model = YOLO(DET_MODEL_PATH)
det_model.set_classes(list(TARGET_CLASSES))

print("📦 Loading MobileSAM...")
sam_predictor = SAM(SAM_MODEL_PATH)

# --- Video Processing ---
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

frame_idx = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
saved_count = 0
print(f"🎥 Video has {total_frames} frames. Starting annotation...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    det_results = det_model(frame, verbose=False, device="cpu", conf=0.25, iou=0.45)
    boxes_xyxy = det_results[0].boxes.xyxy.cpu().numpy()
    cls_ids = det_results[0].boxes.cls.int().cpu().tolist()
    confs = det_results[0].boxes.conf.cpu().numpy()  # ← save confidence now!
    class_names = [det_model.names[i] for i in cls_ids]

    # Filter relevant classes
    valid_indices = [
        i for i, name in enumerate(class_names)
        if any(target in name.lower() for target in TARGET_CLASSES)
    ]

    if not valid_indices:
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"⏭️  Frame {frame_idx}/{total_frames} skipped")
        continue

    # Extract valid detections
    valid_boxes = boxes_xyxy[valid_indices]
    valid_cls_ids = [cls_ids[i] for i in valid_indices]
    valid_confs = [confs[i] for i in valid_indices]

    # Segment for mask labels
    sam_results = sam_predictor(frame, bboxes=valid_boxes, verbose=False, device="cpu")
    segments = sam_results[0].masks.xyn  # normalized polygons

    # ✅ Prepare labels
    seg_lines = []   # for labels_seg/ → segmentation
    det_lines = []   # for labels_det/ → detection

    h, w = frame.shape[:2]

    for i, (cls_id, conf, segment) in enumerate(zip(valid_cls_ids, valid_confs, segments)):
        if segment.size == 0:
            continue

        # ── 1. Segmentation label (polygon) ──
        seg_flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in segment)
        seg_lines.append(f"{cls_id} {seg_flat}")

        # ── 2. Detection label (bbox: cx, cy, w, h normalized) ──
        # Get bounding box from polygon (more accurate than YOLO box if mask is better)
        points = (segment * np.array([w, h])).astype(np.int32)
        x, y, box_w, box_h = cv2.boundingRect(points)
        cx = (x + box_w / 2) / w
        cy = (y + box_h / 2) / h
        nw = box_w / w
        nh = box_h / h
        det_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not det_lines:
        frame_idx += 1
        continue

    # ✅ Save image
    img_path = OUTPUT_DIR / "images" / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(img_path), frame)

    # ✅ Save both label types
    stem = f"frame_{frame_idx:06d}"
    (OUTPUT_DIR / "labels_seg" / f"{stem}.txt").write_text("\n".join(seg_lines))
    (OUTPUT_DIR / "labels_det" / f"{stem}.txt").write_text("\n".join(det_lines))

    saved_count += 1
    if frame_idx % 50 == 0:
        print(f"✅ Saved {len(det_lines)} objects in frame {frame_idx}")

    frame_idx += 1

cap.release()
print(f"\n🎉 Done. {saved_count} frames saved with BOTH label types.")