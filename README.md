# 🤖 Auto-Annotation Toolkit for Computer Vision

[![Ultralytics](https://img.shields.io/badge/Ultralytics-%E2%9C%93-blue)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A lightweight, flexible toolkit to **automatically annotate videos and images** using modern foundation models — no manual labeling required.

Use YOLO (detection) + SAM/MobileSAM (segmentation) to generate:
- ✅ **YOLO detection labels** (`class cx cy w h`)  
- ✅ **YOLO segmentation labels** (`class x1 y1 x2 y2 ...`)  
- ✅ **Visual validation** (overlay images with boxes/masks)

Ideal for bootstrapping datasets in resource-constrained environments (CPU-only, offline, low-bandwidth).

---

## 🌟 Features

| Feature | Description |
|--------|-------------|
| **Open-Vocabulary Detection** | Leverage `YOLO-World` to detect custom objects (e.g., `"drone"`, `"solar panel"`, `"wildlife"`) without retraining |
| **Lightweight Segmentation** | Use `MobileSAM` (~40MB) for fast, CPU-friendly mask generation |
| **Video Support** | Process videos frame-by-frame, skipping empty frames |
| **Dual Label Export** | Generate both detection and segmentation labels simultaneously |
| **Visualization Tools** | Inspect annotations with bounding boxes, masks, or both |
| **CPU-Optimized** | Runs efficiently on modest hardware (e.g., Intel i5, Jetson, Raspberry Pi 4) |
| **No Cloud / No API** | Fully offline — works in restricted/regional networks |

---

## 🛠️ Installation

### Prerequisites
- Python ≥ 3.8
- `ffmpeg` (for video I/O)

### Setup
```bash
# Clone repo
git clone https://github.com/javad71/auto-annotation.git
cd auto-annotation

# Install dependencies
pip install -r requirements.txt

# Optional: Install MobileSAM (if not using Ultralytics >=8.3)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

> 💡 **Note**: As of Ultralytics v8.3+, `mobile_sam` is built-in — no extra install needed.

---

## 📁 Project Structure

```
auto-annotation/
├── video_annotate.py           # Main annotation script (video → labels)
├── visualize_annotations.py    # Generate annotated previews
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Annotate a Video
```bash
python video_annotate.py 
```

> 🔍 Labels saved to `./auto_annotated_data/{images,labels_det,labels_seg}` in standard YOLO format.

### 2. Visualize Results
```bash
python visualize_annotations.py   
```
Outputs `./auto_annotated_data/visual/` with overlay images for quality inspection.

---

## ⚙️ Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video | `./input.mp4` |
| `--det_model` | Detection model (`yolov8s-worldv2.pt`, `yolov8n.pt`, or custom `.pt`) | `yolov8s-worldv2.pt` |
| `--sam_model` | Segmentation model (`mobile_sam`, `sam_b.pt`, or local `.pt`) | `mobile_sam` |
| `--classes` | Comma-separated object names (open-vocab) | `"object"` |
| `--conf` | Detection confidence threshold | `0.25` |
| `--device` | `cpu` or `cuda` | auto-select |
| `--output_dir` | Output directory | `./auto_annotated_data` |

---

## 📊 Output Format

### Directory Layout
```
auto_annotated_data/
├── images/              # Extracted frames (JPEG)
├── labels_det/          # Detection labels (YOLO format)
│   └── frame_000000.txt # → "0 0.5 0.5 0.2 0.3"
├── labels_seg/          # Segmentation labels (YOLO format)
│   └── frame_000000.txt # → "0 0.4 0.3 0.45 0.32 ..."
└── visual/              # Optional: annotated previews
```

### Label Formats
- **Detection**: `class_id center_x center_y width height` (all normalized to `[0,1]`)
- **Segmentation**: `class_id x1 y1 x2 y2 ...` (normalized polygon vertices)

✅ Ready for direct use with `yolo train task=detect` or `task=segment`.

---

## 🧪 Training with Annotated Data

Create `data.yaml`:
```yaml
path: ./auto_annotated_data
train: images
val: images
names:
  0: drone
  1: solar_panel
```

Train a lightweight detector:
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=8 device=cpu
```

Export for edge deployment:
```bash
yolo export model=best.pt format=onnx opset=12
```

---

## 💡 Tips for Best Results

- ✅ **Start with YOLO-World** for open-vocabulary bootstrapping  
- ✅ Use `mobile_sam` for fast CPU segmentation  
- ✅ Filter low-confidence frames manually for high-stakes applications  
- ✅ Retrain on auto-annotated data → iterate for higher accuracy  
- ✅ Downscale frames (`imgsz=480`) to speed up annotation on low-end CPUs


## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)  
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

---

> 📬 Feedback? Open an [issue](https://github.com/javad71/auto-annotation/issues) or submit a PR!  
> 🔒 Designed for **offline, privacy-sensitive, and resource-limited** environments.

--- 

✅ **Ready to accelerate your CV pipeline — no labeling fatigue.**