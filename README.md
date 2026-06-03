<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>

# Modular Hybrid Perception Framework
### Real-Time Lane Detection & Traffic Object Recognition for ADAS / Autonomous Driving

*Fusing classical computer vision with deep learning — designed for reproducibility, modularity, and real-world deployment.*

<br/>

[Overview](#-overview) · [Architecture](#-architecture) · [Performance](#-performance) · [Installation](#-installation) · [Usage](#-usage) · [Dataset](#-dataset) · [API Reference](#-api-reference) · [Roadmap](#-roadmap)

</div>

---

## Overview

This framework is a research-grade, production-ready perception system combining **classical computer vision** and **deep learning** into a unified, modular pipeline. It is built for developers and researchers working on autonomous driving, ADAS, or hybrid perception research who need a clean, extensible foundation.

**Why this framework?**

Most open-source lane detection systems either rely purely on classical CV (brittle under lighting changes) or on heavy deep networks (hard to interpret and deploy). This project takes a hybrid approach — pairing the speed and interpretability of classical techniques with the robustness of YOLOv8, all wired through a clean REST API and a real-time dashboard.

**Key highlights:**

- Classical lane detection (CLAHE → ROI → BEV → Sliding Window → Polynomial Fit) running at **28 FPS on GPU** and **11 FPS on CPU**
- YOLOv8-based traffic object detection with COCO road-class filtering
- Flask REST backend with CLI and API training modes
- Live camera inference with real-time visualization dashboard
- Achieves **mAP@0.5 of 0.91** and **lane mIoU of 0.86** on validation data

---

## Architecture

The system is composed of four fully decoupled modules. Each module can be independently replaced, upgraded, or benchmarked.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Input Layer                                  │
│           Live Camera Feed  │  Uploaded Image/Video              │
└─────────────────────┬────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Preprocessing Module                            │
│       Frame normalization · Resize · Color space conversion      │
└───────────────────┬──────────────────────┬───────────────────────┘
                    │                      │
          ┌─────────▼──────────┐  ┌────────▼──────────────┐
          │  Lane Detection    │  │   Object Detection     │
          │  (Classical CV)    │  │   (YOLOv8 Deep DL)     │
          │                    │  │                        │
          │ • CLAHE            │  │ • Backbone feature     │
          │ • Adaptive ROI     │  │   extraction           │
          │ • Bird's Eye View  │  │ • Road-class filtering │
          │ • Sliding Window   │  │ • BBox regression      │
          │ • Polynomial Fit   │  │ • GPU acceleration     │
          │ • Temporal Smooth  │  │                        │
          └─────────┬──────────┘  └────────┬───────────────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
          ┌────────────────────────────────────────────┐
          │        Fusion & Visualization Module        │
          │  Lane overlay · BBox rendering · Telemetry  │
          └──────────────────────┬─────────────────────┘
                                 ▼
          ┌────────────────────────────────────────────┐
          │           Real-Time Dashboard              │
          │  Live feed · Stats panel · Detection log   │
          └────────────────────────────────────────────┘
```

### Component Responsibilities

| Module | Technology | Responsibility |
|---|---|---|
| Preprocessing | OpenCV | Normalization, resizing, color conversion |
| Lane Detection | Classical CV (CLAHE, BEV, Polynomial Fit) | Lane boundary localization, curvature & offset estimation |
| Object Detection | YOLOv8s (Ultralytics) | Real-time traffic object recognition and bounding box prediction |
| Fusion & Viz | OpenCV + HTML Canvas | Overlay rendering, telemetry display, output composition |
| Backend | Flask + REST | Inference serving, training orchestration, API routing |
| Frontend | Vanilla JS + CSS | Live camera feed, upload interface, stats dashboard |

---

## Performance

Benchmarks measured on the validation split of the training dataset.

| Metric | Score |
|---|---|
| Object Detection mAP@0.5 | **0.91** |
| Object Detection mAP@0.5:0.95 | **0.67** |
| Lane Detection mIoU | **0.86** |
| Inference Speed — GPU (NVIDIA RTX 3050) | **28 FPS** |
| Inference Speed — CPU (Intel i5 12th Gen) | **11 FPS** |

> Results are reported on the validation set. Performance may vary depending on hardware configuration, dataset distribution, and lighting conditions.

---

## Repository Structure

```
Modular-Hybrid-Perception-Framework/
│
├── backend/
│   ├── app.py                  # Flask server: serve / train / test modes
│   ├── lanenet_model.py        # LaneNet deep model (optional)
│   ├── postprocess.py          # Lane post-processing utilities
│   ├── requirements.txt        # Python dependencies
│   └── yolov8s.pt              # Pre-trained YOLOv8s weights
│
├── frontend/
│   ├── index.html              # Main dashboard UI
│   ├── styles.css              # Dashboard styling
│   └── script.js               # Live feed, upload, and stats logic
│
├── models/
│   └── lanenet.pth             # LaneNet weights (optional deep lane model)
│
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
│
├── docs/
│   ├── output_highway.png
│   ├── output_night.png
│   ├── output_intersection.png
│   └── architecture.png
│
├── data.yaml                   # Dataset configuration for YOLO training
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+, pip, and optionally an NVIDIA GPU with CUDA.

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/Modular-Hybrid-Perception-Framework.git
cd Modular-Hybrid-Perception-Framework
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

**4. Verify GPU availability (optional)**

```python
import torch
print(torch.cuda.is_available())   # True if GPU is ready
```

---

## Dataset

Due to storage constraints, datasets are not bundled in this repository.

**Lane Segmentation Dataset**
Download from Kaggle: [Road Lane Instance Segmentation](https://www.kaggle.com/datasets/sovitrath/road-lane-instance-segmentation)

**Expected directory structure after download:**

```
dataset/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

Update `data.yaml` with the correct paths before training.

---

## Usage

### Start the Inference Server

```bash
python backend/app.py --mode serve
```

Then open your browser at `http://localhost:5000`. The dashboard will load with the live camera interface and upload panel.

### Train the Object Detection Model

```bash
python backend/app.py --mode train --data data.yaml --epochs 50
```

Training results are saved to `train_results/`. You can monitor live metrics from the dashboard.

### Run Evaluation

```bash
python backend/app.py --mode test --model best.pt
```

Outputs mAP, precision, recall, mIoU, and pixel accuracy to the console and saves a results summary.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `YOLO_DEVICE` | `cpu` | Set to `cuda` to enable GPU inference |
| `YOLO_MODEL` | `yolov8s.pt` | Path to YOLOv8 model weights |
| `LANENET_WEIGHTS` | `models/lanenet.pth` | Path to optional LaneNet weights |

Example with GPU:
```bash
YOLO_DEVICE=cuda python backend/app.py --mode serve
```

---

## API Reference

The backend exposes a REST API for integration with external systems.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Run inference on a base64-encoded image |
| `GET` | `/stream` | MJPEG live camera stream |
| `POST` | `/train` | Trigger model training programmatically |
| `GET` | `/stats` | Retrieve real-time detection statistics |

**Example `/detect` request:**

```python
import requests, base64, cv2

_, buf = cv2.imencode('.jpg', frame)
b64 = base64.b64encode(buf).decode()

resp = requests.post('http://localhost:5000/detect',
                     json={'image': b64})
print(resp.json())   # { "lanes": [...], "objects": [...], "fps": 28.4 }
```

---

## Detected Road Classes

The object detector is filtered to road-relevant COCO classes only, reducing false positives from irrelevant categories.

| Class | COCO ID | Color (BGR) |
|---|---|---|
| Person | 0 | Orange-Yellow |
| Bicycle | 1 | Light Blue |
| Car | 2 | Red |
| Motorcycle | 3 | Orange |
| Bus | 5 | Dark Red |
| Truck | 7 | Darker Red |
| Traffic Light | 9 | Green |
| Stop Sign | 11 | Crimson |
| Fire Hydrant | 10 | Orange |
| Parking Meter | 12 | Teal |

---

## Evaluation Metrics

**Object Detection**
- Mean Average Precision at IoU 0.5 (mAP@0.5)
- Mean Average Precision at IoU 0.5–0.95 (mAP@0.5:0.95)
- Precision and Recall per class

**Lane Detection**
- Mean Intersection over Union (mIoU)
- Pixel Accuracy

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| GPU | — | NVIDIA GPU with CUDA |
| OS | Windows / Linux / macOS | Ubuntu 20.04+ |

---

## Roadmap

Planned extensions for future versions:

- [ ] Multi-camera surround perception
- [ ] Sensor fusion with LiDAR and Radar inputs
- [ ] Model export to TensorRT and ONNX for optimized inference
- [ ] Object tracking integration (SORT, DeepSORT)
- [ ] 3D bounding box estimation
- [ ] Edge deployment optimization for Jetson Nano / Raspberry Pi
- [ ] Docker containerization for one-command deployment

---

## Research Applications

This framework is suitable as a starting point or baseline for:

- Autonomous driving perception systems
- Advanced Driver Assistance Systems (ADAS)
- Real-time hybrid perception research
- Robotics and intelligent transportation
- Academic benchmarking of lane and object detection methods

---

## Author

**M V Karthikeya**
B.Tech — Computer Science (AI & ML)
SRM Institute of Science and Technology

Research interests: Computer Vision · Autonomous Systems · Deep Learning · Perception Systems

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with precision for the autonomous driving research community.</sub>
</div>
