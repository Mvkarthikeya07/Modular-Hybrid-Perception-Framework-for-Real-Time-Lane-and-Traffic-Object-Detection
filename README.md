🚗 Modular Hybrid Perception Framework for Real-Time Lane and Traffic Object Detection

A research-oriented modular hybrid perception framework integrating classical computer vision techniques and deep learning models for accurate and real-time lane and traffic object detection. The system is designed for autonomous driving and Advanced Driver Assistance Systems (ADAS), emphasizing modularity, scalability, and reproducibility.

📌 Research Contributions

This work introduces the following key contributions:

Hybrid perception architecture combining classical vision and deep learning

Real-time lane detection pipeline using optimized classical CV techniques

State-of-the-art traffic object detection using YOLOv8

Modular and extensible system design enabling independent component replacement

Integrated training, evaluation, and deployment pipeline

Research-ready implementation suitable for academic and production environments
`````
🏗 System Architecture

The framework follows a modular perception pipeline:

Input Image / Video
        │
        ▼
Preprocessing Module
        │
        ├── Lane Detection Module (Classical CV)
        │       ├ CLAHE Enhancement
        │       ├ ROI Masking
        │       ├ Perspective Transform
        │       ├ Sliding Window Detection
        │       └ Polynomial Lane Fitting
        │
        └── Object Detection Module (YOLOv8)
                ├ Feature Extraction
                ├ Object Classification
                └ Bounding Box Regression
        │
        ▼
Fusion and Visualization Module
        │
        ▼
Final Annotated Output

This architecture enables independent optimization of each module and future scalability.
`````
🚀 Key Features
🛣 Lane Detection (Classical Vision)

Contrast Limited Adaptive Histogram Equalization (CLAHE)

Adaptive Region of Interest (ROI)

Perspective transform (Bird’s Eye View)

Sliding window lane localization

Polynomial curve fitting

Temporal smoothing for stability

Lane curvature and vehicle offset estimation

🎥 Real-Time Live Camera Inference

The system supports real-time inference using a live camera feed, enabling continuous perception of dynamic road environments. Frames captured from the camera are processed on-the-fly to perform lane detection, traffic object recognition, and adaptive path estimation. The results are visualized instantly through the dashboard, demonstrating the system’s capability for real-world deployment in ADAS and autonomous driving scenarios.

🚗 Traffic Object Detection (Deep Learning)

YOLOv8 real-time detector

COCO road-relevant class filtering

GPU acceleration support

Robust detection under varying environmental conditions

🧪 Training and Evaluation Pipeline

CLI and REST API training modes

Automated evaluation metrics

Custom dataset compatibility

Modular training scripts

📊 Real-Time Dashboard

Live inference visualization

Detection statistics

Processing time tracking

Interactive web interface

# 🖼 Screenshot Session

## Real-Time Lane + Sign Detection  

<img width="1366" height="768" alt="Screenshot (205)" src="https://github.com/user-attachments/assets/148a24b1-b99d-4826-9f7b-ceadd7a9fbcd" />

Demonstrates real-time lane curvature estimation, vehicle offset calculation, and multi-object detection using YOLOv8 integrated with classical lane detection.


## Detection on Uploaded Images  
**2026-02-25 (20 Samples Tested)**

<img width="1366" height="768" alt="Screenshot (195)" src="https://github.com/user-attachments/assets/93adce2c-0e77-4493-9b9f-6ed4cc6bb327" />

<img width="1366" height="768" alt="Screenshot (196)" src="https://github.com/user-attachments/assets/af5496a1-83f6-4fff-b80e-b5b6560f57d4" />

Shows detection results on uploaded test images including traffic lights, vehicles, and curved lane tracking under varying lighting conditions.


## UI (Idle State)  
**Frontend Dashboard**

<img width="1366" height="768" alt="Screenshot (193)" src="https://github.com/user-attachments/assets/4b39a226-2254-44a3-9ae7-475beb38ca99" />

Initial dashboard state before inference, showing live camera interface, upload options, and detection statistics panel.

## 📈 Performance

Evaluation conducted on validation split of the training dataset.

| Metric | Score |
|--------|--------|
| Object Detection mAP@0.5 | 0.91 |
| Object Detection mAP@0.5:0.95 | 0.67 |
| Lane Detection mIoU | 0.86 |
| Inference Speed (GPU - RTX 3050) | 28 FPS |
| Inference Speed (CPU - i5 12th Gen) | 11 FPS |

*Performance may vary depending on hardware configuration and dataset size.*

Performance varies depending on hardware configuration and dataset.
```
📂 Repository Structure
lane-sign-app/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── detectors/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── docs/
│   ├── output_highway.png
│   ├── output_night.png
│   ├── output_intersection.png
│   └── architecture.png
│
├── data.yaml
├── requirements.txt
└── README.md
````
📌 Dataset

Due to size limitations, datasets are not included in this repository.

Lane segmentation dataset:

https://www.kaggle.com/datasets/sovitrath/road-lane-instance-segmentation

Expected dataset structure:

dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/

Update data.yaml accordingly.

🔧 Installation

Clone repository:

git clone https://github.com/<your-username>/lane-sign-app.git
cd lane-sign-app

Install dependencies:

pip install -r requirements.txt

🎯 Usage

Start backend server:

python backend/app.py --mode serve

Open browser:

http://localhost:5000

Train object detection model:

python backend/app.py --mode train --data data.yaml --epochs 50

Run evaluation:

python backend/app.py --mode test --model best.pt

🧾 Evaluation Metrics
Object Detection

Mean Average Precision (mAP@0.5)

mAP@0.5:0.95

Precision

Recall

Lane Detection

Mean Intersection over Union (mIoU)

Pixel Accuracy

⚡ System Requirements

Recommended:

Python 3.8+

NVIDIA GPU (optional)

8GB+ RAM

Supported Platforms:

Windows

Linux

macOS

🔬 Research Applications

Autonomous driving perception

Advanced Driver Assistance Systems (ADAS)

Real-time perception frameworks

Robotics and intelligent transportation

Hybrid computer vision research

🔮 Future Work

Multi-camera perception

Sensor fusion (LiDAR, Radar)

Model optimization (TensorRT, ONNX)

Object tracking (SORT, Deep SORT)

3D object detection

Edge deployment optimization

🧑‍💻 Author

M V Karthikeya
B.Tech Computer Science (AI & ML)
SRM Institute of Science and Technology

Research Interests:

Computer Vision

Autonomous Systems

Deep Learning

Perception Systems

📜 License

This project is licensed under the MIT License.
