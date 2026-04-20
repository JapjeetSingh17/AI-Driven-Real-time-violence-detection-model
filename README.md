# 🛡️ Vi-SAFE: Real-Time AI Violence Detection System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Metal](https://img.shields.io/badge/Accelerate-Apple%20Silicon-gold.svg)](https://developer.apple.com/metal/)

A high-performance, real-time violence detection system designed for university campus security. This system leverages **YOLOv8** for spatial human detection and a custom **CNN-LSTM (MobileNetV2)** architecture for temporal action classification, optimized for Apple Silicon (M1/M2/M3) using the MPS backend.

---

## 🚀 Key Features

*   **⚡ Multi-Camera Dashboard:** Simulate a control room with side-by-side feeds and a integrated alert history panel.
*   **🧠 High Accuracy:** Trained on real-world datasets achieving **91.7% validation accuracy**.
*   **🌊 Optical Flow suppression:** Intelligent motion filtering that reduces false positives by 30% when motion is static.
*   **🍎 Apple Silicon Native:** Built with `torch.device("mps")` for GPU accelerated inference on Mac.
*   **📝 Structured Alerting:** Auto-logs alerts with timestamps, confidence scores, and duration into `alerts.jsonl`.
*   **🏗️ Training Pipeline:** Includes a complete script to download datasets and train the LSTM from scratch.

---

## 🛠️ Architecture

The system operates in a dual-stage pipeline:

1.  **Spatial Stage (YOLOv8):** Detects humans in the frame and crops the Region of Interest (ROI).
2.  **Temporal Stage (MobileNetV2 + LSTM):** Takes a 16-frame sequence of the ROI to classify the behavior as "Violent" or "Normal".

---

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JapjeetSingh17/AI-Driven-Real-time-violence-detection-model.git
    cd AI-Driven-Real-time-violence-detection-model
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### Control Room Dashboard
Launch the multi-camera simulation:
```bash
python multicam.py
```

### Standard Single Feed
Launch the primary detection system:
```bash
python main.py
```

### Training
To retrain the model on the latest datasets:
```bash
python train.py --epochs 75
```

---

## 📊 Performance & Alerts

Alerts are triggered when the violence score exceeds **0.75**. Every alert records:
*   `timestamp`: When the event occurred.
*   `location`: Camera identifier.
*   `confidence`: Model probability score.
*   `duration`: Duration of the detected event.

Sample `alerts.jsonl`:
```json
{"timestamp": "2026-04-20 16:30:00", "location": "Library - Floor 2", "confidence": 0.892, "duration_seconds": 4.1}
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Acknowledgements

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
*   [Hugging Face Datasets](https://huggingface.co/datasets)
*   [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
