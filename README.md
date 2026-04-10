# AI-Driven Real-time Violence Detection Model

This repository contains the PyTorch implementation for fine-tuning a pre-trained Video Classification 3D CNN (ResNet3D) model for real-time violence detection.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Add your video datasets to the appropriately named folders in the `dataset` directory:
```text
dataset/
├── violence/
│   ├── video1.mp4
│   └── video2.mp4
└── non_violence/
    ├── video1.mp4
    └── video2.mp4
```

## Training

The training script automatically performs a 2-phase fine-tuning strategy:
1. Frozen backbone with training only on the classification head.
2. Unfrozen full-network fine-tuning with a reduced learning rate.

To start training, simply run:
```bash
python train.py
```

## Model Evaluation

During training, the script evaluates the model on a validation split, providing:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The final trained model is saved as `violence_detection_model.pth`.
