from ultralytics import YOLO

# Downloads automatically on first use
model = YOLO('yolov8n.pt')  # 'n' = nano, lightest for M1 8GB
print("YOLOv8 downloaded and ready")