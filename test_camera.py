import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)  # 0 = built-in Mac camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 - detect only 'person' class (class 0)
    results = model(frame, classes=[0], verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Person Detection - Press Q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()