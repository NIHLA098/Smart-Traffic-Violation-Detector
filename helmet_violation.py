import cv2
from ultralytics import YOLO
import os
import datetime
import pandas as pd

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Pretrained model

# CSV file for logging
csv_file = 'violations_log.csv'
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['Type', 'Timestamp', 'Image'])
    df.to_csv(csv_file, index=False)

# Set webcam or video file
use_webcam = False
cap = cv2.VideoCapture(0 if use_webcam else 'videos/helmet_test.mp4')

if not cap.isOpened():
    print("❌ Error: Could not open the video/camera.")
    exit()

while cap.isOpened():
    start_time = datetime.datetime.now()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    persons = []
    bikes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls]

        # Draw boxes
        color = (0, 255, 0) if cls == 0 else (0, 0, 255) if cls == 2 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls == 0:
            persons.append((x1, y1, x2, y2))
        elif cls == 2:
            bikes.append((x1, y1, x2, y2))

    # Only detect largest person (most likely rider)
    if persons:
        px1, py1, px2, py2 = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx, cy = (px1 + px2) // 2, (py1 + py2) // 2

        for bx1, by1, bx2, by2 in bikes:
            if bx1 < cx < bx2 and by1 < cy < by2:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"evidence/helmet_violation_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                try:
                    df = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame(columns=['Type', 'Timestamp', 'Image'])

                df.loc[len(df.index)] = ["Helmet Violation", timestamp, filename]
                df.to_csv(csv_file, index=False)

                print(f"[!] Helmet Violation Logged at {timestamp}")
                break

    # Display FPS
    end_time = datetime.datetime.now()
    fps = 1 / (end_time - start_time).total_seconds()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show result
    cv2.imshow("Helmet Violation Detection - Live", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
