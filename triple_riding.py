import cv2
from ultralytics import YOLO
import datetime
import os
import pandas as pd
import numpy as np

print("🚦 Triple Riding Detection Started")

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 model loaded")

# Ensure folders and log file
os.makedirs('evidence', exist_ok=True)
csv_file = 'violations_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w') as f:
        f.write("Type,Timestamp,Image\n")
    print("📄 Created violations_log.csv")

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def log_violation(violation_type, frame):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"violation_{violation_type}_{timestamp}.jpg"
    filepath = os.path.join('evidence', filename)
    cv2.imwrite(filepath, frame)

    # Append to CSV directly
    with open(csv_file, 'a') as f:
        f.write(f"{violation_type},{timestamp},{filename}\n")

    print(f"⚠️ Logged violation: {violation_type} at {timestamp}")

def detect_triple_riding(frame):
    results = model(frame)[0]
    bikes = []
    persons = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        if conf < 0.3:
            continue

        if cls == 2:  # bike
            bikes.append(xyxy)
        elif cls == 0:  # person
            persons.append(xyxy)

    print(f"🔍 Found {len(bikes)} bikes and {len(persons)} persons")

    violation_found = False

    for bike_box in bikes:
        count = 0
        for person_box in persons:
            iou = bb_intersection_over_union(bike_box, person_box)
            if iou > 0.1:
                count += 1

        if count >2:

            violation_found = True
            cv2.putText(frame, f"Triple Riding Detected ({count})",
                        (int(bike_box[0]), int(bike_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(bike_box[0]), int(bike_box[1])),
                          (int(bike_box[2]), int(bike_box[3])), (0, 0, 255), 2)
            log_violation("Triple Riding", frame)

    if not violation_found:
        cv2.putText(frame, "No Triple Riding Detected",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def main():
    video_path = 'videos/triple_riding.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    print(f"🎥 Video Loaded: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🛑 End of video")
            break

        output_frame = detect_triple_riding(frame)
        cv2.imshow("Triple Riding Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quit pressed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
