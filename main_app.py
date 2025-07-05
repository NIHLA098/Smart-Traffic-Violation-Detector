import cv2
from ultralytics import YOLO
import datetime
import os
import pandas as pd
import numpy as np

print("🚦 Smart Traffic Violation Detection Started")

# Load YOLO model
model = YOLO('yolov8n.pt')

# Folder setup
os.makedirs('evidence', exist_ok=True)
csv_file = 'violations_log.csv'
if not os.path.exists(csv_file):
    pd.DataFrame(columns=['Type', 'Timestamp', 'Image']).to_csv(csv_file, index=False)

# -------------------------
# Utility Functions
# -------------------------

def log_violation(vtype, frame):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{vtype}_{timestamp}.jpg"
    filepath = os.path.join('evidence', filename)
    cv2.imwrite(filepath, frame)

    df = pd.read_csv(csv_file)
    df.loc[len(df.index)] = [vtype, timestamp, filename]
    df.to_csv(csv_file, index=False)
    print(f"⚠️ Violation Logged: {vtype} at {timestamp}")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# -------------------------
# Violation Detection Functions
# -------------------------

def detect_helmet_violation(frame):
    result = model(frame)[0]
    persons, helmets, bikes = [], [], []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        if conf < 0.3:
            continue
        if cls == 0: persons.append(xyxy)
        elif cls == 35: helmets.append(xyxy)  # Custom class for helmet
        elif cls == 2: bikes.append(xyxy)

    for person in persons:
        near_helmet = any(iou(person, h) > 0.2 for h in helmets)
        if not near_helmet:
            cv2.putText(frame, "No Helmet", (int(person[0]), int(person[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (0, 0, 255), 2)
            log_violation("Helmet Violation", frame)
    return frame

def detect_triple_riding(frame):
    result = model(frame)[0]
    bikes, persons = [], []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        if conf < 0.3:
            continue
        if cls == 2: bikes.append(xyxy)
        elif cls == 0: persons.append(xyxy)

    for bike in bikes:
        count = 0
        for person in persons:
            if iou(bike, person) > 0.1:
                count += 1
        if count > 2:
            cv2.putText(frame, f"Triple Riding ({count})", (int(bike[0]), int(bike[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(bike[0]), int(bike[1])), (int(bike[2]), int(bike[3])), (0, 0, 255), 2)
            log_violation("Triple Riding", frame)
    return frame

# -------------------------
# Main App Logic
# -------------------------

def main():
    print("🎥 Available videos:")
    print("1. Helmet Violation Video (helmet_test.mp4)")
    print("2. Triple Riding Video (traffic_rider.mp4)")
    print("3. Custom Video")

    choice = input("Enter choice (1/2/3): ").strip()
    if choice == '1':
        video_path = 'videos/helmet_test.mp4'
        detection = 'helmet'
    elif choice == '2':
        video_path = 'videos/traffic_rider.mp4'
        detection = 'triple'
    else:
        video_path = input("Enter full video path: ")
        detection = input("Which detection to run? (helmet/triple/both): ").strip().lower()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return

    print("🔁 Processing... Press Q to exit window")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🛑 End of video.")
            break

        if detection == 'helmet':
            frame = detect_helmet_violation(frame)
        elif detection == 'triple':
            frame = detect_triple_riding(frame)
        elif detection == 'both':
            frame = detect_helmet_violation(frame)
            frame = detect_triple_riding(frame)

        cv2.imshow("Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quit pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
