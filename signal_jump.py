import cv2
import os
import datetime
import pandas as pd
from ultralytics import YOLO

print("🚦 Signal Jump Detection Started")

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  
print("✅ YOLOv8 model loaded")

# Setup folders
os.makedirs("evidence", exist_ok=True)
csv_file = "violations_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("Type,Timestamp,Image\n")
    print("📄 Created violations_log.csv")

# Virtual stop line Y position
STOP_LINE_Y = 300  # Adjust based on your video

def log_violation(violation_type, frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"violation_{violation_type}_{timestamp}.jpg"
    filepath = os.path.join("evidence", filename)

    print(f"📸 Saving image to: {filepath}")
    frame_copy = frame.copy()
    success = cv2.imwrite(filepath, frame_copy)
    if success:
        print("✅ Image saved successfully!")
    else:
        print("❌ Failed to save image!")

    with open(csv_file, "a") as f:
        f.write(f"{violation_type},{timestamp},{filename}\n")

    print(f"⚠️ Logged {violation_type} at {timestamp}")

def detect_signal_jump(frame):
    result = model(frame)[0]
    lights = []
    vehicles = []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        if conf < 0.3:
            continue

        if cls == 9:  # traffic light
            lights.append((xyxy, conf))
        elif cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            vehicles.append(xyxy)

    red_light_detected = len(lights) > 0  # simplified assumption

    # Draw stop line
    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)

    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle
        center_y = int((y1 + y2) / 2)

        if center_y < STOP_LINE_Y:
            continue  # No violation

        if red_light_detected and center_y > STOP_LINE_Y:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, "Signal Jump", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            log_violation("Signal Jump", frame)

    return frame

def main():
    video_path = "videos/signal_test.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    print("🎥 Video Loaded")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🛑 End of video")
            break

        output_frame = detect_signal_jump(frame)
        cv2.imshow("Signal Jump Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
