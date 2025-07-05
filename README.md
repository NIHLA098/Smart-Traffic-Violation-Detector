# 🚦 AutoGuard AI: Real-Time Traffic Violation Detection System

**Detects helmet violations, triple riding, and signal jumping using YOLOv8 and OpenCV**

---

## 🧠 Overview

**AutoGuard AI** is an intelligent traffic monitoring system that uses **YOLOv8 object detection** and **OpenCV** to identify and log road rule violations in real-time. It detects:

* 🪖 Helmet violations
* 👨‍👩‍👧 Triple riding on two-wheelers
* 🚦 Signal jumping (crossing stop line on red light)

Violations are automatically captured, annotated, and logged with a timestamp and saved image evidence.

---

## 💡 Features

| Feature                | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| 🎯 Helmet Detection    | Detects riders without helmets using object detection       |
| 👨‍👩‍👧 Triple Riding | Identifies 3+ persons on a two-wheeler                      |
| 🚦 Signal Violation    | Detects vehicles crossing a virtual stop line on red signal |
| 📸 Evidence Capture    | Saves violation frames in `/evidence/` folder               |
| 📊 CSV Logging         | Logs violation type, timestamp, and image filename          |

---

## 📁 Project Structure

```
autoguard-ai/
├── videos/                # Input traffic videos
├── evidence/              # Captured violation images
├── violations_log.csv     # CSV file logging all violations
├── helmet_violation.py    # Helmet detection script
├── triple_riding.py       # Triple riding detection script
├── signal_jump.py         # Signal jumping detection
├── main_app.py            # (Optional) unified launcher
└── README.md              # You're here!
```

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python pandas
```

Make sure you have:

* Python 3.8+
* OpenCV (cv2)
* YOLOv8 (Ultralytics)

---

## ▶️ How to Run

### 1. Helmet Violation Detection

```bash
python helmet_violation.py
```

### 2. Triple Riding Detection

```bash
python triple_riding.py
```

### 3. Signal Jump Detection

```bash
python signal_jump.py
```

Make sure video files are placed inside the `/videos/` folder and paths in scripts are correct.

---

## 📸 Sample Output

> Screenshots of violations will be saved in the `evidence/` folder

| Violation Type   | Image                                 |
| ---------------- | ------------------------------------- |
| Helmet Violation | ![helmet](evidence/sample_helmet.jpg) |
| Triple Riding    | ![triple](evidence/sample_triple.jpg) |
| Signal Jump      | ![signal](evidence/sample_signal.jpg) |

---

## 🚀 Future Enhancements

* 🚗 Number plate recognition
* 🌍 GPS-based location tagging
* 📡 Live webcam support with Streamlit dashboard

---

## 🏁 Credits

* YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
* OpenCV for real-time video processing

---

## 📬 Contact

For queries or collaboration: `fathimanihla841@gmail.com`

---

**Made with ❤️ to make streets safer using AI.**
