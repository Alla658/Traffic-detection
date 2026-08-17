# Smart Traffic Management System 🚦📊

An AI-powered traffic monitoring and analytics system built using **YOLOv8**, **OpenCV**, **DeepSORT**, and **Flask**.

The system detects and tracks vehicles from a traffic video, classifies them into different categories, calculates real-time traffic density, maintains vehicle counts, and displays the results through a web-based dashboard.

---

## 🚀 Features

- 🚗 Real-time vehicle detection using YOLOv8
- 🚌 Vehicle classification:
  - Cars
  - Buses
  - Trucks
  - Motorcycles
  - People
- 🎯 Multi-object tracking using DeepSORT
- 📊 Real-time vehicle count
- 🔢 Unique vehicle counting to avoid repeated counting
- 📈 Traffic density calculation
- 📉 Vehicle detection trend visualization
- 🌐 Flask-based web dashboard
- 🎨 Color-coded vehicle bounding boxes
- 📄 Detection history stored in CSV format
- 🔁 Automatic video looping
- ⚡ Optimized CPU-based video processing

---

## 🛠️ Technologies Used

- **Python 3.12**
- **YOLOv8**
- **OpenCV**
- **DeepSORT**
- **Flask**
- **NumPy**
- **PyTorch**
- **HTML / CSS / JavaScript**

---

## 📁 Project Structure

```text
Smart-Traffic-Management-System/
│
├── templates/
│   └── index.html
│
├── video/
│   └── traffic.mp4
│
├── .gitignore
├── README.md
├── requirements.txt
├── traffic_detection.py
├── yolov8n.pt
└── report.csv