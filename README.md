# 🚦 Smart Traffic Management System

Smart Traffic Management System is an AI-powered **traffic monitoring and analytics application** built with Python, YOLOv8, OpenCV, DeepSORT, and Flask. It detects and tracks vehicles from traffic video, maintains unique vehicle counts, calculates traffic density, records detection data, and displays the results through a web-based dashboard.

## ✨ Features

* 🚗 Vehicle detection using YOLOv8
* 🚌 Vehicle classification:

  * Cars
  * Buses
  * Trucks
  * Motorcycles
  * People
* 🎯 Multi-object tracking using DeepSORT
* 🔢 Unique vehicle counting using tracking IDs
* 📊 Real-time vehicle count tracking
* 📈 Traffic density calculation
* 🎨 Color-coded vehicle bounding boxes
* 🌐 Flask-based web dashboard
* 📄 Detection history stored in CSV format
* 📥 Download complete or filtered traffic reports
* 🔄 Real-time video streaming through the dashboard
* ⚡ CPU-based video processing

## 🛠️ Technologies Used

* **Python 3.12**
* **YOLOv8** – Object detection
* **OpenCV** – Video processing and visualization
* **DeepSORT** – Multi-object tracking
* **Flask** – Web application and dashboard
* **NumPy** – Numerical operations
* **PyTorch** – Deep learning framework used by YOLOv8
* **HTML / CSS / JavaScript** – Dashboard interface

## 🔬 Traffic Processing Pipeline

The system processes traffic video through multiple stages:

```text
Input Traffic Video
        ↓
YOLOv8 Vehicle Detection
        ↓
Vehicle Classification
        ↓
DeepSORT Multi-Object Tracking
        ↓
Unique Track ID Assignment
        ↓
Vehicle Counting
        ↓
Traffic Density Calculation
        ↓
CSV Detection Recording
        ↓
Flask Web Dashboard
```

### 1. Vehicle Detection

YOLOv8 processes each video frame and detects objects with a confidence threshold of `0.3`.

The system focuses on the following classes:

```text
car
bus
truck
motorbike
person
```

### 2. Vehicle Tracking

DeepSORT is used to track detected objects across consecutive frames.

Each tracked object receives a unique **track ID**, which helps prevent the same vehicle from being counted repeatedly.

### 3. Unique Vehicle Counting

The system maintains a set of previously counted track IDs.

When a new track ID appears, the corresponding vehicle count is increased.

```text
New Vehicle
     ↓
New Track ID
     ↓
Check Counted IDs
     ↓
Not Previously Counted
     ↓
Increase Vehicle Count
```

### 4. Traffic Density

Traffic density is calculated using the number of currently active tracked objects.

The system calculates the active vehicle count from confirmed DeepSORT tracks.

### 5. Count Smoothing

A smoothing window is used to reduce sudden fluctuations in displayed vehicle counts.

The current implementation uses a smoothing window of:

```text
10 frames
```

### 6. Detection History

Detection information is recorded in `report.csv`.

The recorded information includes:

```text
timestamp
vehicle_type
confidence
density
track_id
```

### 7. Web Dashboard

Flask provides the web interface for monitoring the traffic system.

The dashboard provides:

* Current vehicle counts
* Traffic density
* Total vehicle count
* Live processed video feed
* CSV report download
* Vehicle-type filtering for reports

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
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Alla658/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

If PowerShell prevents activation, you can directly use the Python executable inside the environment:

```bash
venv\Scripts\python.exe
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask application with:

```bash
python traffic_detection.py
```

The application will start the Flask server at:

```text
http://127.0.0.1:5000
```

Open the URL in your web browser to access the traffic monitoring dashboard.

## 🖥️ How to Use

1. Make sure `yolov8n.pt` is present in the project root.
2. Place the traffic video inside the `video/` directory.
3. Start the application using `traffic_detection.py`.
4. Open `http://127.0.0.1:5000` in your browser.
5. View the live processed traffic video.
6. Monitor vehicle counts and traffic density.
7. Use the dashboard to view traffic analytics.
8. Download the complete traffic report or filter the report by vehicle type.

## 📦 Requirements

The project dependencies are listed in `requirements.txt`:

```text
Flask==3.0.3
numpy==1.26.4
opencv-python==4.10.0.84
deep-sort-realtime==1.3.2
ultralytics==8.3.27
setuptools==80.9.0
```

## 📊 CSV Report

The system automatically maintains a CSV file containing detection information.

Each record contains:

```text
timestamp
vehicle_type
confidence
density
track_id
```

The dashboard also provides an option to download:

* Complete traffic detection report
* Filtered report for a selected vehicle type

## 📌 Future Improvements

Possible future improvements include:

* Real-time CCTV camera integration
* GPU acceleration for faster processing
* Traffic congestion level classification
* Traffic prediction using historical data
* Multiple camera support
* Vehicle speed estimation
* License plate recognition
* Advanced traffic analytics
* Real-time alerts for high traffic density
* Improved dashboard visualizations
* Cloud-based traffic monitoring
* Database integration for long-term traffic records

## 👨‍💻 Project

**Smart Traffic Management System**

GitHub: https://github.com/Alla658/Smart-Traffic-Management-System
