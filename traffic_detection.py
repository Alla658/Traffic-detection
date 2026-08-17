import csv
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"
VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"
CSV_PATH = BASE_DIR / "report.csv"

# Set to True to wipe report.csv on every app startup (clean demo runs).
# Set to False to keep accumulating history across restarts.
RESET_CSV_ON_START = True


# ============================================================
# YOLO + DeepSORT
# ============================================================

model = YOLO(str(MODEL_PATH))

tracker = DeepSort(
    max_age=30
)


# ============================================================
# Vehicle Configuration
# ============================================================
# NOTE: YOLOv8's default COCO model uses the class name
# "motorcycle" (NOT "motorbike"). Using "motorbike" here
# silently filtered out every motorcycle detection, which is
# why that category was always stuck at 0.

VEHICLE_TYPES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "person"
}

COLOR_MAP = {
    "car": (0, 0, 255),
    "truck": (0, 255, 0),
    "bus": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "person": (0, 0, 0)
}


# ============================================================
# Persistent Vehicle Counts
# ============================================================

persistent_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0,
    "person": 0
}


# ============================================================
# Track IDs
# ============================================================

# Keeps a vehicle from being counted repeatedly.
counted_track_ids = set()


# ============================================================
# CSV Lock
# ============================================================

csv_lock = threading.Lock()


# ============================================================
# CSV Setup
# ============================================================

def initialize_csv():

    needs_fresh_write = (
        RESET_CSV_ON_START
        or not CSV_PATH.exists()
        or CSV_PATH.stat().st_size == 0
    )

    if needs_fresh_write:

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "vehicle_type",
                "confidence",
                "density",
                "track_id"
            ])

        if RESET_CSV_ON_START:

            print(
                f"report.csv reset for a fresh run:\n{CSV_PATH}"
            )


initialize_csv()


# ============================================================
# Save Detection to CSV
# ============================================================

def record_detection(
    vehicle_type,
    confidence,
    density,
    track_id
):

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with csv_lock:

        with open(
            CSV_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                vehicle_type,
                round(float(confidence), 3),
                round(float(density), 3),
                track_id
            ])


# ============================================================
# Calculate Vehicle Density
# ============================================================

def calculate_density(area_km2=1):

    active_vehicle_count = sum(
        1
        for track in tracker.tracker.tracks
        if (
            track.is_confirmed()
            and track.time_since_update <= 1
        )
    )

    if area_km2 <= 0:
        return 0

    return active_vehicle_count / area_km2


# ============================================================
# Video Streaming
# ============================================================

def video_stream():

    global persistent_counts

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print(
            f"ERROR: Unable to open video:\n{VIDEO_PATH}"
        )

        return


    print(
        f"Video opened:\n{VIDEO_PATH}"
    )


    try:

        while True:

            ret, frame = cap.read()

            if not ret or frame is None:

                print(
                    "End of video stream."
                )

                break


            # ==================================================
            # YOLO DETECTION
            # ==================================================

            results = model(
                frame,
                conf=0.3,
                verbose=False
            )


            detections = []


            for detection in results[0].boxes:

                class_id = int(
                    detection.cls.item()
                )

                confidence = float(
                    detection.conf.item()
                )

                class_name = model.names[
                    class_id
                ]

                # --------------------------------------------
                # TEMP DEBUG: uncomment to see every raw class
                # YOLO detects in this video, before filtering.
                # Remove this line once you've verified your
                # video actually contains all vehicle types.
                # --------------------------------------------
                # print("Detected raw class:", class_name)


                if class_name not in VEHICLE_TYPES:
                    continue

                if confidence < 0.3:
                    continue


                x1, y1, x2, y2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )


                bbox_xywh = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1
                ]


                detections.append(
                    (
                        bbox_xywh,
                        confidence,
                        class_name
                    )
                )


            # ==================================================
            # DEEP SORT TRACKING
            # ==================================================

            if detections:

                tracks = tracker.update_tracks(
                    detections,
                    frame=frame
                )

            else:

                tracks = tracker.update_tracks(
                    [],
                    frame=frame
                )


            # ==================================================
            # PROCESS TRACKS
            # ==================================================

            for track in tracks:

                if not track.is_confirmed():
                    continue

                if track.time_since_update > 1:
                    continue


                track_id = track.track_id

                class_name = track.det_class


                if class_name not in VEHICLE_TYPES:
                    continue


                # ------------------------------------------------
                # Count each track only once
                # ------------------------------------------------

                if track_id not in counted_track_ids:

                    persistent_counts[
                        class_name
                    ] += 1


                    counted_track_ids.add(
                        track_id
                    )


                    density = calculate_density()


                    record_detection(
                        vehicle_type=class_name,
                        confidence=0.0,
                        density=density,
                        track_id=track_id
                    )


            # ==================================================
            # DRAW YOLO DETECTIONS
            # ==================================================

            for detection in results[0].boxes:

                class_id = int(
                    detection.cls.item()
                )

                confidence = float(
                    detection.conf.item()
                )

                class_name = model.names[
                    class_id
                ]


                if class_name not in VEHICLE_TYPES:
                    continue

                if confidence < 0.3:
                    continue


                x1, y1, x2, y2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )


                color = COLOR_MAP.get(
                    class_name,
                    (255, 255, 255)
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )


                cv2.putText(
                    frame,
                    f"{class_name} ({confidence:.2f})",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )


            # ==================================================
            # ENCODE FRAME
            # ==================================================

            success, jpeg = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue


            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n\r\n"
            )


    except Exception as e:

        print(
            f"Video stream error: {e}"
        )


    finally:

        cap.release()

        print(
            "Video capture released."
        )


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def index():

    density = calculate_density()

    total_vehicles = sum(
        persistent_counts.values()
    )

    return render_template(
        "index.html",
        vehicle_count=persistent_counts,
        density=density,
        total_vehicles=total_vehicles
    )


# ============================================================
# Video Feed
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        video_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ============================================================
# Current Count API
# ============================================================

@app.route("/current_count")
def current_count():

    density = calculate_density()

    total_vehicles = sum(
        persistent_counts.values()
    )

    return jsonify({
        "vehicle_count": persistent_counts,
        "density": density,
        "total_vehicles": total_vehicles
    })


# ============================================================
# Download CSV
# ============================================================

@app.route("/download_csv")
def download_csv():

    vehicle_type = request.args.get(
        "vehicle_type",
        ""
    ).strip().lower()


    # ========================================================
    # No vehicle filter → Download complete CSV
    # ========================================================

    if not vehicle_type:

        return send_file(
            CSV_PATH,
            as_attachment=True,
            download_name="traffic_report.csv",
            mimetype="text/csv"
        )


    # ========================================================
    # Filter CSV by vehicle type
    # ========================================================

    filtered_path = (
        BASE_DIR /
        "filtered_traffic_report.csv"
    )


    with csv_lock:

        with open(
            CSV_PATH,
            "r",
            newline="",
            encoding="utf-8"
        ) as source:

            reader = csv.DictReader(source)


            with open(
                filtered_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as destination:

                fieldnames = [
                    "timestamp",
                    "vehicle_type",
                    "confidence",
                    "density",
                    "track_id"
                ]


                writer = csv.DictWriter(
                    destination,
                    fieldnames=fieldnames
                )

                writer.writeheader()


                for row in reader:

                    row_vehicle_type = (
                        row["vehicle_type"]
                        .strip()
                        .lower()
                    )


                    if row_vehicle_type != vehicle_type:

                        continue


                    writer.writerow(row)


    # ========================================================
    # Download filtered CSV
    # ========================================================

    return send_file(
        filtered_path,
        as_attachment=True,
        download_name="traffic_report.csv",
        mimetype="text/csv"
    )

# ============================================================
# Start Flask Application
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        TRAFFIC AI - SMART TRAFFIC MONITORING")
    print("=" * 60)
    print()
    print(
        "Dashboard: http://127.0.0.1:5000"
    )
    print(
        f"Video: {VIDEO_PATH}"
    )
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"CSV: {CSV_PATH}"
    )
    print()
    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )