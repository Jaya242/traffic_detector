import cv2
import os
import csv
from ultralytics import YOLO
import tempfile
import re
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (required for server / Gradio)
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

model = YOLO('yolov8n.pt')

ALLOWED_CLASSES = {"car", "bus", "truck", "motorcycle"}
ALLOWED_CLASS_IDS = [2, 3, 5, 7]   # COCO ids for car, motorcycle, bus, truck
CONF_THRESHOLD = 0.5

LINE_Y_RATIO = 0.7
LINE_X_RATIO = 0.5

MANUAL_UNIQUE = 26  # actual unique vehicles in the clip (camera angle: diagonal SE traffic)


def detect(frame):
    results = model.track(frame, persist=True, verbose=False, classes=ALLOWED_CLASS_IDS)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]

        if cls_name not in ALLOWED_CLASSES:
            continue
        if conf < CONF_THRESHOLD:
            continue                  
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        track_id = int(box.id[0]) if box.id is not None else None

        detections.append({
            "track_id": track_id,     
            "class": cls_name,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
        })
    annotated = results[0].plot()
    return detections, annotated      


def make_per_class_chart(crossings):
    """Bar chart: count of crossings per vehicle class."""
    class_counts = Counter(c["class"] for c in crossings)
    if not class_counts:
          # Empty fallback
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No crossings detected", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig  

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(classes, counts, color=["#3498db", "#e67e22", "#2ecc71", "#9b59b6"][:len(classes)])
    ax.set_title("Crossings by Vehicle Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Vehicle Class")
    ax.set_ylabel("Number of Crossings")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return fig

def make_time_series_chart(crossings, fps, total_frames):
    """Time-series: crossings per 5-second window."""
    if not crossings:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No crossings detected", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
  
    bucket_size_sec = 5
    total_duration_sec = total_frames / fps if fps > 0 else 0

    buckets = defaultdict(int)
    for c in crossings:
        bucket = int(c["timestamp_sec"] // bucket_size_sec)
        buckets[bucket] += 1  

    max_bucket = int(total_duration_sec // bucket_size_sec) + 1
    x = list(range(max_bucket))
    y = [buckets.get(b, 0) for b in x]
    x_labels = [f"{b * bucket_size_sec}s" for b in x]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_labels, y, marker="o", linewidth=2, color="#e67e22")
    ax.fill_between(range(len(x_labels)), y, alpha=0.2, color="#e67e22")
    ax.set_title("Crossings Over Time (5-second windows)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time into clip")
    ax.set_ylabel("Crossings in window")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def process_video(input_video_path):
    output_dir = tempfile.mkdtemp(prefix="traffic_")
    output_video_path = os.path.join(output_dir, "traffic_tracked.mp4")
    output_csv_path = os.path.join(output_dir, "crossings.csv")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ Couldn't open video")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)   
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    LINE_Y = int(height * LINE_Y_RATIO)
    LINE_X = int(width * LINE_X_RATIO)

    previous_centres = {}
    counted_horizontal = set()
    counted_vertical = set()
    crossings = []

    frame_count = 0
    while True:
        ret, frame = cap.read()       
        if not ret:
            break

        detections, annotated = detect(frame)

        cv2.line(annotated, (0, LINE_Y), (width, LINE_Y), (0, 255, 255), 2)
        cv2.line(annotated, (LINE_X, 0), (LINE_X, height), (0, 255, 255), 2)

        for det in detections:
            track_id = det["track_id"]
            if track_id is None:      
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            prev = previous_centres.get(track_id)
            if prev is not None:
                prev_cx, prev_cy = prev

                # Horizontal line — catches north/south
                if track_id not in counted_horizontal:
                    if prev_cy < LINE_Y <= cy:
                        counted_horizontal.add(track_id)
                        crossings.append({
                            "track_id": track_id, "class": det["class"],
                            "line": "horizontal", "direction": "south",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })
                    elif prev_cy > LINE_Y >= cy:
                        counted_horizontal.add(track_id)
                        crossings.append({
                            "track_id": track_id, "class": det["class"],
                            "line": "horizontal", "direction": "north",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })

                # Vertical line — catches east/west
                if track_id not in counted_vertical:
                    if prev_cx < LINE_X <= cx:
                        counted_vertical.add(track_id)
                        crossings.append({
                            "track_id": track_id, "class": det["class"],
                            "line": "vertical", "direction": "east",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })
                    elif prev_cx > LINE_X >= cx:
                        counted_vertical.add(track_id)
                        crossings.append({
                            "track_id": track_id, "class": det["class"],
                            "line": "vertical", "direction": "west",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })

            previous_centres[track_id] = (cx, cy)

        current_unique = len(counted_horizontal | counted_vertical)
        counter_text = f"Vehicles: {current_unique}"
        cv2.putText(annotated, counter_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        writer.write(annotated)
        frame_count += 1
        print(f"Frame {frame_count}: {len(detections)} objects | Unique so far: {current_unique}")

    cap.release()
    writer.release()

    with open(output_csv_path, "w", newline="") as f:
        fieldnames = ["track_id", "class", "line", "direction", "frame", "timestamp_sec"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(crossings)

    unique_vehicles = counted_horizontal | counted_vertical
    turning_vehicles = counted_horizontal & counted_vertical

    summary = f"""### 🚦 Detection Results

    - **Frames processed:** {frame_count}
    - 🚗 **Unique vehicles detected:** {len(unique_vehicles)}
    - ↪️ **Turning vehicles** (crossed both lines): {len(turning_vehicles)}
    - 📊 **Total crossing events:** {len(crossings)}

    _Built with YOLOv8 + ByteTrack. See repo for methodology._
    """

    per_class_chart = make_per_class_chart(crossings)
    time_series_chart = make_time_series_chart(crossings, fps, frame_count)

    return output_video_path, output_csv_path, summary, per_class_chart, time_series_chart
  


if __name__ == "__main__":
    # CLI mode — runs on the local test video
    script_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(script_dir)
    VIDEO_PATH = os.path.join(project_dir, "data", "traffic.mp4")

    annotated, csv_out, summary, _, _ = process_video(VIDEO_PATH)

    print(f"\n✅ Done!")              
    print(f"📹 Annotated video: {annotated}")
    print(f"📊 Crossings CSV: {csv_out}")
    print(summary)

    if MANUAL_UNIQUE > 0:
        # Parse unique count from summary for accuracy print
        m = re.search(r"Unique vehicles detected:\*\* (\d+)", summary)
        if m:
            auto = int(m.group(1))
            acc = round(100 * (1 - abs(auto - MANUAL_UNIQUE) / MANUAL_UNIQUE), 1)
            print(f"\n🎯 Unique vehicle accuracy vs manual ({MANUAL_UNIQUE}): {acc}%")


    