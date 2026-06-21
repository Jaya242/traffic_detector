import cv2
import os
import csv
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

ALLOWED_CLASSES = {"car", "bus", "truck", "motorcycle"}
ALLOWED_CLASS_IDS = [2, 3, 5, 7]   # COCO ids for car, motorcycle, bus, truck
CONF_THRESHOLD = 0.5

LINE_Y_RATIO = 0.7
LINE_X_RATIO = 0.5

MANUAL_UNIQUE = 26  # actual unique vehicles in the clip (camera angle: diagonal SE traffic)

  
def detect(frame):
    results = model.track(frame, persist=True, verbose=False,classes=ALLOWED_CLASS_IDS)
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


if __name__ == "__main__":
      script_dir = os.path.dirname(__file__)
      project_dir = os.path.dirname(script_dir)
      VIDEO_PATH = os.path.join(project_dir, "data", "traffic.mp4")
      OUTPUT_PATH = os.path.join(project_dir, "data", "traffic_tracked.mp4")
      CSV_PATH = os.path.join(project_dir, "data", "crossings.csv")

      cap = cv2.VideoCapture(VIDEO_PATH)
      if not cap.isOpened():
          print("❌ Couldn't open video")
          exit()

      fps = cap.get(cv2.CAP_PROP_FPS)   
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

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
          print(f"Frame {frame_count}: {len(detections)} objects |  Unique so far:{current_unique}")

      cap.release()
      writer.release()

      with open(CSV_PATH, "w", newline="") as f:
          fieldnames = ["track_id", "class", "line", "direction", "frame","timestamp_sec"]
          csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
          csv_writer.writeheader()
          csv_writer.writerows(crossings)

      unique_vehicles = counted_horizontal | counted_vertical
      turning_vehicles = counted_horizontal & counted_vertical
  
      print(f"\n✅ Done! Processed {frame_count} frames.")
      print(f"📹 Annotated video: {OUTPUT_PATH}")
      print(f"📊 Crossings CSV: {CSV_PATH}")
      print(f"\n🚗 Unique vehicles: {len(unique_vehicles)}")
      print(f"↪️   Turning vehicles: {len(turning_vehicles)}")

      if MANUAL_UNIQUE > 0:
              unique_accuracy = round(100 * (1 - abs(len(unique_vehicles) - MANUAL_UNIQUE) /MANUAL_UNIQUE), 1)
              print(f"\n🎯 Unique vehicle accuracy: auto={len(unique_vehicles)} manual={MANUAL_UNIQUE}  → {unique_accuracy}%")