import cv2
import os
from ultralytics import YOLO
import csv  

model = YOLO('yolov8n.pt')

ALLOWED_CLASSES = {"car", "bus", "truck", "motorcycle"}
CONF_THRESHOLD = 0.5

LINE_Y_RATIO=0.7
LINE_X_RATIO=0.5

def detect(frame):
    results = model.track(frame, persist=True, verbose=False)
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
    writer= cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    LINE_Y = int(height * LINE_Y_RATIO)
    LINE_X = int(width * LINE_X_RATIO)

    previous_centres={}
    counted_horizontal=set()
    counted_vertical=set()
    counts = {"north": 0, "south": 0, "east": 0, "west": 0}                                    
    crossings = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break   
        detections,annotated = detect(frame)

        cv2.line(annotated, (0, LINE_Y), (width, LINE_Y), (0, 255, 255), 2)
        cv2.line(annotated, (LINE_X, 0), (LINE_X, height), (0, 255, 255), 2)

        for det in detections:
            track_id = det["track_id"]
            if track_id is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx=(x1+x2)//2
            cy=(y1+y2)//2

            prev = previous_centres.get(track_id)
            if prev is not None:                
                prev_cx, prev_cy = prev

                # Horizontal line — catches north/south (vertical traffic)
                if track_id not in counted_horizontal:
                    if prev_cy < LINE_Y <= cy:
                        # Was above the line, now at/below → moved DOWN → south
                        counts["south"] += 1
                        counted_horizontal.add(track_id)
                        crossings.append({      
                            "track_id": track_id,
                            "class": det["class"],
                            "line": "horizontal",
                            "direction": "south",
                            "frame": frame_count,
                              "timestamp_sec": round(frame_count / fps, 2),
                        })
                    elif prev_cy > LINE_Y >= cy:
                        # Was below, now at/above → moved UP → north
                        counts["north"] += 1
                        counted_horizontal.add(track_id)
                        crossings.append({      
                            "track_id": track_id,
                            "class": det["class"],
                            "line": "horizontal",
                            "direction": "north",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })

                  # Vertical line — catches east/west (horizontal traffic)
                if track_id not in counted_vertical:
                    if prev_cx < LINE_X <= cx:
                          # Was left of line, now at/right → moved RIGHT → east
                        counts["east"] += 1
                        counted_vertical.add(track_id)
                        crossings.append({
                            "track_id": track_id,
                            "class": det["class"],
                            "line": "vertical",
                            "direction": "east",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })
                    elif prev_cx > LINE_X >= cx:
                          # Was right, now at/left → moved LEFT → west
                        counts["west"] += 1
                        counted_vertical.add(track_id)
                        crossings.append({
                            "track_id": track_id,
                            "class": det["class"],
                            "line": "vertical", 
                            "direction": "west",
                            "frame": frame_count,
                            "timestamp_sec": round(frame_count / fps, 2),
                        })
            previous_centres[track_id] = (cx, cy)
        writer.write(annotated)
        frame_count += 1
        print(f"Frame {frame_count}: {len(detections)} objects | "
                f"N/S/E/W = {counts['north']}/{counts['south']}/{counts['east']}/{counts['west']}")

    cap.release()   
    writer.release()


    # Write all crossings to CSV
    with open(CSV_PATH, "w", newline="") as f:
        fieldnames = ["track_id", "class", "line", "direction", "frame", "timestamp_sec"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(crossings)

    
    print(f"\n✅ Done! Processed {frame_count} frames.")
    print(f"📹 Annotated video: {OUTPUT_PATH}")
    print(f"📊 Total counts → North: {counts['north']}, South: {counts['south']}, "
    f"East: {counts['east']}, West: {counts['west']}")
    print(f"📊 Crossings CSV: {CSV_PATH}")


