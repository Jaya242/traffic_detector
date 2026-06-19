import cv2
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

ALLOWED_CLASSES = {"car", "bus", "truck", "motorcycle", "person", "bicycle"}
CONF_THRESHOLD = 0.5


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
    return detections


if __name__ == "__main__":
      script_dir = os.path.dirname(__file__)     
      project_dir = os.path.dirname(script_dir)
      VIDEO_PATH = os.path.join(project_dir, "data", "traffic.mp4")
  
      cap = cv2.VideoCapture(VIDEO_PATH)
      if not cap.isOpened():
          print("❌ Couldn't open video")
          exit()

      frame_count = 0
      while True:
          ret, frame = cap.read()
          if not ret:
              break   
          detections = detect(frame)
          frame_count += 1
          ids = [d["track_id"] for d in detections]
          print(f"Frame {frame_count}: {len(detections)} objects | IDs: {ids}")

      cap.release()   
      print(f"\n✅ Done! Processed {frame_count} frames.")