import cv2
from ultralytics import YOLO
model= YOLO('yolov8n.pt')
# print("Model loaded ✅",model.names)
# cap=cv2.VideoCapture("/Users/jayaarora/Desktop/traffic-sdc/traffic-sdc/data/traffic.mp4")
# ret,frame=cap.read()
# print("Frame read:", ret)                                                                                                                                   
# if ret:
#       print("Frame shape:", frame.shape)
# else:
#       print("Can't read frame")

# cap.release()
# result=model(frame)
# print("\n--- RAW DETECTIONS ---")
# print(result[0].boxes)
ALLOWED_CLASSES = {"car", "bus", "truck", "motorcycle", "person", "bicycle"}
CONF_THRESHOLD = 0.5
def detect(frame):
        results=model(frame,verbose=False)
        # verbose=False → YOLO stays silent
        detections=[]
        for box in results[0].boxes:
            cls_id=int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]

            if cls_name not in ALLOWED_CLASSES:
                continue
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #  bounding box coordinates
            detections.append({                                                                                                                                 
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
            })
        return detections
if __name__ == "__main__":
    cap = cv2.VideoCapture("/Users/jayaarora/Desktop/traffic-sdc/traffic-sdc/data/traffic.mp4")
    ret, frame = cap.read()

    if not cap.isOpened():
        print("❌ Couldn't open video")
        exit()

    frame_count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect(frame)
        frame_count+=1
        print(f"Frame {frame_count}: {len(detections)} objects detected")

    cap.release()
    print(f"\n✅ Done! Processed {frame_count} frames.")