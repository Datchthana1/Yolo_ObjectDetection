import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO("yolov8n.pt")

# เปิดกล้อง (Webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ทำ Object Detection
    results = model(frame)

    # วาดผลลัพธ์ลงบนภาพ
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # ตำแหน่ง Bounding Box
            conf = box.conf[0].item()  # ค่าความมั่นใจ
            label = f"{result.names[int(box.cls[0].item())]} {conf:.2f}"

            # วาดกรอบสี่เหลี่ยม
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # แสดงภาพ
    cv2.imshow("YOLOv8 Detection", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
