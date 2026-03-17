import config
import cv2
import threading
from ultralytics import YOLO
import os

# -------------------------------
# Threaded Video Capture
# -------------------------------
class VideoCaptureAsync:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source {src}")
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# -------------------------------
# Main
# -------------------------------
# Load YOLO model (smallest for speed, can switch to yolov8s.pt if you want more accuracy)
MODEL_PATH = os.path.join("..", "models", config.MODEL)
model = YOLO(MODEL_PATH)  # downloaded automatically

# RTSP stream
rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam?rtsp_transport=tcp"

# Start threaded capture
cap = VideoCaptureAsync(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLO
    results = model(frame, device=config.DEVICE, verbose=False, imgsz=config.YOLO_INPUT_SIZE, half=True)[0]

     # Draw detection boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show full-resolution frame with boxes
    cv2.imshow("YOLO RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()