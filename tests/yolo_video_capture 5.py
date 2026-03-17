import config
import cv2
import threading
from ultralytics import YOLO
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

# -------------------------------
# Threaded Video Capture
# -------------------------------
class VideoCaptureAsync:
    def __init__(self, src, reconnect_interval=1.0):
        self.src = src
        self.reconnect_interval = reconnect_interval
        self.cap = None
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def open_capture(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            return False
        ret, frame = self.cap.read()
        if not ret:
            return False
        with self.lock:
            self.ret, self.frame = ret, frame
        return True

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                if not self.open_capture():
                    time.sleep(self.reconnect_interval)
                    continue
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_interval)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None

    def release(self):
        self.stopped = True
        self.thread.join()
        if self.cap is not None:
            self.cap.release()

# -------------------------------
# Main
# -------------------------------

# Load YOLO
MODEL_PATH = os.path.join("..", "models", config.MODEL)
model = YOLO(MODEL_PATH)

rtsp_url = f"rtsp://{config.JOSHPI_IP}:{config.JOSHPI_PORT}/cam"

# Start threaded capture
cap = VideoCaptureAsync(rtsp_url)

frame_count = 0
last_boxes = []
last_detect_time = 0
timeout = 0.1

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        continue

    frame_height, frame_width = frame.shape[:2]

    # Only run inference every 3 frames
    if frame_count % 3 == 0:
        results = model(
            frame,
            device=config.DEVICE,
            imgsz=config.YOLO_INPUT_SIZE,
            half=True,
            verbose=False
        )[0]

        boxes = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name != "bottle":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            label = f"{model.names[cls]} {conf:.2f}"

        # Update last_boxes only if we detected something
        if boxes:
            last_boxes = boxes
            last_detect_time = time.time()
            
    if time.time() - last_detect_time > timeout:
        last_boxes = []

    # Draw last detected boxes on every frame
    for x1, y1, x2, y2 in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()