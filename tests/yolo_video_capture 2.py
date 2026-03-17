import config
import cv2
import threading
from ultralytics import YOLO
import os
import threading
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

class VideoCaptureAsync:
    def __init__(self, src, reconnect_interval=2.0):
        self.src = src
        self.reconnect_interval = reconnect_interval  # seconds to wait before retry
        self.cap = None
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def open_capture(self):
        """Open or reopen the VideoCapture."""
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
                success = self.open_capture()
                if not success:
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                # Stream temporarily down, release and try reconnect
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
# Load YOLO model
MODEL_PATH = os.path.join("..", "models", config.MODEL)
model = YOLO(MODEL_PATH)

# RTSP stream
#rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam?rtsp_transport=udp"
rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"

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