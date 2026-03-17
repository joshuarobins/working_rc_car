import config
import cv2
import threading
from ultralytics import YOLO
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

class VideoCaptureAsync:
    def __init__(self, src, reconnect_interval=1.0):
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


class SharedControlState:
    def __init__(self):
        self.cx = None
        self.area = 0
        self.frame_width = 0
        self.lock = threading.Lock()

    def update(self, cx, area, frame_width):
        with self.lock:
            self.cx = cx
            self.area = area
            self.frame_width = frame_width

    def read(self):
        with self.lock:
            return self.cx, self.area, self.frame_width
        

# -------------------------------
# Main
# -------------------------------

# Load YOLO model
MODEL_PATH = os.path.join("..", "models", config.MODEL)
model = YOLO(MODEL_PATH)

# RTSP stream
rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"

# Start threaded capture
cap = VideoCaptureAsync(rtsp_url)

control_state = SharedControlState()

frame_count = 0
last_best_cx = None
last_best_area = 0

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        continue

    frame_height, frame_width = frame.shape[:2]

    best_area = 0
    best_cx = None

    # Only infer every 3rd frame
    if frame_count % 3 == 0:
        results = model(frame,
                        device=config.DEVICE,
                        verbose=False,
                        imgsz=config.YOLO_INPUT_SIZE,
                        half=True)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name != "bottle":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) // 2

            if area > best_area:
                best_area = area
                best_cx = cx

            # draw box
            label = f"bottle {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update last-known positions
        if best_cx is not None:
            last_best_cx, last_best_area = best_cx, best_area

    # Use last known positions for skipped frames
    control_state.update(last_best_cx, last_best_area, frame_width)

    cv2.imshow("YOLO RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break