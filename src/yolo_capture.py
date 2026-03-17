import config
import cv2
from ultralytics import YOLO
import os
import time
import socket

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

# -------------------------------
# Main YOLO + autonomous control
# -------------------------------
MODEL_PATH = os.path.join("..", "models", config.MODEL)
model = YOLO(MODEL_PATH)

rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"

frame_height, frame_width = config.JOSHPI_DISPLAY_SIZE

for result in model.track(
source=rtsp_url,
stream=True,
conf=config.CONFIDENCE,
classes=config.CLASS_IDS,
device=config.DEVICE,
imgsz=config.YOLO_INPUT_SIZE,
verbose=False,
half=True,
persist=True):

    frame = result.orig_img
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        last_detect_time = time.time()

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        largest = xyxy[areas.argmax()]
        x1, y1, x2, y2 = largest.astype(int) 
        box_x_center = (x1 + x2) // 2

    annotated_frame = result.plot() 
    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()