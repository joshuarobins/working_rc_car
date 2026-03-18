import config
import cv2
from ultralytics import YOLO
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

cv2.setNumThreads(1)

def plot_result(frame, box=None, label=None):
    if box is not None:
        x1, y1, x2, y2 = box
        if label is not None:
            cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

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
    largest_box = None
    label = None

    if boxes is not None and len(boxes) > 0:

        smoothed_throttle = config.AUTONOMOUS_THROTTLE_SPEED
        smoothed_steering = config.AUTONOMOUS_STEERING_SPEED

        xyxy = boxes.xyxy
        areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        idx = areas.argmax()
        largest = xyxy[idx]
        x1, y1, x2, y2 = map(int, largest)
        box_x_center = (x1 + x2) // 2

        # --- Metadata ---
        largest_box = (x1, y1, x2, y2)

        cls_id = int(boxes.cls[idx])
        conf = float(boxes.conf[idx])
        class_name = model.names[cls_id]

        track_id = None
        if boxes.id is not None:
            track_id = int(boxes.id[idx])

        if track_id is not None:
            label = f"{class_name} ID:{track_id} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

    annotated_frame = plot_result(frame, largest_box, label)
    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()