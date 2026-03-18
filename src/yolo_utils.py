import config
import cv2
from ultralytics import YOLO
import os

def setup():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

    cv2.setNumThreads(1)

    MODEL_PATH = os.path.join("..", "models", config.MODEL)
    model = YOLO(MODEL_PATH)

    rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"

    frame_height, frame_width = config.JOSHPI_DISPLAY_SIZE

    return model, rtsp_url, frame_height, frame_width

def get_largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None, None, None, None
    xyxy = boxes.xyxy
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = areas.argmax()
    x1, y1, x2, y2 = map(int, xyxy[idx])
    
    cls_id = int(boxes.cls[idx])
    conf = float(boxes.conf[idx])
    track_id = int(boxes.id[idx]) if boxes.id is not None else None
    largest_box = (x1, y1, x2, y2)

    return largest_box, cls_id, conf, track_id

def annotate_frame(frame, box=None, cls_id=None, conf=None, track_id=None, model=None):
    label = None
    if box is not None:
        x1, y1, x2, y2 = box
        if cls_id is not None:
            class_name = model.names[cls_id]
            if track_id is not None:
                label = f"{class_name} ID:{track_id} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

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