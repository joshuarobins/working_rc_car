import config
import cv2
from utils import yolo_setup, get_box, annotate_frame


model, rtsp_url, frame_height, frame_width = yolo_setup()

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
    box, cls_id, conf, track_id = get_box(boxes)
    annotated_frame = annotate_frame(frame, box, cls_id, conf, track_id, model)
    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()