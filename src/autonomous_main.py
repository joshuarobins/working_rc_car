import config
import cv2
from ultralytics import YOLO
from utils import yolo_setup, autonomous_setup, send_motor_command, get_largest_box, autonomous_logic, annotate_frame

model, rtsp_url, frame_height, frame_width = yolo_setup()
sock, last_detect_time, smoothed_throttle, smoothed_steering =  autonomous_setup()

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

    box, cls_id, conf, track_id = get_largest_box(boxes)
    annotated_frame = annotate_frame(frame, box, cls_id, conf, track_id, model)

    last_detect_time, smoothed_throttle, smoothed_steering = autonomous_logic(
    sock, box, last_detect_time, smoothed_throttle, smoothed_steering
    )
        

    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        send_motor_command(0, 0, do_print=False, sock=sock)
        break
        
cv2.destroyAllWindows()