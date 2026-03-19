import config
import cv2
from ultralytics import YOLO
import os
import socket
import time

def yolo_setup():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"
    cv2.setNumThreads(1)
    MODEL_PATH = os.path.join("..", "models", config.MODEL)
    model = YOLO(MODEL_PATH)
    rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"
    frame_height, frame_width = config.JOSHPI_DISPLAY_SIZE

    return model, rtsp_url, frame_height, frame_width

def autonomous_setup():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_detect_time = time.time()
    smoothed_throttle = 0
    smoothed_steering = 0
    track_id = None

    return sock, last_detect_time, smoothed_throttle, smoothed_steering, track_id

def get_box(boxes, track_id=None):
    if boxes is None or len(boxes) == 0:
        return None, None, None, track_id

    xyxy = boxes.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    if track_id is not None and boxes.id is not None:
        for i, tid in enumerate(boxes.id):
            if int(tid) == track_id:
                x1, y1, x2, y2 = map(int, xyxy[i])
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                return (x1, y1, x2, y2), cls_id, conf, track_id

    idx = areas.argmax()
    x1, y1, x2, y2 = map(int, xyxy[idx])
    cls_id = int(boxes.cls[idx])
    conf = float(boxes.conf[idx])
    new_track_id = int(boxes.id[idx]) if boxes.id is not None else None

    return (x1, y1, x2, y2), cls_id, conf, new_track_id

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

def send_motor_command(throttle, steering, sleep=False, do_print=True, sock=None):
    """Send throttle and steering values to ESP32 via UDP"""
    message = f"{throttle},{steering}"
    try:
        sock.sendto(message.encode(), (config.ESP_IP, config.ESP_PORT))
        if do_print:
            print((throttle, steering))
        if sleep:
            time.sleep(config.SLEEP_TIME)
    except Exception as e:
        print(f"Error sending UDP message: {e}")

def autonomous_logic(sock, box, last_detect_time, smoothed_throttle, smoothed_steering):
    frame_width = config.JOSHPI_DISPLAY_SIZE[0]
    if box is not None:
        last_detect_time = time.time()
        box_x_center = (box[0] + box[2]) // 2
        steering = (box_x_center - frame_width//2) / (frame_width//2)
        steering = max(-1, min(1, steering))

        smoothed_steering = steering * config.AUTONOMOUS_STEERING_SPEED
        if abs(smoothed_steering) > 40:
            smoothed_steering*=2
            smoothed_throttle = 0 #TODO: make this a config somehow
        
        else:
            smoothed_throttle = config.AUTONOMOUS_THROTTLE_SPEED


        send_motor_command(int(smoothed_throttle), int(smoothed_steering), sock=sock)

    elif time.time() - last_detect_time > config.TIMEOUT:
        smoothed_throttle *= config.DECAY
        smoothed_steering *= config.DECAY

        if abs(smoothed_throttle) < 5:
            smoothed_throttle = 0
        if abs(smoothed_steering) < 5:
            smoothed_steering = 0

        send_motor_command(int(smoothed_throttle), int(smoothed_steering),
                        sleep=True, do_print=False, sock=sock)

    return last_detect_time, smoothed_throttle, smoothed_steering