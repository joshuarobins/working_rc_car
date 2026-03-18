import config
import cv2
from ultralytics import YOLO
import os
import time
import socket

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

cv2.setNumThreads(1)

# -------------------------------
# UDP setup for ESP32 control
# -------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_motor_command(throttle, steering, sleep=False, do_print=True):
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

last_detect_time = time.time()

frame_height, frame_width = config.JOSHPI_DISPLAY_SIZE

smoothed_throttle = 0
smoothed_steering = 0

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
        last_detect_time = time.time()

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

        '''
        if box_x_center > frame_width//2:
            steering = config.AUTONOMOUS_STEERING_SPEED
        elif box_x_center < frame_width//2:
            steering = -config.AUTONOMOUS_STEERING_SPEED
        else:
            steering = 0
        
        # Throttle: larger box means closer -> slower
        throttle = 1 - (box_width / frame_width)
        throttle = max(0, min(1, throttle))  # clamp to [0,1]

        # Convert to PWM scale (0-255)
        throttle_pwm = int(throttle * 255)
        '''
        
        steering = (box_x_center - frame_width//2) / (frame_width//2)
        steering = max(-1, min(1, steering))

        raw_steering = steering * config.AUTONOMOUS_STEERING_SPEED
        raw_throttle = config.AUTONOMOUS_THROTTLE_SPEED

        smoothed_steering = raw_steering
        smoothed_throttle = raw_throttle

        send_motor_command(int(smoothed_throttle), int(smoothed_steering))
        
        
    # Timeout: stop if no detection for a while
    elif time.time() - last_detect_time > config.TIMEOUT:
        smoothed_throttle *= config.DECAY
        smoothed_steering *= config.DECAY
        
        # Clamp small values to zero so it fully stops
        if abs(smoothed_throttle) < 5:
            smoothed_throttle = 0
        if abs(smoothed_steering) < 5:
            smoothed_steering = 0

        send_motor_command(int(smoothed_throttle), int(smoothed_steering), sleep=True, do_print=False)

    annotated_frame = plot_result(frame, largest_box, label)
    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        send_motor_command(0, 0, do_print=False)
        break

cv2.destroyAllWindows()