import config
import cv2
from ultralytics import YOLO
import os
import time
import socket

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay"

# -------------------------------
# UDP setup for ESP32 control
# -------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ESP_ADDR = (config.ESP_IP, config.ESP_PORT)

def send_motor_command(throttle, steering, sleep=True):
    """Send throttle and steering values to ESP32 via UDP"""
    message = f"{throttle},{steering}"
    try:
        sock.sendto(message.encode(), ESP_ADDR)
        if sleep:
            time.sleep(config.SLEEP_TIME)
    except Exception as e:
        print(f"Error sending UDP message: {e}")

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

    if boxes is not None and len(boxes) > 0:
        last_detect_time = time.time()

        smoothed_throttle = config.AUTONOMOUS_THROTTLE_SPEED
        smoothed_steering = config.AUTONOMOUS_STEERING_SPEED

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        largest = xyxy[areas.argmax()]
        x1, y1, x2, y2 = largest.astype(int) 
        box_x_center = (x1 + x2) // 2

        if box_x_center > frame_width//2:
            steering = config.AUTONOMOUS_STEERING_SPEED
        elif box_x_center < frame_width//2:
            steering = -config.AUTONOMOUS_STEERING_SPEED
        else:
            steering = 0
        
        '''
        steering = (box_x_center - frame_width//2) / (frame_width//2)
        steering = -max(-1, min(1, steering))
        
        # Throttle: larger box means closer -> slower
        throttle = 1 - (box_width / frame_width)
        throttle = max(0, min(1, throttle))  # clamp to [0,1]

        # Convert to PWM scale (0-255)
        throttle_pwm = int(throttle * 255)
        '''
        throttle_pwm = int(config.AUTONOMOUS_THROTTLE_SPEED)
        steering_pwm = int(steering)
    
        
        send_motor_command(throttle_pwm, steering_pwm)
        print((throttle_pwm, steering_pwm))
        
    # Timeout: stop if no detection for a while
    elif time.time() - last_detect_time > config.TIMEOUT:
        smoothed_throttle *= config.DECAY
        smoothed_steering *= config.DECAY
        
        # Clamp small values to zero so it fully stops
        if abs(smoothed_throttle) < 50:
            smoothed_throttle = 0
        if abs(smoothed_steering) < 50:
            smoothed_steering = 0

        send_motor_command(int(smoothed_throttle), int(smoothed_steering))

    annotated_frame = result.plot() 
    cv2.imshow("YOLO RTSP Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        send_motor_command(0, 0)
        break

cv2.destroyAllWindows()