# Motor Control Settings
DEADZONE = 10
THROTTLE_AXIS = 3
STEERING_AXIS = 2
ESP_IP = "192.168.137.27"
ESP_PORT = 4210
SLEEP_TIME = 0.1

# Video Capture Settings
JOSHPI_DISPLAY_SIZE = (640, 480)
JOSHPI_IP = "192.168.137.114"
JOSHPI_PORT = "8554"

# Autonomous Control settings
YOLO_INPUT_SIZE = 640 #expects multiple of 32
MODEL = "yolov8n.pt"
DEVICE = "cuda"
AUTONOMOUS_SPEED = 0 # 0-255
CLASS_ID = 39 # YOLO Bottle Class
TIMEOUT = 0.1
CONFIDENCE = 0.5