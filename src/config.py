# Motor Control Settings
DEADZONE = 10
THROTTLE_AXIS = 3  # right stick vertical
STEERING_AXIS = 2  # right stick horizontal
ESP_IP = "192.168.137.27"
ESP_PORT = 4210
SLEEP_TIME = 0.1

AUTONOMOUS_SPEED = 0

# Video Capture Settings
JOSHPI_DISPLAY_SIZE = (640, 480)
JOSHPI_IP = "192.168.137.114"
JOSHPI_PORT = "8554"

# YOLO settings
YOLO_INPUT_SIZE = 640 #expects multiple of 32
MODEL = "yolov8n.pt"
DEVICE = "cuda"