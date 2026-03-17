import config
import pygame
import socket
import time

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Socket created for UDP communication to {config.ESP_IP}:{config.ESP_PORT}")

# Initialize Pygame and joystick
pygame.init()
pygame.joystick.init()
print("Pygame and joystick module initialized.")

try:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick '{joystick.get_name()}' connected and initialized.")
except pygame.error:
    print("No joystick found. Exiting.")
    exit()

# Previous values for change detection
prev_throttle = None
prev_steering = None

while True:
    pygame.event.pump()

    throttle = joystick.get_axis(config.THROTTLE_AXIS)
    steering = joystick.get_axis(config.STEERING_AXIS)

    # Scale and apply deadzone
    throttle_scaled = int(-throttle * 255) if abs(int(-throttle * 255)) > config.DEADZONE else 0
    steering_scaled = int(steering * 255) if abs(int(steering * 255)) > config.DEADZONE else 0

    # Only send if values changed
    if throttle_scaled != prev_throttle or steering_scaled != prev_steering:
        message = f"{throttle_scaled},{steering_scaled}"
        try:
            sock.sendto(message.encode(), (config.ESP_IP, config.ESP_PORT))
            print(f"Sent message: {message}")
        except Exception as e:
            print(f"Error sending message: {e}")

        prev_throttle = throttle_scaled
        prev_steering = steering_scaled

    time.sleep(config.SLEEP_TIME)