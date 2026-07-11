import config
import pygame
import socket
import time


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Socket created for UDP communication to {config.ESP_IP}:{config.ESP_PORT}")

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found", end="", flush=True)
    while pygame.joystick.get_count() == 0:
        pygame.event.pump()
        time.sleep(.2)
        print(".", end="", flush=True)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick '{joystick.get_name()}' connected and initialized. ")
print("Press q to quit")

prev_throttle = None
prev_steering = None

while True:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            exit()

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