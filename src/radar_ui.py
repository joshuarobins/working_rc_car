import math
import queue
import threading
import pygame
import json
import sys
from lidar_receiver import stream_lidar_data 

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
SCALE = 0.08

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    scan_points = {}

    # 1. Create a thread-safe Queue in memory
    shared_queue = queue.Queue()

    # 2. Start the network receiver on a background thread
    network_thread = threading.Thread(
        target=stream_lidar_data, 
        args=(shared_queue,), 
        daemon=True # Daemon ensures the thread closes when the UI window closes
    )
    network_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 3. Pull ALL available items out of the queue instantly
        while not shared_queue.empty():
            try:
                line = shared_queue.get_nowait()
                packet = json.loads(line)
                
                # Math conversions
                angle_deg = packet["angle"]
                distance_mm = packet["distance_mm"]
                angle_int = int(angle_deg) % 360
                
                angle_rad = math.radians(angle_deg)
                x_mm = distance_mm * math.cos(angle_rad)
                y_mm = distance_mm * math.sin(angle_rad)
                
                screen_x = CENTER_X + int(x_mm * SCALE)
                screen_y = CENTER_Y - int(y_mm * SCALE)
                
                # Overwrite point
                scan_points[angle_int] = (screen_x, screen_y)
                
            except (queue.Empty, json.JSONDecodeError):
                break

        # 4. Draw Step
        screen.fill((10, 15, 20))
        for point_coord in scan_points.values():
            pygame.draw.circle(screen, (0, 255, 100), point_coord, 2)
        pygame.draw.circle(screen, (255, 50, 50), (CENTER_X, CENTER_Y), 5)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()