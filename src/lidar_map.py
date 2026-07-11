import pygame
import math

class TestLidar:
    def __init__(self):
        self.angle = 0
        # Define obstacles as (angle, distance)
        self.obstacles = [
            (45, 300), (180, 400), (270, 200), (90, 250), (315, 350)
        ]

    def get_next_reading(self):
        # Rotate the angle by 5 degrees each call
        self.angle = (self.angle + 5) % 360
        
        # Default distance (500) if no obstacle is found
        dist = 500
        # Check if current angle is hitting any obstacle
        for obs_angle, obs_dist in self.obstacles:
            if abs(self.angle - obs_angle) < 5:
                dist = obs_dist
                break
        return self.angle, dist

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
lidar = TestLidar()
persistent_map = {}

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. Update Lidar reading
    angle, dist = lidar.get_next_reading()
    persistent_map[int(angle)] = dist
    
    # 2. Draw
    screen.fill((0, 0, 0))
    center = (300, 300)
    
    # Draw all stored obstacles
    for a, d in persistent_map.items():
        if d < 500:
            rad = math.radians(a)
            # Scaling distance for display
            x = center[0] + int((d / 2) * math.cos(rad))
            y = center[1] + int((d / 2) * math.sin(rad))
            
            # Choose color based on angle
            color = (0, 255, 0) if a < 120 else (0, 0, 255) if a < 240 else (255, 0, 0)
            
            # Draw the line to the obstacle
            pygame.draw.line(screen, color, center, (x, y), 1)
            # Draw the point
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 3)

    # 3. Draw the active scanner arm (the "beam")
    rad = math.radians(lidar.angle)
    arm_end_x = center[0] + int(250 * math.cos(rad))
    arm_end_y = center[1] + int(250 * math.sin(rad))
    pygame.draw.line(screen, (255, 255, 0), center, (arm_end_x, arm_end_y), 2)

    pygame.display.flip()
    clock.tick(60) 

pygame.quit()