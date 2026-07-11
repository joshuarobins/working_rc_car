import socket
import json
import config
import math

def receive_lidar():
    print(f"🔗 Connecting to Raspberry Pi at {config.JOSHPI_IP}:{config.LIDAR_PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((config.JOSHPI_IP, config.LIDAR_PORT))
        print("✅ Connected! Receiving Lidar data stream...\n")
        
        buffer = ""
        while True:
            # Read chunks of data from the network
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                print("❌ Connection closed by the Pi.")
                break
                
            buffer += data
            # Process complete lines separated by our '\n' character
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    try:
                        data = json.loads(line)

                        angle_deg = data["angle"]
                        distance = data["distance_mm"]
                        angle_rad = math.radians(angle_deg)

                        x = distance * math.cos(angle_rad)
                        y = distance * math.sin(angle_rad)

                        print(f"Plotting Point -> X: {x:.2f} mm, Y: {y:.2f} mm")
                            
                    except json.JSONDecodeError:
                        print("Skipping malformed packet...")
                    
    except KeyboardInterrupt:
        print("\nStopping receiver.")
    except Exception as e:
        print(f"Network Error: {e}")
    finally:
        client_socket.close()

if __name__ == '__main__':
    receive_lidar()