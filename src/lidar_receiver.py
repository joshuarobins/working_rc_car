import socket
import json
import config

def stream_lidar_data(data_queue):
    """Connects to the Pi and pushes raw incoming lines into a shared queue."""
    print(f"🔗 Network Thread: Connecting to Pi at {config.JOSHPI_IP}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((config.JOSHPI_IP, config.LIDAR_PORT))
        print("✅ Network Thread: Connected!")
        
        buffer = ""
        while True:
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                break
            buffer += data
            
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    # Safely push the line into our shared memory queue
                    data_queue.put(line)
                    
    except Exception as e:
        print(f"Network Thread Error: {e}")
    finally:
        client_socket.close()