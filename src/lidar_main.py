import socket
import json
import config

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
                    print(line) # Prints the raw JSON data string: {"a": 124.5, "d": 450.0}
                    
    except KeyboardInterrupt:
        print("\nStopping receiver.")
    except Exception as e:
        print(f"Network Error: {e}")
    finally:
        client_socket.close()

if __name__ == '__main__':
    receive_lidar()