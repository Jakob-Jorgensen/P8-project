import socket
import numpy as np
import time

# Function to create a homogeneous transformation matrix
def create_transformation_matrix(frame: str):
    # Create a 3x3 rotation matrix (e.g., identity matrix for simplicity)
    R = np.eye(3)
    # Create a 3x1 translation vector (e.g., [1.0, 2.0, 3.0])
    t = np.array([100.0, 0.0, 0.0])
    # Combine R and t into a 4x4 homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    # Return as a tuple (matrix, frame)
    return transformation_matrix, frame

# TCP Server Setup
def start_tcp_server(host='localhost', port=20001):
    # Create the TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)  # Start listening for connections
    
    print(f"Server started, listening on {host}:{port}...")
    
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")
    
    time.sleep(2)

    # Homogeneous Transformation Matrix (Just as a sample, no longer needed for this specific task)
    transformation_matrix, frame = create_transformation_matrix("world")
    print(f"Sending transformation matrix:\n{transformation_matrix}, Frame: {frame}")
    
    # Convert the 4x4 matrix to a flat list and pack it as binary data
    matrix_data = transformation_matrix.flatten().astype(np.float32).tobytes()
    
    # Convert the string frame to bytes (null-terminated)
    frame_data = frame.encode('utf-8') + b'\0'

    # Send the matrix data followed by the string frame data
    client_socket.sendall(matrix_data + frame_data)
    print("Sent transformation matrix and frame to client.")
    
    # Start receiving string messages (e.g., motion planning result)
    try:
        while True:
            # Receive the message from the client
            data = client_socket.recv(1024)  # Buffer size of 1024 bytes
            
            if data:
                # Decode the message (null-terminated string)
                message = data.decode('utf-8').strip('\0')
                print(f"Received message from client: {message}")
            else:
                # No more data received (client disconnected)
                print("Client has disconnected.")
                break

    except Exception as e:
        print(f"Error receiving data: {e}")

    except KeyboardInterrupt:
        print("\nServer interrupted by user (Ctrl+C).")

    finally:
        # Gracefully close the client connection and server socket
        print("Closing client connection.")
        client_socket.close()
        print("Closing server socket.")
        server_socket.close()

if __name__ == "__main__":
    start_tcp_server()
