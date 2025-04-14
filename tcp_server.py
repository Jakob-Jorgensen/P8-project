import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rob8_interfaces.msg import Command
import time
import numpy as np

class TcpServerNode(Node):
    def __init__(self):
        super().__init__('tcp_server_node')

        # Subscriber for /command topic
        self.command_subscriber = self.create_subscription(
            Command, '/command', self.command_callback, 10
        )

        # Publisher for /movement_result topic
        self.movement_result_publisher = self.create_publisher(String, '/movement_result', 10)

        # TCP client setup
        self.socket = None
        self.server_address = ('100.114.98.19', 20001)  # Update with your server IP and port
        
        self.get_logger().info("test")

        self.connect_to_tcp_server()

        a_command = Command()
        a_command.htm = [1.0, 0.0, 0.0, 100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        a_command.frame = "world"

        self.get_logger().info(f"a_command: {a_command}")

        self.command_publisher = self.create_publisher(Command, '/command', 10)

        self.command_publisher.publish(a_command)


    def command_callback(self, msg: Command):
        """Callback function for the /command topic."""
        self.get_logger().info(f"Received command: {msg}")
    
        # Access the fields of the Command message
        self.get_logger().info(f"htm: {msg.htm}")
        self.get_logger().info(f"frame: {msg.frame}")

        homogenous_transformation_matrix = np.array([[ msg.htm[0],  msg.htm[1],  msg.htm[2],   msg.htm[3]  ],
                                                     [ msg.htm[4],  msg.htm[5],  msg.htm[6],   msg.htm[7]  ],
                                                     [ msg.htm[8],  msg.htm[9],  msg.htm[10],  msg.htm[11] ],
                                                     [ msg.htm[12], msg.htm[13], msg.htm[14],  msg.htm[15] ]])
    
        frame = msg.frame

        # Convert the 4x4 matrix to a flat list and pack it as binary data
        matrix_data = homogenous_transformation_matrix.flatten().astype(np.float32).tobytes()
    
        # Convert the string frame to bytes (null-terminated)
        frame_data = frame.encode('utf-8') + b'\0'
    
        # Send the command message to the TCP server
        self.send_tcp_message(matrix_data + frame_data)  # Send the string representation of msg (or adjust as needed)


    def send_tcp_message(self, message: str):
        """Send the message via TCP."""
        if self.socket is None:
            self.connect_to_tcp_server()
        
        try:
            # Send the message
            self.socket.sendall(message)
            self.get_logger().info(f"Sent message to server: {message}")

        except Exception as e:
            self.get_logger().error(f"Error while sending TCP message: {e}")
        except KeyboardInterrupt:
            self.get_logger().info("Node interrupted. Closing socket...")
            self.close_tcp_connection()

        try:
            # Wait for the server's response
            response = self.socket.recv(1024)  # Receive response (e.g., movement result)
            if response:
                # Decode the received data
                result_message = response.decode('utf-8').strip()
                self.get_logger().info(f"Received response from server: {result_message}")

                # Publish the result to /movement_result
                self.publish_movement_result(result_message)
        except Exception as e:
            self.get_logger().error(f"Error while receiving TCP message: {e}")
        except KeyboardInterrupt:
            self.get_logger().info("Node interrupted. Closing socket...")
            self.close_tcp_connection()

    def connect_to_tcp_server(self):
        """Establish TCP connection with the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind(self.server_address)  # Correct binding
            self.socket.listen(1)  # Start listening for connections
            
            self.get_logger().info(f"Server started, listening on {self.server_address[0]}:{self.server_address[1]}...")

            # Accept a client connection
            client_socket, client_address = self.socket.accept()
            self.get_logger().info(f"Connection established with {client_address}")
            
            time.sleep(2)
            
        except Exception as e:
            self.get_logger().error(f"Failed to connect to TCP server: {e}")

    def publish_movement_result(self, result: str):
        """Publish the received movement result to the /movement_result topic."""
        msg = String()
        msg.data = result
        self.movement_result_publisher.publish(msg)
        self.get_logger().info(f"Published movement result: {result}")

    def close_tcp_connection(self):
        """Gracefully close the TCP connection."""
        if self.socket:
            self.socket.close()
            self.get_logger().info("Closed TCP connection")


def main(args=None):
    rclpy.init(args=args)

    node = TcpServerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_tcp_connection()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
