import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rob8_interfaces.msg import Command
import time
import numpy as np

class UDP_transmitter_node(Node):
    def __init__(self):
        super().__init__('UDP_transmitter_node')

        # Subscriber for /command topic
        self.command_subscriber = self.create_subscription(
            Command, '/grasp_positions', self.command_callback, 10
        )

        # UDP setup
        #self.address = None
        self.server_address = ('100.106.51.5', 20000) 
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            #self.socket.bind(self.server_address)  # Correct binding
            
            self.get_logger().info(f"Server started, transmiting on {self.server_address[0]}:{self.server_address[1]}...")
                        
        except Exception as e:
            self.get_logger().error(f"Failed to connect UDP: {e}")


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
    
        gripper_distance = np.array(msg.gripper_distance)
        frame = msg.frame

        # Convert the 4x4 matrix to a flat list and pack it as binary data
        matrix_data = homogenous_transformation_matrix.flatten().astype(np.float32).tobytes()
    
        gripper_distance_data = gripper_distance.flatten().astype(np.float32).tobytes()

        # Convert the string frame to bytes (null-terminated)
        frame_data = frame.encode('utf-8') + b'\0'
    
        # Send the command message to the UDP server
        self.udp_send(matrix_data + gripper_distance_data + frame_data)  # Send the string representation of msg (or adjust as needed)

    def udp_send(self, message: str):
        """Send the message via UDP."""
        
        try:
            # Send the message
            self.socket.sendto(message, self.server_address)
            self.get_logger().info(f"Sent message to server: {message}")

        except Exception as e:
            self.get_logger().error(f"Error while sending UDP message: {e}")
        except KeyboardInterrupt:
            self.get_logger().info("Node interrupted. Closing socket...")
            self.close_UDP_connection()

    def close_UDP_connection(self):
        """Gracefully close the UDP connection."""
        if self.socket:
            self.socket.close()
            self.get_logger().info("Closed UDP connection")


def main(args=None):
    rclpy.init(args=args)

    node = UDP_transmitter_node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_UDP_connection()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
