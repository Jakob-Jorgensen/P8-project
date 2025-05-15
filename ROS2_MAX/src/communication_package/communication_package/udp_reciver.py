import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import select

class UDP_reciver_node(Node):
    def __init__(self):
        super().__init__('UDP_reciver_node')

        self.aaulab_output_publisher = self.create_publisher(String, '/aaulab_output', 10)

        self.server_address = ('100.114.98.19', 20001) 
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(self.server_address)
            self.socket.setblocking(False)  # Non-blocking mode
            self.get_logger().info(f"Server started, listening on {self.server_address[0]}:{self.server_address[1]}...")

            # ROS 2 timer to check for new UDP messages
            self.timer = self.create_timer(0.1, self.udp_receive)
                        
        except Exception as e:
            self.get_logger().error(f"Failed to connect UDP: {e}")

    def udp_receive(self):
        try:
            # Use select to check for available data (non-blocking)
            ready = select.select([self.socket], [], [], 0.0)
            if ready[0]:
                response, address = self.socket.recvfrom(4096)
                if response:
                    result_message = response.decode('utf-8').strip()
                    self.get_logger().info(f"Received response from server: {result_message}")

                    msg = String()
                    msg.data = result_message
                    self.aaulab_output_publisher.publish(msg)
                    self.get_logger().info(f"Published movement result: {result_message}")

        except Exception as e:
            self.get_logger().error(f"Error while receiving UDP message: {e}")

    def close_UDP_connection(self):
        if self.socket:
            self.socket.close()
            self.get_logger().info("Closed UDP connection")


def main(args=None):
    rclpy.init(args=args)
    node = UDP_reciver_node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_UDP_connection()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
