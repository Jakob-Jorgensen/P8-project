import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import pyttsx3


class UX(Node):
    def __init__(self):
        super().__init__('ux')

        self.get_logger().info('UX node has started.')

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()

        self.timer = self.create_subscription(String, "/aaulab_output", self.aaulab_output_callback, 10)


    def aaulab_output_callback(self, msg):
        self.get_logger().info(f"Received message: {msg.data}")

        data = msg.data

        self.speak_text(data)

    def speak_text(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


def main(args=None):

    rclpy.init(args=args)
    
    node = UX()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
