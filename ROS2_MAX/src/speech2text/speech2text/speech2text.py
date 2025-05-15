import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import speech_recognition as sr
import pyttsx3


class SpeechToTextPublisher(Node):
    def __init__(self):
        super().__init__('speech_to_text_publisher')

        # ROS2 Publisher
        self.publisher_ = self.create_publisher(String, 'human_command', 10)

        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()

        # Start microphone listening
        self.listen_loop()

    def speak_text(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_loop(self):
        with sr.Microphone() as source:
            self.get_logger().info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=4)
            self.get_logger().info("Listening for speech...")

            while rclpy.ok():
                try:
                    self.get_logger().info("Waiting for speech...")
                    audio = self.recognizer.listen(source)

                    # Google Speech Recognition
                    recognized_text = self.recognizer.recognize_google(audio).lower()
                    self.get_logger().info(f"Recognized: {recognized_text}")

                    if "max" in recognized_text:
                        print("Sucsess ............................................................")
                        # Publish to ROS topic
                        msg = String()
                        msg.data = recognized_text
                        self.publisher_.publish(msg)

                        # Speak it back (optional)
                        self.speak_text(recognized_text)

                except sr.UnknownValueError:
                    self.get_logger().warn("Could not understand the audio.")
                except sr.RequestError as e:
                    self.get_logger().error(f"Speech recognition request failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
