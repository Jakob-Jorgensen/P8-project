import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge 
# llm imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub
import json
import re

model_id= "IDEA-Research/grounding-dino-base"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

class VLM_NODE(Node)
    def __init__(self):
        super().__init__('vlm_interface')

        # Create subscription to depth image topic
        self.subscription = self.create_subscription(
            Image,
            '/segmented_depth_img', #TODO change to correct topic
            self.gg_cnn_callback,
            10
        )

        # Publisher to output grasping information
        self.publisher_ = self.create_publisher(Float32MultiArray, '/object_mask', 10)
        self.bridge = CvBridge()

def main(args=None):
    rclpy.init(args=args)
    node = VLM_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()