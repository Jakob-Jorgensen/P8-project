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

def msg_to_json(msg): #TODO Write this function
    continue

class VLM_NODE(Node)
    def __init__(self):
        super().__init__('vlm_interface')

        # Create subscription to depth image topic
        self.subscription = self.create_subscription(
            Image,
            '/segmented_depth_img', #FIXME change to correct topic
            self.vlm_callback,
            10
        )

        # Publisher to output grasping information
        self.publisher_ = self.create_publisher(Float32MultiArray, '/object_mask', 10)
        self.bridge = CvBridge()

    def vlm_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        text_label = json_process(msg)

        # Process the image with the VLM model
        inputs = processor(images=cv_image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[cv_image.size[::-1]]
        )

        # Publish the results
        self.publisher_.publish(results)


def main(args=None):
    rclpy.init(args=args)
    node = VLM_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()