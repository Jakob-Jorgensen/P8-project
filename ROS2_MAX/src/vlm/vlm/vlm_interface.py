# ROS2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, String
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge 

# VLM Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import json
import re

model_id= "IDEA-Research/grounding-dino-base"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Function to create the image mask ggcnn needs
def ggcnn_mask_creator():
    return

# Function to handle the JSON message
def json_process():
    return

class VLM_NODE(Node):
    def __init__(self):
        super().__init__('vlm_interface')
        self.cv_image = None

        # Create subscription to image topic
        self.subscription = self.create_subscription(
            Image,
            '/segmented_depth_img', #FIXME change to correct topic
            self.save_image,
            10
        )
        
        self.subscription = self.create_subscription(
            String,
            '/llm_output',
            self.vlm_callback,
            10
        )

        # Publisher to output grasping information
        self.publisher_ = self.create_publisher(Image, '/object_mask', 10)
        self.bridge = CvBridge()

    def save_image(self, msg):
        # Convert ROS Image message to OpenCV image
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 
                                                  desired_encoding='passthrough')

    def vlm_callback(self, msg):
        # If no image has been captured yet, return while doing nothing
        if self.cv_image == None: return

        text_input = json_process(msg) #TODO write json_process function

        # Process the image with the VLM model
        inputs = processor(images=self.cv_image, text=text_input,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[self.cv_image.size[::-1]]
        )

        #TODO write mask creation function
        mask = ggcnn_mask_creator()

        # Publish the results
        self.publisher_.publish(results)


def main(args=None):
    rclpy.init(args=args)
    node = VLM_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()