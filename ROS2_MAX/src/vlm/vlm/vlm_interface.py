# This is the the node which contains the grounding dino model
# It subscribes to the topics from the LLM node and the realsense image 
# and then processes the information and image into a black or white 
# mask which is then published to the /ggcnn_mask topic. 
# This node does NOT provide an image to ggcnn with the objects visible
# it only provides a mask which can be used to generate such image 
#

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
def ggcnn_mask_creator(bbox: list, image):
    # Image dimensions
    h = image.shape[0]
    w = image.shape[1]

    mask = image.copy()
    for y in range(0, h):
        for x in range(0, w):
            if y < bbox[0] or y > bbox[2]:
                if x < bbox[1] or x > bbox[3]:
                    mask[y, x] = 0
            if y > bbox[0] and y < bbox[2]:
                if x > bbox[1] and x < bbox[3]:
                    mask[y, x] = 255

    return mask

class VLM_NODE(Node):
    def __init__(self):
        super().__init__('vlm_interface')
        self.cv_image = None
        self.LLM_output = ["", "", "", "", ""]


        # Create subscription to image topic
        self.subscription = self.create_subscription(
            Image,
            '/segmented_depth_img', #FIXME change to correct topic
            self.save_image,
            10
        ) 

        # Subscriber for output from LLM
        self.subscription = self.create_subscription(
            String, '/object', self.save_object, 10
        )
        self.subscription = self.create_subscription(
            String, '/description', self.save_desc, 10
        )
        self.subscription = self.create_subscription(
            String, '/side', self.save_side, 10
        )
        self.subscription = self.create_subscription(
            String, '/distance', self.save_distance, 10
        )
        self.subscription = self.create_subscription(
            String, '/size', self.save_size, 10
        )

        # Publisher to output grasping information
        self.publisher_ = self.create_publisher(Image, '/ggcnn_mask', 10)
        self.bridge = CvBridge()

    def save_image(self, msg):
        # Convert ROS Image message to OpenCV image
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 
                                                  desired_encoding='passthrough')
        
    #Save outputs from LLM node
    def save_object(self, msg): self.LLM_output[0] = msg
    def save_desc(self, msg): self.LLM_output[1] = msg
    def save_side(self, msg): self.LLM_output[2] = msg
    def save_distance(self, msg): self.LLM_output[3] = msg
    def save_size(self, msg): self.LLM_output[4] = msg

    def vlm_callback(self, msg):
        # If no image has been captured yet, return while doing nothing
        if self.cv_image == None: return

        # Process the image with the VLM model using description from LLM
        inputs = processor(images=self.cv_image, text=self.LLM_output[1],
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

        # Convert the list into a mask for gg-cnn
        mask = ggcnn_mask_creator(results, self.cv_image)

        # Publish the results
        self.publisher_.publish(mask)


def main(args=None):
    rclpy.init(args=args)
    node = VLM_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()