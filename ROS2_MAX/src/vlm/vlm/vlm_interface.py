# This is the the node which contains the grounding dino model
# It subscribes to the topics from the LLM node and the realsense image 
# and then processes the information and image into a black or white 
# mask which is then published to the /ggcnn_mask topic. 
# This node does NOT provide an image to ggcnn with the objects visible
# it only provides a mask which can be used to generate such image 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rob8_interfaces.msg import LlmCommands
from cv_bridge import CvBridge

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-base"

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_interface')
        self.get_logger().info("VLM node starting up…")

        # Load model & processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(self.device)

        # State
        self.cv_image = None
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.save_image, 10)
        self.llm_sub = self.create_subscription(
            LlmCommands, '/LLM_output',
            self.vlm_callback, 10)

        # Publisher
        self.mask_pub = self.create_publisher(Image, '/ggcnn_mask', 10) 
        self.get_logger().info(">>  VLM Node Ready  <<.")

    def save_image(self, msg: Image):
        # Convert to OpenCV BGR8 (or whatever your color topic uses)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def vlm_callback(self, msg: LlmCommands):
        if self.cv_image is None:
            self.get_logger().warn("No image yet, skipping detection.")
            return

        # Forward VLM
        inputs = self.processor(
            images=self.cv_image,
            text=msg.object,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Must pass the correct target size: (height, width)
        h, w = self.cv_image.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.4, text_threshold=0.3,
            target_sizes=[(h, w)]
        )
        
        # results is a list (one per image); we only have one:
        det = results[0]
        if det["boxes"].numel() == 0:
            self.get_logger().info("No objects found above threshold.")
            return

        # Pick the highest‑score box (or iterate if you want all)
        scores = det["scores"]
        best_idx = torch.argmax(scores).item()
        x0, y0, x1, y1 = det["boxes"][best_idx].int().tolist()

        # Create a binary mask: 255 inside box, 0 outside
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255

        # Convert back to ROS Image and publish
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(mask_msg)
        self.get_logger().info(f"Published mask for box {(x0,y0,x1,y1)}")

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
