# This is the the node which contains the grounding dino model
# It subscribes to the topics from the LLM node and the realsense image 
# and then processes the information and image into a black or white 
# mask which is then published to the /ggcnn_mask topic. 
# This node does NOT provide an image to ggcnn with the objects visible
# it only provides a mask which can be used to generate such image 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from std_msgs.msg import String
from rob8_interfaces.msg import LlmCommands
from cv_bridge import CvBridge 
import cv2

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
        self.outputs =  self.create_publisher(String,'/aaulab_output',10)
        self.get_logger().info(">>  VLM Node Ready  <<.")

    def save_image(self, msg: Image):
        # Convert to OpenCV BGR8 (or whatever your color topic uses)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')  


    def contextual_filter(self, msg: LlmCommands, box_tensor, score_tensor, image_width, image_height): 
        
        def get_area(box):
            x1, y1, x2, y2 = box
            return (x2 - x1) * (y2 - y1)

        def get_center_x(box):
            x1, _, x2, _ = box
            return (x1 + x2) / 2

        def get_center_y(box):
            _, y1, _, y2 = box
            return (y1 + y2) / 2

        boxes_list = box_tensor.tolist()
        scores_list = score_tensor.tolist()

        size_idx = None
        side_idx = None
        distance_idx = None


        #areas = [(i, get_area(box)) for i, box in enumerate(boxes_list)]     
        #print(f"box before: {boxes_list}") 
        #print(f"box area: {areas}")
        #boxes_list = [box for i, box in enumerate(boxes_list) if get_area(box) < 10000]
        #print(f"box after: {boxes_list}") 

        # --- Early fallback if all filters are unspecified ---
        if (
            "unspecified" in msg.size and 
            "unspecified" in msg.side and 
            "unspecified" in msg.distance
        ):
            print("here")
            if len(score_tensor) == 0:
                return None
            
            best_idx = score_tensor.argmax().item()  # Get index of highest score
            return box_tensor[best_idx]

        

        # --- SIZE FILTER ---
        if msg.size != "unspecified":
            areas = [(i, get_area(box)) for i, box in enumerate(boxes_list)] 
            print(areas)
            if  "big" in msg.size  :
                size_idx, _ = max(areas, key=lambda x: x[1])
            elif "small" in msg.size :
                size_idx, _ = min(areas, key=lambda x: x[1])
            elif "medium" in msg.size :
                sorted_areas = sorted(areas, key=lambda x: x[1])
                mid = len(sorted_areas) // 2
                size_idx = sorted_areas[mid - 1][0] if len(sorted_areas) % 2 == 0 else sorted_areas[mid][0]

        # --- SIDE FILTER (horizontal) ---
        if msg.side != "unspecified":
            centers_x = [(i, get_center_x(box)) for i, box in enumerate(boxes_list)]
            if "left" in msg.side :
                side_idx, _ = min(centers_x, key=lambda x: x[1])
            elif "right" in  msg.side:
                side_idx, _ = max(centers_x, key=lambda x: x[1])
            elif "middle" in msg.side:
                image_center_x = image_width / 2
                side_idx, _ = min(centers_x, key=lambda x: abs(x[1] - image_center_x))

        # --- DISTANCE FILTER (vertical) ---
        if msg.distance != "unspecified":
            centers_y = [(i, get_center_y(box)) for i, box in enumerate(boxes_list)]
            if "closest" in msg.distance :
                # Closest = lower in image = highest center_y
                distance_idx, _ = max(centers_y, key=lambda x: x[1])
            elif "furthest"  in msg.distance :
                # Furthest = higher in image = lowest center_y
                distance_idx, _ = min(centers_y, key=lambda x: x[1])
            elif "middle" in msg.distance :
                image_center_y = image_height / 2
                distance_idx, _ = min(centers_y, key=lambda x: abs(x[1] - image_center_y))

        # --- COMPARE BY SCORE ---
        candidates = []
        if size_idx is not None:
            candidates.append((size_idx, scores_list[size_idx]))
        if side_idx is not None:
            candidates.append((side_idx, scores_list[side_idx]))
        if distance_idx is not None:
            candidates.append((distance_idx, scores_list[distance_idx]))

        if not candidates:   
            error_msg = String()
            error_msg.data = "Sorry, I could not find the described object, please try again or provide more information"
            self.outputs.publish(error_msg)
            return None  # No valid candidate found
            

        # Return the box with the highest score
        best_idx, _ = max(candidates, key=lambda x: x[1])
        return box_tensor[best_idx]

    



    def vlm_callback(self, msg: LlmCommands):
        if self.cv_image is None:
            self.get_logger().warn("No image yet, skipping detection.")
            return

        # Forward VLM
        inputs = self.processor(
            images=self.cv_image,
            text=msg.description,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Must pass the correct target size: (height, width)
        h, w = self.cv_image.shape[:2] 
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.3, text_threshold=0.3,
            target_sizes=[(h, w)]
        )
        
        # results is a list (one per image); we only have one:
        det = results[0]
        if det["boxes"].numel() == 0:
            self.get_logger().info("No objects found above threshold.")
            return

        # Get the bounding boxes and scores
        boxes = det["boxes"]
        scores = det["scores"]
        print(f"boxes: {len(boxes)}")
        # Apply the contextual filter to the results
        filtered_box = self.contextual_filter(msg, boxes, scores, w, h)
        if filtered_box is None:
            self.get_logger().info("No box passed the filter.")
            return

        # Convert filtered_box to a mask
        x0, y0, x1, y1 = filtered_box.int().tolist()

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
