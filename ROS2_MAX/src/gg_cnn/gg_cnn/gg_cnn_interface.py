""" Step-by-step guide to using GG-CNN for object grasping with an Intel RealSense camera.
This code leverages GG-CNN, a deep learning model for grasp detection, to capture depth and color images from an Intel RealSense camera. It processes these images through the GG-CNN model and visualizes the predicted grasp positions. 
The code is designed to easily switch between different GG-CNN models (GG-CNN and GG-CNN2). It includes functions for image preprocessing, model output post-processing, and drawing the predicted grasps on the images. 

Workflow:
1. Camera configuration
2. Chose model
3. Image preprocessing
4. Predict grasping positions
5. Post-process GG-CNN model outputs
6.Draw predicted grasps on the images
7. Visualize results
"""   


"""  
On startup it should intilaise our model 
Then our model should sit and wait for an input. 
    Lets bind the input to the function call 

Then it procsses that image, output in a topic and then wait for another image 
"""

#ROS2 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge 
# Code framework
import cv2
import numpy as np
import torch
import math
from skimage.filters import gaussian
from ggcnn.models.ggcnn import GGCNN
from ggcnn.models.ggcnn2 import GGCNN2

# Device configuration  
GRASP_WIDTH_MAX = 200.0  # Maximum grasp width for visualization 

### CHOSE THE MODEL TO USE ###
MODEL_CHOSE = 'ggcnn' # 'GGCNN' or 'GGCNN2'  # Choose the model to use
HOME_PATH = '/home/max/Documents/P8-project/ROS2_MAX/src/gg_cnn/gg_cnn/' 

### Camera Calibration ### 
INTRINSIC_MATRIX = np.array([[1.34772092e+03, 0.00000000e+00, 9.62833091e+02],
 [0.00000000e+00, 1.34663271e+03, 5.45299335e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

ROBOT_TO_CAM_EXTRNSIC = np.array([[ 0.79110791,  0.22299801,  0.56957893, -0.1363679 ],
 [ 0.13523332,  0.84436975, -0.51841265,  0.14765188],
 [-0.59654021,  0.4871464,   0.63783083,  0.58239576],
 [ 0. ,        0. ,         0. ,         1.        ]])


if MODEL_CHOSE == 'ggcnn': 
    MODEL_PATH =  HOME_PATH+'pretraind-models/pretraind_ggccn.pt' # The GGCNN is trained on the cornell dataset 
    NETWORK = GGCNN()

elif MODEL_CHOSE == 'ggcnn2':  
    MODEL_PATH = HOME_PATH +'pretraind-models/pretraind_ggccn2.pth' # The GGCNN2 is trained on the jacquard dataset 
    NETWORK = GGCNN2()
else: 
    raise ValueError('Please choose a valid model')
#### END OF MODEL CHOICE ####

def input_img(img, out_size=300):
    """
    Crop the image, keeping the middle (320, 320) portion
        :param file: rgb file
        :return: tensor ready for network input, coordinates of top-left corner of cropped area
    """

    assert img.shape[0] >= out_size and img.shape[1] >= out_size, 'Input depth image must be greater than or equal to (320, 320)'

    # Crop middle image block
    crop_x1 = int((img.shape[1] - out_size) / 2)
    crop_y1 = int((img.shape[0] - out_size) / 2)
    crop_x2 = crop_x1 + out_size
    crop_y2 = crop_y1 + out_size
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Normalize
    img = np.clip(img - img.mean(), -1., 1.).astype(np.float32)

    # Adjust order to match network input
    tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :])  # numpy to tensor

    return tensor, crop_x1, crop_y1


class GGCNNNet:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        # Load model
        print('>> loading gg-CNN model')
        self.net = NETWORK  # GGCNN2() or  GGCNN()
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)   # True: exact match, False: load only matching key-value parameters, others load default values.
        # self.net = self.net.to(device)
        print('>> load done')

    def predict(self, img, input_size=300):
        """
        Predict grasp model
            :param img: input depth image np.array (h, w)
            :return:
        pred_grasps: list([row, col, angle, width])  width in meters
        """
        # Predict
        input, self.crop_x1, self.crop_y1 = input_img(img, input_size)

        self.pos_out, self.cos_out, self.sin_out, self.wid_out = get_pred(self.net, input.to(self.device))
        pos_pred, ang_pred, wid_pred = post_process_output(self.pos_out, self.cos_out, self.sin_out, self.wid_out)

        # Point with maximum confidence
        loc = np.argmax(pos_pred)
        row = loc // pos_pred.shape[0]
        col = loc % pos_pred.shape[0]
        angle = (ang_pred[row, col] + 2 * math.pi) % math.pi
        width = wid_pred[row, col]    # length in pixels
        row += self.crop_y1
        col += self.crop_x1

        return row, col, angle, width 

def get_pred(net, xc): 
    """  
    Get predictions from the network 
        :param net: network model 
        :param xc: input tensor 
        :return: predicted position, cos, sin, width 
    """
    with torch.no_grad():
        pred_pos, pred_cos, pred_sin, pred_wid = net(xc)
        
        pred_pos = torch.sigmoid(pred_pos)
        pred_cos = torch.sigmoid(pred_cos)
        pred_sin = torch.sigmoid(pred_sin)
        pred_wid = torch.sigmoid(pred_wid)

    return pred_pos, pred_cos, pred_sin, pred_wid

def post_process_output(quality_map, cos_map, sin_map, width_map):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
        :param quality_map: Q output of GG-CNN (as torch Tensors)
        :param cos_map: cos output of GG-CNN
        :param sin_map: sin output of GG-CNN
        :param width_map: Width output of GG-CNN
        :return: Filtered quality map, filtered angle map, filtered width map
    """
    quality_map = quality_map.cpu().numpy().squeeze()
    cos_map = cos_map * 2 - 1
    sin_map = sin_map * 2 - 1
    angle_map = (torch.atan2(sin_map, cos_map) / 2.0).cpu().numpy().squeeze()  

    width_map = width_map.cpu().numpy().squeeze() * GRASP_WIDTH_MAX

    quality_map = gaussian(quality_map, 2.0, preserve_range=True)
    angle_map = gaussian(angle_map, 2.0, preserve_range=True)
    width_map = gaussian(width_map, 1.0, preserve_range=True)

    return quality_map, angle_map, width_map


def drawGrasps(img, grasps, mode='line'):
    """
    Draw grasps
        img:    RGB image
        grasps: list()	elements are [row, col, angle, width]
        mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            # Calculate the opposite angle(orthagonal) for a given angle in 
            angle2 = angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)

        else:
            img[row, col] = [color_b, color_g, color_r]
    return img  

def depth2Gray3(im_depth):
    """
    Convert depth image to 3-channel 8-bit grayscale image
    Args:
        im_depth: Input depth image
    Returns:
        numpy.ndarray: 3-channel 8-bit grayscale image of shape (h, w, 3)
    Raises:
        EOFError: If image rendering fails (max depth equals min depth)
    """
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('Image rendering error...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret

# ROS 2 Node Implementation
class GGCNNNode(Node):
    def __init__(self):
        super().__init__('gg_cnn_image_processing') 
        # Load GGCNN neural network model
        self.model_loader = GGCNNNet(MODEL_PATH)

        # Create subscription to depth image topic
        self.subscription = self.create_subscription(
            Image,
            '/segmented_depth_img',
            self.gg_cnn_callback,
            10
        )

        # Publisher to output grasping information
        self.publisher_ = self.create_publisher(Float32MultiArray, '/grasp_positions', 10)
        self.bridge = CvBridge()

    def gg_cnn_callback(self, msg):  
        # Convert ROS Image message to OpenCV image
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Process the image using the GG-CNN model
        row, col, grasp_angle, grasp_width_pixels = self.model_loader.predict(depth_image)
        
        z_pix = depth_image[col,row]
        x_pix = (col - INTRINSIC_MATRIX[0, 2]) * z / INTRINSIC_MATRIX[0, 0]
        y_pix = (row - INTRINSIC_MATRIX[1, 2]) * z / INTRINSIC_MATRIX[1, 1]
        pix_coord = np.array([x_pix,y_pix,z_pix])
        # In camera frame, assume Z points forward, X right, Y down
        x_axis = np.array([np.cos(grasp_angle_rad), np.sin(grasp_angle_rad), 0])
        z_axis = np.array([0, 0, 1])  # approach direction
        y_axis = np.cross(z_axis, x_axis)
        
        # Re-orthogonalize (in case of rounding)
        x_axis = np.cross(y_axis, z_axis)
        
        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3x3 rotation matrix
        
        T = np.eye(4) 
        T[:3,:3] = R  # Rotation matrix 
        T[:3, 3] = pix_coord # camrea coordiates
        
        transform_came_to_robot_matrix = np.dot(ROBOT_TO_CAM_EXTRNSIC,T)
        
        print(transform_came_to_robot_matrix) 
        

        # Prepare the output message with the grasp position (row, col), angle, and width
        grasp_msg = Float32MultiArray()
        grasp_msg.data = [row, col, grasp_angle, grasp_width_pixels]
        
        # Publish the grasp information
        self.publisher_.publish(grasp_msg)
        self.get_logger().info(f"Published grasp position: {grasp_msg.data}")    
  

def main(args=None):
    rclpy.init(args=args) 
    node = GGCNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
