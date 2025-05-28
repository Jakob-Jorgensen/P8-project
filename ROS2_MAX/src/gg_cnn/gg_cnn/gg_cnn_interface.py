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
from cv_bridge import CvBridge 
# Code framework
import cv2
import numpy as np
import torch
import math 
from skimage.draw import line
from skimage.filters import gaussian
from gg_cnn.ggcnn_structur import GGCNN
from gg_cnn.ggcnn2_structur import GGCNN2 
from rob8_interfaces.msg import Command


# Device configuration  
GRASP_WIDTH_MAX = 200.0  # Maximum grasp width for visualization   
FINGER_1L =  15.0 #mm
FINGER_2L = 5.0 #mm v
#world_offset = FINGER_1L + 5.0 # table offset(17mm) and finger hight

### CHOSE THE MODEL TO USE ###
MODEL_CHOSE = 'ggcnn' # 'GGCNN' or 'GGCNN2'  # Choose the model to use
HOME_PATH = '/home/max/Documents/P8-project/ROS2_MAX/src/gg_cnn/gg_cnn/' 

### background depth image ### 
ground_depth_img = cv2.imread(HOME_PATH + 'grounded-depth.png', cv2.IMREAD_UNCHANGED)

### Camera Calibration ### 
INTRINSIC_MATRIx = np.array([[909.0791015625,0.0,640.423583984375],[0.0,907.5655517578125,364.3811340332031],[0.0,0.0,1.0]])

#np.array([[1.34772092e+03, 0.00000000e+00, 9.62833091e+02],[0.00000000e+00, 1.34663271e+03, 5.45299335e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) 

#np.array([[909.0791015625,0.0,640.423583984375],[0.0,907.5655517578125,364.3811340332031],[0.0,0.0,1.0]])


""" K
- 909.0791015625
- 0.0
- 640.423583984375
- 0.0
- 907.5655517578125
- 364.3811340332031
- 0.0
- 0.0
- 1.0
"""



home_made_extrnsic=np.eye(4)   

home_made_translation = np.eye(4)
home_made_translation[:3, 3]  = np.array([-35,-62,-79]) # x, y, z 
#home_made_extrnsic[:3, 3] = home_made_translation  
print(home_made_translation)
#a = 0

home_made_roation = np.eye(4) 
home_made_roation [:-1, :-1] = np.array([[np.cos(((1/2)*np.pi)),-np.sin(((1/2)*np.pi)),0],[np.sin(((1/2)*np.pi)),np.cos(((1/2)*np.pi)),0],[0,0,1]])  
print(home_made_roation)

home_made_extrnsic =   home_made_roation @ home_made_translation 
#home_made_extrnsic[0,0] = 0
#home_made_extrnsic[1,1] = 0
print(home_made_extrnsic) 
#rot_z = np.eye(4)  
#b = -90
#rot_z[:3, :3] = np.array([[np.cos((np.pi/180)*(b)),-np.sin((np.pi/180)*(b)),0],[np.sin((np.pi/180)*(b)),np.cos((np.pi/180)*(b)),0],[0,0,1]])   

#rot_x = np.eye(4)
#c = 180
#rot_x[:3, :3] = np.array([[1,0,0],[0,np.cos((np.pi/180)*(c)),-np.sin((np.pi/180)*(c))],[0,np.sin((np.pi/180)*(c)),np.cos((np.pi/180)*(c))]]) 




CAM_TO_ROBOT_EXTRNSIC =  home_made_extrnsic 

""" 
np.array([[0.05478242149131124, 0.9728897060947174, 0.2246875743571134, 0.01687956257455467],
 [0.9936994802894625, -0.03108559732977296, -0.10768021411152463, 0.3283410621467196],
 [0.09777642439770845, -0.22917090874189552, 0.9684629396205167, 0.4041309174793619],
 [0, 0, 0, 1]])
"""    
    
""" [[ 0.79110791,  0.22299801,  0.56957893, -0.1363679 ],
  [ 0.13523332,  0.84436975, -0.51841265,  0.14765188],
  [-0.59654021,  0.4871464,   0.63783083,  0.58239576],
  [ 0. ,        0. ,         0. ,         1.        ]])
"""
if MODEL_CHOSE == 'ggcnn': 
    MODEL_PATH =  HOME_PATH+'pretraind-models/pretraind_ggccn.pt' # The GGCNN is trained on the cornell dataset 
    NETWORK = GGCNN()

elif MODEL_CHOSE == 'ggcnn2':  
    MODEL_PATH = HOME_PATH +'pretraind-models/pretraind_ggccn2.pth' # The GGCNN2 is trained on the jacquard dataset 
    NETWORK = GGCNN2()
else: 
    raise ValueError('Please choose a valid model')
#### END OF MODEL CHOICE ####

def input_img(img, out_size=320):
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
        # Load model 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        print('>> loading gg-CNN model')
        self.net = NETWORK  # GGCNN2() or  GGCNN()
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)   # True: exact match, False: load only matching key-value parameters, others load default values.
        self.net = self.net.to(self.device)
        print('>> load done')  
        


    def predict(self, img, input_size=320):
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
            print(f"gg-cnn angle {angle*180/np.pi}")
            print(f"cross angle{angle2*180/np.pi}")
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

 
def inpaint(img, missing_value=0):
    
    #Inpaint missing values in depth image.
    #:param missing_value: Value to fill in the depth image.
    
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

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


def median_depth_filter(image, mask):

    # Apply the mask to the image
    masked_image = image * mask

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask resulted in an empty object.")

    # Get bounding rect and center
    concat_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(concat_contours)
    center_x = x + w // 2
    center_y = y + h // 2

    # Create a list of 8-connected neighbor coordinates (including center)
    kernel_offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),  (0, 0),  (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

    # Collect depth values from the 3x3 kernel
    depth_values = []
    for dy, dx in kernel_offsets:
        ny, nx = center_y + dy, center_x + dx
        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
            val = masked_image[ny, nx]
            if val > 0:
                depth_values.append(val)

    if len(depth_values) == 0:
        raise ValueError("No valid depth points found in 3x3 kernel.")

    return float(np.median(depth_values))


def apply_mask_and_center(image, mask, output_size=(1280, 720)):
    # Use cv2.findContours to get bounding box of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask resulted in an empty object.")
    concat_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(concat_contours)

    # Extract the masked region from the original image
    masked_crop = image[y:y+h, x:x+w] * (mask[y:y+h, x:x+w])

    # Find center of the masked region
    ys, xs = np.where(mask[y:y+h, x:x+w] > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask has no foreground pixels.")

    # Background value estimation
    h_crop, w_crop = masked_crop.shape[:2]
    corners = [
            masked_crop[0, 0],
            masked_crop[0, w_crop - 1],
            masked_crop[h_crop - 1, 0],
            masked_crop[h_crop - 1, w_crop - 1]
        ]
    background_value = np.median(corners)

    # Calculate top-left corner of where to paste the masked object
    target_x = output_size[0] // 2
    target_y = output_size[1] // 2
    offset_x = target_x - int(xs.mean())
    offset_y = target_y - int(ys.mean())

    # Create a blank canvas filled with background value
    canvas = np.full((output_size[1], output_size[0]), background_value)

    # Compute destination coordinates
    paste_x1 = max(0, offset_x)
    paste_y1 = max(0, offset_y)
    paste_x2 = min(output_size[0], paste_x1 + w)
    paste_y2 = min(output_size[1], paste_y1 + h)

    # Compute source coordinates
    crop_x1 = 0
    crop_y1 = 0
    crop_x2 = paste_x2 - paste_x1
    crop_y2 = paste_y2 - paste_y1

    # Prepare cropped mask and masked part
    cropped_mask = mask[y:y+h, x:x+w][crop_y1:crop_y2, crop_x1:crop_x2]
    masked_part = masked_crop[crop_y1:crop_y2, crop_x1:crop_x2]

    # Paste the masked object onto the canvas
    region = canvas[paste_y1:paste_y2, paste_x1:paste_x2]
    canvas[paste_y1:paste_y2, paste_x1:paste_x2] = np.where(
        cropped_mask > 0,
        masked_part,
        region
    )

    # Return canvas and transformation info
    transformation = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "crop_origin_x": x,
        "crop_origin_y": y
    }

    return canvas, transformation

def map_canvas_to_original(x_canvas, y_canvas, transformation):
    x_orig = x_canvas - transformation["offset_x"] + transformation["crop_origin_x"]
    y_orig = y_canvas - transformation["offset_y"] + transformation["crop_origin_y"]
    return x_orig, y_orig

class GGCNNNode(Node):
    def __init__(self):
        super().__init__('gg_cnn_image_processing') 
        self.get_logger().info("GGCNN node starting up...")

        # Load GG-CNN model
        self.model_loader = GGCNNNet(MODEL_PATH)

        # Store the latest depth image
        self.latest_depth_image = None
        self.bridge = CvBridge()  
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10) 
        
        self.RGB_sub = self.create_subscription(
            Image, 'camera/camera/color/image_raw',
            self.RGB_callback, 10)


        # Subscriptions
        self.mask_sub = self.create_subscription(
            Image, '/ggcnn_mask',
            self.gg_cnn_callback, 10) 
        
        self.count = 0
        
        # Publisher
        self.publisher_ = self.create_publisher(Command, '/grasp_positions', 10) 
        self.get_logger().info('>>  gg_CNN System ready  <<') 

    def depth_callback(self, msg:Image):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}") 

    def RGB_callback(self, msg:Image):
        try:
            self.latest_RGB_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   
            #self.latest_RGB_image = cv2.cvtColor(self.latest_RGB_image,cv2.COLOR_BGR2RGB)

        except Exception as e:
            self.get_logger().error(f"Error converting RGB image: {e}")


    def gg_cnn_callback(self, msg:Image):
        if self.latest_depth_image is None or self.latest_RGB_image is None:
            self.get_logger().warn("No depth or RGB image available yet.")
            return

        #try:
        # Convert the incoming mask image
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        mask = np.where(mask==0,0,mask).astype(np.uint8)
        # Ensure mask and depth image sizes match
        if mask.shape != self.latest_depth_image.shape:
            self.get_logger().error(f"Mask and depth image shape mismatch: {mask.shape} vs {self.latest_depth_image.shape}")
            return
        
        inpainted_depth=inpaint(self.latest_depth_image)    
        
        masked_inpainted,transform = apply_mask_and_center(inpainted_depth, mask)
        
        row, col, object_angle, grasp_width_pixels = self.model_loader.predict(masked_inpainted)#,input_size=320)  

        #print(f"object_angle: {object_angle}")

        x_orig ,y_orig = map_canvas_to_original(col,row,transform)
    
       
        
        #inpainted_depth3d = depth2Gray3(self.latest_depth_image)
        grasp_img = drawGrasps(self.latest_RGB_image,[[y_orig,x_orig,object_angle,grasp_width_pixels]],mode='line')
        
        
        cv2.imshow("inpainted_grasp_img", grasp_img)    
        cv2.imwrite(f"/home/max/Documents/tests/{self.count}_rgb.png",grasp_img)
        #cv2.imshow("test",grasp_test)
        self.count += 1 
        cv2.waitKey(1)


        z_cam = self.latest_depth_image[y_orig,x_orig] # The depth is in mm 
        if z_cam == 0:
            self.get_logger().warn("Depth value is zero at predicted point.")
            return
     
        
        # Convert pixel to camera 3D coordinates
        x_coord = (x_orig - INTRINSIC_MATRIx[0, 2]) * z_cam / INTRINSIC_MATRIx[0, 0]
        y_coord = (y_orig - INTRINSIC_MATRIx[1, 2]) * z_cam / INTRINSIC_MATRIx[1, 1]  
        cam_coord = np.array([x_coord, y_coord,z_cam])  
        
        """ 
    
        if object_angle <  np.pi:
            grasp_angle = (object_angle + np.pi - int((object_angle + np.pi) // (2 * np.pi)) * 2 * np.pi ) #+ np.pi
            print("angle under pi")
        else:
            grasp_angle = (object_angle + np.pi - int((object_angle + np.pi) // (2 * np.pi)) * 2 * np.pi) - np.pi # + 1.57079 
            print("angle over pi")

        print(f"graph angle_our {grasp_angle}")

        R = np.array([[ np.cos(grasp_angle), -np.sin(grasp_angle), 0.0 ],
                      [ np.sin(grasp_angle), np.cos(grasp_angle),  0.0 ],
                      [ 0.0,                 0.0,                  1.0 ]])
        
        print(f"R: \n{R}")
        """
        #T_cam = np.eye(4) 
        T_coord = np.eye(4) 
        #T_rot = np.eye(4)
        T_coord[:3, 3] = cam_coord  
        #T_rot[:3,:3] = R 

        #T_cam = T_rot @ 

        # Transform to robot frame 
        #print(f" cam {T} \n")  
        #print(f"inverse cam{np.linalg.inv(T)} \n")
        transform_camera_to_robot = home_made_extrnsic @ T_coord       
        transform_camera_to_robot[3, 0] = object_angle

        print(transform_camera_to_robot) 
      
        # grasp_width_pixels greater than 80mm is not possible
        grasp_width_mm = (grasp_width_pixels * z_cam / INTRINSIC_MATRIx[0, 0]) 
        if grasp_width_mm > 80.0 : 
            grasp_width_mm = 79.0

     
        # Publish
        grasp_msg = Command()  
        
        #print(transform_camera_to_robot_copy) 
        grasp_msg.htm = transform_camera_to_robot.flatten().tolist() 
        grasp_msg.gripper_distance = [grasp_width_mm] 
        grasp_msg.frame = "world"

        self.publisher_.publish(grasp_msg)
        self.get_logger().info(f"Published grasp command.") 
        

        #except Exception as e:
        #    self.get_logger().error(f"Error in gg_cnn_callback: {e}")

  
def main(args=None):
    rclpy.init(args=args) 
    node = GGCNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
