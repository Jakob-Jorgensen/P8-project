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

if MODEL_CHOSE == 'ggcnn': 
    MODEL_PATH = 'pretraind-models/pretraind_ggccn.pt' # The GGCNN is trained on the cornell dataset 
    NETWORK = GGCNN()

elif MODEL_CHOSE == 'ggcnn2':  
    MODEL_PATH = 'pretrainind-models/pretraind_ggcnn2.pth' # The GGCNN2 is trained on the jacquard dataset 
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
        print('>> loading AFFGA')
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

# Load GGCNN neural network model
model_loader=GGCNNNet(MODEL_PATH)


# Main loop to capture frames from the camera and process them 
while True:
 
    # Convert images to numpy arrays
    depth_image = 
    color_image = 


    #outputs model predictions, input_size can be adjusted
    row, col, grasp_angle, grasp_width_pixels  = model_loader.predict(depth_image)

    # Convert depth image to 3-channel 8-bit grayscale image 
    depth_3d = depth2Gray3(depth_image) 

    # Draw grasps 
    img_grashp=drawGrasps(depth_3d, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')
    img_grashp_RGB = drawGrasps(color_image, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')
    
    cv2.imshow('img_grasp', np.hstack((img_grashp, img_grashp_RGB)) )
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

pipeline.stop()
cv2.destroyAllWindows()