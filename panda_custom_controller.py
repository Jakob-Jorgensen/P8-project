import rospy
import moveit_commander
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R  # SciPy for quaternion math
import numpy as np
import socket
import threading

class Panda_Custom_Controller(): 
    def __init__(self):

        rospy.init_node("panda_custom_controller", anonymous=True)

        # # Initialize MoveIt!
        moveit_commander.roscpp_initialize(rospy.myargv(argv=[]))
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.manipulator_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("panda_hand")

        # Initialize tf2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.sleep(1)  # Allow tf to populate

        rospy.logdebug("Hello world!")
        rospy.loginfo("Hello world!")
        rospy.logwarn("Hello world!")
        rospy.logerr("Hello world!")

        self.host = 'localhost'
        self.port = 20002

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        self.sock.connect((self.host, self.port))
        
        rospy.loginfo(f"Connected to {self.host}:{self.port}")

        self.command = np.array([[ 1.0, 0.0, 0.0, 0.0 ],
                                 [ 0.0, 1.0, 0.0, 0.0 ],
                                 [ 0.0, 0.0, 1.0, 0.0 ],
                                 [ 0.0, 0.0, 0.0, 1.0 ]])
        
        self.received_command = np.array([[ 1.0, 0.0, 0.0, 0.0 ],
                                          [ 0.0, 1.0, 0.0, 0.0 ],
                                          [ 0.0, 0.0, 1.0, 0.0 ],
                                          [ 0.0, 0.0, 0.0, 1.0 ]])
        

       # tcp_thread = threading.Thread(target=self.tcp_receive, args=())
       # tcp_thread.start()

        ################
        #### HOMING ####
        ################

        self.base_home_transformation = np.array([[ 1.0, 0.0, 0.0, 400.0 ],
                                                  [ 0.0, 0.0, 1.0, 0.0   ],
                                                  [ 0.0, 1.0, 0.0, 400.0 ],
                                                  [ 0.0, 0.0, 0.0, 1.0   ]])
        
        theta = np.deg2rad(90.0)
        self.rotate_end_effector_around_z = np.array([[ np.cos(theta), -np.sin(theta), 0.0, 0.0 ], 
                                                      [ np.sin(theta), np.cos(theta),  0.0, 0.0 ],
                                                      [ 0.0,           0.0,            1.0, 0.0 ],
                                                      [ 0.0,           0.0,            0.0, 1.0 ]])

        self.home()
        
        #######################
        #### END OF HOMING ####
        #######################

        while True:
            self.tcp_receive()
                
            try:
              #  rospy.loginfo(f"received_command: \n{self.received_command}\ncommand: \n{self.command}")
                if not np.array_equal(self.received_command[0], self.command[0]):
                    self.command = self.received_command
                    pose = self.homogenous_transformation_matrix_to_pose(self.command[0], input_frame_of_reference=self.command[1], move_when_done=True, relative_movement=True)

            except Exception as e:
                rospy.logerr(f"Failed with error: \n{e}")


    def homogenous_transformation_matrix_to_pose(self, homogenous_transformation_matrix: np.ndarray, input_frame_of_reference: str, move_when_done: bool=False, relative_movement: bool=True):
        """
        Takes a homogenous transformation matrix and a frame of reference (see the ROS tf tree).
        - The two most important frames are "world" which is at the same position as the base (which is panda_link0), and "panda_hand" which is the tool.

        Returns geometry_msgs.msg.Pose()
        """

        # If the frame of reference is world (same as panda_link0, ie base, but for reasons I like this more), then get current pose and add the translation values to xyz. 
        if input_frame_of_reference == "world":
            
            current_pose = self.manipulator_group.get_current_pose().pose

            rotation_matrix = homogenous_transformation_matrix[:3, :3]
            translation = homogenous_transformation_matrix[:3, 3]
            
            if relative_movement == True:
                translation[0] += current_pose.position.x * 1000
                translation[1] += current_pose.position.y * 1000
                translation[2] += current_pose.position.z * 1000

            # If rotation is identity, grab current orientation and use for new pose. Else use rotation matrix from input.
            if self.is_identity(rotation_matrix):
                rotation = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]

            else:
                rotation = self.rotation_matrix_to_quaternion(rotation_matrix)

        # If frame of reference is something else, likely to be panda_hand, then we do some math to get the frames transformed correctly.
        else:            
            # Gets a transformation T which converts the homogenous transformation matrix from input frame of reference (inputFoR) to the robot base frame "panda_link0"
            T_inputFoR_to_base = self.get_transform(target_frame="world", source_frame=input_frame_of_reference)

            # converts the transform into a homogenous transformation matrix (htm), so it can be matrix multiplied with the input homogenous transformation matrix
            htm_T_inputFoR_to_base = self.transform_to_homogeneous_matrix(T_inputFoR_to_base)
            rospy.logdebug(f"htm_T_inputFoR_to_base: \n{htm_T_inputFoR_to_base}")

            # Unit conversion of translation vector from m to mm
            htm_T_inputFoR_to_base[0, 3] = htm_T_inputFoR_to_base[0, 3] * 1000
            htm_T_inputFoR_to_base[1, 3] = htm_T_inputFoR_to_base[1, 3] * 1000
            htm_T_inputFoR_to_base[2, 3] = htm_T_inputFoR_to_base[2, 3] * 1000
            rospy.logdebug(f"htm_T_inputFoR_to_base after unit conversion from m to mm: \n{htm_T_inputFoR_to_base}")

            # This is just a matrix multiplication
            htm_transformation_result = htm_T_inputFoR_to_base @ homogenous_transformation_matrix

            rospy.logdebug(f"htm_transformation_result: \n{htm_transformation_result}")

            rotation_matrix = htm_transformation_result[:3, :3]
            translation = htm_transformation_result[:3, 3]
            rotation = self.rotation_matrix_to_quaternion(rotation_matrix)

        # Construct the pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = translation[0] / 1000
        target_pose.position.y = translation[1] / 1000
        target_pose.position.z = translation[2] / 1000
        target_pose.orientation.x = rotation[0]
        target_pose.orientation.y = rotation[1]
        target_pose.orientation.z = rotation[2]
        target_pose.orientation.w = rotation[3]

        if move_when_done:
            rospy.logdebug(f"pose: {target_pose}")
            self.move_to_target(target_pose)
            return target_pose

        else:
            return target_pose


    def move_to_target(self, target_pose):
        """
        Move the robot end effector to the given target pose using MoveIt.
        """

        self.manipulator_group.set_pose_target(target_pose)
        plan = self.manipulator_group.go(wait=True)
        self.manipulator_group.stop()
        self.manipulator_group.clear_pose_targets()
        return plan


    def grip(self, target=800.0, min_gripper=0.0, max_gripper=800.0):
        """
        Opens the gripper to a target mm distance between the claws.
        Default max open (800 mm).
        """

        # maps from a custom range (the mm dist between the claws of the gripper) to the grippers standard range (0.0:0.04), so we can control it in mm.
        target = self.map_range(target, min_gripper, max_gripper, 0.0, 0.04)
        grip_l = target
        grip_r = target

        self.gripper_group.set_joint_value_target([grip_l, grip_r])

        #Execute the movement
        self.gripper_group.go(wait=True)

        #Stop movement
        self.gripper_group.stop()


    def home(self):
        """
        Homes the manipulator to a position set in init, rotates the gripper to face towards world x, fully closes and opens the gripper.
        """

        _ = self.homogenous_transformation_matrix_to_pose(self.base_home_transformation, input_frame_of_reference="panda_link0", move_when_done=True)
        _ = self.homogenous_transformation_matrix_to_pose(self.rotate_end_effector_around_z, input_frame_of_reference="panda_hand", move_when_done=True)

        self.grip(0.0)
        self.grip()



    def get_transform(self, target_frame, source_frame):
        """
        Get the current transform between two frames. Default is from base to tool, ie. panda_link8 relative to panda_link0.
        """
        
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            return transform
        except tf2_ros.LookupException as e:
            rospy.logwarn("Could not get transform: %s", e)
            return None
        

    def transform_to_homogeneous_matrix(self, transform):
        """
        Converts a transform (translation and rotation) to a 4x4 homogeneous transformation matrix.
        """

        # Extract translation (x, y, z)
        translation = transform.transform.translation
        tx, ty, tz = translation.x, translation.y, translation.z

        # Extract rotation (x, y, z, w) and convert to rotation matrix
        quat = transform.transform.rotation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()  # 3x3 rotation matrix

        # Construct the 4x4 homogeneous transformation matrix
        homogenous_matrix = np.eye(4)  # Start with identity matrix
        homogenous_matrix[:3, :3] = rot  # Set rotation part
        homogenous_matrix[:3, 3] = [tx, ty, tz]  # Set translation part

        return homogenous_matrix


    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """
        Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).
        
        :param R_matrix: 3x3 numpy array (rotation matrix)
        :return: Quaternion (x, y, z, w) as a numpy array
        """

        rot = R.from_matrix(rotation_matrix)  # Convert to Rotation object
        return rot.as_quat()  # Get quaternion (x, y, z, w)


    def map_range(self, x, custom_range_min, custom_range_max, target_range_min, target_range_max):
        """
        Maps from custom range to a target range. Eg. custom range 0:800 mm -> gripper arbitrary range 0:0.04.
        """
        return (x - custom_range_min) * (target_range_max - target_range_min) / (custom_range_max - custom_range_min) + target_range_min


    def is_identity(self, matrix, tol=1e-6):
        return np.allclose(matrix, np.eye(matrix.shape[0]), atol=tol)


    def tcp_receive(self, buffer_size=1024):
        rospy.loginfo(f"Receiving data from {self.host}:{self.port}...")

        try:
            # Receive message from the server
            data = self.sock.recv(buffer_size)
        
            if data:
                rospy.loginfo(f"Received message: {data}")

                # Rebuild the data and separate the matrix and frame
                matrix, frame = self.rebuild_data(data)

                rospy.loginfo(f"Rebuilt matrix: \n{matrix}")
                rospy.loginfo(f"Rebuilt frame: {frame}")

                self.received_command = [matrix, frame]  # You now only have the matrix, as the frame is separate

            else:
                rospy.logwarn("No data received or connection closed by the server.")

        except Exception as e:
            rospy.logerr(f"Client stopped with error: {e}.")
            self.sock.close()
            rospy.loginfo("Connection closed.")


    def rebuild_matrix(self, matrix_data):
        # Convert the byte data back into a numpy array and reshape it into a 4x4 matrix
        transformation_matrix = np.frombuffer(matrix_data, dtype=np.float32).copy().reshape(4, 4)
        return transformation_matrix
        

    def rebuild_data(self, data):
        """ Function to rebuild the data received from the server """
        # The first 16 bytes represent the 4x4 matrix (4*4=16 float32 elements)
        matrix_data = data[:16 * 4]  # First 16 elements (4x4 matrix)
        
        # Convert matrix data back into a 4x4 NumPy array
        matrix = np.frombuffer(matrix_data, dtype=np.float32).reshape((4, 4)).copy()

        # The remaining data is the string (null-terminated)
        frame_data = data[16 * 4:]  # The rest is the string
        frame = frame_data.decode('utf-8').strip('\0')  # Decode and remove null-termination
        
        # Instead of returning a list with mixed data, let's ensure they are returned separately.
        return matrix, frame



if __name__ == "__main__":

    panda_custom_controller = Panda_Custom_Controller()

