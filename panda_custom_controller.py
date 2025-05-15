import rospy
import moveit_commander
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R  # SciPy for quaternion math
import numpy as np
import socket
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, GraspEpsilon
from sensor_msgs.msg import JointState
from tf.transformations import euler_matrix
import time

class Panda_Custom_Controller(): 
    def __init__(self):

        rospy.init_node("panda_custom_controller", anonymous=True)

        # # Initialize MoveIt!
        moveit_commander.roscpp_initialize(rospy.myargv(argv=[]))
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.manipulator_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("panda_hand")

        # Create action client
        self.client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        self.client.wait_for_server()
        
        self.manipulator_group.set_planner_id("RRTConnect")

        self.manipulator_group.set_planning_time(10.0)
        self.manipulator_group.set_goal_tolerance(0.01)
        self.manipulator_group.set_goal_orientation_tolerance(0.01)

        self.manipulator_group.set_max_velocity_scaling_factor(0.4)     # 0.4  # 20% of max velocity
        self.manipulator_group.set_max_acceleration_scaling_factor(0.5) # 0.5  # 10% of max acceleration

        # Initialize tf2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.sleep(1)  # Allow tf to populate

        rospy.logdebug("Hello world!")
        rospy.loginfo("Hello world!")
        rospy.logwarn("Hello world!")
        rospy.logerr("Hello world!")

        self.adress = None
        self.server_address_receive_from = ('100.106.51.5', 20000)
        self.server_address_send_to =      ('100.114.98.19', 20001)


        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server_address_receive_from)
        self.sock.setblocking(False)
        self.sock.settimeout(1.0)


        self.command = np.array([[ 1.0, 0.0, 0.0, 0.0 ],
                                 [ 0.0, 1.0, 0.0, 0.0 ],
                                 [ 0.0, 0.0, 1.0, 0.0 ],
                                 [ 0.0, 0.0, 0.0, 1.0 ]])
        
        self.received_command = np.array([[ 1.0, 0.0, 0.0, 0.0 ],
                                          [ 0.0, 1.0, 0.0, 0.0 ],
                                          [ 0.0, 0.0, 1.0, 0.0 ],
                                          [ 0.0, 0.0, 0.0, 1.0 ]])
        
        self.cam_to_gripper_extrinsic = np.array([[ 0.79110791,  0.22299801,  0.56957893, -0.1363679  ],
                                                  [ 0.13523332,  0.84436975, -0.51841265,  0.14765188 ],
                                                  [-0.59654021,  0.4871464,   0.63783083,  0.58239576 ],
                                                  [ 0. ,         0. ,         0. ,         1.         ]])

        ################
        #### HOMING ####
        ################

        self.base_home_transformation = np.array([[ 1.0, 0.0, 0.0, 600.0 ],
                                                  [ 0.0, 0.0, 1.0, 0.0   ],
                                                  [ 0.0, 1.0, 0.0, 550.0 ],
                                                  [ 0.0, 0.0, 0.0, 1.0   ]])
        

        theta_home = np.deg2rad(0.0)
        self.rotate_end_effector_around_z = np.array([[ np.cos(theta_home), -np.sin(theta_home), 0.0, 0.0 ], 
                                                      [ np.sin(theta_home), np.cos(theta_home),  0.0, 0.0 ],
                                                      [ 0.0,                0.0,                 1.0, 0.0 ],
                                                      [ 0.0,                0.0,                 0.0, 1.0 ]])

       # self.rotate_end_effector_around_z_world = self.combined_frame_to_world_frame(self.rotate_end_effector_around_z)

       # self.base_home_transformation[:3,:3] = self.rotate_end_effector_around_z_world[:3,:3] @ self.base_home_transformation[:3,:3] 

        self.home()
        self.grip(0.0)
        self.grip(79.0)
        
        #######################
        #### END OF HOMING ####
        #######################

        ##########################
        #### OTHER TRANSFORMS ####
        ##########################

        theta_hand_over = np.deg2rad(0.0)
        self.hand_over_combined = np.array([[ np.cos(theta_hand_over), -np.sin(theta_hand_over), 0.0, 200.0  ],
                                   [ np.sin(theta_hand_over), np.cos(theta_hand_over), 0.0, -300.0 ],
                                   [ 0.0, 0.0, 1.0, 300.0  ],
                                   [ 0.0, 0.0, 0.0, 1.0    ]])
        
        self.hand_over_world = self.combined_frame_to_world_frame(self.hand_over_combined)

        theta_home_transformation = np.deg2rad(0.0)
        self.home_transformation_combined = np.array([[ np.cos(theta_home_transformation), -np.sin(theta_home_transformation), 0.0, 600.0 ],
                                                      [ np.sin(theta_home_transformation), np.cos(theta_home_transformation),  0.0, 0.0   ],
                                                      [ 0.0,                               0.0,                                1.0, 550.0 ],
                                                      [ 0.0,                               0.0,                                0.0, 1.0   ]])
        
        self.home_transformation_world = self.combined_frame_to_world_frame(self.home_transformation_combined)

        self.world_home_transformation = np.array([[ 1.0, 0.0, 0.0, 600.0 ],
                                                   [ 0.0, 1.0, 0.0, 0.0   ],
                                                   [ 0.0, 0.0, 1.0, 550.0 ],
                                                   [ 0.0, 0.0, 0.0, 1.0   ]])
        
        

    def run(self):
        state = 1
        counter = 0
        while not rospy.is_shutdown(): 

            try:

                if state == 0: # Testing state
                    rospy.loginfo_throttle(10, f"State {state} - Testing state")
 
                    # z = 0.0
                    # rotz = np.array([[ np.cos(z*np.pi/180), -np.sin(z*np.pi/180), 0.0, 0.0 ],
                    #                  [ np.sin(z*np.pi/180), np.cos(z*np.pi/180),  0.0, 0.0 ],
                    #                  [ 0.0,                 0.0,                  1.0, 0.0 ],
                    #                  [ 0.0,                 0.0,                  0.0, 1.0 ]])
                    # rospy.loginfo(f"z: {z}")
                    # rotz_world = self.combined_frame_to_world_frame(rotz)
                    # _ = self.homogenous_transformation_matrix_to_pose(rotz_world, input_frame_of_reference="world", move_when_done=True, relative_movement= True)
                    # time.sleep(1)

                    # cur_pose = self.manipulator_group.get_current_pose().pose
                    # rospy.loginfo(f"\ncur_pose: \n{cur_pose}")

 
                    # z = 90.0
                    # rotz = np.array([[ np.cos(z*np.pi/180), -np.sin(z*np.pi/180), 0.0, 0.0 ],
                    #                  [ np.sin(z*np.pi/180), np.cos(z*np.pi/180),  0.0, 0.0 ],
                    #                  [ 0.0,                 0.0,                  1.0, 0.0 ],
                    #                  [ 0.0,                 0.0,                  0.0, 1.0 ]])
                    # rospy.loginfo(f"z: {z}")
                    # rotz_world = self.combined_frame_to_world_frame(rotz)
                    # _ = self.homogenous_transformation_matrix_to_pose(rotz_world, input_frame_of_reference="world", move_when_done=True, relative_movement= True)
                    # time.sleep(1)

                    cur_pose = self.manipulator_group.get_current_pose().pose
                    rospy.loginfo(f"\ncur_pose: \n{cur_pose}")

                    self.grasp(target= 0.0)
                    finger_distance =  self.get_finger_distance()
                
                    self.grasp(target= finger_distance - 0.70)

                    time.sleep(5)

                    self.client.cancel_goal()
                    self.grip(79.0)

                    state = 0


                if state == 1: # Go to object

                    if counter % 60 == 0:
                        self.udp_send("Waiting for command.")

                    rospy.loginfo_throttle(60, f"State {state} - Go to object")
                    self.udp_receive()

                    # try:
                    if not np.array_equal(self.received_command[0], self.command[0]):
                        self.command = self.received_command

                        (self.received_command[0])[2, 3] = (self.received_command[0])[2, 3] - 109.0

                        # Get the object angle data from bottom left cornor of the recieved matrix
                        object_angle = (self.received_command[0])[3, 0]
                        object_angle = (object_angle * -1) - (45*np.pi/180)
                        rospy.loginfo(f"object_angle: \n{object_angle}")

                        # Replace the object angle data from bottom left cornor of the received matrix with zero, so it becomes htm
                        (self.received_command[0])[3, 0] = 0
                        rospy.loginfo(f"received_command: \n{self.received_command[0]}")

                        wrist_angle = self.get_joint_angle("panda_joint7")
                        rospy.loginfo(f"wrist_angle: \n{wrist_angle}")

                        best_angle = self.choose_best_grasp_angle(object_angle, wrist_angle)
                        rospy.loginfo(f"best_angle: \n{best_angle}")

                        best_angle_rot = np.array([[ np.cos(best_angle), -np.sin(best_angle), 0],
                                                   [ np.sin(best_angle), np.cos(best_angle),  0],
                                                   [ 0,                   0,                  1]])
                        rospy.loginfo(f"best_angle_rot: \n{best_angle_rot}")

                        # best_angle_rot = euler_matrix(0, 0, best_angle)[:3, :3]

                        command_world = self.panda_hand_frame_to_world_frame(self.received_command[0])
                        rospy.loginfo(f"command_world: \n{command_world}")

                        command_world[:3, :3] = command_world[:3, :3] @ best_angle_rot[:3, :3]
                        rospy.loginfo(f"command_world: \n{command_world}")

                        pose = self.homogenous_transformation_matrix_to_pose(
                            homogenous_transformation_matrix= command_world,
                            grip = True,
                            gripper_distance= self.command[1],
                            input_frame_of_reference= "world",
                            relative_movement= False,
                            move_when_done= True,
                            z_offset= True,
                            z_offset_value= 0.30
                        )

                        command_world_without_rotation = np.eye(4)
                        command_world_without_rotation[:3, 3] = command_world[:3, 3]


                        self.manipulator_group.set_max_velocity_scaling_factor(0.1)     # 0.4  # 20% of max velocity
                        self.manipulator_group.set_max_acceleration_scaling_factor(0.1) # 0.5  # 10% of max acceleration


                        pose = self.homogenous_transformation_matrix_to_pose(
                            homogenous_transformation_matrix= command_world_without_rotation,
                            input_frame_of_reference= "world",
                            relative_movement= False,
                            move_when_done= True,
                            grip= False
                        )


                        self.manipulator_group.set_max_velocity_scaling_factor(0.4)     # 0.4  # 20% of max velocity
                        self.manipulator_group.set_max_acceleration_scaling_factor(0.5) # 0.5  # 10% of max acceleration

                        state = 2


                if state == 2: # Grasp object
                    rospy.loginfo(f"State {state} - Grasp object")
                    self.grasp(target= 0.0)
                    finger_distance =  self.get_finger_distance()
                    
                    if finger_distance < 1.0:
                        rospy.loginfo("I missed the object. Please give me a new command.")
                        self.udp_send("I missed the object. Please give me a new command.")
                        self.home()
                        
                        state = 1
                    

                    else:
                        self.grasp(target= finger_distance - 0.70)

                        state = 3


                if state == 3: # Home
                    rospy.loginfo(f"State {state} - Home")
                    
                    self.home()

                    state = 4


                if state == 4: # Go to hand over
                    rospy.loginfo(f"State {state} - Go to hand over")
                    _ = self.homogenous_transformation_matrix_to_pose( 
                            homogenous_transformation_matrix= self.hand_over_world,
                            input_frame_of_reference= "world",  
                            move_when_done= True,  
                            relative_movement= False,  
                            grip= False)                    
                    
                    state = 5


                if state == 5: # release grasp
                    rospy.loginfo(f"State {state} - Release grasp")
                    self.client.cancel_goal()
                    self.grip(79.0)

                    state = 6


                if state == 6: # Home
                    rospy.loginfo(f"State {state} - Home")

                    self.home()

                    state = 1


            except socket.timeout:
                counter += 1 

            except KeyboardInterrupt:
                rospy.loginfo("Shutting down due to KeyboardInterrupt.") 
                if socket:
                    socket.close()
                    rospy.loginfo("Socket closed cleanly.")
                moveit_commander.roscpp_shutdown()
                rospy.loginfo("MoveIt closed cleanly.")  
                quit()
                break             


    def combined_frame_to_world_frame(self, homogenous_transformation_matrix_combined: np.ndarray):
        """
        Takes a homogenous transformation matrix. The system expects any translation to be in frame of
        reference "world" and any rotation (of the gripper) to be in frame of reference "panda_hand".
        This function therefore splits the two actions into two matricies. 
        Then converts the translation to be in "panda_hand" and then combines the translation and rotaion again.
        Then it converts back to "world".
        """

        rospy.loginfo(f"combined_frame_to_world_frame - homogenous_transformation_matrix_combined: \n{homogenous_transformation_matrix_combined}")

        rot_component_hand = homogenous_transformation_matrix_combined[:3, :3]
        trans_component_world = homogenous_transformation_matrix_combined[:3, 3]
        rospy.loginfo(f"combined_frame_to_world_frame - _hand: \n{rot_component_hand}")
        rospy.loginfo(f"combined_frame_to_world_frame - trans_component: \n{trans_component_world}")

        rot_matrix_hand = np.identity(4) # "panda_hand"
        rot_matrix_hand[:3, :3] = rot_component_hand
        #  zero_vec = np.zeros((3, 1))
        #  rot_matrix_hand[:3, 3] = zero_vec
        rospy.loginfo(f"combined_frame_to_world_frame - rot_matrix_hand: \n{rot_matrix_hand}")
        
        T_hand_to_world = self.get_transform(target_frame="world", source_frame="panda_hand")

        htm_T_hand_to_world = self.transform_to_homogeneous_matrix(T_hand_to_world)
        rospy.loginfo(f"combined_frame_to_world_frame - htm_T_hand_to_world: \n{htm_T_hand_to_world}")

        rot_component_htm_T_hand_to_world = htm_T_hand_to_world[:3, :3]
        rospy.loginfo(f"combined_frame_to_world_frame - rot_component_htm_T_hand_to_world: \n{rot_component_htm_T_hand_to_world}")

        rot_component_world = rot_component_htm_T_hand_to_world @ rot_component_hand

        combined_matrix_world = np.identity(4)
        combined_matrix_world[:3, :3] = rot_component_world
        combined_matrix_world[:3, 3] = trans_component_world
        rospy.loginfo(f"combined_frame_to_world_frame - combined_matrix_world: \n{combined_matrix_world}")

        return combined_matrix_world


    def panda_hand_frame_to_world_frame(self, homogenous_transformation_matrix_hand: np.ndarray):
        """
        TBD
        """

        rospy.loginfo(f"panda_hand_frame_to_world_frame - homogenous_transformation_matrix_world: \n{homogenous_transformation_matrix_hand}")

        T_hand_to_world = self.get_transform(target_frame= "world", source_frame= "panda_hand_tcp")

        htm_T_hand_to_world = self.transform_to_homogeneous_matrix(T_hand_to_world)
        rospy.loginfo(f"panda_hand_frame_to_world_frame - htm_T_hand_to_world: \n{htm_T_hand_to_world}")

        htm_T_hand_to_world[0, 3] = htm_T_hand_to_world[0, 3] * 1000
        htm_T_hand_to_world[1, 3] = htm_T_hand_to_world[1, 3] * 1000
        htm_T_hand_to_world[2, 3] = htm_T_hand_to_world[2, 3] * 1000

        rospy.loginfo(f"panda_hand_frame_to_world_frame - htm_T_hand_to_world: \n{htm_T_hand_to_world}")

        homogenous_transformation_matrix_world = htm_T_hand_to_world @ homogenous_transformation_matrix_hand
        rospy.loginfo(f"panda_hand_frame_to_world_frame - homogenous_transformation_matrix_world: \n{homogenous_transformation_matrix_world}")

        return homogenous_transformation_matrix_world


    def homogenous_transformation_matrix_to_pose(self, homogenous_transformation_matrix: np.ndarray, input_frame_of_reference: str, relative_movement: bool, gripper_distance: float= 79.0, move_when_done: bool= False, grip: bool= False, grasp: bool= False, z_offset: bool= False, z_offset_value: float= 0.0):
        """
        Takes a homogenous transformation matrix and a frame of reference (see the ROS tf tree).
        - The two most important frames are "world" which is at the same position as the base (which is panda_link0), and "panda_hand" which is the tool.

        Returns geometry_msgs.msg.Pose()
        """

        # If the frame of reference is world (same as panda_link0, ie base, but for reasons I like this more), then get current pose and add the translation values to xyz. 
        if input_frame_of_reference == "world":
            
            rospy.loginfo(f"homogenous_transformation_matrix: \n{homogenous_transformation_matrix}")

            current_pose = self.manipulator_group.get_current_pose().pose

            rotation_matrix = homogenous_transformation_matrix[:3, :3]
            translation = homogenous_transformation_matrix[:3, 3]
            
            rospy.loginfo(f"rotation_matrix: \n{rotation_matrix}\ntranslation: \n{translation}")

            rospy.loginfo(f"\ncurrent_pose.position.x: {current_pose.position.x}\ncurrent_pose.position.y: {current_pose.position.y}\ncurrent_pose.position.z: {current_pose.position.z}")

            if relative_movement == True:
                translation[0] += current_pose.position.x * 1000
                translation[1] += current_pose.position.y * 1000
                translation[2] += current_pose.position.z * 1000
                
                rospy.loginfo(f"relative_movement=true: translation: \n{translation}")

            # If rotation is identity, grab current orientation and use for new pose. Else use rotation matrix from input.
            if self.is_identity(rotation_matrix):
                rotation = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
                print("is a identity mattrix") 
                
            else:
                rotation = self.rotation_matrix_to_quaternion(rotation_matrix) 
                print("is not a identity matrix")

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

        rospy.loginfo(f"translation: {translation}")

        if z_offset:
            # Construct the pose
            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x = translation[0] / 1000
            target_pose.position.y = translation[1] / 1000
            target_pose.position.z = (translation[2] / 1000) + z_offset_value
            target_pose.orientation.x = rotation[0]
            target_pose.orientation.y = rotation[1]
            target_pose.orientation.z = rotation[2]
            target_pose.orientation.w = rotation[3]

        else:
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
            rospy.loginfo(f"pose: \n{target_pose}")
            self.move_to_target(target_pose)
            rospy.loginfo(f"gripper_distance: \n{gripper_distance}")
            if grip:
                self.grip(gripper_distance)
            if grasp:
                self.grasp(target= gripper_distance)
            return target_pose

        else:
            return target_pose


    def move_to_target(self, target_pose, waitkey: bool= True, recursion_counter: int= 0):
        """
        Move the robot end effector to the given target pose using MoveIt.
        """
        try:
            # self.manipulator_group.clear_pose_targets()
            # self.manipulator_group.set_start_state_to_current_state()

                if recursion_counter == 5:
                    rospy.logerr(f"Did not succed in creating a motion plan after {recursion_counter} attemtps.")
                    self.udp_send(f"Did not succed in creating a motion plan after {recursion_counter} attemtps.")
                    #quit()

                rospy.loginfo(f"target_pose: {target_pose}")

                self.manipulator_group.set_pose_target(target_pose)

                success, plan, planning_time, error_code = self.manipulator_group.plan()

                rospy.logerr(f"Move to target error: {error_code}")
                #rospy.loginfo(f"Plan: {plan}")

                if success:  # plan[0] is the success flag
                    result = self.manipulator_group.execute(plan, wait= waitkey)

                    if result:
                        rospy.loginfo("Motion execution succeeded.")
                        # Send the result as a string over UDP
                        #self.udp_send("Motion execution succeeded.")
                    else:
                        rospy.logwarn("Motion execution failed during execution.")
                        # Send the failure message over UDP
                        self.udp_send("Motion execution failed during execution.")
                else:
                    rospy.logwarn("Motion planning failed. Retrying...")
                    # Send the failure message over UDP
                    self.udp_send("Motion planning failed. Retrying...")

                    #self.move_to_target(target_pose)#, recursion_counter= recursion_counter + 1)

            # self.manipulator_group.stop()
            # self.manipulator_group.clear_pose_targets() 
        except KeyboardInterrupt: 
            quit()

        return plan


    def grip(self, target= 80.0, min_gripper= 0.0, max_gripper= 80.0):
        """
        Opens the gripper to a target mm distance between the claws.
        Default max open (800 mm).
        """

        rospy.loginfo(f"target: {target}")

        # maps from a custom range (the mm dist between the claws of the gripper) to the grippers standard range (0.0:0.04), so we can control it in mm.
        target = self.map_range(target, min_gripper, max_gripper, 0.0, 0.04)
        grip_l = target
        grip_r = target

        rospy.loginfo(f"target: {target}\ngrip_l: {grip_l}\ngrip_r: {grip_r}")

        self.gripper_group.set_joint_value_target([grip_l, grip_r])

        #Execute the movement
        self.gripper_group.go(wait=True)

        #Stop movement
        self.gripper_group.stop()


    def grasp(self, target=80.0, min_gripper=0.0, max_gripper=80.0, speed=0.2, force=10.0):
        """
        Closes the gripper to a target mm distance between the claws using force-based control.
        Uses the franka_gripper action interface.

        :param target: Desired gripper opening in mm (default: 80 mm).
        :param min_gripper: Minimum gripper range in mm (default: 0.0 mm).
        :param max_gripper: Maximum gripper range in mm (default: 80.0 mm).
        :param speed: Gripper closing speed in m/s (default: 0.05).
        :param force: Gripper grasping force in N (default: 20.0).
        """
        rospy.loginfo(f"[grip] target: {target} mm")

        # Convert target from mm to meters and map to [0.0, 0.08] (gripper's real opening range in meters)
        width = self.map_range(target, min_gripper, max_gripper, 0.0, 0.08)

        # Define grasp goal
        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.epsilon = GraspEpsilon(inner=0.005, outer=0.005)
        grasp_goal.speed = speed
        grasp_goal.force = force

        # Send goal
        self.client.send_goal(grasp_goal)
        self.client.wait_for_result()

        result = self.client.get_result()
        rospy.loginfo(f"[grip] Grasp result: {result}")
        

    def home(self):
        """
        Homes the manipulator to a position set in init, rotates the gripper to face towards world x, fully closes and opens the gripper.
        """

        #_ = self.homogenous_transformation_matrix_to_pose(self.base_home_transformation, input_frame_of_reference="world", move_when_done=True, relative_movement= False, grip= True, gripper_distance= 79.0)

        
        joint_goal = self.manipulator_group.get_current_joint_values()
        rospy.loginfo(f"joint_goal: {joint_goal}")

        joint_goal[0] = 0.0
        joint_goal[1] = 0.0
        joint_goal[2] = 0.0
        joint_goal[3] = -90.0*np.pi/180
        joint_goal[4] = 0.0
        joint_goal[5] = 90.0*np.pi/180
        joint_goal[6] = 0.0

        # "panda_joint1"
        # "panda_joint2"
        # "panda_joint3"
        # "panda_joint4"
        # "panda_joint5"
        # "panda_joint6"
        # "panda_joint7"

        self.manipulator_group.set_joint_value_target(joint_goal)
        
        success, plan, planning_time, error_code = self.manipulator_group.plan()

        self.manipulator_group.execute(plan, wait=True)
        self.manipulator_group.stop()

        trans_plan = self.homogenous_transformation_matrix_to_pose(self.base_home_transformation, input_frame_of_reference="panda_link0", move_when_done=True, relative_movement= False)
        rot_plan = self.homogenous_transformation_matrix_to_pose(self.rotate_end_effector_around_z, input_frame_of_reference="panda_hand", move_when_done=True, relative_movement= False)


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


    def is_identity(self, matrix, tol= 1e-6):
        return np.allclose(matrix, np.eye(matrix.shape[0]), atol=tol)


    def udp_receive(self, buffer_size= 4096):
        rospy.loginfo_throttle_identical(10, f"Receiving data from {self.server_address_receive_from[0]}:{self.server_address_receive_from[1]}...")

        #try:
            # Receive message from the server
        data, self.adress = self.sock.recvfrom(buffer_size)
    
        if data:
            rospy.loginfo(f"Received message: {data}")

            # Rebuild the data and separate the matrix and frame
            matrix, gripper_distance, frame = self.rebuild_data(data)

            rospy.loginfo(f"Rebuilt matrix: \n{matrix}")
            rospy.loginfo(f"Rebuilt gripper_distance: \n{gripper_distance}")
            rospy.loginfo(f"Rebuilt frame: {frame}")

            self.received_command = [matrix, gripper_distance, frame]  # You now only have the matrix, as the frame is separate

        else:
            rospy.logwarn_throttle(10, "No data received or connection closed by the server.")

        # except Exception as e:
        #     rospy.logerr_throttle(10, f"Client stopped with error: {e}.")
        #     self.sock.close()


    def udp_send(self, message):
        """
        Sends a message (motion plan/result) as a string to the server over UDP.

        :param message: The message to be sent (e.g., result or motion plan)
        """
        try:
            rospy.logdebug(f"Sending message to {self.server_address_send_to[0]}:{self.server_address_send_to[1]}...")

            # Encode the message as bytes (null-terminated string)
            message_bytes = message.encode('utf-8') + b'\0'  # Null-terminate string

            # Send the message
            self.sock.sendto(message_bytes, self.server_address_send_to)
            rospy.logdebug(f"Sent message to {self.server_address_send_to[0]}:{self.server_address_send_to[1]}")

        except Exception as e:
            rospy.logerr(f"Failed to send data: {e}")


    def rebuild_data(self, data):
        """ Function to rebuild the data received from the server """
        # The first 16 bytes represent the 4x4 matrix (4*4=16 float32 elements)
        matrix_data = data[:16 * 4]  # First 16 elements (4x4 matrix)
        
        # Convert matrix data back into a 4x4 NumPy array
        matrix = np.frombuffer(matrix_data, dtype=np.float32).reshape((4, 4)).copy()

        gripper_distance_data = data[16 * 4 : 17 * 4]  # get 4 bytes for 1 float32
        gripper_distance = np.frombuffer(gripper_distance_data, dtype=np.float32)[0]

        # The remaining data is the string (null-terminated)
        frame_data = data[17 * 4:]  # The rest is the string
        frame = frame_data.decode('utf-8').strip('\0')  # Decode and remove null-termination
        
        # Instead of returning a list with mixed data, let's ensure they are returned separately.
        return matrix, gripper_distance, frame



    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def choose_best_grasp_angle(self, grasp_angle, current_joint_7):
        """
        Choose between grasp_angle + 90° and grasp_angle - 90° based on joint 7 limits and shortest path.
        
        Parameters:
            grasp_angle (float): Object-predicted grasp angle (radians)
            current_joint_7 (float): Current joint 7 angle (radians)
            joint_7_limits (tuple): The joint 7 limits (min, max) in radians
            pi_2 (float): The value of pi/2 (defaults to np.pi / 2)
        
        Returns:
            float: Chosen grasp angle (radians)
        """
        # Two candidates: +90° and -90° rotation
        angle_plus_90 = self.normalize_angle(grasp_angle)
        angle_minus_90 = self.normalize_angle(grasp_angle - (np.pi))

        # Simulate joint 7 values for both rotations
        joint_7_plus_90 = self.normalize_angle(current_joint_7 + angle_plus_90)
        joint_7_minus_90 = self.normalize_angle(current_joint_7 + angle_minus_90)

        # Check if both angles are valid within joint 7 limits 
        joint_7_limits=(-2.8973, 2.8973)
        is_plus_90_valid = joint_7_limits[0] <= joint_7_plus_90 <= joint_7_limits[1]
        is_minus_90_valid = joint_7_limits[0] <= joint_7_minus_90 <= joint_7_limits[1]

        if is_plus_90_valid and not is_minus_90_valid:
            return angle_plus_90
        elif is_minus_90_valid and not is_plus_90_valid:
            return angle_minus_90
        elif is_plus_90_valid and is_minus_90_valid:
            # Both valid — pick the one with the shortest path for joint 7
            delta_plus_90 = abs(joint_7_plus_90 - current_joint_7)
            delta_minus_90 = abs(joint_7_minus_90 - current_joint_7)
            return angle_plus_90 if delta_plus_90 <= delta_minus_90 else angle_minus_90
        else:
            # Neither are valid, fallback to original
            print("Warning: No valid angle keeps joint 7 within limits. Returning original grasp angle.")
            return grasp_angle 
    

    def get_joint_angle(self, joint: str):
        try:
            msg = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
            idx = msg.name.index(joint)
            return msg.position[idx]
        except (rospy.ROSException, ValueError) as e:
            rospy.logerr("Error getting joint 7 angle: %s", e)
            return None


    def get_finger_distance(self):
        try:
            msg = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
            left_idx = msg.name.index("panda_finger_joint1")
            right_idx = msg.name.index("panda_finger_joint2")

            rospy.loginfo(f"\nleft_idx: {left_idx}\nright_idx: {right_idx}")

            left_finger = msg.position[left_idx]
            right_finger = msg.position[right_idx]

            rospy.loginfo(f"\nleft_finger: {left_finger}\nright_finger: {right_finger}")

            return (left_finger + right_finger) * 1000

        except (rospy.ROSException, ValueError) as e:
            rospy.logerr("Error getting finger distance: %s", e)
            return None


if __name__ == "__main__":
    try:
        controller = Panda_Custom_Controller()
        controller.run() 
        exit()
    except rospy.ROSInterruptException:
        pass