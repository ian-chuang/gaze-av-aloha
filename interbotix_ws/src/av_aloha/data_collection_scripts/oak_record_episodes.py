import depthai as dai
import time
import numpy as np
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from constants import (
    XML_DIR, REAL_DT, 
    LEFT_GRIPPER_JOINT_UNNORMALIZE_FN, RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
    LEFT_GRIPPER_JOINT_NORMALIZE_FN, RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    LEFT_GRIPPER_VELOCITY_NORMALIZE_FN, RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN,
    LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE,
)
from diff_ik import DiffIK
from grad_ik import GradIK
from webrtc_headset import WebRTCHeadset
from image_recorders import ROSImageRecorder
from oak_recorder import OAKImageRecorder
from robot_utils import setup_puppet_bot, move_arms, move_grippers, sleep, torque_off, torque_on, get_arm_gripper_positions
from transform_utils import xyzw_to_wxyz, mat2pose, pose2mat, wxyz_to_xyzw
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rospy
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
from kinematics import create_fk_fn, create_safety_fn
import mujoco
import os
from tqdm import tqdm

DT = REAL_DT

class RealEnv(gym.Env):

    def __init__(self, init_node=True, headset: WebRTCHeadset = None):
        self.headset = headset
        self.oak_recorder.start()
        # setup observation and action space
        self.observation_space = spaces.Dict({
            'joints': spaces.Dict({
                'position': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
                'velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,))  # 21 joint velocities
            }),
            'control': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
            'poses': spaces.Dict({
                'left': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # left arm pose
                'right': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # right arm pose
                'middle': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))  # middle arm pose
            }),
            'images': spaces.Dict({
                'oak_left': spaces.Box(low=0, high=255, shape=(480, 640, 3)),
                'oak_right': spaces.Box(low=0, high=255, shape=(480, 640, 3)),
                'wrist_cam_left': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # left wrist camera image
                'wrist_cam_right': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # right wrist camera image
                # 'overhead_cam': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # high camera image
                # 'worms_eye_cam': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # low camera image
            }),
        })
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        )   

        # setup mujoco for forward kinematics
        self._mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'aloha_real.xml'))
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        self._left_eef_site = self._mjcf_root.find('site', LEFT_EEF_SITE)
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        self._left_fk_fn = create_fk_fn(self._physics, self._left_joints, self._left_eef_site)
        self._right_fk_fn = create_fk_fn(self._physics, self._right_joints, self._right_eef_site)
        self._middle_fk_fn = create_fk_fn(self._physics, self._middle_joints, self._middle_eef_site)
        
        self._middle_controller = DiffIK(
            physics=self._physics,
            joints=self._middle_joints,
            actuators=self._middle_actuators,
            eef_site=self._middle_eef_site,
            k_pos=0.3,
            k_ori=0.3,
            damping=1.0e-4,
            k_null=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
            q0=np.array(MIDDLE_ARM_POSE),
            max_angvel=3.14,
            integration_dt=DT,
            iterations=10
        )

        # setup ROS image recorder
        self.image_recorder = ROSImageRecorder(init_node=init_node, camera_names=['cam_left_wrist', 'cam_right_wrist'])
        # setup OAK image recorder
        self.oak_recorder = OAKImageRecorder()

        # setup bots
        self.left_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
        self.right_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
        self.middle_bot = InterbotixManipulatorXS(robot_model="vx300s_7dof", group_name="arm", gripper_name=None, robot_name=f"puppet_middle", init_node=False)
        sleep(self.left_bot, self.right_bot, self.middle_bot)
        setup_puppet_bot(self.left_bot)
        setup_puppet_bot(self.right_bot)
        setup_puppet_bot(self.middle_bot)
        
        # cmd buffer
        self.left_ctrl = np.array(self.left_bot.arm.core.joint_states.position[:7])
        self.right_ctrl = np.array(self.right_bot.arm.core.joint_states.position[:7])
        self.middle_ctrl = np.array(self.middle_bot.arm.core.joint_states.position[:7])  

        # normalize the gripper joints
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(self.left_ctrl[6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6])  

    def get_obs(self) -> np.ndarray:
        # get joint positions and velocities
        left_joint_pos = np.array(self.left_bot.arm.core.joint_states.position[:7])
        left_joint_pos[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(left_joint_pos[6])
        right_joint_pos = np.array(self.right_bot.arm.core.joint_states.position[:7])
        right_joint_pos[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(right_joint_pos[6])
        middle_joint_pos = np.array(self.middle_bot.arm.core.joint_states.position[:7])

        try:
            left_joint_vel = np.array(self.left_bot.arm.core.joint_states.velocity[:7])
            left_joint_vel[6] = LEFT_GRIPPER_VELOCITY_NORMALIZE_FN(left_joint_vel[6])
            right_joint_vel = np.array(self.right_bot.arm.core.joint_states.velocity[:7])
            right_joint_vel[6] = RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN(right_joint_vel[6])
            middle_joint_vel = np.array(self.middle_bot.arm.core.joint_states.velocity[:7])
        except IndexError:
            left_joint_vel = np.zeros(7)
            right_joint_vel = np.zeros(7)
            middle_joint_vel = np.zeros(7)

        # do forward kinematics
        # send back ctrl instead of qpos because we want to send back the commanded position
        # real position might be affected by gravity and other forces
        left_pos, left_quat = mat2pose(self._left_fk_fn(self.left_ctrl[:6]))
        right_pos, right_quat = mat2pose(self._right_fk_fn(self.right_ctrl[:6]))
        middle_pos, middle_quat = mat2pose(self._middle_fk_fn(self.middle_ctrl))
        left_quat = xyzw_to_wxyz(left_quat)
        right_quat = xyzw_to_wxyz(right_quat)
        middle_quat = xyzw_to_wxyz(middle_quat)

        # get images
        image_dict = self.image_recorder.get_images()
        images = self.oak_recorder.get_images()

        if images is not None:
            oak_left, oak_right = images
        else:
            oak_left = np.zeros((480, 640, 3), dtype=np.uint8)
            oak_right = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return {
            'joints': {
                'position': np.concatenate([left_joint_pos, right_joint_pos, middle_joint_pos]),
                'velocity': np.concatenate([left_joint_vel, right_joint_vel, middle_joint_vel]),
            },
            'control': np.concatenate([self.left_ctrl, self.right_ctrl, self.middle_ctrl]),
            'poses': {
                'left': np.concatenate([left_pos, left_quat]),
                'right': np.concatenate([right_pos, right_quat]),
                'middle': np.concatenate([middle_pos, middle_quat]),
            },
            'images': {
                'oak_left': oak_left,
                'oak_right': oak_right,
                'wrist_cam_left': image_dict['cam_left_wrist'],
                'wrist_cam_right': image_dict['cam_right_wrist'],
                # 'overhead_cam': image_dict['cam_high'],
                #'worms_eye_cam': image_dict['cam_low'],
            },
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # Reboot puppet robot gripper motors
        self.left_bot.dxl.robot_reboot_motors("single", "gripper", True)
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
        time.sleep(1.0)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)
        move_arms([self.left_bot, self.right_bot, self.middle_bot], [LEFT_ARM_POSE[:6], RIGHT_ARM_POSE[:6], MIDDLE_ARM_POSE[:7]], move_time=2.5)

        self.left_ctrl[:6] = np.array(LEFT_ARM_POSE[:6])
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(LEFT_GRIPPER_JOINT_CLOSE)
        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(RIGHT_GRIPPER_JOINT_CLOSE)
        self.middle_ctrl = np.array(MIDDLE_ARM_POSE[:7])

        observation = self.get_obs()
        info = {}

        return observation, info

    def step(self, action: np.ndarray, all_joints=False) -> tuple:
        left_ctrl = action[:6]
        left_gripper = action[6] # val from 0 to 1
        right_ctrl = action[7:13]
        right_gripper = action[13] # val from 0 to 1
        middle_target = action[14:21]

        # set vals
        self.left_ctrl[:6] = left_ctrl
        self.right_ctrl[:6] = right_ctrl
        if all_joints:
            self.middle_ctrl = middle_target
        else:
            self.middle_ctrl = self._middle_controller.run(self.middle_ctrl, middle_target[:3], middle_target[3:])
        self.left_ctrl[6] = left_gripper
        self.right_ctrl[6] = right_gripper

        # move the robots
        self.left_bot.arm.set_joint_positions(self.left_ctrl[:6], blocking=False)
        self.left_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", 
                                                                         cmd=LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(self.left_ctrl[6])))
        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", 
                                                                          cmd=RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(self.right_ctrl[6])))
        self.middle_bot.arm.set_joint_positions(self.middle_ctrl, blocking=False) 
        
        # get observation
        observation = self.get_obs()

        # send OAK images to headset
        if self.headset is not None:
            left_img = observation["images"]["oak_left"]
            right_img = observation["images"]["oak_right"]

        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
    
    def close(self):
        self.oak_recorder.stop()

def make_real_env(init_node):
    env = RealEnv(init_node, headset=None)
    return env

def main():
    from real_env import RealEnv
    from webrtc_headset import WebRTCHeadset

    # setup the headset
    headset = WebRTCHeadset()
    headset.run_in_thread()

    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    env = RealEnv(init_node=False, headset=headset)

    print("Starting...")
    action = np.zeros(23)

    while True:
        step_start = time.time()

        ts, _, _, _, _ = env.step(action)

        # --- CONTROL ---
        headset_data = headset.receive_data()
        if headset_data is not None:
            action, feedback = headset_control.run(
                headset_data,
                ts['poses']['left'],
                ts['poses']['right'],
                ts['poses']['middle']
            )

        # --- VIDEO (ONLY PLACE THIS HAPPENS) ---
        left_img = ts["images"]["oak_left"]
        right_img = ts["images"]["oak_right"]

        # convert BGR → RGB (VERY IMPORTANT)
        left_img = left_img[:, :, ::-1]
        right_img = right_img[:, :, ::-1]

        left_img = np.zeros((480,640,3), dtype=np.uint8)
        left_img[:, :, 1] = 255  # GREEN

        right_img = left_img.copy()

        headset.send_images(left_img, right_img)

        # --- FEEDBACK ---
        headset.send_feedback(feedback)

        time.sleep(max(0, REAL_DT - (time.time() - step_start)))

# if __name__ == "__main__":
#     import rospy
#     import os

#     def shutdown():
#         print("Shutting down...")
#         os._exit(42)
#     rospy.on_shutdown(shutdown)

#     try:
#         main()
#     except KeyboardInterrupt:
#         print("Shutting down...")
#         os._exit(42)

if __name__ == "__main__":
    try:
        # --- INIT ---
        headset = WebRTCHeadset()
        headset.run_in_thread()

        env = RealEnv(init_node=False, headset=headset)

        headset_control = HeadsetControl()
        feedback = HeadsetFeedback()

        action = np.zeros(23)

        print("Starting OAK stream...")

        while True:
            start = time.time()

            # --- STEP ENV (CRITICAL) ---
            ts, _, _, _, _ = env.step(action)

            # --- CONTROL ---
            headset_data = headset.receive_data()
            if headset_data is not None:
                action, feedback = headset_control.run(
                    headset_data,
                    ts['poses']['left'],
                    ts['poses']['right'],
                    ts['poses']['middle']
                )

            # --- GET OAK IMAGES ---
            left_img = ts["images"]["oak_left"]
            right_img = ts["images"]["oak_right"]

            # DEBUG: check if frames are real
            print("OAK mean:", left_img.mean())

            # --- CONVERT BGR → RGB (VERY IMPORTANT) ---
            left_img = left_img[:, :, ::-1]
            right_img = right_img[:, :, ::-1]

            # --- SEND TO HEADSET ---
            headset.send_images(left_img, right_img)

            # --- FEEDBACK ---
            headset.send_feedback(feedback)

            # ~30 FPS
            time.sleep(max(0, 1/30 - (time.time() - start)))


            #headset.send_images(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)