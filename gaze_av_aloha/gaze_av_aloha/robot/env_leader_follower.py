import time
import numpy as np
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from gaze_av_aloha.data_collection_scripts.constants import (
    XML_DIR,
    REAL_DT,
    RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
    RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN,
    RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_ARM_POSE,
    RIGHT_GRIPPER_JOINT_OPEN,
    RIGHT_GRIPPER_JOINT_CLOSE,
    RIGHT_MASTER_GRIPPER_JOINT_OPEN,
    RIGHT_MASTER_GRIPPER_JOINT_CLOSE,
    RIGHT_JOINT_NAMES,
    RIGHT_EEF_SITE,
)
from gaze_av_aloha.robot.cameras import ROSImageRecorder
from gaze_av_aloha.robot.robot import (
    setup_puppet_bot,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from gaze_av_aloha.data_collection_scripts.transform_utils import xyzw_to_wxyz, mat2pose
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from gaze_av_aloha.data_collection_scripts.kinematics import create_fk_fn
import mujoco
import os
import rospy

DT = REAL_DT

class RealEnv(gym.Env):

    def __init__(self, init_node=True):
        # setup observation and action space
        # Action space: 7 (right arm only)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )   

        self.observation_space = spaces.Dict({
            'joints': spaces.Dict({
                'position': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # 7 joint positions
                'velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))   # 7 joint velocities
            }),
            'control': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # 7 joint positions
            'poses': spaces.Dict({
                'right': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # right arm pose
            }),
            'images': spaces.Dict({
                'wrist_cam_right': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # right wrist camera image
                'overhead_cam': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # high camera image
            }),
        })

        # setup mujoco for forward kinematics
        self._mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'aloha_real.xml'))
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        
        self._right_fk_fn = create_fk_fn(self._physics, self._right_joints, self._right_eef_site)
        
        # setup ROS image recorder
        self.image_recorder = ROSImageRecorder(
            init_node=init_node,
            camera_names=['cam_high', 'cam_right_wrist'],
            wait_for_messages=False,
        )

        # setup bot
        self.right_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
        setup_puppet_bot(self.right_bot)
        
        # cmd buffer
        self.right_ctrl = np.array(self.right_bot.arm.core.joint_states.position[:7])

        # normalize the gripper joints
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6])  

    def get_obs(self) -> np.ndarray:
        # get joint positions and velocities
        right_joint_pos = np.array(self.right_bot.arm.core.joint_states.position[:7])
        right_joint_pos[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(right_joint_pos[6])

        try:
            right_joint_vel = np.array(self.right_bot.arm.core.joint_states.velocity[:7])
            right_joint_vel[6] = RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN(right_joint_vel[6])
        except IndexError:
            right_joint_vel = np.zeros(7)

        # do forward kinematics
        right_pos, right_quat = mat2pose(self._right_fk_fn(self.right_ctrl[:6]))
        right_quat = xyzw_to_wxyz(right_quat)

        # get images
        image_dict = self.image_recorder.get_images()
        
        return {
            'joints': {
                'position': right_joint_pos,
                'velocity': right_joint_vel,
            },
            'control': self.right_ctrl,
            'poses': {
                'right': np.concatenate([right_pos, right_quat]),
            },
            'images': {
                'wrist_cam_right': image_dict['cam_right_wrist'],
                'overhead_cam': image_dict['cam_high'],
            },
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # Reboot puppet robot gripper motors
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        
        move_grippers([self.right_bot], [RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
        time.sleep(1.0)
        move_grippers([self.right_bot], [RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)
        
        move_arms([self.right_bot], [RIGHT_ARM_POSE[:6]], move_time=2.5)

        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(RIGHT_GRIPPER_JOINT_CLOSE)

        observation = self.get_obs()
        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # Action shape is now (7,) -> 7 for right
        right_ctrl = action[:6]
        right_gripper = action[6] # val from 0 to 1

        # set vals
        self.right_ctrl[:6] = right_ctrl
        self.right_ctrl[6] = right_gripper

        # move the robot
        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", 
                                                                          cmd=RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(self.right_ctrl[6])))
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
    
    def close(self):
        pass


def reset_env(env: RealEnv, master_bot_right):
    print("Resetting the puppet arm...")
    ts, info = env.reset()
    action = ts['control']
    env.step(action) # compile numba

def reset_master_arm(master_bot_right):
    print("Resetting the master arm...")

    """ Move master robot to a pose where it is easy to start demonstration """
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(master_bot_right)

    # move arm to starting position
    move_arms([master_bot_right], 
              [RIGHT_ARM_POSE[:6]],
              move_time=1.5)
    
    # halfway open master gripper position
    right_master_gripper_middle = (RIGHT_MASTER_GRIPPER_JOINT_OPEN + RIGHT_MASTER_GRIPPER_JOINT_CLOSE) / 2

    # move gripper to starting position
    move_grippers([master_bot_right], 
                  [right_master_gripper_middle],
                  move_time=0.5)

def wait_for_user(master_bot_right):
    print("\nTo Start: Close the master gripper.")

    # almost closed master gripper position (90% closed)
    right_master_gripper_almost_close = RIGHT_MASTER_GRIPPER_JOINT_CLOSE + 0.1 * (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE)

    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)

    last_log = 0
    while True:
        start_time = time.time()

        gripper_pos_right = master_bot_right.dxl.joint_states.position[6]
        gripper_norm = RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN(gripper_pos_right)

        # break once user closes the gripper past ~90% closed
        if gripper_pos_right < right_master_gripper_almost_close or gripper_norm < 0.2:
            break

        # periodic feedback so user knows the current reading
        if time.time() - last_log > 1.0:
            print(f"Waiting for master gripper to close... current={gripper_pos_right} (normalized {gripper_norm}), target<{right_master_gripper_almost_close}")
            last_log = time.time()

        time_until_next_step = REAL_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))

    torque_off(master_bot_right)
    print(f'Started!')

def get_master_bot_action(master_bot_right):
    action = np.zeros(7) # 6 joint + 1 gripper, for one arm
    # Arm actions
    action[:6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action

def main():
    rospy.init_node("leader_follower", anonymous=True)

    # source of data
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name=f'master_right',
        init_node=False
    )

    # setup the environment
    print("Setting up the environment...")
    env = RealEnv(init_node=False)
    print("Environment set up.")

    reset_env(env, master_bot_right)
    print("Environment reset.")

    reset_master_arm(master_bot_right)
    print("Master arm reset.")

    wait_for_user(master_bot_right)
    print("User ready.")

    # run 
    print(f"Starting Teleoperation...")
    ts = env.get_obs()
    action = ts['control']
    
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        ts, reward, terminated, truncated, info = env.step(action)

        # update master bot actions
        action = get_master_bot_action(master_bot_right)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))  

if __name__ == "__main__":
    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
