import time
import os
import numpy as np
from dm_control import mjcf

from gaze_av_aloha.robot.config import (
    XML_DIR, REAL_DT, 
    LEFT_GRIPPER_JOINT_UNNORMALIZE_FN, RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
    LEFT_GRIPPER_JOINT_NORMALIZE_FN, RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE,
)

from gym_av_aloha.kinematics.diff_ik import DiffIK, DiffIKConfig
from gym_av_aloha.kinematics.grad_ik import GradIK, GradIKConfig
from gym_av_aloha.vr.headset import WebRTCHeadset
from gym_av_aloha.vr.headset_control import HeadsetControl
from gym_av_aloha.vr.headset_utils import HeadsetFeedback, HeadsetData

from gaze_av_aloha.robot.cameras import StereoImageRecorder, ROSImageRecorder
from gaze_av_aloha.robot.robot import setup_puppet_bot, move_arms, move_grippers, sleep

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rospy

DT = REAL_DT
FPS = round(1.0 / DT)

class RealEnv():

    def __init__(self, init_node=True, headset: WebRTCHeadset = None, stereo_cam_idx=24):
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
        # set up controllers
        self.left_controller = GradIK(
            config=GradIKConfig(),
            physics=self._physics,
            joints=self._left_joints,
            eef_site=self._left_eef_site,
        )
        self.right_controller = GradIK(
            config=GradIKConfig(),
            physics=self._physics,
            joints=self._right_joints,
            eef_site=self._right_eef_site,
        )
        self.middle_controller = DiffIK(
            config=DiffIKConfig(),
            physics=self._physics,
            joints=self._middle_joints,
            eef_site=self._middle_eef_site,
        )

        # setup ROS image recorder
        self.stereo_cam = StereoImageRecorder(cam_index=stereo_cam_idx, headset=headset, auto_start=True)
        self.image_recorder = ROSImageRecorder(init_node=init_node, camera_names=['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'])
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

    def get_obs(self) -> np.ndarray:
        # get joint positions and velocities
        left_joint_pos = np.array(self.left_bot.arm.core.joint_states.position[:7])
        left_joint_pos = np.concatenate([[LEFT_GRIPPER_JOINT_NORMALIZE_FN(left_joint_pos[6])], left_joint_pos[:6]])

        right_joint_pos = np.array(self.right_bot.arm.core.joint_states.position[:7])
        right_joint_pos = np.concatenate([[RIGHT_GRIPPER_JOINT_NORMALIZE_FN(right_joint_pos[6])], right_joint_pos[:6]])

        middle_joint_pos = np.array(self.middle_bot.arm.core.joint_states.position[:7])

        left_ctrl = np.concatenate([[LEFT_GRIPPER_JOINT_NORMALIZE_FN(self.left_ctrl[6])], self.left_ctrl[:6]])
        right_ctrl = np.concatenate([[RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6])], self.right_ctrl[:6]])
        middle_ctrl = self.middle_ctrl

        # get images
        left_image, right_image = self.stereo_cam.get_images()
        image_dict = self.image_recorder.get_images()
        
        return {
            'joints': np.concatenate([left_joint_pos, right_joint_pos, middle_joint_pos]),
            'control': np.concatenate([left_ctrl, right_ctrl, middle_ctrl]),
            'images': {
                'left_eye_cam': left_image,
                'right_eye_cam': right_image,
                'wrist_cam_left': image_dict['cam_left_wrist'],
                'wrist_cam_right': image_dict['cam_right_wrist'],
                'overhead_cam': image_dict['cam_high'],
                'worms_eye_cam': image_dict['cam_low'],
            },
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # Reboot puppet robot gripper motors
        self.left_bot.dxl.robot_reboot_motors("single", "gripper", True)
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)

        offset_middle_ctrl = MIDDLE_ARM_POSE.copy()
        # horrible
        offset_middle_ctrl[5] += 0.03
        offset_middle_ctrl[6] += -0.01

        move_arms([self.left_bot, self.right_bot, self.middle_bot], [LEFT_ARM_POSE[:6], RIGHT_ARM_POSE[:6], offset_middle_ctrl[:7]], move_time=2.5)

        self.left_ctrl[:6] = np.array(LEFT_ARM_POSE[:6])
        self.left_ctrl[6] = np.array(LEFT_GRIPPER_JOINT_OPEN)
        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = np.array(RIGHT_GRIPPER_JOINT_OPEN)
        self.middle_ctrl = np.array(MIDDLE_ARM_POSE[:7])

        return self.get_obs(), self.get_info()
    
    def sleep(self):
        sleep(self.left_bot, self.right_bot, self.middle_bot)

    def get_info(self):
        return {
            'left_pose': self.left_controller.fk(self.left_ctrl[:6]),
            'left_gripper': LEFT_GRIPPER_JOINT_NORMALIZE_FN(self.left_ctrl[6]),
            'right_pose': self.right_controller.fk(self.right_ctrl[:6]),
            'right_gripper': RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6]),
            'middle_pose': self.middle_controller.fk(self.middle_ctrl),
        }

    def step_pose(
        self,
        left_pose,
        left_gripper,
        right_pose,
        right_gripper,
        middle_pose,
    ):
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper)
        self.left_ctrl[:6] = self.left_controller.run(
            q=self.left_ctrl[:6],
            target_pos=left_pose[:3, 3],
            target_mat=left_pose[:3, :3],
        )
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper)
        self.right_ctrl[:6] = self.right_controller.run(
            q=self.right_ctrl[:6],
            target_pos=right_pose[:3, 3],
            target_mat=right_pose[:3, :3],
        )
        self.middle_ctrl = self.middle_controller.run(
            q=self.middle_ctrl,
            target_pos=middle_pose[:3, 3],
            target_mat=middle_pose[:3, :3],
        )

        # horrible
        offset_middle_ctrl = self.middle_ctrl.copy()
        offset_middle_ctrl[5] += 0.03
        offset_middle_ctrl[6] += -0.01

        self.left_bot.arm.set_joint_positions(self.left_ctrl[:6], blocking=False)
        self.left_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", cmd=self.left_ctrl[6]))
        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", cmd=self.right_ctrl[6]))
        self.middle_bot.arm.set_joint_positions(offset_middle_ctrl, blocking=False) 

        return self.get_obs(), self.get_info()

    def step_action(self, action) -> tuple:
        left_gripper = action[0]
        left_target = action[1:7]
        right_gripper = action[7]
        right_target = action[8:14]
        middle_target = action[14:21]

        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper)
        self.left_ctrl[:6] = left_target
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper)
        self.right_ctrl[:6] = right_target
        self.middle_ctrl = middle_target

        offset_middle_ctrl = self.middle_ctrl.copy()
        offset_middle_ctrl[5] += 0.03
        offset_middle_ctrl[6] += -0.01

        self.left_bot.arm.set_joint_positions(self.left_ctrl[:6], blocking=False)
        self.left_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", cmd=self.left_ctrl[6]))
        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", cmd=self.right_ctrl[6]))
        self.middle_bot.arm.set_joint_positions(offset_middle_ctrl, blocking=False)
        
        return self.get_obs(), self.get_info()
    
    def __del__(self):
        self.stereo_cam.stop()

def main():
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # setup the environment
    env = RealEnv(init_node=True, headset=headset, stereo_cam_idx=24)
    obs, info = env.reset()
    init_action = {
        'left_pose': info['left_arm_pose'],
        'left_gripper': info['left_gripper'],
        'right_pose': info['right_arm_pose'],
        'right_gripper': info['right_gripper'],
        'middle_pose': info['middle_arm_pose'],
    }
    action = init_action.copy()

    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_data = HeadsetData()
    headset_control.reset()
    while True:
        start_time = time.time()
        obs, info = env.step_pose(**action)

        # get the headset data
        headset_data = headset.receive_data()
        if headset_data is not None:
            headset_action, feedback = headset_control.run(
                headset_data=headset_data, 
                left_arm_pose=info['left_arm_pose'],
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            ) 
            # start the episode if the user clicks the right button and the headset is in sync
            if headset_data.r_button_one == True and not headset_control.is_running():
                headset_control.start(
                    headset_data, 
                    info['middle_arm_pose'],
                )
            
            if headset_control.is_running():
                action = headset_action

            if headset_control.is_running() and headset_data.r_button_one == False:
                action = init_action.copy()
                headset_control.reset()
                obs, info = env.reset()

        # send feedback to the headset
        headset.send_feedback(feedback)

        end_time = time.time()
        print(f"Step time: {end_time - start_time:.4f} seconds")
        time.sleep(max(0, 1.0 / FPS - (end_time - start_time)))

if __name__ == "__main__":
    import traceback
    import rospy

    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        main()
    except Exception as e:
        print(f"An error occured: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down...")
        os._exit(42)