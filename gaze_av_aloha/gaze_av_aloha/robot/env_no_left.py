import time
import os
import numpy as np
from dm_control import mjcf

from gaze_av_aloha.robot.config import (
    XML_DIR, REAL_DT, 
    RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
    RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    RIGHT_GRIPPER_JOINT_OPEN,
    RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    RIGHT_EEF_SITE, MIDDLE_EEF_SITE,
)

from gym_av_aloha.kinematics.diff_ik import DiffIK, DiffIKConfig
from gym_av_aloha.kinematics.grad_ik import GradIK, GradIKConfig
from gym_av_aloha.vr.headset import WebRTCHeadset
from gym_av_aloha.vr.headset_control_no_left import HeadsetControl
from gym_av_aloha.vr.headset_utils import HeadsetFeedback, HeadsetData

from gaze_av_aloha.robot.cameras import StereoImageRecorder, ROSImageRecorder
from gaze_av_aloha.robot.robot import setup_puppet_bot, move_arms, move_grippers, sleep_no_left

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rospy

DT = REAL_DT
FPS = round(1.0 / DT)

class RealEnv():

    def __init__(self, init_node=True, headset: WebRTCHeadset = None, stereo_cam_idx=18):
        # setup mujoco for forward kinematics
        self._mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'aloha.xml'))
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        # set up controllers
        cfg = GradIKConfig()
        cfg.joint_p = 0.3
        self.right_controller = GradIK(
            config=cfg,
            physics=self._physics,
            joints=self._right_joints,
            eef_site=self._right_eef_site,
        )
        cfg = DiffIKConfig()
        cfg.joint_p = 0.3
        self.middle_controller = DiffIK(
            config=cfg,
            physics=self._physics,
            joints=self._middle_joints,
            eef_site=self._middle_eef_site,
        )

        # setup ROS image recorder
        self.stereo_cam = StereoImageRecorder(cam_index=stereo_cam_idx, headset=headset, auto_start=True)
        self.image_recorder = ROSImageRecorder(init_node=init_node, camera_names=['cam_high', 'cam_low', 'cam_right_wrist'])
        # setup bots
        self.right_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
        self.middle_bot = InterbotixManipulatorXS(robot_model="vx300s_7dof", group_name="arm", gripper_name=None, robot_name=f"puppet_middle", init_node=False)
        sleep_no_left(self.right_bot, self.middle_bot)
        setup_puppet_bot(self.right_bot)
        setup_puppet_bot(self.middle_bot)
        
        # cmd buffer
        self.right_ctrl = np.array(self.right_bot.arm.core.joint_states.position[:7])
        self.middle_ctrl = np.array(self.middle_bot.arm.core.joint_states.position[:7])    

    def get_obs(self) -> np.ndarray:
        # get joint positions and velocities
        right_joint_pos = np.array(self.right_bot.arm.core.joint_states.position[:7])
        right_joint_pos = np.concatenate([[RIGHT_GRIPPER_JOINT_NORMALIZE_FN(right_joint_pos[6])], right_joint_pos[:6]])

        middle_joint_pos = np.array(self.middle_bot.arm.core.joint_states.position[:7])

        right_ctrl = np.concatenate([[RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6])], self.right_ctrl[:6]])
        middle_ctrl = self.middle_ctrl

        # get images
        left_image, right_image = self.stereo_cam.get_images()
        image_dict = self.image_recorder.get_images()
        
        return {
            'joints': np.concatenate([right_joint_pos, middle_joint_pos]),
            'control': np.concatenate([right_ctrl, middle_ctrl]),
            'images': {
                'left_eye_cam': left_image,
                'right_eye_cam': right_image,
                'wrist_cam_right': image_dict['cam_right_wrist'],
                'overhead_cam': image_dict['cam_high'],
                'worms_eye_cam': image_dict['cam_low'],
            },
        }

    def reset(self, seed=None) -> tuple:

        # Reboot puppet robot gripper motors
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        move_grippers([self.right_bot], [RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)

        offset_middle_ctrl = MIDDLE_ARM_POSE.copy()
        # horrible
        offset_middle_ctrl[5] += 0.03
        offset_middle_ctrl[6] += -0.01

        move_arms([self.right_bot, self.middle_bot], [RIGHT_ARM_POSE[:6], offset_middle_ctrl[:7]], move_time=2.5)

        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = np.array(RIGHT_GRIPPER_JOINT_OPEN)
        self.middle_ctrl = np.array(MIDDLE_ARM_POSE[:7])

        return self.get_obs(), self.get_info()
    
    def sleep_no_left(self):
        sleep_no_left(self.right_bot, self.middle_bot)

    def get_info(self):
        return {
            'right_arm_pose': self.right_controller.fk(self.right_ctrl[:6]),
            'right_gripper': RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6]),
            'middle_arm_pose': self.middle_controller.fk(self.middle_ctrl),
        }

    def step_pose(
        self,
        right_pose,
        right_gripper,
        middle_pose,
    ):
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

        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", cmd=self.right_ctrl[6]))
        self.middle_bot.arm.set_joint_positions(offset_middle_ctrl, blocking=False) 

        return self.get_obs(), self.get_info()

    def step_action(self, action) -> tuple:
        right_gripper = action[0]
        right_target = action[1:7]
        middle_target = action[7:14]

        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper)
        self.right_ctrl[:6] = right_target
        self.middle_ctrl = middle_target

        offset_middle_ctrl = self.middle_ctrl.copy()
        offset_middle_ctrl[5] += 0.03
        offset_middle_ctrl[6] += -0.01

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
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            ) 
            # start the episode if the user clicks the right button and the headset is in sync
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.right_out_of_sync == False and \
                not headset_control.is_running():
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
        # print(f"Step time: {end_time - start_time:.4f} seconds")
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