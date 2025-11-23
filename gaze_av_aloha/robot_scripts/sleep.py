from interbotix_xs_modules.arm import InterbotixManipulatorXS
from gaze_av_aloha.robot.robot import torque_on, move_arms


def sleep_two_rights(puppet_bot_right, master_bot_right):
    # torque on arms
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    sleep_pose = (0, -1.7, 1.55, 0, 0.65, 0)
    sleep_pose_2 = (0, -1.85, 1.6, 0, 0.65, 0)

    move_arms([puppet_bot_right, master_bot_right], [sleep_pose, sleep_pose], move_time=3)
    move_arms([puppet_bot_right, master_bot_right], [sleep_pose_2, sleep_pose_2], move_time=3)


def main():
    puppet_bot_right = InterbotixManipulatorXS(
        robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name="puppet_right", init_node=True
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name="master_right", init_node=False
    )

    sleep_two_rights(puppet_bot_right, master_bot_right)


if __name__ == "__main__":
    main()
