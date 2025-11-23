import argparse
import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from gaze_av_aloha.robot.robot import get_arm_gripper_positions, move_arms, move_grippers
from gaze_av_aloha.data_collection_scripts.constants import (
    RIGHT_ARM_POSE,
    RIGHT_GRIPPER_JOINT_OPEN,
    RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_MASTER_GRIPPER_JOINT_OPEN,
    RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN,
)


def setup_bot(is_master: bool):
    """Create and home the requested bot, leaving arm torqued and gripper free."""
    model = "wx250s" if is_master else "vx300s"
    name = "master_right" if is_master else "puppet_right"
    bot = InterbotixManipulatorXS(
        robot_model=model,
        group_name="arm",
        gripper_name="gripper",
        robot_name=name,
        init_node=False,
    )
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    bot.dxl.robot_torque_enable("group", "arm", True)

    gripper_open = RIGHT_MASTER_GRIPPER_JOINT_OPEN if is_master else RIGHT_GRIPPER_JOINT_OPEN
    move_arms([bot], [RIGHT_ARM_POSE[:6]], move_time=2.0)
    move_grippers([bot], [gripper_open], move_time=1.0)
    bot.dxl.robot_torque_enable("single", "gripper", False)
    return bot


def main():
    parser = argparse.ArgumentParser(description="Read gripper positions for master or puppet.")
    parser.add_argument("--master", action="store_true", help="Read the master (leader) gripper instead of puppet.")
    args = parser.parse_args()

    rospy.init_node("read_grippers", anonymous=True)
    is_master = args.master

    bot = setup_bot(is_master)
    normalize_fn = RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN if is_master else RIGHT_GRIPPER_JOINT_NORMALIZE_FN

    label = "master_right" if is_master else "puppet_right"
    while not rospy.is_shutdown():
        raw = get_arm_gripper_positions(bot)
        norm = normalize_fn(raw)
        print(f"{label} gripper pos: {raw} (normalized {norm})")
        rospy.sleep(0.2)


if __name__ == "__main__":
    main()
