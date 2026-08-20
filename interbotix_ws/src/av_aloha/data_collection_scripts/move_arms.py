import numpy as np
import rospy

import argparse

from robot_control import (
    create_and_configure_robots,
    interpolate_to_pose,
    get_pose,
)

from data_col_config import ARM_MODES


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--mode",
        choices=["left", "right", "middle", "bimanual", "all", "av"],
        default="right",
    )

    group.add_argument(
        "--high",
        action="store_true",
        help="Move to high reset pose",
    )

    group.add_argument(
        "--forward",
        action="store_true",
        help="Move to forward-facing reset pose",
    )

    group.add_argument(
        "--rest",
        action="store_true",
        help="Move to rest pose",
    )

    group.add_argument(
        "--low",
        action="store_true",
        help="Move to lowered pose",
    )

    return parser.parse_args()

def main():
    rospy.init_node("move_arm", anonymous=True)

    # print("rospy node initialized")

    args = parse_args()

    # print(f"Read args: {args}")

    pose_name = "forward"

    if args.high:
        pose_name = "high"
    elif args.rest:
        pose_name = "rest"
    elif args.low:
        pose_name = "low"

    # print(f"Read pose name: {pose_name}")

    arm_names = ARM_MODES[args.mode]

    # print(f"Read arm names: {arm_names}")

    robots = create_and_configure_robots(arm_names)

    # for arm_name, bot in robots.items():
    #     print(f"Joint names for {arm_name}: {bot.arm.group_info.joint_names}")

    # print(f"Created and configured robots for arms: {arm_names}")

    rospy.sleep(1)

    # print(f"Moving {args.mode} arms to pose '{pose_name}'")

    for i, arm_name in enumerate(arm_names):
        interpolate_to_pose(
            robots[arm_name],
            arm_name,
            get_pose(arm_name, pose_name),
            blocking=(i == len(arm_names) - 1),
        )

if __name__ == "__main__":
    main()