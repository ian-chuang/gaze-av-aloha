import rospy
import time

from interbotix_xs_modules.arm import InterbotixManipulatorXS

rospy.init_node("test_arm")

bot = InterbotixManipulatorXS(
    robot_model="wx250s",
    group_name="arm",
    gripper_name="gripper",
    robot_name="puppet_left",
    moving_time=2.0,
    accel_time=0.3,
    init_node=False,
)

time.sleep(1)

print("moving")

bot.arm.set_joint_positions(
    [0.5, 0, 0, 0, 0, 0],
    blocking=True,
)

time.sleep(1)

bot.arm.set_joint_positions(
    [0, 0.5, 0, 0, 0, 0],
    blocking=True,
)

print("done")

time.sleep(5)