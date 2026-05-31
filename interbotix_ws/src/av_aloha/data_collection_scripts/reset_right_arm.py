import numpy as np
import rospy

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

GRIPPER_CURRENT_LIMIT = 200
# RIGHT_RESET_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
# RIGHT_RESET_Q = np.array([0.1, -0.6, 0.1, 0.0, 1.5, 0.1], dtype=float) # new reset pose looking down higher up
RIGHT_RESET_Q = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float) # new reset pose

GRIPPER_OPEN = 0.1
GRIPPER_CLOSED = -1.7

def set_register(robot_name, motor_name, reg_name, value):
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value

    return srv(req)

def main():
    rospy.init_node("reset_right_arm", anonymous=True)

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=4.0,
        accel_time=1.5,
        init_node=False,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)

    set_register(
        "puppet_right",
        "gripper",
        "Current_Limit",
        GRIPPER_CURRENT_LIMIT,
    )

    right_bot.dxl.robot_set_operating_modes(
        "single",
        "gripper",
        "current_based_position",
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    print("Moving right arm to reset pose...")
    right_bot.arm.set_joint_positions(
        RIGHT_RESET_Q.tolist(),
        moving_time=4.0,
        accel_time=1.5,
        blocking=True,
    )

    rospy.sleep(1)

    # cmd = JointSingleCommand(name="gripper")
    # cmd.cmd = GRIPPER_OPEN
    # right_bot.gripper.core.pub_single.publish(cmd)

    # rospy.sleep(0.5)

    # cmd = JointSingleCommand(name="gripper")
    # cmd.cmd = GRIPPER_CLOSED
    # right_bot.gripper.core.pub_single.publish(cmd)\
    
    # rospy.sleep(0.5)

    # cmd = JointSingleCommand(name="gripper")
    # cmd.cmd = GRIPPER_OPEN
    # right_bot.gripper.core.pub_single.publish(cmd)

    rospy.sleep(1.0)
    print("Done.")


if __name__ == "__main__":
    main()