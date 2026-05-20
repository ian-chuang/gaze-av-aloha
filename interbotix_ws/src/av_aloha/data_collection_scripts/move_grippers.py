from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import move_grippers, torque_on
from constants import (
    LEFT_GRIPPER_JOINT_CLOSE,
    LEFT_GRIPPER_JOINT_OPEN,
    RIGHT_GRIPPER_JOINT_CLOSE,
    RIGHT_GRIPPER_JOINT_OPEN,
)
import rospy
from interbotix_xs_msgs.msg import (JointSingleCommand)

def main():
    #puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=True)    
    #puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    #puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    puppet_bot_right.dxl.robot_torque_enable("single", "gripper", True)


    print("Hard coding move")

    cmd = JointSingleCommand(
        name="gripper"
    )

    cmd.cmd = 0.1

    print("OPEN")

    puppet_bot_right.gripper.core.pub_single.publish(
        cmd
    )

    rospy.sleep(3)

    cmd.cmd = -1.7

    print("CLOSE")

    puppet_bot_right.gripper.core.pub_single.publish(
        cmd
    )

    rospy.sleep(3)

    print("move_grippers code")

    #torque_on(puppet_bot_left)
    #torque_on(puppet_bot_right)

    # move_grippers([puppet_bot_right], [RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
    # rospy.sleep(1.0)
    # move_grippers([puppet_bot_right], [RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)

if __name__ == '__main__':
    main()
