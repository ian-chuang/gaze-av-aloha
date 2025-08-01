from interbotix_xs_modules.arm import InterbotixManipulatorXS
from gaze_av_aloha.robot.robot import get_arm_gripper_positions, torque_off
import rospy

def main():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)    
    torque_off(puppet_bot_left)
    torque_off(puppet_bot_right)
    while not rospy.is_shutdown():
        print(f"left gripper pos: {get_arm_gripper_positions(puppet_bot_left)}, right gripper pos: {get_arm_gripper_positions(puppet_bot_right)}")
        rospy.sleep(0.2)

if __name__ == '__main__':
    main()