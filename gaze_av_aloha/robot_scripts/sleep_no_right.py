from interbotix_xs_modules.arm import InterbotixManipulatorXS
from gaze_av_aloha.robot.robot import sleep,sleep_2arms

def main():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    #puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
    puppet_bot_middle = InterbotixManipulatorXS(robot_model="vx300s_7dof", group_name="arm", gripper_name=None, robot_name=f"puppet_middle", init_node=False)

    sleep_2arms(puppet_bot_left, puppet_bot_middle)

if __name__ == '__main__':
    main()