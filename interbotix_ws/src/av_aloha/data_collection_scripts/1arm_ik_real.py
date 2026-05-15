import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import pyroki as pk
import sys
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks
from viser.extras import ViserUrdf
from yourdfpy import URDF
from webrtc_headset import WebRTCHeadset

from transform_utils import (
    pose2mat,
    transform_coordinates,
    align_rotation_to_z_axis,
    xyzw_to_wxyz,
)

from interbotix_xs_modules.arm import InterbotixManipulatorXS

# from sensor_msgs.msg import JointState
# from interbotix_xs_msgs.msg import JointGroupCommand

# CONFIG

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK  = "leftgripper_base"

def wxyz_from_matrix(R_mat):

    quat_xyzw = (
        R.from_matrix(R_mat)
        .as_quat()
    )

    return np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])


def matrix_from_wxyz(wxyz):

    return R.from_quat([
        wxyz[1],
        wxyz[2],
        wxyz[3],
        wxyz[0],
    ]).as_matrix()

def main():

    # HEADSET
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # ROBOT
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    # ROBOT STATE
    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    # INITIAL ROBOT TARGET
    T_robot_target = np.eye(4)

    T_robot_target[:3,3] = np.array([
        0.1,
        0,
        0.45,
    ])

    initial_rot = R.from_euler(
        "xyz",
        #[0, 90, 0],
        [-90, 0, 90],
        degrees=True,
    )

    quat_xyzw = initial_rot.as_quat()

    R_robot_target = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    teleop_active = False

    start_controller_pose = None
    start_robot_pose = None

    rospy.init_node("vr_left_arm_teleop")

    """

    bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        init_node=False,
    )

    bot.core.robot_set_operating_modes(
        "group",
        "arm",
        "position",
    )

    bot.core.robot_set_motor_registers(
        "group",
        "arm",
        "Profile_Velocity",
        2000,
    )

    bot.core.robot_set_motor_registers(
        "group",
        "arm",
        "Profile_Acceleration",
        300,
    )

    """

    bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=0.2,
        accel_time=0.05,
        init_node=False,
    )

    t = 0.0

    # TELEOP STATE

    # LEFT

    left_start_controller_pose = None
    left_start_robot_pose = None

    left_start_controller_rot = None
    left_start_robot_rot = None

    # TRANSLATION REMAP

    # R_remap = np.array([
    #     [0, 1, 0],
    #     [-1, 0, 0],
    #     [0, 0, 1],
    # ])


    current_joint_state = None

    def joint_state_callback(msg):
        global current_joint_state
        current_joint_state = msg

    while not rospy.is_shutdown():

        start_time = time.time()

        # RECEIVE HEADSET DATA
        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # CURRENT CONTROLLER POSE
        current_controller_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        # BUTTON STATE
        button_pressed = (
            headset_data.r_button_one
        )

        # TELEOP START
        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            # CALIBRATE CONTROLLER FRAME
            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller_right[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller_right[:3,3]
            )

            # SAVE REFERENCE FRAMES
            start_controller_pose = (
                aligned_controller.copy()
            )

            start_robot_pose = (
                T_robot_target.copy()
            )

            start_controller_rot = (
                current_controller_right[:3,:3].copy()
            )

            start_robot_rot = (
                R.from_quat([
                    R_robot_target[1],
                    R_robot_target[2],
                    R_robot_target[3],
                    R_robot_target[0],
                ]).as_matrix()
            )

            print("Teleop ENABLED")

        # TELEOP STOP
        elif ( not button_pressed and teleop_active ):

            teleop_active = False

            print("Teleop DISABLED")

        # RUN TELEOP
        if teleop_active:

            # MAP CONTROLLER MOTION INTO ROBOT TARGET FRAME

            if teleop_active:

                # controller motion since teleop start

                delta = (
                    current_controller_right[:3,3]
                    - start_controller_pose[:3,3]
                )

                # remap controller axes

                # R_remap = np.array([
                #     [0, 1, 0],
                #     [-1, 0, 0],
                #     [0, 0, 1],
                # ])

                # delta = R_remap @ delta

                # apply to robot target

                T_robot_target[:3,3] = np.array([
                    0.15 + 0.05*np.sin(time.time()),
                    0.0,
                    0.45,
                ])

                """ TEMPORARILY COMMENT OUT 
                T_robot_target[:3,3] = (
                    start_robot_pose[:3,3]
                    + delta
                ) """

                """ COMMENTING OUT ROT TO DEBUG TRANS 
                R_delta = (
                    start_controller_rot.T
                    @ current_controller_right[:3,:3]
                )

                controller_rpy = (
                    R.from_matrix(R_delta)
                    .as_euler("xyz", degrees=False)
                )

                rotvec = (
                    R.from_matrix(R_delta)
                    .as_rotvec()
                )

                print("controller semantic rpy")
                print(controller_rpy)

                pitch = rotvec[1]
                yaw   = rotvec[0]
                roll  = rotvec[2]

                # tried 012, 210, 120 (pitch makes sense but is directionally reversed!), 102 (this makes sense but yes pitch still dir rev)

                rotvec_ee = np.array([
                    -pitch,
                    yaw,
                    roll,
                ])

                R_local_delta = (
                    R.from_rotvec(rotvec_ee)
                    .as_matrix()
                )

                R_target = (
                    start_robot_rot
                    # @ R_local_delta
                ) """
                
               # R_target = R_target @ R_offset

                # matrix -> quaternion
                # scipy gives xyzw----

                """ quat_xyzw = (
                    R.from_matrix(R_target)
                    .as_quat()
                )

                R_robot_target = np.array([
                    quat_xyzw[3],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                ])

                R_ee = R.from_quat([
                    ee_wxyz[1],
                    ee_wxyz[2],
                    ee_wxyz[3],
                    ee_wxyz[0],
                ]).as_matrix()

                R_error = (
                    R_target
                    @ R_ee.T
                )

                error_euler = (
                    R.from_matrix(R_error)
                    .as_euler("xyz", degrees=True)
                ) """

                """ controller_x = current_controller_right[:3,0]
                controller_y = current_controller_right[:3,1]
                controller_z = current_controller_right[:3,2]

                ee_x = R_ee[:,0]
                ee_y = R_ee[:,1]
                ee_z = R_ee[:,2]

                print("\nCONTROLLER AXES")
                print("x:", controller_x)
                print("y:", controller_y)
                print("z:", controller_z)

                print("\nEE AXES")
                print("x:", ee_x)
                print("y:", ee_y)
                print("z:", ee_z) """

        print("\n=== TARGET ===")
        print("target pos:", T_robot_target[:3,3])
        print("target quat:", R_robot_target)
        
        # SOLVE IK

        q = np.zeros(23)

        q[0] = np.sin(2*t)

        bot.arm.set_joint_positions(
            q[:6].tolist(),
            blocking=False,
        )

        t += 0.03

        # q_new = pks.solve_ik(
        #     robot=robot,
        #     target_link_name=LEFT_EE_LINK,

        #     # POSITION TARGET

        #     target_position=(
        #         T_robot_target[:3,3]
        #     ),

        #     # TARGET ORIENTATION

        #     target_wxyz=R_robot_target,
        # )

        # # print("\n=== FK RESULT ===")
        # # print("ee pos:", ee_position)
        # # print("ee quat:", ee_wxyz)

        # # UPDATE ROBOT STATE

        # if q_new is not None:
        #     q = q_new

        #     # q = np.array([
        #     #     1.0*np.sin(2*t),
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     #     0,
        #     # ])

        #     bot.arm.set_joint_positions(
        #         q[:6],
        #         blocking=False,
        #     )

        #     """ msg = JointGroupCommand()

        #     msg.name = "arm"

        #     msg.cmd = [
        #         float(q[0]),
        #         float(q[1]),
        #         float(q[2]),
        #         float(q[3]),
        #         float(q[4]),
        #         float(q[5]),
        #     ]

        #     left_arm_pub.publish(msg)"""

        #     fk = robot.forward_kinematics(q)

        #     ee_index = robot.links.names.index(
        #         LEFT_EE_LINK
        #     )

        #     ee_pose = np.array(
        #         fk[ee_index]
        #     )

        #     ee_wxyz = ee_pose[:4]
        #     ee_position = ee_pose[4:]

        #     t += 0.03

        # # UPDATE VISUALIZATION

        # joint_dict = {
        #     name: value
        #     for name, value in zip(
        #         robot.joints.actuated_names,
        #         q,
        #     )
        # }

        # # GRIPPER CONTROL

        # if headset_data.r_hand_trigger > 0.001:
        #     print(f"right trigger is pressed! {headset_data.r_hand_trigger}")

        #     gripper_value = 0.0

        # else:

        #     gripper_value = 0.041

        # joint_dict["rightright_finger"] = gripper_value

        for k, v in vars(headset_data).items():
            if isinstance(v, (int, float)):
                print(k, v)

        time.sleep(0.01)

if __name__ == "__main__":
    main()