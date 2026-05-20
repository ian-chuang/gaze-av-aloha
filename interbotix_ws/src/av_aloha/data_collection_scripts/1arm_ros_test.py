import rospy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from interbotix_xs_modules.arm import InterbotixManipulatorXS

import pyroki as pk
from yourdfpy import URDF

from transform_utils import (
    pose2mat,
    align_rotation_to_z_axis,
)
import jaxlie
import sys
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks

import os

# CONFIG

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "leftgripper_base"


POSITION_SCALE = 0.1

MOVING_TIME = 0.25
CONTROL_DT = MOVING_TIME * 0.5
ACCEL_TIME = 0.08

ALPHA = 0.15

COMMAND_DEADBAND = 0.01

ROT_DEG = 2.0
ROT_RAD = np.deg2rad(ROT_DEG)

# ============================================================
# TRAJECTORY TESTING
# ============================================================

TRAJ_AMPLITUDE = 0.01
TRAJ_FREQUENCY = 0.10

TEST_SEQUENCE = [

    ("x", 3.0),

    ("y", 3.0),

    ("z", 3.0),

    # ("circle_xy", 20.0),

    # ("square_xy", 20.0),

    # ("rot_x_pos", 8.0),
    # ("rot_x_neg", 8.0),

    # ("rot_y_pos", 8.0),
    # ("rot_y_neg", 8.0),

    # ("rot_z_pos", 8.0),
    # ("rot_z_neg", 8.0),
]

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "ros_ik_trajectory_logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True,
)

def compute_trajectory_target(
    mode,
    t,
    amplitude,
    frequency,
    base_euler_deg,
):
    """
    Computes target EE trajectory.

    Returns:
        delta_position: (3,)
        target_rotation: scipy Rotation
    """

    omega = (
        2.0
        * np.pi
        * frequency
    )

    delta = np.zeros(3)

    rotvec = np.zeros(3)

    # ========================================================
    # TRAJECTORY MODES
    # ========================================================

    match mode:

        # ----------------------------------------------------
        # TRANSLATION
        # ----------------------------------------------------

        case "x":

            delta[0] = (
                amplitude
                * np.sin(omega * t)
            )

        case "y":

            delta[1] = (
                amplitude
                * np.sin(omega * t)
            )

        case "z":

            delta[2] = (
                amplitude
                * np.sin(omega * t)
            )

        # ----------------------------------------------------
        # CIRCLE
        # ----------------------------------------------------

        case "circle_xy":

            theta = omega * t

            delta[0] = (
                amplitude
                * np.cos(theta)
            )

            delta[1] = (
                amplitude
                * np.sin(theta)
            )

        # ----------------------------------------------------
        # SQUARE
        # ----------------------------------------------------

        case "square_xy":

            period = (
                1.0 / frequency
            )

            phase = (
                (t % period)
                / period
            )

            a = amplitude

            if phase < 0.25:

                s = phase / 0.25

                delta = np.array([
                    -a + 2*a*s,
                    -a,
                    0.0,
                ])

            elif phase < 0.5:

                s = (
                    (phase - 0.25)
                    / 0.25
                )

                delta = np.array([
                    a,
                    -a + 2*a*s,
                    0.0,
                ])

            elif phase < 0.75:

                s = (
                    (phase - 0.5)
                    / 0.25
                )

                delta = np.array([
                    a - 2*a*s,
                    a,
                    0.0,
                ])

            else:

                s = (
                    (phase - 0.75)
                    / 0.25
                )

                delta = np.array([
                    -a,
                    a - 2*a*s,
                    0.0,
                ])

        # ----------------------------------------------------
        # ORIENTATION TESTS
        # ----------------------------------------------------

        # ----------------------------------------------------
        # ROTATION TESTS
        # ----------------------------------------------------

        case "rot_x_pos":

            rotvec = np.array([
                ROT_RAD,
                0.0,
                0.0,
            ])

        case "rot_x_neg":

            rotvec = np.array([
                -ROT_RAD,
                0.0,
                0.0,
            ])

        case "rot_y_pos":

            rotvec = np.array([
                0.0,
                ROT_RAD,
                0.0,
            ])

        case "rot_y_neg":

            rotvec = np.array([
                0.0,
                -ROT_RAD,
                0.0,
            ])

        case "rot_z_pos":

            rotvec = np.array([
                0.0,
                0.0,
                ROT_RAD,
            ])

        case "rot_z_neg":

            rotvec = np.array([
                0.0,
                0.0,
                -ROT_RAD,
            ])

        case _:

            raise ValueError(
                f"Unknown trajectory mode: {mode}"
            )

    base_rot = R.from_rotvec(
        np.array([
            -np.pi / 2,
            0.0,
            -np.pi / 2,
        ])
    )

    delta_rot = R.from_rotvec(rotvec)

    rot = base_rot * delta_rot

    return delta, rot

def main():

    rospy.init_node("vr_left_arm_teleop")

    # --------------------------------------------------------
    # ROBOT MODEL
    # --------------------------------------------------------

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    LEFT_ARM_NAMES = [
        "leftwaist",
        "leftshoulder",
        "leftelbow",
        "leftforearm_roll",
        "leftwrist_angle",
        "leftwrist_rotate",
    ]

    left_arm_indices = [
        robot.joints.actuated_names.index(name)
        for name in LEFT_ARM_NAMES
    ]

    left_ee_link_index = (
        robot.links.names.index(
            LEFT_EE_LINK
        )
    )

    # --------------------------------------------------------
    # INTERBOTIX
    # --------------------------------------------------------

    bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    print("interbotix joints:")
    print(bot.arm.group_info.joint_names)

    input("\nMove arm to desired startup pose, then press ENTER...\n")

    home_q = np.array(
        bot.arm.get_joint_commands()
    )

    print("\nCaptured HOME_Q:")
    print(home_q)

    HOME_Q = home_q

    # ========================================================
    # MOVE TO KNOWN HOME CONFIG
    # ========================================================

    print("\nMoving to deterministic home configuration...\n")

    bot.arm.set_joint_positions(
        HOME_Q.tolist(),
        moving_time=3.0,
        accel_time=1.0,
        blocking=True,
    )

    time.sleep(1.0)

    # ========================================================
    # INITIAL IK STATE
    # ========================================================

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    q[left_arm_indices] = HOME_Q

    # --------------------------------------------------------
    # INITIAL TARGET
    # --------------------------------------------------------

    T_robot_target = np.eye(4)

    T_robot_target[:3,3] = np.array([
        0.32,
        -0.05,
        0.22,
    ])

    initial_rot = R.from_euler(
        "xyz",
        #[0, 90, 0],
        [-90, 0, -90],
        degrees=True,
    )

    quat_xyzw = initial_rot.as_quat()

    R_robot_target = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    start_robot_pose = (
        T_robot_target.copy()
    )

    print("\nSolving initial IK...\n")

    joint_velocity_limits = np.zeros(
        robot.joints.num_actuated_joints
    )

    joint_velocity_limits[left_arm_indices] = np.array([
        2.0,
        2.0,
        2.0,
        4.0,
        4.0,
        4.0,
    ])

    q_init = pks.solve_trajectory_ik(

        robot=robot,

        target_link_name=LEFT_EE_LINK,

        target_position=(
            T_robot_target[:3,3]
        ),

        target_wxyz=R_robot_target,

        prev_q=q,

        dt=CONTROL_DT,

        joint_velocity_limits=joint_velocity_limits,

        left_arm_indices=left_arm_indices,
    )

    if q_init is None:

        raise RuntimeError(
            "Failed to solve initial IK"
        )

    q = q_init.copy()

    left_arm_q = q[
        left_arm_indices
    ]

    print(
        "moving to IK start config..."
    )

    bot.arm.set_joint_positions(

        left_arm_q.tolist(),

        moving_time=4.0,

        accel_time=1.0,

        blocking=True,
    )

    time.sleep(1.0)
    
    print("\n================================================")
    print("Robot at initial pose.")
    print("Commands:")
    print("  x  -> oscillate along X")
    print("  y  -> oscillate along Y")
    print("  z  -> oscillate along Z")
    print("  rx -> rotate +X")
    print("  ry -> rotate +Y")
    print("  rz -> rotate +Z")
    print("  s  -> stop motion")
    print("  q  -> quit")
    print("================================================\n")

    current_mode = None
    motion_start_time = None

    # --------------------------------------------------------
    # IK STATE
    # --------------------------------------------------------

    """ q = np.zeros(
        robot.joints.num_actuated_joints
    )

    left_arm_q = np.array(
        bot.arm.get_joint_commands()
    )

    q[left_arm_indices] = left_arm_q """

    q = q_init.copy()

    # ========================================================
    # LOGGING
    # ========================================================

    log = {

        "time": [],

        "trajectory_mode": [],

        "target_position": [],

        "actual_position": [],

        "target_quat_wxyz": [],

        "actual_quat_wxyz": [],

        "q": [],

        "dq": [],

        "solve_time": [],

        "target_quat_log": [],

        "actual_quat_log": [],

        "orientation_error_log": [],
    }

    # ========================================================
    # TEST SEQUENCER
    # ========================================================

    global_start_time = time.time()

    try:

        # ========================================================
        # LOOP
        # ========================================================

        while not rospy.is_shutdown():

            loop_start = time.time()

            # ====================================================
            # USER COMMAND
            # ====================================================

            if current_mode is None:

                cmd = input(
                    "\nEnter command: "
                ).strip().lower()

                if cmd == "q":

                    print("\nQuitting.\n")
                    break

                elif cmd == "s":

                    continue

                elif cmd in [
                    "x",
                    "y",
                    "z",
                    "rot_x_pos",
                    "rot_y_pos",
                    "rot_z_pos",
                    "rx",
                    "ry",
                    "rz",
                ]:

                    mode_map = {
                        "rx": "rot_x_pos",
                        "ry": "rot_y_pos",
                        "rz": "rot_z_pos",
                    }

                    current_mode = mode_map.get(
                        cmd,
                        cmd,
                    )

                    motion_start_time = time.time()

                    print(
                        f"\nRunning mode: "
                        f"{current_mode}"
                    )

                else:

                    print(
                        "\nUnknown command.\n"
                    )

                    continue

            test_elapsed = (
                time.time()
                - motion_start_time
            )

            # ====================================================
            # TRAJECTORY TARGET
            # ====================================================

            delta, rot = (
                compute_trajectory_target(
                    mode=current_mode,
                    t=test_elapsed,
                    amplitude=TRAJ_AMPLITUDE,
                    frequency=TRAJ_FREQUENCY,
                    base_euler_deg=[
                        -90,
                        0,
                        -90,
                    ],
                )
            )

            # ----------------------------------------------------
            # TARGET POSITION
            # ----------------------------------------------------

            T_robot_target[:3,3] = (
                start_robot_pose[:3,3]
                + delta
            )

            # ----------------------------------------------------
            # TARGET ORIENTATION
            # ----------------------------------------------------

            quat_xyzw = (
                rot.as_quat()
            )

            R_robot_target = np.array([
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            ])

            joint_velocity_limits = np.zeros(
                robot.joints.num_actuated_joints
            )

            joint_velocity_limits[left_arm_indices] = np.array([
                2.0,
                2.0,
                2.0,
                4.0,
                4.0,
                4.0,
            ])

            # ====================================================
            # IK
            # ====================================================

            ik_start = time.time()

            q_new = pks.solve_trajectory_ik(

                robot=robot,

                target_link_name=LEFT_EE_LINK,

                target_position=(
                    T_robot_target[:3,3]
                ),

                prev_q=q,

                dt=CONTROL_DT,

                joint_velocity_limits=joint_velocity_limits,

                left_arm_indices=left_arm_indices,

                target_wxyz=R_robot_target,
            )

            solve_time = (
                time.time() - ik_start
            )

            # ------------------------------------------------
            # VALID IK
            # ------------------------------------------------

            if q_new is not None:

                dq = (
                    q_new - q
                ) / CONTROL_DT

                if np.any(np.abs(dq) > 10.0):

                    print("\nDQ EXPLOSION DETECTED\n")

                    continue

                q = q_new

                left_arm_q = q[
                    left_arm_indices
                ]

                # ------------------------------------------------
                # SEND COMMAND
                # ------------------------------------------------

                bot.arm.set_joint_positions(
                    left_arm_q.tolist(),
                    moving_time=MOVING_TIME,
                    accel_time=ACCEL_TIME,
                    blocking=False,
                )

                # =================================================
                # FK VALIDATION
                # =================================================

                fk = robot.forward_kinematics(
                    q
                )

                T_actual = jaxlie.SE3(
                    fk[left_ee_link_index]
                )

                # T_actual = robot.forward_kinematics(q)[
                #     left_ee_link_index
                # ]

                actual_position = (
                    T_actual.translation()
                )

                actual_quat_wxyz = (
                    T_actual.rotation()
                    .wxyz
                )

                R_target = R.from_quat([
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                    quat_xyzw[3],
                ])

                R_actual = R.from_quat([
                    actual_quat_wxyz[1],
                    actual_quat_wxyz[2],
                    actual_quat_wxyz[3],
                    actual_quat_wxyz[0],
                ])

                R_err = (
                    R_target.as_matrix()
                    @ R_actual.as_matrix().T
                )

                err_rotvec = (
                    R.from_matrix(R_err)
                    .as_rotvec()
                )

                print(
                    "orientation error rotvec:",
                    np.round(err_rotvec, 3)
                )

                R_target = jaxlie.SO3(
                    R_robot_target
                )

                R_actual = jaxlie.SO3(
                    actual_quat_wxyz
                )

                R_error = (
                    R_target.inverse()
                    @ R_actual
                )

                orientation_error = np.linalg.norm(
                    np.array(
                        R_error.log()
                    )
                )

                # =================================================
                # LOGGING
                # =================================================

                global_t = (
                    time.time()
                    - global_start_time
                )

                log["time"].append(
                    global_t
                )

                log["trajectory_mode"].append(
                    current_mode
                )

                log["target_position"].append(
                    T_robot_target[:3,3].copy()
                )

                log["actual_position"].append(
                    actual_position.copy()
                )

                log["target_quat_wxyz"].append(
                    R_robot_target.copy()
                )

                log["actual_quat_wxyz"].append(
                    actual_quat_wxyz.copy()
                )

                log["q"].append(
                    q.copy()
                )

                log["dq"].append(
                    dq.copy()
                )

                log["solve_time"].append(
                    solve_time
                )

                log["target_quat_log"].append(
                    R_robot_target
                )

                log["actual_quat_log"].append(
                    actual_quat_wxyz
                )

                log["orientation_error_log"].append(
                    orientation_error
                )

                # =================================================
                # PRINT
                # =================================================

                print(
                    f"[{current_mode}]"
                )

                print(
                    "target:",
                    np.round(
                        T_robot_target[:3,3],
                        3,
                    )
                )

                print(
                    "actual:",
                    np.round(
                        actual_position,
                        3,
                    )
                )

                print(
                    "dq:",
                    np.round(
                        dq,
                        3,
                    )
                )

            else:

                print(
                    f"[{current_mode}] IK FAILED"
                )

            # ====================================================
            # NONBLOCKING STOP CHECK
            # ====================================================

            import select
            import sys

            print(
                "\nPress 's' + ENTER to stop motion.\n"
            )

            if select.select([sys.stdin], [], [], 0)[0]:

                stop_cmd = sys.stdin.readline().strip()

                if stop_cmd == "s":

                    print("\nStopping motion.\n")

                    current_mode = None

                    continue

            # ----------------------------------------------------
            # LOOP TIMING
            # ----------------------------------------------------

            elapsed = (
                time.time() - loop_start
            )

            time.sleep(
                max(
                    0,
                    CONTROL_DT - elapsed
                )
            )
    finally:

        print(
            "\nSaving logs...\n"
        )

        np.savez(
            os.path.join(
                LOG_DIR,
                "orientation_debug.npz",
            ),

            time=np.array(
                log["time"]
            ),

            trajectory_mode=np.array(
                log["trajectory_mode"]
            ),

            target_position=np.array(
                log["target_position"]
            ),

            actual_position=np.array(
                log["actual_position"]
            ),

            target_quat_wxyz=np.array(
                log["target_quat_wxyz"]
            ),

            actual_quat_wxyz=np.array(
                log["actual_quat_wxyz"]
            ),

            q=np.array(
                log["q"]
            ),

            dq=np.array(
                log["dq"]
            ),

            solve_time=np.array(
                log["solve_time"]
            ),

            target_quaternion=np.array(
                log["target_quat_log"]
            ),

            actual_quaternion=np.array(
                log["actual_quat_log"]
            ),

            orientation_error=np.array(
                log["orientation_error_log"]
            ),
        )

        print(
            "\nLogs saved.\n"
        )

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
