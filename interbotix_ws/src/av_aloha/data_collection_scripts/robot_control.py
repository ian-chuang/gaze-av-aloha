"""
Robot creation, configuration, state queries, and motion helpers.
"""
import rospy
import pyroki as pk
import numpy as np

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from yourdfpy import URDF

from arm_config import ARM_CONFIG, DEFAULT_RESET_POSE, POSES, URDF_PATH
from data_col_config import ARM_MODES
from gripper import configure_gripper, command_gripper

REPLAY_MAX_JOINT_STEP_LR = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14], dtype=float)
REPLAY_MAX_JOINT_STEP_M = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14, 0.16], dtype=float)

# Robot creation
def create_robot(arm_name, moving_time=0.14, accel_time=0.04):
    cfg = ARM_CONFIG[arm_name]

    return InterbotixManipulatorXS(
        robot_model=cfg["robot_model"],
        group_name="arm",
        gripper_name="gripper" if cfg["has_gripper"] else None,
        robot_name=cfg["robot_name"],
        moving_time=moving_time,
        accel_time=accel_time,
        init_node=False,
    )
    

def create_and_configure_robot(arm_name, moving_time=0.14, accel_time=0.04):
    # print(f"In function call create_and_configure_robot with arm_name: {arm_name}")
    bot = create_robot(arm_name, moving_time, accel_time)

    if arm_name == "middle":
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
        rospy.sleep(0.5)
        bot.dxl.robot_torque_enable("group", "arm", True)
        rospy.sleep(0.5)

    # print(f"Created robot for arm: {arm_name}")
    if ARM_CONFIG[arm_name]["has_gripper"]:
        # print(f"Configuring gripper for arm: {arm_name}")
        configure_gripper(bot, ARM_CONFIG[arm_name]["robot_name"])
    # print(f"Robot for arm {arm_name} created and configured.")
    return bot


def create_and_configure_robots(arm_names=("left", "right", "middle")):
    # print(f"In function call create_and_configure_robots with arm_names: {arm_names}")
    return {arm_name: create_and_configure_robot(arm_name) for arm_name in arm_names}


# State query helper: get current joint positions as a NumPy array.
def get_joint_positions(bot):
    return np.array(bot.arm.core.joint_states.position[:len(bot.arm.group_info.joint_names)], 
                    dtype=float)

def sync_robot_state(robots,
    robot,
    arm_data,
    arm_names,
    cmd_kin,
    cmd_state,
):
    """
    Synchronize software command state with the robots' measured joint states.

    Call this after any motion that occurs outside the normal teleoperation
    command loop, such as resetting the arms or moving to a named pose.
    """
    measured_q_full = cmd_kin.q_cmd.copy()

    for arm in arm_names:
        joint_idx = arm_data[arm]["joint_indices"]
        num_joints = ARM_CONFIG[arm]["num_joints"]

        measured_q_arm = np.asarray(
            robots[arm].dxl.joint_states.position[:num_joints],
            dtype=float,
        ).copy()

        if measured_q_arm.shape[0] != len(joint_idx):
            raise RuntimeError(
                f"{arm}: measured {measured_q_arm.shape[0]} joints, "
                f"but robot model expects {len(joint_idx)}."
            )

        measured_q_full[joint_idx] = measured_q_arm
        cmd_state.last_cmds[arm] = measured_q_arm.copy()

        # Permit the next command immediately.
        cmd_state.last_arm_cmd_time[arm] = 0.0

    cmd_kin.q_cmd[:] = measured_q_full

    # Recompute end-effector poses from the synchronized joint configuration.
    _, measured_ee = compute_fk_and_ee(
        robot,
        cmd_kin.q_cmd,
        arm_data,
    )

    for arm in arm_names:
        cmd_kin.T_cmd[arm] = measured_ee[arm].copy()

def command_state_is_stale(
    robots,
    arm_data,
    arm_names,
    cmd_state,
    tolerance=0.08,
):
    """
    Return True when measured joints differ substantially from the command
    state stored by the teleoperation loop.
    """
    for arm in arm_names:
        num_joints = ARM_CONFIG[arm]["num_joints"]

        measured_q = np.asarray(
            robots[arm].dxl.joint_states.position[:num_joints],
            dtype=float,
        )

        expected_q = np.asarray(
            cmd_state.last_cmds[arm],
            dtype=float,
        )

        max_error = float(np.max(np.abs(measured_q - expected_q)))

        if max_error > tolerance:
            print(
                f"\n[{arm}] Command state is stale: "
                f"maximum joint difference = {max_error:.3f} rad."
            )
            return True

    return False

# Motion

# Moves all arms simultaneously but does  not properly break the motion into segments
# def interpolate_to_pose(bot, pose, moving_time=3.0, accel_time=1.5, blocking=True):
#     current_q = get_joint_positions(bot)
#     target_q = np.asarray(pose, dtype=float)
#     delta = np.abs(target_q - current_q)
#     num_steps = max(1, int(np.ceil(np.max(delta / REPLAY_MAX_JOINT_STEP))))
#     waypoints = np.linspace(current_q, target_q, num_steps + 1)[1:]
#     segment_time = max(0.25, moving_time / num_steps)
#     segment_accel = min(accel_time / num_steps, 0.5 * segment_time)

#     print(f"max_delta={np.max(delta):.3f}, num_steps={num_steps}, segment_time={segment_time:.3f}")

#     for i, q in enumerate(waypoints):
#         bot.arm.set_joint_positions(q.tolist(), moving_time=segment_time, accel_time=segment_accel, blocking=(blocking and i == len(waypoints) - 1))

def interpolate_to_pose(bot, arm, pose, moving_time=0.2, accel_time=0.1, blocking=True):

    # # print("--------------------------------")
    # # print(bot.arm.group_info.joint_names)
    # # print(bot.arm.core.joint_states.name)
    # # print("--------------------------------")

    # current_q = get_joint_positions(bot)
    # target_q = np.asarray(pose, dtype=float)
    # delta = np.abs(target_q - current_q)

    # # print("interpolate_to_pose: current_q =", current_q)
    # # print("interpolate_to_pose: target_q =", target_q)

    # num_steps = max(1, int(np.ceil(np.max(delta / REPLAY_MAX_JOINT_STEP))))

    current_q = np.asarray(bot.dxl.joint_states.position[:len(pose)], dtype=float)
    target_q = np.asarray(pose, dtype=float)
    # print("current:", np.round(current_q, 3))
    # print("target :", np.round(target_q, 3))
    # print("delta  :", np.round(target_q - current_q, 3))
    delta = np.abs(target_q - current_q)

    if arm == "middle":
        max_joint_step = REPLAY_MAX_JOINT_STEP_M
    else:
        max_joint_step = REPLAY_MAX_JOINT_STEP_LR

    num_steps = max(1, int(np.ceil(np.max(delta / max_joint_step))))
    waypoints = np.linspace(current_q, target_q, num_steps + 1)[1:]

    print(f"max_delta={np.max(delta):.3f}, num_steps={num_steps}, total_time={num_steps * moving_time:.1f}")

    for i, q in enumerate(waypoints):
        # print(bot.arm.group_info.joint_names)
        bot.arm.set_joint_positions(
            q.tolist(),
            moving_time=moving_time,
            accel_time=min(accel_time, 0.5 * moving_time),
            blocking=True,
        )

def move_to_named_pose(bot, arm_name, pose_name, moving_time=0.2, accel_time=0.1, blocking=True):
    interpolate_to_pose(bot, arm_name, get_pose(arm_name, pose_name), moving_time, accel_time, blocking)

def reset_arm(bot, arm_name, pose_name=DEFAULT_RESET_POSE):
    move_to_named_pose(bot, arm_name, pose_name)

def reset_arms(robots, pose_name=DEFAULT_RESET_POSE):
    for arm_name, bot in robots.items():
        reset_arm(bot, arm_name, pose_name)

def stop_arm(bot):
    bot.arm.set_joint_positions(
        get_joint_positions(bot).tolist(),
        moving_time=0.05,
        accel_time=0.02,
        blocking=False,
    )

def stop_robots(robots):
    for bot in robots.values():
        stop_arm(bot)

# Set gripper position directly.
def set_gripper(bot, position):
    if not hasattr(bot, "gripper"):
        return

    command_gripper(bot, float(position))

# Send a joint command during dataset replay. 
# Returns the commanded joint vector so replay.py can compute tracking error.
def replay_arm_command(bot, target_q, moving_time=0.03, accel_time=0.01):
    target_q = np.asarray(target_q, dtype=float)

    bot.arm.set_joint_positions(
        target_q.tolist(),
        moving_time=moving_time,
        accel_time=accel_time,
        blocking=False,
    )

    return target_q

# -----------------------------------------------------------------------------
# Torque
# -----------------------------------------------------------------------------

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)

    if hasattr(bot, "gripper"):
        bot.dxl.robot_torque_enable("single", "gripper", True)


def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)

    if hasattr(bot, "gripper"):
        bot.dxl.robot_torque_enable("single", "gripper", False)

def safe_move_arm_joints(bot, target_q, total_time=3.0, step_time=0.25, accel_ratio=0.35):
    n = len(bot.arm.group_info.joint_names)
    current_q = np.array(bot.dxl.joint_states.position[:n], dtype=float)
    target_q = np.array(target_q, dtype=float)
    max_step = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

    delta = target_q - current_q
    n_steps = int(np.ceil(np.max(np.abs(delta) / max_step)))
    n_steps = max(n_steps, int(np.ceil(total_time / step_time)), 1)

    waypoints = np.linspace(current_q, target_q, n_steps + 1)[1:]
    move_t = step_time
    accel_t = min(accel_ratio * move_t, 0.5 * move_t)

    for q_cmd in waypoints:
        bot.arm.set_joint_positions(q_cmd.tolist(), moving_time=move_t, accel_time=accel_t, blocking=True)

def build_robot_model(mode_idx):
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    # print("Actuated joints in URDF:")
    # print(robot.joints.actuated_names)

    arm_data = {}

    for arm in ARM_MODES[mode_idx]:
        cfg = ARM_CONFIG[arm]

        # print(f"\nChecking arm: {arm}")
        # print("Expected joints:")
        # print(cfg["joint_names"])

        arm_data[arm] = {
            "joint_indices": [
                robot.joints.actuated_names.index(name)
                for name in cfg["joint_names"]
            ],
            "ee_index": robot.links.names.index(cfg["ee_link"]),
        }

    return robot, arm_data

# def build_robot_model(mode_idx):
#     urdf = URDF.load(URDF_PATH)
#     robot = pk.Robot.from_urdf(urdf)

#     arm_data = {}

#     for arm in ARM_MODES[mode_idx]:
#         cfg = ARM_CONFIG[arm]

#         arm_data[arm] = {
#             "joint_indices": [
#                 robot.joints.actuated_names.index(name)
#                 for name in cfg["joint_names"]
#             ],
#             "ee_index": robot.links.names.index(
#                 cfg["ee_link"]
#             ),
#         }

#     return robot, arm_data

# Return a named pose for an arm; raises ValueError if not defined.
def get_pose(arm_name, pose_name):
    if pose_name not in POSES[arm_name]:
        raise ValueError(
            f"Pose '{pose_name}' not defined for arm '{arm_name}'"
        )

    return POSES[arm_name][pose_name]

def compute_fk_and_ee(robot, q, arm_data):
    fk = robot.forward_kinematics(q)
    ee_positions = {arm: fk[arm_data[arm]["ee_index"]] for arm in arm_data}
    return fk, ee_positions