"""
Robot creation, configuration, state queries, and motion helpers.
"""
import warnings

# ROS noetic ships python2-era docstrings ("\p", "\*") that python 3.12
# flags as SyntaxWarning on every run (system dirs are root-owned, so the
# bytecode is never cached).  Harmless — silence just that warning, before
# the interbotix/actionlib/tf import chain below compiles them.
warnings.filterwarnings("ignore", message=r"invalid escape sequence",
                        category=SyntaxWarning)

import numpy as np
try:
    import rospy
except ImportError:
    rospy = None

try:
    import pyroki as pk
except ImportError:
    pk = None

try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
except ImportError:
    InterbotixManipulatorXS = None

try:
    from yourdfpy import URDF
except ImportError:
    URDF = None

if __package__:
    from .arm_config import ARM_CONFIG, DEFAULT_RESET_POSE, POSES, URDF_PATH
    from .data_col_config import ARM_MODES
    from .gripper import configure_gripper, command_gripper
else:
    from arm_config import ARM_CONFIG, DEFAULT_RESET_POSE, POSES, URDF_PATH
    from data_col_config import ARM_MODES
    from gripper import configure_gripper, command_gripper

REPLAY_MAX_JOINT_STEP_LR = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14], dtype=float)
REPLAY_MAX_JOINT_STEP_M = np.array([0.06, 0.06, 0.08, 0.12, 0.12, 0.14, 0.16], dtype=float)

# Robot creation
def create_robot(arm_name, moving_time=0.14, accel_time=0.04):
    if InterbotixManipulatorXS is None:
        raise ImportError("interbotix_xs_modules is required to create robots.")
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
        if rospy is None:
            raise ImportError("rospy is required to configure the middle arm.")
        # Torque on only -- do NOT set operating modes here.
        #
        # robot_set_operating_modes torques EVERY motor off to write the
        # EEPROM mode registers, then torques back on (xs_sdk_obj.cpp,
        # robot_set_joint_operating_mode: "torqued off").  Under gravity that
        # is the visible ~1 cm sag the middle arm showed at every program
        # start.  Modes are now set once at launch by puppet_modes_middle.yaml
        # (same as left/right), so the per-run write was redundant EEPROM wear
        # plus a mechanical glitch.  Torque-enable alone is idempotent: writing
        # 1 to an already-torqued motor causes no blip.
        bot.dxl.robot_torque_enable("group", "arm", True)
        rospy.sleep(0.2)

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
    to_urdf=None,
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

        # Reset the DRIVER's own command reference too.
        #
        # interbotix validates every command against `arm.joint_commands` --
        # its last *accepted* command -- not against the measured position:
        #     speed = |goal - joint_commands| / moving_time  >  velocity_limit
        # When a command is rejected, `joint_commands` is never updated, so it
        # stays frozen wherever it was.  Resyncing only our own bookkeeping
        # therefore does nothing: the driver keeps measuring every new goal
        # against a stale reference that may be radians away, rejects it, and
        # the arm is stuck permanently.
        #
        # interbotix's own `capture_joint_positions()` does this, but it is dead
        # code in this workspace: `import modern_robotics as mr` is commented
        # out at the top of arm.py (as is `self.robot_des`), so its final
        # `mr.FKinSpace(...)` line raises NameError.  We therefore set
        # `joint_commands` directly -- that FK line only maintains `T_sb`,
        # which nothing in this pipeline reads.
        arm_iface = robots[arm].arm
        try:
            core = arm_iface.core
            arm_iface.joint_commands = [
                core.joint_states.position[core.js_index_map[name]]
                for name in arm_iface.group_info.joint_names
            ]
        except Exception as exc:  # never let bookkeeping kill the session
            print(f"[{arm}] could not reset driver joint_commands: {exc}")

        # Permit the next command immediately.
        cmd_state.last_arm_cmd_time[arm] = 0.0

    cmd_kin.q_cmd[:] = measured_q_full

    # Recompute end-effector poses from the synchronized joint configuration.
    #
    # q_cmd is in DRIVER coordinates, and the middle arm's waist is offset by pi
    # between the driver and URDF frames.  Forward kinematics must therefore run
    # on the converted vector -- the main control loop already does this via
    # coupled_ik.driver_to_urdf().  Without it, T_cmd["middle"] lands pi away
    # from where the camera arm actually is, so the first teleop command after a
    # reset or a stale-state resync is computed against a bogus start pose.
    q_for_fk = cmd_kin.q_cmd if to_urdf is None else to_urdf(cmd_kin.q_cmd)
    _, measured_ee = compute_fk_and_ee(
        robot,
        q_for_fk,
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

    # SAFETY GUARD (wrapped multiturn encoder): after a power cycle a
    # multiturn joint (the middle waist, driver range [-6.35, 0]) can report
    # its position wrapped into [-pi, pi] (e.g. physical -3.22 reads +3.06).
    # Interpolating from the fictitious reading would command a full physical
    # revolution and wind the cables.  If the reading is outside the driver's
    # own limits, refuse to move this arm and tell the operator to re-home.
    # A joint slightly past its soft limit (arm sagging in the cradle) is
    # normal and safe to move FROM — interpolation just pulls it back in
    # range.  Only a large excursion (comparable to a turn) means the
    # encoder actually wrapped and moving would wind the cables.
    WRAP_GUARD_MARGIN = 0.5  # rad
    try:
        lower = np.asarray(bot.arm.group_info.joint_lower_limits, dtype=float)
        upper = np.asarray(bot.arm.group_info.joint_upper_limits, dtype=float)
        excess = np.maximum(lower - current_q, current_q - upper)
        slight = (excess > 1e-3) & (excess <= WRAP_GUARD_MARGIN)
        wrapped = excess > WRAP_GUARD_MARGIN
        names = list(bot.arm.group_info.joint_names)
        for i in np.nonzero(slight)[0]:
            print(f"[startup] {arm}: joint '{names[i]}' reads "
                  f"{current_q[i]:+.3f}, {excess[i]:.3f} past driver limits "
                  f"[{lower[i]:+.3f}, {upper[i]:+.3f}] — small excursion, "
                  "moving back into range.")
        if wrapped.any():
            for i in np.nonzero(wrapped)[0]:
                print(f"[SAFETY] {arm}: joint '{names[i]}' reads "
                      f"{current_q[i]:+.3f}, {excess[i]:.3f} rad outside "
                      f"driver limits [{lower[i]:+.3f}, {upper[i]:+.3f}] — "
                      "encoder likely wrapped.")
            print(f"[SAFETY] {arm}: SKIPPING startup move. Re-home / "
                  "power-cycle the servo at an in-range orientation, then "
                  "restart.")
            return
    except AttributeError:
        pass  # group_info without limit fields: proceed as before

    # FRAME-PROOF startup for the multiturn middle waist: the driver frame
    # can boot 2pi-shifted, so move to the 2pi-equivalent of the target
    # nearest to the current reading instead of sweeping a full turn.
    if arm == "middle":
        k = np.round((current_q[0] - target_q[0]) / (2 * np.pi))
        if k != 0:
            shifted = target_q[0] + 2 * np.pi * k
            print(f"[frame] middle waist target {target_q[0]:+.3f} -> "
                  f"{shifted:+.3f} (nearest 2pi-equivalent to current "
                  f"{current_q[0]:+.3f})")
            target_q = target_q.copy()
            target_q[0] = shifted
        delta = np.abs(target_q - current_q)
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
    """Halt an arm in place. Torque stays on, so nothing drops.

    moving_time is sized from the ACTUAL distance rather than fixed at
    0.05 s.  interbotix validates every command (arm.py:check_joint_limits)
    as

        speed = |goal - self.joint_commands| / moving_time
        if speed > joint_velocity_limits: return False

    where joint_commands is the driver's last ACCEPTED command, not the
    measured position -- so mid-motion that difference is the whole
    remaining travel.  A fixed 0.05 s therefore asks for 4 rad/s to cancel
    0.2 rad, gets refused against the 3.14 rad/s limit, and
    set_joint_positions returns False rather than raising.  Nothing checked
    that return value, so the stop silently did nothing in exactly the case
    it was needed: a fast, large motion.

    Sizing moving_time to the distance keeps every halt inside the limit.
    The return value is checked, and the command escalated, so a refusal
    can no longer pass unnoticed."""
    n = len(bot.arm.group_info.joint_names)
    measured = np.asarray(bot.arm.core.joint_states.position[:n], dtype=float)

    ref = measured
    getter = getattr(bot.arm, "get_joint_commands", None)
    if getter is not None:
        try:
            ref = np.asarray(getter(), dtype=float)[:n]
        except Exception:
            pass

    try:
        vel = np.asarray(bot.arm.group_info.joint_velocity_limits,
                         dtype=float)[:n]
        vel = np.where(vel > 1e-6, vel, np.pi)
    except Exception:
        vel = np.full(n, np.pi)

    # 0.7 of the limit leaves room for the driver's rounding to 3 decimals.
    moving_time = max(
        0.05,
        float(np.max(np.abs(measured - ref) / np.maximum(0.7 * vel, 1e-6))))

    for _ in range(4):
        if bot.arm.set_joint_positions(
                measured.tolist(), moving_time=moving_time,
                accel_time=min(0.02, 0.5 * moving_time), blocking=False):
            return True
        moving_time *= 2.0

    print("[SAFETY] stop_arm: the driver refused every halt command. "
          "Kill the roslaunch or use the physical power switch.")
    return False

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
    if URDF is None or pk is None:
        raise ImportError("yourdfpy and pyroki are required to build the robot model.")
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

        joint_indices = [
            robot.joints.actuated_names.index(name)
            for name in cfg["joint_names"]
        ]
        # Per-arm position limits, used to keep commands inside what the driver
        # will accept (it rejects the whole group command otherwise).
        arm_data[arm] = {
            "joint_indices": joint_indices,
            "ee_index": robot.links.names.index(cfg["ee_link"]),
            "lower_limits": np.asarray(robot.joints.lower_limits, dtype=float)[joint_indices],
            "upper_limits": np.asarray(robot.joints.upper_limits, dtype=float)[joint_indices],
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

# Middle-waist frame shift, in radians, mirroring the servo's Homing_Offset
# register (reported = actual + offset).  The pose tables in arm_config.py
# store LEGACY driver values recorded with offset 0; when a Homing_Offset has
# been written (set_waist_homing_offset.py), every commanded waist value must
# shift by the same amount.  data_collection reads the register at startup and
# calls set_middle_waist_shift(); 0.0 keeps historical behavior exactly.
MIDDLE_WAIST_DRIVER_SHIFT = 0.0


def set_middle_waist_shift(shift_rad):
    global MIDDLE_WAIST_DRIVER_SHIFT
    MIDDLE_WAIST_DRIVER_SHIFT = float(shift_rad)
    if abs(MIDDLE_WAIST_DRIVER_SHIFT) > 1e-9:
        print(f"[frame] middle waist Homing_Offset shift "
              f"{MIDDLE_WAIST_DRIVER_SHIFT:+.3f} rad -- pose tables and "
              "URDF offset adjusted to match")


def read_middle_waist_shift(bot):
    """Read the waist Homing_Offset from the servo, in radians (0.0 if unset)."""
    try:
        resp = bot.dxl.robot_get_motor_registers("single", "waist", "Homing_Offset")
        ticks = int(resp.values[0]) if resp.values else 0
        if ticks >= (1 << 31):
            ticks -= 1 << 32
        return ticks * 2.0 * np.pi / 4096.0
    except Exception as exc:
        print(f"[frame] could not read waist Homing_Offset ({exc}); assuming 0")
        return 0.0


# Return a named pose for an arm; raises ValueError if not defined.
def get_pose(arm_name, pose_name):
    if pose_name not in POSES[arm_name]:
        raise ValueError(
            f"Pose '{pose_name}' not defined for arm '{arm_name}'"
        )

    pose = POSES[arm_name][pose_name]
    if arm_name == "middle" and abs(MIDDLE_WAIST_DRIVER_SHIFT) > 1e-9:
        pose = np.asarray(pose, dtype=float).copy()
        pose[0] += MIDDLE_WAIST_DRIVER_SHIFT  # waist is joint 0
    return pose

def compute_fk_and_ee(robot, q, arm_data):
    fk = robot.forward_kinematics(q)
    ee_positions = {arm: fk[arm_data[arm]["ee_index"]] for arm in arm_data}
    return fk, ee_positions
