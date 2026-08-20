import os

# The coupled IK study runs the solver on CPU (measured faster than GPU at
# this problem size, ~5 ms/solve).  Must be set before jax is imported
# anywhere (jaxlie below pulls it in).  Override by exporting JAX_PLATFORMS.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
import threading

## Collision model selection MUST happen before study_ik and capsule_gate
## are imported below: both read their configuration from the environment
## at module import, so setting it later would silently do nothing while
## the startup banner claimed otherwise. See collision_modes.py.
try:
    from .collision_modes import banner as _collision_banner
    from .collision_modes import select as _collision_select
except ImportError:
    from collision_modes import banner as _collision_banner
    from collision_modes import select as _collision_select
COLLISION_MODE = _collision_select()
from pathlib import Path

import numpy as np

# Diagnostic switches (cheap when off).
LOG_CLEARANCE = os.environ.get("GIAVA_LOG_CLEARANCE", "0") == "1"
DEBUG_HEAD = os.environ.get("GIAVA_DEBUG_HEAD", "0") == "1"
DEBUG_HANDS = os.environ.get("GIAVA_DEBUG_HANDS", "0") == "1"
TRACK_LOG = os.environ.get("GIAVA_TRACK_LOG", "0") == "1"
# Middle-arm activation: "y" (default, dedicated Y button) or "either"
# (middle follows whenever X or A is held -- the original coupled behavior).
MIDDLE_BUTTON = os.environ.get("GIAVA_MIDDLE_BUTTON", "y")
# Default OFF (2026-08, revised): on-hardware testing with the debug tool
# showed the gripper arms HIGHLY affected by head motion with composition ON,
# and behaving correctly with raw streams.  (An earlier stand-up test pointed
# the opposite way -- the evidence conflicts, so the switch remains:
# GIAVA_HANDS_HEAD_RELATIVE=1 restores composition if arms ever track head
# height again.)
HANDS_HEAD_RELATIVE = os.environ.get("GIAVA_HANDS_HEAD_RELATIVE", "0") == "1"

import torch
from scipy.spatial.transform import Rotation as R

for ros_path in (
    Path("/opt/ros/noetic/lib/python3/dist-packages"),
    Path("/home/devi/giava/interbotix_ws/devel/lib/python3/dist-packages"),
):
    if ros_path.is_dir():
        ros_path_str = str(ros_path)
        if ros_path_str not in sys.path:
            sys.path.append(ros_path_str)

try:
    import jaxlie
except ImportError:
    jaxlie = None
try:
    import rospy
except ImportError:
    rospy = None

if __package__:
    from .webrtc_headset import WebRTCHeadset
    from .arm_config import ARM_CONFIG, URDF_PATH
    try:
        from .three_arm_ik import make_three_arm_ik_solver
    except ImportError:
        make_three_arm_ik_solver = None
    try:
        from .study_ik import COLLISION_MARGIN, CoupledStudyIK
    except ImportError:
        CoupledStudyIK = None
    from .camera_manager import (
        adjust_eye_view,
        SYNC_FRAMES,
        select_synchronized_frames,
        CameraConfig,
        CAMERA_SERIALS,
        CAMERA_INTRINSICS,
        REQUIRE_CALIBRATION,
        save_camera_intrinsics,
        setup_cameras,
        get_active_cameras,
        digital_zoom,
    )
    from .robot_control import (
        create_and_configure_robots,
        build_robot_model,
        move_to_named_pose,
        read_middle_waist_shift,
        reset_arms,
        set_middle_waist_shift,
        compute_fk_and_ee,
        sync_robot_state,
        command_state_is_stale
    )
    from .gripper import update_gripper
    from .dataset import (
        BackgroundEpisodeSaver,
        create_dataset,
        build_frame,
    )
    from .data_col_config import (
        anchor_arm_state,
        session_yaw_remap,
        TASKS,
        ARM_MODES,
        ArmTeleopState,
        TeleopConfig,
        TeleopSessionState,
        RobotCommandState,
        CommandKinematicsState,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
        solve_single_arm_ik,
        clamp_joint_step,
    )
    from .log import (
        SessionStats,
        reset_episode_log,
        log_episode_info,
    )
else:
    from webrtc_headset import WebRTCHeadset

    from arm_config import ARM_CONFIG, URDF_PATH
    try:
        from three_arm_ik import make_three_arm_ik_solver
    except ImportError:
        make_three_arm_ik_solver = None
    try:
        from study_ik import COLLISION_MARGIN, CoupledStudyIK
    except ImportError:
        CoupledStudyIK = None
    from camera_manager import (
        adjust_eye_view,
        SYNC_FRAMES,
        select_synchronized_frames,
        CameraConfig,
        CAMERA_SERIALS,
        CAMERA_INTRINSICS,
        REQUIRE_CALIBRATION,
        save_camera_intrinsics,
        setup_cameras,
        get_active_cameras,
        digital_zoom,
    )
    from robot_control import (
        create_and_configure_robots,
        build_robot_model,
        move_to_named_pose,
        read_middle_waist_shift,
        reset_arms,
        set_middle_waist_shift,
        compute_fk_and_ee,
        sync_robot_state,
        command_state_is_stale
    )
    from gripper import update_gripper
    from dataset import (
        BackgroundEpisodeSaver,
        create_dataset,
        build_frame,
    )
    from data_col_config import (
        anchor_arm_state,
        session_yaw_remap,
        TASKS,
        ARM_MODES,
        ArmTeleopState,
        TeleopConfig,
        TeleopSessionState,
        RobotCommandState,
        CommandKinematicsState,
        compute_camera_arm_target,
        compute_gripper_arm_target,
        start_teleop_session,
        stop_teleop_session,
        solve_single_arm_ik,
        clamp_joint_step,
    )
    from log import (
        SessionStats,
        reset_episode_log,
        log_episode_info,
    )


# quat2mat / pose2mat come from transform_utils (same math; one home).
try:
    from .transform_utils import pose2mat, quat2mat  # noqa: F401
except ImportError:
    from transform_utils import pose2mat, quat2mat  # noqa: F401

latest_frames = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
latest_timestamps = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
frame_lock = threading.Lock()
camera_shutdown = threading.Event()
latest_key = None

camera_config = CameraConfig(
    top_active=True,
    low_active=True,
)

def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()

def select_mode():
    mode_names = list(ARM_MODES.keys())

    print("\nAVAILABLE MODES\n")

    for mode, arm_names in enumerate(mode_names):
        print(f"{mode}: {arm_names}")

    while True:
        try:
            idx = int(input("\nSelect mode: "))
            return mode_names[idx]
        except:
            print("Invalid selection")

def now():
    return time.monotonic()

def mark_episode_discarded(dataset_root, episode_idx):
    discard_file = dataset_root / "discarded_episodes.txt"
    with open(discard_file, "a") as f:
        f.write(f"{episode_idx}\n")

def safe_shutdown(robots, pipelines, dataset, episode_saver, collecting_episode, dataset_root, episode_idx):
    """Minimal safe shutdown helper to avoid undefined-symbol crashes."""
    try:
        if collecting_episode and dataset is not None:
            # CHANGED: queue the in-progress episode so shutdown can wait on the
            # single background writer instead of blocking the control loop first.
            episode_saver.save_episode_async()
            collecting_episode = False
    except Exception:
        pass

    try:
        # CHANGED: make shutdown wait until all queued episode saves finish.
        if episode_saver is not None:
            episode_saver.close()
    except Exception:
        pass

    try:
        # try to reset all robots to rest
        reset_arms(robots)
    except Exception:
        pass

    # close/cleanup pipelines if they expose stop/close (best-effort)
    try:
        for p in pipelines.values():
            stop = getattr(p, "stop", None)
            if callable(stop):
                stop()
    except Exception:
        pass

    return episode_idx, True

def main():
    if rospy is None:
        raise ImportError("rospy is required for data collection.")

    rospy.init_node("data_collection")

    global latest_key
    collecting_episode = False
    cfg = TeleopConfig()

    stats = SessionStats()

    # # Print tasks and get selection
    # print("\nAVAILABLE TASKS:\n")
    # for idx, name in TASKS.items():
    #     print(f"{idx}: {name}")

    # while True:
    #     try:
    #         task_idx = int(input("\nSelect task number: ").strip())
    #         if task_idx not in TASKS:
    #             print("Invalid task number. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter an integer task number.")

    task_name = TASKS[7]  # active_vision_data_collection — coupled-IK study deployment. Hardcoded for now.
    print(f"\nSelected task: {task_name}")

    # # Print modes and get selection
    # print("\nAVAILABLE MODES:\n")
    # for mode, arm_names in ARM_MODES.items():
    #     print(f"{mode}: {arm_names}")

    # while True:
    #     try:
    #         mode = input("\nSelect mode: ").strip()
    #         if mode not in ARM_MODES:
    #             print("Invalid mode. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter one of the listed modes.")

    ## --mode selects which arms are active.  Default "av" (all three) is
    ## the coupled-IK study deployment.  Consumed from argv the same way
    ## --collision is, so the positional episode index still works.
    try:
        from .collision_modes import take_option as _take_option
    except ImportError:
        from collision_modes import take_option as _take_option
    mode = str(_take_option("mode", default="av")).strip().lower()
    if mode not in ARM_MODES:
        raise SystemExit(
            f"--mode must be one of {sorted(ARM_MODES)}, got '{mode}'")

    # # Print modes and get selection
    # print("\nWhich scene cameras to activate?\nEnter l for low and t for top (e.g. 'lt' for both, 'l' for low only, 't' for top only):")

    # while True:
    #     try:
    #         camera_selection = input().strip()
    #         if not camera_selection:
    #             print("Invalid selection. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter a valid selection.")

    # camera_config = CameraConfig(
    #     top_active="t" in camera_selection,
    #     low_active="l" in camera_selection,
    # )

    camera_config = CameraConfig(
        top_active=True,
        low_active=True,
    )

    # background keyboard thread to get user input
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # headset thread
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # camera pipelines
    # determine active cameras (use camera manager helper if present; fallback to configured serials)
    active_cameras = get_active_cameras(mode, camera_config)

    print(f"active cam: {active_cameras}")

    # initialize latest frame/timestamp containers now that active_cameras is known
    latest_frames = {cam: None for cam in active_cameras}
    latest_timestamps = {cam: None for cam in active_cameras}

    # setup cameras (headset passed so the OAK worker streams the stereo
    # feed to the VR display while also filling latest_frames for the dataset)
    pipelines = setup_cameras(
        active_cameras,
        camera_shutdown,
        frame_lock,
        latest_frames,
        latest_timestamps,
        headset=headset,
    )

    arm_names = ARM_MODES[mode]

    episode_stats = reset_episode_log(active_cameras, arm_names)
    tick_counter = 0
    _gate_last_print = [-10**9]
    _frozen_prev = {"left": None, "right": None}
    _frozen_ticks = {"left": 0, "right": 0}

    robots = create_and_configure_robots(arm_names)

    # Middle-waist frame: read the servo's Homing_Offset so pose tables and the
    # driver<->URDF conversion track whatever set_waist_homing_offset.py wrote.
    # Reads 0.0 on a stock servo, in which case nothing changes.
    waist_shift = 0.0
    if "middle" in arm_names:
        waist_shift = read_middle_waist_shift(robots["middle"])
        if abs(waist_shift) > 1e-6:
            # In ext_position mode the servo does not apply Homing_Offset (study
            # measurement), so a nonzero register would desynchronize the pose
            # tables from the servo's actual frame.  Refuse to guess.
            print("=" * 60)
            print(f"[frame] waist Homing_Offset is {waist_shift:+.3f} rad but the")
            print("        waist now runs in ext_position mode, which ignores it.")
            print("        Revert it before collecting data:")
            print("            python set_waist_homing_offset.py --degrees 0")
            print("=" * 60)
            waist_shift = 0.0
        set_middle_waist_shift(waist_shift)

    robot, arm_data = build_robot_model(mode)

    # Coupled three-arm IK (ik_study winner: smoothing 0.05 + centering 0.5 +
    # sphere self-collision, margin 20 mm / weight 100; pose 50/10).  One
    # solve per tick for all arms — replaces per-arm solve_single_arm_ik.
    if CoupledStudyIK is None:
        raise ImportError("study_ik.CoupledStudyIK unavailable — check ik_study/.")
    print()
    print(_collision_banner(COLLISION_MODE))
    print()

    coupled_ik = CoupledStudyIK(
        robot,
        URDF_PATH,
        ee_links={a: ARM_CONFIG[a]["ee_link"] for a in ("left", "right", "middle")},
        # Smoothing is scaled against the *actual* control period, so changing
        # the loop rate keeps the study's physical velocity budget.
        control_dt=cfg.control_dt,
        waist_driver_shift=waist_shift,
    )
    print("Compiling coupled IK solver (a few seconds)...")
    _t0 = now()
    coupled_ik.warmup(np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32))
    print(f"Coupled IK ready in {now() - _t0:.1f} s")

    ## Hard inter-arm collision gate on the FINAL clamped command (the
    ## solver's collision term is a soft cost checked BEFORE the clamps;
    ## this is a circumscribed-capsule proof-of-separation checked after
    ## them, immediately before set_joint_positions).  GIAVA_CAPSULE_GATE=0
    ## disables; see capsule_gate.py for margins and scope.
    try:
        from .capsule_gate import build_gate as _build_capsule_gate
    except ImportError:
        from capsule_gate import build_gate as _build_capsule_gate
    from yourdfpy import URDF as _URDF_for_gate
    capsule_gate = _build_capsule_gate(robot, _URDF_for_gate.load(URDF_PATH))
    stats.collision_mode = COLLISION_MODE

    for arm_name in arm_names:
        move_to_named_pose(robots[arm_name], arm_name, "forward")

    q = np.zeros(robot.joints.num_actuated_joints, dtype=float)

    for arm_name in arm_names:
        n = ARM_CONFIG[arm_name]["num_joints"]
        q[arm_data[arm_name]["joint_indices"]] = np.asarray(
            robots[arm_name].dxl.joint_states.position[:n],
            dtype=float,
        )

    fk, ee = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(q), arm_data)

    # for arm in arm_names:
    #     print(
    #         arm,
    #         type(ee[arm]),
    #         np.shape(ee[arm]),
    #     )

    cmd_kin = CommandKinematicsState(q_cmd=q.copy(),
        T_cmd={
            arm: ee[arm].copy() for arm in arm_names
        },
    )

    # print(type(cmd_kin.T_cmd["left"]))
    # print(np.shape(cmd_kin.T_cmd["left"]))
    # print(cmd_kin.T_cmd["left"])

    # cmd_state = RobotCommandState(
    #     last_cmds={
    #         arm: q[arm_data[arm]["joint_indices"]].copy() for arm in arm_names
    #     }
    # )

    cmd_state = RobotCommandState(
        last_arm_cmd_time={arm: 0.0 for arm in arm_names},
        last_cmds={
            arm: q[arm_data[arm]["joint_indices"]].copy()
            for arm in arm_names
        },
    )

    teleop_state = TeleopSessionState()

    full_joint_velocity_limits = np.ones(robot.joints.num_actuated_joints) * cfg.full_joint_velocity_limits_value

    ## Episode index is positional.  Skip anything flag-shaped so an
    ## unrecognised option produces a clear message rather than an
    ## int('--flag') traceback after the arms have already been energised.
    _pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    _flags = [a for a in sys.argv[1:] if a.startswith("-")]
    if _flags:
        raise SystemExit(
            f"unrecognised option(s): {' '.join(_flags)}\n"
            f"  usage: python data_collection.py [episode_index] "
            f"[--collision sphere|capsule|gjk] [--mode av|bimanual|left|"
            f"right|middle]\n"
            f"  (--collision is consumed before startup; see "
            f"collision_modes.py)")
    try:
        episode_idx = int(_pos[0]) if _pos else 0
    except ValueError:
        raise SystemExit(
            f"episode index must be an integer, got '{_pos[0]}'")
    gripper_actions = {arm: 0.1 for arm in ARM_MODES[mode]}
    
    dataset, dataset_root = create_dataset(task_name, mode, active_cameras, cfg.control_dt)

    # Camera calibration travels WITH the dataset.  Intrinsics are specific to
    # the device and the resolution, so a recording that does not carry its own
    # is not undistortable afterwards -- you can no longer tell which camera at
    # which settings produced it.  Written once, at session start.
    try:
        intr_path = save_camera_intrinsics(dataset_root / "camera_intrinsics.json")
        print(f"[calib] camera intrinsics -> {intr_path} "
              f"({len(CAMERA_INTRINSICS)} camera(s))")
    except Exception as exc:
        print(f"[calib] could not save camera intrinsics: {exc}")
        if REQUIRE_CALIBRATION:
            raise

    # CHANGED: background saver lets collection continue while a single worker
    # serializes episodes into the shared dataset folder.
    episode_saver = BackgroundEpisodeSaver(dataset)

    # print(f"Dataset: {dataset}")
    # print(f"Dataset root: {dataset_root}")

    next_tick = now()

    print("READY!")

    while not rospy.is_shutdown():
        loop_start = now()

        key = latest_key
        latest_key = None

        if key == "i":
            stop_teleop_session(teleop_state)

            if collecting_episode:
                print("\nIn the middle of collecting episode. Press s to save and stop.")
                continue

            for arm in arm_names:
                move_to_named_pose(robots[arm], arm, "forward")

            # Allow the latest ROS joint-state message to arrive after the final motion.
            rospy.sleep(0.05)

            sync_robot_state(robots=robots, robot=robot, arm_data=arm_data,
                             arm_names=arm_names, cmd_kin=cmd_kin, cmd_state=cmd_state,
                             to_urdf=coupled_ik.driver_to_urdf)

            print("\nREADY")
            continue

        if key == "s":
            if not collecting_episode:
                print("\nNo active episode to save.")
            else:
                print("\nStarting the save")
                stop_teleop_session(teleop_state)
                collecting_episode = False
                # CHANGED: hand off saving to the background worker and continue
                # collecting in the same dataset folder.
                episode_saver.save_episode_async()
                log_episode_info(episode_idx, episode_stats)
                if TRACK_LOG:
                    _ld = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "trajectories", "logs")
                    os.makedirs(_ld, exist_ok=True)
                    try:
                        from study_ik import describe_weights as _dw
                        _w = _dw()
                    except Exception:
                        _w = "unknown"
                    for _a in arm_names:
                        _rows = getattr(episode_stats.arms[_a], "track_rows", [])
                        if len(_rows) > 5:
                            _lp = os.path.join(
                                _ld, f"dc_ep{episode_idx:04d}_{_a}_"
                                f"{1.0 / cfg.arm_cmd_dt:g}hz_"
                                f"{time.strftime('%H%M%S')}.npz")
                            np.savez(_lp, rows=np.asarray(_rows),
                                     rate=1.0 / cfg.arm_cmd_dt,
                                     scale=cfg.position_scale, arm=_a,
                                     traj=f"episode_{episode_idx}", weights=_w)
                            print(f"[track] wrote {os.path.basename(_lp)}")
                episode_idx += 1
                print(f"\nEPISODE QUEUED FOR SAVE")

                # LOGGING TBD
            continue

        if key == "d":
            if not collecting_episode:
                print("\nNo active episode to discard.")
            else:
                print("\nDiscarding the current episode.")
                stop_teleop_session(teleop_state)
                collecting_episode = False
                # CHANGED: discard only the active foreground buffer while
                # leaving any queued background saves untouched.
                episode_saver.discard_current_episode()
                episode_idx += 1
                episode_stats = None
                reset_episode_log(active_cameras, arm_names)
                print(f"\nEPISODE DISCARDED")
            continue

        if key == "r":
            if collecting_episode:
                print("\nAlready recording episode. Press s to save or d to discard.")
            else:
                collecting_episode = True

                episode_stats = reset_episode_log(active_cameras, arm_names)

                print(f"\nCOLLECTING: episode_{episode_idx:04d}")

        # Live stereo-comfort tuning (headset view only; dataset unaffected).
        #   , / .  : pull the eye images together / apart (convergence)
        #   - / =  : shrink / grow the per-eye image (zoom comfort)
        # Values print on every press; paste the final ones into
        # camera_manager.py EYE_VIEW defaults.
        if key == ",":
            adjust_eye_view(dinward=+0.005)
            continue
        if key == ".":
            adjust_eye_view(dinward=-0.005)
            continue
        if key == "-":
            adjust_eye_view(dscale=-0.025)
            continue
        if key == "=":
            adjust_eye_view(dscale=+0.025)
            continue

        if key == "q":
            print("\nEXITING")

            episode_idx, did_shutdown = safe_shutdown(
                robots, pipelines, dataset, episode_saver, collecting_episode, dataset_root, episode_idx)

            if did_shutdown:
                break

        # Teleoperating
        tick_counter += 1

        t_headset_start = now()
        headset_data = headset.receive_data()
        episode_stats.headset.append(now() - t_headset_start)

        if headset_data is None:
            continue

        head_mat_raw = pose2mat(headset_data.h_pos, headset_data.h_quat)
        # The probe (2026-08) measured the app world as z-up with an arbitrary
        # per-session yaw; the old fixed basis conversion here was wrong and is
        # gone.  All frame handling now happens through the session-yaw
        # calibration built at anchor time (data_col_config.session_yaw_remap).
        head_mat = head_mat_raw
        if HANDS_HEAD_RELATIVE:
            # The headset app reports CONTROLLER poses in the HEAD's frame, not
            # the world frame (symptom: standing up moves the gripper arms DOWN
            # -- the hands' head-relative height drops when the head rises).
            # Compose through the head pose to recover world-frame hands.
            # Never noticed while seated: with the head still, head-relative
            # deltas equal world deltas exactly.
            # Compose with the RAW head pose.  The probe (2026-08) measured the
            # app's world frame as z-up, so HEAD_BASIS_FIX (built on the y-up
            # hypothesis) must NOT be applied here -- with it, the step-3 leak
            # test showed up to 810 mm of fake hand motion from pure head
            # rotation, which was the gripper-arm drift.  Raw composition is
            # also the configuration the stand-up test validated.
            controller_poses = {
                "right": head_mat_raw @ pose2mat(headset_data.r_pos, headset_data.r_quat),
                "left": head_mat_raw @ pose2mat(headset_data.l_pos, headset_data.l_quat),
                "middle": head_mat,
            }
        else:
            controller_poses = {
                "right": pose2mat(headset_data.r_pos, headset_data.r_quat),
                "left": pose2mat(headset_data.l_pos, headset_data.l_quat),
                "middle": head_mat,
            }
        # GIAVA_DEBUG_HANDS=1: move ONLY your head (hands still), then ONLY one
        # hand, and read which numbers change.  If head-only motion changes
        # l_pos/r_pos, the app is reporting head-relative hands -- enable
        # GIAVA_HANDS_HEAD_RELATIVE=1.
        if DEBUG_HANDS and (tick_counter % 25 == 0):
            print(f"[hands] h={np.round(np.asarray(headset_data.h_pos), 3)} "
                  f"l={np.round(np.asarray(headset_data.l_pos), 3)} "
                  f"r={np.round(np.asarray(headset_data.r_pos), 3)}")

        for _side, _p in (("left", headset_data.l_pos), ("right", headset_data.r_pos)):
            _pv = np.asarray(_p, dtype=float).copy()
            if _frozen_prev[_side] is not None and np.array_equal(_pv, _frozen_prev[_side]):
                _frozen_ticks[_side] += 1
            else:
                _frozen_ticks[_side] = 0
            _frozen_prev[_side] = _pv
            if _frozen_ticks[_side] == 25:  # ~0.5 s bit-identical = asleep/untracked
                print(f"[tracking] {_side} controller pose is FROZEN -- asleep or "
                      "untracked. Wake it / bring it into view before enabling "
                      "that arm (probe measured 0.5 m of drift on re-acquire).")

        button_pressed = (headset_data.r_button_one or headset_data.l_button_one
                          or headset_data.l_button_two)

        # Per-arm buttons: X (left one) = left arm, A (right one) = right arm,
        # Y (left two) = middle/camera arm ONLY.  The camera stays parked
        # unless Y is held, so incidental head motion moves nothing.
        arm_active = {
            "left": headset_data.l_button_one,
            "right": headset_data.r_button_one,
            "middle": (headset_data.l_button_two if MIDDLE_BUTTON == "y" else
                       (headset_data.l_button_one or headset_data.r_button_one)),
        }
        # Tool parity: a frozen device (asleep / untracked, probe-measured 0.5 m
        # drift on re-acquire) must not drive an arm.  Refuse activation and
        # hold mid-session rather than only warning.
        for _arm, _side, _pos in (("left", "left", headset_data.l_pos),
                                  ("right", "right", headset_data.r_pos)):
            # exact zeros = the headset lost this controller (hands above the
            # head / out of volume) -- hold the arm, do not chase (0,0,0)
            if arm_active.get(_arm) and float(np.linalg.norm(
                    np.asarray(_pos, dtype=float))) < 1e-6:
                arm_active[_arm] = False
                if tick_counter % 50 == 0:
                    print(f"[tracking] {_side} controller reads zeros (LOST) -- "
                          f"{_arm} arm held")
                continue
            if arm_active.get(_arm) and _frozen_ticks.get(_side, 0) > 25:
                arm_active[_arm] = False
                if tick_counter % 50 == 0:
                    print(f"[tracking] {_side} controller frozen -- {_arm} arm "
                          "held (wiggle the controller)")

        for arm in arm_names:
            if arm not in teleop_state.arms:
                teleop_state.arms[arm] = ArmTeleopState()
            was_active = teleop_state.arms[arm].active
            if arm_active[arm] and not was_active and teleop_state.active:
                # Arm activated MID-session: re-anchor its reference frames to
                # now, else it would jump by everything the controller/head
                # moved since the session started.
                anchor_arm_state(
                    teleop_state.arms[arm], controller_poses[arm], cmd_kin.T_cmd[arm],
                    head_pose=controller_poses.get("middle"),
                    base_remap=(cfg.R_cam_remap if arm == "middle" else cfg.R_arm_remap),
                )
            teleop_state.arms[arm].active = arm_active[arm]

        # teleop_state.active = any(
        #     arm_state.active
        #     for arm_state in teleop_state.arms.values()
        # )

        if button_pressed and not teleop_state.active:
            # Re-anchor UNCONDITIONALLY at every teleop enable, not only when
            # the stale heuristic fires.  Any motion outside the loop (named
            # poses, scripts, hand-moving a torqued-off arm) desynchronizes
            # three references at once -- our last_cmds, our T_cmd, and the
            # DRIVER's joint_commands (its velocity check compares against its
            # last accepted command, not the encoders).  One sync per enable
            # costs a joint-state read and one FK; missing one costs a stuck
            # arm rejecting every command.
            if command_state_is_stale(robots=robots, arm_data=arm_data, arm_names=arm_names, cmd_state=cmd_state, tolerance=0.08):
                print("Resynchronizing teleoperation state with measured joints.")
            sync_robot_state(
                robots=robots,
                robot=robot,
                arm_data=arm_data,
                arm_names=arm_names,
                cmd_kin=cmd_kin,
                cmd_state=cmd_state,
                to_urdf=coupled_ik.driver_to_urdf,
            )

            ## The gate must not be armed already-violated -- it would hold
            ## every command and read as "teleop is broken".  Checked at the
            ## just-synced commanded pose; the error names the pair to move.
            if capsule_gate is not None:
                ## Report only. Starting a session with the grippers close
                ## is normal -- after a handover, for instance -- and the
                ## gate holds the arms until they separate. It must never
                ## take the session down with it.
                capsule_gate.validate(
                    coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                    where="teleop enable")

            start_teleop_session(
                teleop_state,
                mode,
                controller_poses,
                cmd_kin,
                cfg=cfg,
            )

            print("\nTeleop ENABLED")
            episode_stats.teleop_enable_count += 1
            episode_stats.teleop_enable_count += 1
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")
            episode_stats.teleop_disable_count += 1

        gripper_actions = {}
        joint_positions = {}
        gripper_states = {}

        for arm in arm_names:
            joint_state_msg = robots[arm].dxl.joint_states

            # joint_positions[arm] = np.asarray(joint_state_msg.position[:6], dtype=np.float32)
            n = ARM_CONFIG[arm]["num_joints"]

            joint_positions[arm] = np.asarray(
                joint_state_msg.position[:n],
                dtype=np.float32,
            )

            if ARM_CONFIG[arm]["has_gripper"]:
                gripper_states[arm] = np.asarray([joint_state_msg.position[6]], dtype=np.float32)

        if "left" in arm_names:
            gripper_actions["left"] = update_gripper(robots["left"], headset_data.l_index_trigger > 0)

        if "right" in arm_names:
            gripper_actions["right"] = update_gripper(robots["right"], headset_data.r_index_trigger > 0)

        gripper_actions["middle"] = None

        targets = {}

        for arm in arm_names:

            arm_state = teleop_state.arms[arm]

            if not arm_state.active:
                continue

            controller_pose = controller_poses[arm]

            if arm == "middle":
                # print("target_pos", target_pos)
                # print("target_wxyz", target_wxyz)
                # print("middle active:", teleop_state.arms["middle"].active)
                # print("camera target")
                # print(target_pos)
                target_pos, target_wxyz = compute_camera_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_cam_remap,
                )
                # GIAVA_DEBUG_HEAD=1: print the raw headset delta and the
                # remapped robot delta (~2 Hz) so the axis mapping can be
                # verified empirically: nod / shake / lean in one direction at
                # a time and read off which robot axis moves.
                if DEBUG_HEAD and (tick_counter % 25 == 0):
                    _dh = controller_pose[:3, 3] - arm_state.start_controller_pos
                    _rm = (arm_state.session_remap
                           if arm_state.session_remap is not None else cfg.R_cam_remap)
                    _dr = _rm @ _dh
                    _cmd = np.asarray(target_pos) - arm_state.start_robot_pos
                    _ach = np.asarray(cmd_kin.T_cmd[arm][4:]) - arm_state.start_robot_pos
                    print(f"[head] d_raw={np.round(_dh, 3)} d_robot={np.round(_dr, 3)} "
                          f"cmd={np.round(_cmd, 3)} achieved={np.round(_ach, 3)} "
                          f"(x=op-left, y=op-back, z=up; cmd<<d_robot = starved)")
            else:
                target_pos, target_wxyz = compute_gripper_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_arm_remap,
                )

            targets[arm] = (target_pos, target_wxyz)

        if teleop_state.active:
            # start_teleop_session(teleop_state, mode, controller_poses, cmd_kin)
            t_solve_start = now()

            # Coupled three-arm IK (ik_study winner): ONE solve for all due
            # arms per tick, replacing the previous per-arm solve loop.  Arms
            # without a fresh target hold their commanded pose inside the
            # same problem, which is what makes the inter-arm collision term
            # meaningful.
            due_arms = [
                arm for arm in arm_names
                if teleop_state.arms[arm].active
                and (t_solve_start - cmd_state.last_arm_cmd_time[arm]) >= cfg.arm_cmd_dt
                and arm in targets
            ]

            if due_arms:
                for arm in due_arms:
                    episode_stats.arms[arm].ik_attempts += 1

                _t_ik = now()
                q_new = coupled_ik.solve(
                    cmd_kin.q_cmd,
                    {arm: targets[arm] for arm in due_arms},
                )

                # Solve-time accounting: with sphere collision active the
                # study measured 5.2 ms median but 64 ms worst-trajectory p95,
                # against a 20 ms budget at 50 Hz.  Overruns are the first thing
                # to check when the motion feels bad.
                _ik_ms = (now() - _t_ik) * 1e3
                episode_stats.ik_solve_ms.append(_ik_ms)
                if _ik_ms > cfg.control_dt * 1e3:
                    episode_stats.ik_overrun_ticks += 1

                # Optional: prove the collision terms see what the operator
                # sees.  GIAVA_LOG_CLEARANCE=1 prints min sphere clearance
                # whenever it drops near the margin (throttled to ~2 Hz).
                if LOG_CLEARANCE and (tick_counter % 25 == 0):
                    _clr = coupled_ik.min_clearance(q_new)
                    if _clr < COLLISION_MARGIN + 0.02:
                        state = "INSIDE MARGIN" if _clr < COLLISION_MARGIN else "near"
                        print(f"[collision] min sphere clearance "
                              f"{_clr * 1e3:+.1f} mm ({state}, margin "
                              f"{COLLISION_MARGIN * 1e3:.0f} mm)")

                fk_sol, ee_sol = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(q_new), arm_data)

                pending_cmds = []  # (arm, joint_idx, q_arm_cmd, prev_cmd)
                for arm in due_arms:
                    target_pos, target_wxyz = targets[arm]

                    episode_stats.arms[arm].ik_successes += 1

                    pos_err = np.linalg.norm(ee_sol[arm][4:] - target_pos)
                    episode_stats.arms[arm].ik_position_errors.append(pos_err)

                    quat = np.asarray(ee_sol[arm][:4], dtype=np.float64).copy()
                    R_sol = quat2mat(quat)
                    R_target = quat2mat(target_wxyz)
                    R_err = R_sol.T @ R_target
                    trace = np.clip(np.trace(R_err), -1.0, 3.0)
                    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
                    episode_stats.arms[arm].ik_orientation_errors.append(angle_rad)

                    joint_idx = arm_data[arm]["joint_indices"]
                    target_q = np.asarray(q_new[joint_idx], dtype=float)

                    prev_cmd = (cmd_state.last_cmds[arm] if arm in cmd_state.last_cmds else cmd_kin.q_cmd[joint_idx])

                    # Post-solve step clamp -- off by default to match the
                    # ik_study conditions (see TeleopConfig.enable_joint_clamp).
                    if cfg.enable_joint_clamp:
                        max_step = (cfg.max_joint_step_middle if arm == "middle"
                                    else cfg.max_joint_step)
                        q_arm_cmd = clamp_joint_step(prev_cmd, target_q, max_step)
                        if np.any(
                            np.abs(target_q - prev_cmd) > np.asarray(max_step) * 0.99
                        ):
                            episode_stats.clamp_saturated_ticks += 1
                    else:
                        q_arm_cmd = target_q

                    # Driver-feasibility clamp: interbotix rejects the ENTIRE
                    # group command if any joint would exceed its position or
                    # velocity limit, so one infeasible joint freezes the whole
                    # arm.  Clamp into the feasible set so every command is
                    # accepted; the operator feels a slower joint rather than a
                    # dead arm.
                    if cfg.enable_driver_clamp:
                        # The URDF and the driver disagree slightly (e.g. right
                        # wrist_angle: URDF +2.234, driver rejected +2.243), and
                        # it is the DRIVER that refuses commands -- so clamp
                        # against ITS limit arrays, pulled in by the cushion.
                        lo = np.asarray(
                            robots[arm].arm.group_info.joint_lower_limits, dtype=float
                        )[: len(prev_cmd)]
                        hi = np.asarray(
                            robots[arm].arm.group_info.joint_upper_limits, dtype=float
                        )[: len(prev_cmd)]
                        step = float(cfg.driver_max_step)
                        # Clamp against the DRIVER's reference, not ours.  The
                        # driver validates against its last *accepted* command;
                        # if one of ours was rejected the two references have
                        # already diverged, and clamping against our own would
                        # keep proposing steps it will keep refusing.
                        ref = prev_cmd
                        getter = getattr(robots[arm].arm, "get_joint_commands", None)
                        if getter is not None:
                            try:
                                ref = np.asarray(getter(), dtype=float)[:len(prev_cmd)]
                            except Exception:
                                ref = prev_cmd
                        clamped = np.clip(q_arm_cmd, ref - step, ref + step)
                        clamped = np.clip(
                            clamped,
                            lo + cfg.driver_limit_margin,
                            hi - cfg.driver_limit_margin,
                        )
                        if np.any(np.abs(clamped - q_arm_cmd) > 1e-6):
                            episode_stats.driver_clamp_ticks += 1
                            hit = int(np.argmax(np.abs(clamped - q_arm_cmd)))
                            episode_stats.driver_clamp_joints[f"{arm}{hit}"] = (
                                episode_stats.driver_clamp_joints.get(f"{arm}{hit}", 0) + 1
                            )
                        q_arm_cmd = clamped

                    ## COMPUTE phase ends here -- nothing has been sent.
                    ## Sends happen below, after the capsule gate has seen
                    ## the assembled command of ALL due arms: the clamps
                    ## just modified what the solver certified, and two
                    ## arms each individually fine can still be about to
                    ## meet each other.
                    pending_cmds.append((arm, joint_idx, q_arm_cmd, prev_cmd))

                ## HARD INTER-ARM GATE on the final command.  Circumscribed
                ## capsules: distance > margin PROVES mesh separation.  On a
                ## violation the whole tick is refused -- every arm holds its
                ## previous command (a partial send could still create the
                ## very pair the gate saw).  Nothing is modified: a gate that
                ## edits commands is a second, unstudied IK solver.
                if capsule_gate is not None and pending_cmds:
                    ## The step is SCALED, not refused.  Find the largest
                    ## fraction of this tick's motion whose whole swept
                    ## segment clears the margin, and send that -- so the
                    ## arms slide up to the boundary and stop there however
                    ## hard the operator pushes, instead of freezing at
                    ## whatever the last accepted tick happened to reach.
                    ##
                    ## Scaling is UNIFORM across arms on purpose.  Scaling
                    ## only the arms in the offending pair would change the
                    ## relative geometry mid-step -- i.e. redirect the
                    ## motion into a path nobody commanded.  A uniform
                    ## factor preserves the shape of the commanded motion
                    ## and only slows it down.
                    q_prev_full = cmd_kin.q_cmd.copy()
                    q_target_full = q_prev_full.copy()
                    for _arm, _idx, _q, _ in pending_cmds:
                        q_target_full[_idx] = _q
                    _alpha, _dist, _pair = capsule_gate.largest_safe_fraction(
                        coupled_ik.driver_to_urdf(q_prev_full),
                        coupled_ik.driver_to_urdf(q_target_full))

                    if _alpha >= 1.0:
                        pass                      # full step is clear
                    elif _alpha <= 0.0:
                        episode_stats.capsule_gate_blocks += 1
                        if (tick_counter - _gate_last_print[0]) >= 25:  # ~2 Hz
                            _gate_last_print[0] = tick_counter
                            print(f"[capsule gate] HOLDING: {_pair[0]} <-> "
                                  f"{_pair[1]} at {_dist * 1e3:+.1f} mm "
                                  f"(margin {capsule_gate.margin * 1e3:.0f} "
                                  f"mm). Move the controllers apart.")
                        pending_cmds = []
                    else:
                        ## Partial step: interpolate in DRIVER space by the
                        ## same alpha.  driver->urdf is affine per joint, so
                        ## interpolating either side gives the identical
                        ## configuration -- the fraction checked is exactly
                        ## the fraction sent.
                        episode_stats.capsule_gate_scaled += 1
                        episode_stats.capsule_gate_min_alpha = min(
                            episode_stats.capsule_gate_min_alpha, float(_alpha))
                        scaled = []
                        for _arm, _idx, _q, _prev in pending_cmds:
                            _qs = (q_prev_full[_idx]
                                   + _alpha * (_q - q_prev_full[_idx]))
                            scaled.append((_arm, _idx, _qs, _prev))
                        pending_cmds = scaled
                        if (tick_counter - _gate_last_print[0]) >= 25:
                            _gate_last_print[0] = tick_counter
                            print(f"[capsule gate] limiting step to "
                                  f"{_alpha * 100:.0f}%: {_pair[0]} <-> "
                                  f"{_pair[1]} stopping at "
                                  f"{_dist * 1e3:+.1f} mm")

                for arm, joint_idx, q_arm_cmd, prev_cmd in pending_cmds:
                    cmd_step = np.linalg.norm(q_arm_cmd - prev_cmd)
                    episode_stats.arms[arm].joint_step_norms.append(cmd_step)

                    t_cmd_start = now()

                    robots[arm].arm.set_joint_positions(
                        q_arm_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )

                    episode_stats.cmd.append(now() - t_cmd_start)

                    episode_stats.arms[arm].waist_cmds.append(float(q_arm_cmd[0]))
                    if arm in targets:
                        episode_stats.arms[arm].target_positions.append(
                            np.asarray(targets[arm][0], dtype=float).copy()
                        )

                    cmd_kin.q_cmd[joint_idx] = q_arm_cmd
                    cmd_state.last_cmds[arm] = q_arm_cmd.copy()
                    cmd_state.last_arm_cmd_time[arm] = now()

            # GIAVA_TRACK_LOG=1: per-tick expected-vs-measured EE, per arm, in
            # the same row format the debug tool logs -- so plot_tracking.py
            # renders data-collection episodes identically (per-axis expected
            # vs achieved, % diff, per-rate overlays).
            if TRACK_LOG and due_arms:
                q_meas_full = cmd_kin.q_cmd.copy()
                for _a in arm_names:
                    if _a in joint_positions:
                        _n = len(arm_data[_a]["joint_indices"])
                        q_meas_full[arm_data[_a]["joint_indices"]] =                             np.asarray(joint_positions[_a], dtype=float)[:_n]
                _, ee_meas = compute_fk_and_ee(
                    robot, coupled_ik.driver_to_urdf(q_meas_full), arm_data)
                for _a in due_arms:
                    _st = teleop_state.arms.get(_a)
                    if _st is None or _st.start_robot_pos is None:
                        continue
                    _tp, _tw = targets[_a]
                    _mp = np.asarray(ee_meas[_a][4:], dtype=float)
                    _mq = np.asarray(ee_meas[_a][:4], dtype=float)  # wxyz
                    _R0 = _st.start_robot_rot
                    _Rt = R.from_quat(np.roll(np.asarray(_tw), -1)).as_matrix()
                    _Rm = R.from_quat(np.roll(_mq, -1)).as_matrix()
                    _rv = lambda Ra: np.degrees(
                        R.from_matrix(Ra @ _R0.T).as_rotvec())
                    if not hasattr(episode_stats.arms[_a], "track_rows"):
                        episode_stats.arms[_a].track_rows = []
                    episode_stats.arms[_a].track_rows.append(np.concatenate([
                        [now(), tick_counter],
                        np.asarray(_tp) - _st.start_robot_pos,
                        _mp - _st.start_robot_pos,
                        _rv(_Rt), _rv(_Rm)]))

            fk_cmd, ee_cmd = compute_fk_and_ee(robot, coupled_ik.driver_to_urdf(cmd_kin.q_cmd), arm_data)

            for arm in arm_names:
                cmd_kin.T_cmd[arm] = ee_cmd[arm]

            episode_stats.ik_solve.append(now() - t_solve_start)

        if collecting_episode:
            obs_ts = now()

            images = {}
            camera_timestamps = {}

            

            # Camera frames.
            #
            # SYNC_FRAMES picks, per camera, the frame nearest a shared
            # reference instant instead of whatever is newest.  The cameras
            # free-run (the D405 has no inter_cam_sync_mode), so without this
            # the frames in one timestep can be a full frame period apart --
            # measured at 5.2 ms vs 12.6 ms old on this rig.  sync_info
            # carries the residual spread so alignment quality is recorded
            # rather than assumed.  GIAVA_SYNC_FRAMES=0 restores
            # latest-frame-wins.
            sync_info = None
            with frame_lock:
                if SYNC_FRAMES:
                    sel_frames, sel_ts, sync_info = select_synchronized_frames(
                        active_cameras)
                    for camera in active_cameras:
                        frame = sel_frames.get(camera)
                        ts = sel_ts.get(camera)
                        if frame is None or ts is None:
                            episode_stats.cameras[camera].frames_missing += 1
                            continue
                        images[camera] = frame.copy()
                        camera_timestamps[camera] = ts
                else:
                    for camera in active_cameras:
                        frame = latest_frames.get(camera)
                        ts = latest_timestamps.get(camera)

                        if frame is None or ts is None:
                            episode_stats.cameras[camera].frames_missing += 1
                            continue

                        images[camera] = frame.copy()
                        camera_timestamps[camera] = ts

            if sync_info is not None and sync_info["spread_s"] is not None:
                episode_stats.sync_spreads.append(sync_info["spread_s"])


            # Robot states
            robot_states = {}

            for arm in arm_names:

                joints = joint_positions[arm].astype(np.float32)
                cmd_track_err = np.linalg.norm(joint_positions[arm] - cmd_state.last_cmds[arm])
                episode_stats.arms[arm].cmd_track_err.append(cmd_track_err)

                robot_states[arm] = {
                    "joints": joints,
                    "gripper": float(gripper_states[arm][0])
                    if arm in gripper_states
                    else 0.0,
                }

            robot_actions = {}

            for arm in arm_names:

                joint_idx = arm_data[arm]["joint_indices"]

                q_cmd = (
                    cmd_state.last_cmds[arm]
                    if arm in cmd_state.last_cmds
                    else cmd_kin.q_cmd[joint_idx]
                )

                robot_actions[arm] = {
                    "joints": np.asarray(q_cmd, dtype=np.float32),
                    "gripper": (gripper_actions[arm] if arm in gripper_actions else 0.0),
                }

            ee_poses = {}

            fk, ee = compute_fk_and_ee(
                robot,
                coupled_ik.driver_to_urdf(cmd_kin.q_cmd),
                arm_data,
            )

            for arm in arm_names:
                ee_poses[arm] = np.asarray(ee[arm], dtype=np.float32)

            timestamps = {
                **camera_timestamps,
            }

            for arm in arm_names:
                timestamps[arm] = obs_ts

            frame = build_frame(
                mode=mode,
                active_cameras=active_cameras,
                robot_states=robot_states,
                robot_actions=robot_actions,
                ee_poses=ee_poses,
                timestamps=timestamps,
                images=images,
            )

            dataset.add_frame(frame, task_name)

            episode_stats.frames_added += 1

        # sleep to reduce drift
        next_tick += cfg.control_dt
        sleep_time = next_tick - now()
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
