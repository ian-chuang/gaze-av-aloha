import argparse
import sys
import warnings
from pathlib import Path
import time

# silence ROS noetic's python2-era docstring SyntaxWarnings (py3.12)
warnings.filterwarnings("ignore", message=r"invalid escape sequence",
                        category=SyntaxWarning)

for _ros in ("/opt/ros/noetic/lib/python3/dist-packages",
             "/home/devi/giava/interbotix_ws/devel/lib/python3/dist-packages"):
    if _ros not in sys.path:
        sys.path.append(_ros)

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies import make_policy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.configs.types import PolicyFeature, FeatureType

import pyroki as pk
import jaxlie
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF


URDF_PATH = "/home/devi/giava/right.urdf"
RIGHT_EE_LINK = "right_gripper_base"

RIGHT_ARM_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]

GRIPPER_CURRENT_LIMIT = 200

CAMERA_SERIALS = {
    "observation.images.right_wrist": "230322270105",
    "observation.images.top_scene": "230322270396",
}

RIGHT_RESET_Q = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)


def resolve_migrated_checkpoint_dir(policy_dir):
    """Return the lerobot v0.6.0 checkpoint directory for `policy_dir`.

    v0.6.0 loads a policy from a directory (clean model.safetensors plus the
    policy_preprocessor / policy_postprocessor files), not from a raw .pt state
    dict. migrate_checkpoints.py writes those next to the original as
    "<original>_lerobot_v06"; if `policy_dir` is already such a directory it is
    used as-is.
    """
    p = Path(policy_dir)
    candidates = [p.parent / f"{p.name}_lerobot_v06", p]
    for c in candidates:
        if (c / "model.safetensors").exists() and (c / "policy_preprocessor.json").exists():
            return c
    raise FileNotFoundError(
        f"No migrated checkpoint for {policy_dir}. Expected "
        f"{p.parent / (p.name + '_lerobot_v06')} containing model.safetensors and "
        "policy_preprocessor.json. Run migrate_checkpoints.py first."
    )


def digital_zoom(frame, zoom=1.6):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def extract_green_object_features(img):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(mask)
    h, w = mask.shape

    print("m00 =", moments["m00"])

    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = w / 2
        cy = h / 2

    centroid = np.array([cx / w, cy / h], dtype=np.float32)
    mask = mask.astype(np.float32) / 255.0
    return mask, centroid


def build_robot_model():
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    right_arm_indices = [
        robot.joints.actuated_names.index(name)
        for name in RIGHT_ARM_NAMES
    ]
    right_ee_index = robot.links.names.index(RIGHT_EE_LINK)

    return robot, right_arm_indices, right_ee_index


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


def setup_camera(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    return pipeline


def get_color_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame received")
    return np.asanyarray(color_frame.get_data())


def make_observation(
    right_bot,
    pipelines,
    device,
    robot,
    right_arm_indices,
    right_ee_index,
):
    joint_state_msg = right_bot.dxl.joint_states

    joint_positions = np.array(joint_state_msg.position[:6], dtype=np.float32)
    gripper_position = np.array([joint_state_msg.position[6]], dtype=np.float32)
    obs_state = np.concatenate([joint_positions, gripper_position], axis=0)

    q = np.zeros(robot.joints.num_actuated_joints, dtype=np.float32)
    q[right_arm_indices] = joint_positions

    fk = robot.forward_kinematics(q)
    T_right = jaxlie.SE3(fk[right_ee_index]).as_matrix()

    position = T_right[:3, 3]
    quat_xyzw = R.from_matrix(T_right[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    ee_pose = np.concatenate([position, quat_wxyz]).astype(np.float32)

    observation = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
        "observation.ee_pose": torch.from_numpy(ee_pose).unsqueeze(0).to(device),
    }

    for key, pipeline in pipelines.items():
        frame_rgb = get_color_frame(pipeline)

        if key == "observation.images.top_scene":
            frame_rgb = digital_zoom(frame_rgb)

        # CHANGED: scale to [0, 1] to match training. LeRobotDataset yields images
        # as float32 in [0, 1], so the normalization stats were computed on that
        # range; .float() alone left these in [0, 255].
        frame = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
            .unsqueeze(0).to(device)
        )
        observation[key] = frame

        if key == "observation.images.top_scene":
            mask, centroid = extract_green_object_features(frame_rgb)
            mask_rgb = np.repeat(mask[None], 3, axis=0)

            observation["observation.object_mask"] = (
                torch.from_numpy(mask_rgb).float().unsqueeze(0).to(device)
            )
            observation["observation.object_centroid"] = (
                torch.from_numpy(centroid).float().unsqueeze(0).to(device)
            )

        cv2.imwrite("debug_rgb.png", frame_rgb)
        

    for key in [
        "observation.images.right_wrist",
        "observation.images.top_scene",
        "observation.object_mask",
    ]:
        if key in observation:
            observation[key] = observation[key][:, [2, 1, 0], :, :]

    return observation


def send_action(right_bot, action):
    if torch.is_tensor(action):
        action = action.detach().float().cpu().numpy()
    action = np.asarray(action).reshape(-1)

    if action.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")

    arm_cmd = action[:6]
    gripper_cmd = float(action[6])

    current_q = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)
    arm_cmd = np.clip(
        arm_cmd,
        current_q - MAX_JOINT_STEP,
        current_q + MAX_JOINT_STEP,
    )

    right_bot.arm.set_joint_positions(
        arm_cmd.tolist(),
        moving_time=0.14,
        accel_time=0.04,
        blocking=False,
    )

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = gripper_cmd
    right_bot.gripper.core.pub_single.publish(cmd)


def load_policy(policy_dir, dataset_root, device):
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="grasp_cube",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",
    }

    input_features = {
        k: v
        for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }

    input_features["observation.object_centroid"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(2,),
    )
    input_features["observation.object_mask"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 480, 640),
    )

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=40,
        n_action_steps=10,
        use_vae=True,
        kl_weight=0.01,
        optimizer_lr=3e-4,
        optimizer_lr_backbone=1e-5,
    )

    migrated_dir = resolve_migrated_checkpoint_dir(policy_dir)

    # CHANGED (lerobot v0.6.0): load the policy together with its normalization
    # pipelines from a migrated checkpoint directory. The old fallback path --
    # torch.load() of a raw .pt state dict -- cannot work any more: those state
    # dicts carry the embedded `normalize_inputs.*` buffers, which the v0.6.0
    # model no longer defines. `migrated_dir` is the directory produced by
    # migrate_checkpoints.py (named "<original>_lerobot_v06").
    policy = ACTPolicy.from_pretrained(migrated_dir)

    # The migrated pipelines serialize device_processor with the device used
    # at migration time ("cpu"), which would silently move every observation
    # OFF the GPU right before the cuda policy runs.  Retarget it here.
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        migrated_dir, config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": str(device)}},
    )
    # postprocessor keeps its saved device ("cpu"): send_action wants numpy.
    # The action converters make it callable on the bare action tensor,
    # matching upstream (lerobot factory / lerobot_eval).
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        migrated_dir, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    policy.to(device)
    policy.eval()
    return policy, preprocessor, postprocessor, dataset_metadata


def reset_robot(right_bot):
    right_bot.arm.set_joint_positions(
        RIGHT_RESET_Q.tolist(),
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes(
        "single",
        "gripper",
        "current_based_position",
    )
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = 0.1
    right_bot.gripper.core.pub_single.publish(cmd)
    rospy.sleep(1.0)


def resolve_policy_dir(path: Path) -> Path:
    """Accept a migrated dir, a raw run dir, or a .pt file inside one, and
    return the loadable (_lerobot_v06) directory — or exit with advice.

    Only each run's FINAL model was migrated (model.safetensors next to
    checkpoint.pt); intermediate checkpoint_<step>.pt snapshots have no
    migrated twin and cannot be loaded by lerobot v0.6.0 directly."""
    import json

    if path.is_file():
        print(f"[policy] {path.name} is a file — using its run directory. "
              "NOTE: only the run's FINAL model is loadable; intermediate "
              "step snapshots were not migrated.")
        path = path.parent

    def is_migrated(d: Path) -> bool:
        # raw run dirs also have a config.json with "type"; only migrated
        # dirs carry the processor pipelines from_pretrained needs
        if not (d / "policy_preprocessor.json").is_file():
            return False
        try:
            return json.load(open(d / "config.json")).get("type") is not None
        except Exception:
            return False

    if is_migrated(path):
        return path
    twin = path.parent / f"{path.name}_lerobot_v06"
    if twin.is_dir() and is_migrated(twin):
        print(f"[policy] {path.name} is a raw checkpoint — "
              f"using migrated twin {twin.name}")
        return twin
    sys.exit(
        f"ERROR: {path} is not a loadable checkpoint directory and no "
        f"migrated twin ({twin.name}) exists.\n"
        "Run:  python ../migrate_checkpoints.py   (from "
        "data_collection_scripts/) to create it, then pass the "
        "*_lerobot_v06 directory."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reset the right arm, then roll out an object-centric "
                    "ACT checkpoint. policy-dir must be a migrated "
                    "(_lerobot_v06) checkpoint directory.")
    parser.add_argument("--policy-dir", type=Path, required=True,
                        help="migrated checkpoint dir (model.safetensors + "
                             "policy_pre/postprocessor.json)")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/devi/giava/interbotix_ws/src/av_aloha/"
                     "data_collection_scripts/dataset/lerobot/"
                     "grasp_cube/20260529_162433"),
        help="dataset the policy was trained on (metadata/fps)")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=15.0)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    policy_dir = resolve_policy_dir(args.policy_dir)
    rollout_seconds = args.seconds
    control_hz = args.hz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.init_node("act_rollout_test", anonymous=True)

    policy, preprocessor, postprocessor, dataset_metadata = load_policy(policy_dir, dataset_root, device)

    print("INPUT FEATURES AFTER CALLING LOAD POLICY:")
    for k in policy.config.input_features:
        print(k)
    

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=0.14,
        accel_time=0.04,
        init_node=False,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes(
        "single",
        "gripper",
        "current_based_position",
    )
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    pipelines = {key: setup_camera(serial) for key, serial in CAMERA_SERIALS.items()}
    robot, right_arm_indices, right_ee_index = build_robot_model()

    try:
        reset_robot(right_bot)

        dt = 1.0 / control_hz
        num_steps = int(rollout_seconds * control_hz)

        print(f"Loaded policy from {policy_dir}")
        print(f"Dataset fps={dataset_metadata.fps}")
        print(f"Running rollout for {num_steps} steps at {control_hz} Hz")
        print("Action stats:")
        print(dataset_metadata.stats["action"])

        obs = make_observation(
            right_bot,
            pipelines,
            device,
            robot,
            right_arm_indices,
            right_ee_index,
        )

        print(obs.keys())
        for k, v in obs.items():
            if torch.is_tensor(v):
                print(k, tuple(v.shape))
            else:
                print(k, type(v))
        print("queue len:", len(policy._action_queue))
        with torch.inference_mode():
            for step in range(num_steps):
                t0 = time.time()

                obs = make_observation(
                    right_bot,
                    pipelines,
                    device,
                    robot,
                    right_arm_indices,
                    right_ee_index,
                )
                print(
                    "top mean",
                    obs["observation.images.top_scene"].mean(),
                    "wrist mean",
                    obs["observation.images.right_wrist"].mean()
                )
                print(obs["observation.object_centroid"])

                mask = obs["observation.object_mask"]


                print(
                    "mask min",
                    mask.min().item(),
                    "mask max",
                    mask.max().item(),
                    "mask sum",
                    mask.sum().item()
                )

                obs["observation.object_centroid"][:] = torch.tensor(
                    [[0.95, 0.50]],
                    device=device
                )

                output = policy.select_action(preprocessor(obs))
                # CRITICAL (lerobot v0.6.0): select_action returns actions in
                # NORMALIZED space; the postprocessor's unnormalizer maps them
                # back to joint radians.  Skipping it sends garbage commands.
                output = postprocessor(output)
                action_np = output.squeeze().cpu().numpy()

                # Useful thresholding for closing gripper
                # checkpoint 3000 0.8
                # checkpoint 6000 0.08
                # checkpoint 10000 0.05
                # if action_np[6] > 0.05:
                #     action_np[6] = -1.5
                # else:
                #     action_np[6] = 0.0
                
                

                print(
                    f"step={step} "
                    f"arm={np.round(action_np[:6], 3)} "
                    f"gripper={action_np[6]:.4f}"
                )

                send_action(right_bot, action_np)

                elapsed = time.time() - t0
                sleep_time = max(0.0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        reset_robot(right_bot)


if __name__ == "__main__":
    main()