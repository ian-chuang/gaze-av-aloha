from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.factory import make_policy
from lerobot.configs.types import FeatureType

GRIPPER_CURRENT_LIMIT = 200

CAMERA_SERIALS = {
    "observation.images.right_wrist": "230322270105",
    "observation.images.top_scene": "230322270396",
}

# RIGHT_RESET_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=np.float32)
RIGHT_RESET_Q = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float) # new reset pose facing down
MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)


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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    pipeline.start(config)
    return pipeline

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

def get_color_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame received")
    return np.asanyarray(color_frame.get_data())


def make_observation(right_bot, pipelines, device):
    joint_state_msg = right_bot.dxl.joint_states
    joint_positions = np.array(joint_state_msg.position[:6], dtype=np.float32)
    gripper_position = np.array([joint_state_msg.position[6]], dtype=np.float32)
    obs_state = np.concatenate([joint_positions, gripper_position], axis=0)

    obs = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
    }
    
    for key, pipeline in pipelines.items():
        frame = get_color_frame(pipeline)
        if key == "observation.images.top_scene":
            frame = digital_zoom(frame)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
        obs[key] = frame

    return obs

def send_action(right_bot, action):
    if torch.is_tensor(action):
        action = action.detach().float().cpu().numpy()
    action = np.asarray(action).reshape(-1)

    if action.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")

    arm_cmd = action[:6]
    gripper_cmd = float(action[6])

    current_q = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)
    arm_cmd = np.clip(arm_cmd, current_q - MAX_JOINT_STEP, current_q + MAX_JOINT_STEP)

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
        repo_id="transfer_flower",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        "observation.ee_pose",
        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",
    }

    input_features = {
        k: v for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=50,
        n_action_steps=50,
        use_vae=True,
        kl_weight=10.0,
        optimizer_lr=2e-5,
        optimizer_lr_backbone=1e-5,
    )

    policy = make_policy(cfg, ds_meta=dataset_metadata)
    

    print("image_features:", policy.config.image_features)
    print("state_feature:", policy.config.robot_state_feature)
    print("action_delta_indices:", policy.config.action_delta_indices)
    print("observation_delta_indices:", policy.config.observation_delta_indices)
    print("chunk_size:", policy.config.chunk_size)
    print("n_action_steps:", policy.config.n_action_steps)
    print("input_features:", policy.config.input_features)
    print("output_features:", policy.config.output_features)

    ckpt = torch.load(Path(policy_dir) / "checkpoint_2000.pt", map_location=device, weights_only=False)
    state_dict = ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt
    # policy.load_state_dict(state_dict, strict=True)

    missing, unexpected = policy.load_state_dict(
        state_dict,
        strict=True,
    )

    print("MISSING")
    for k in missing:
        print(k)

    print("\nUNEXPECTED")
    for k in unexpected:
        print(k)

    policy.to(device)
    policy.eval()
    return policy, dataset_metadata

def reset_robot(right_bot):
    right_bot.arm.set_joint_positions(
        RIGHT_RESET_Q.tolist(),
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = 0.1
    right_bot.gripper.core.pub_single.publish(cmd)
    rospy.sleep(1.0)


def main():
    dataset_root = Path("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_000935"
        #"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433"
        #"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838"
    )
    policy_dir = Path("outputs/act_transfer_flower")
    rollout_seconds = 3
    control_hz = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.init_node("act_rollout_test", anonymous=True)

    policy, dataset_metadata = load_policy(policy_dir, dataset_root, device)

    print("Policy expects image keys:", policy.config.image_features)
    print("Policy expects state keys:", policy.config.robot_state_feature)
    

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
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    pipelines = {key: setup_camera(serial) for key, serial in CAMERA_SERIALS.items()}

    try:
        reset_robot(right_bot)

        dt = 1.0 / control_hz
        num_steps = int(rollout_seconds * control_hz)

        print(f"Loaded policy from {policy_dir}")
        print(f"Dataset fps={dataset_metadata.fps}")
        print(f"Running rollout for {num_steps} steps at {control_hz} Hz")
        print("Action stats:")
        print(dataset_metadata.stats["action"])

        with torch.inference_mode():
            for step in range(num_steps):
                t0 = time.time()

                obs = make_observation(right_bot, pipelines, device)
                print("Obs keys:", list(obs.keys()))
                print(obs["observation.state"])

                # Match training-time image channel swap if needed.
                # for key in ["observation.images.right_wrist", "observation.images.top_scene"]:
                #     if key in obs:
                #         obs[key] = obs[key][:, [2, 1, 0], :, :]

                output = policy.select_action(obs)
                print("output information after output = policy.select_action(obs)")
                print(type(output))
                print(output.shape)
                print(output)
                action_np = output.squeeze().cpu().numpy()
                print("output information after action_np = output.squeeze().cpu().numpy()")
                print(type(action_np))
                print(action_np.shape)
                print(action_np)

                print(
                    f"step={step} "
                    f"arm={np.round(action_np[:6], 3)} "
                    f"gripper={action_np[6]:.4f}"
                )

                # gripper_cmd = -1.5 if action_np[6] > 0.01 else 0.0

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