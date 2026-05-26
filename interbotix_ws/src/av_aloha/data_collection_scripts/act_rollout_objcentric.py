from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.factory import make_policy
from lerobot.configs.types import PolicyFeature, FeatureType

import pyroki as pk
import jaxlie

from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

URDF_PATH = "/home/devi/giava/right.urdf"
RIGHT_EE_LINK = "rightgripper_base"

RIGHT_ARM_NAMES = [
    "rightwaist",
    "rightshoulder",
    "rightelbow",
    "rightforearm_roll",
    "rightwrist_angle",
    "rightwrist_rotate",
]

GRIPPER_CURRENT_LIMIT = 200

CAMERA_SERIALS = {
    "observation.images.right_wrist": "230322270105",
    "observation.images.top_scene": "230322270396",
}

RIGHT_RESET_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=np.float32)

def build_robot_model():
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    right_arm_indices = [
        robot.joints.actuated_names.index(name)
        for name in RIGHT_ARM_NAMES
    ]

    right_ee_index = robot.links.names.index(RIGHT_EE_LINK)

    return robot, right_arm_indices, right_ee_index

def extract_green_object_features(img):
    """
    img: RGB uint8 image (H,W,3)

    returns:
        mask_rgb: (3,H,W) float32
        centroid: (2,) float32 normalized to [0,1]
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(mask)

    h, w = mask.shape

    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = w / 2
        cy = h / 2

    centroid = np.array([
        cx / w,
        cy / h,
    ], dtype=np.float32)

    mask = mask.astype(np.float32) / 255.0

    mask_rgb = np.repeat(mask[None], 3, axis=0)

    return mask_rgb, centroid, mask

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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_color_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame received")
    frame = np.asanyarray(color_frame.get_data())
    return frame


def make_observation(
    right_bot,
    pipelines,
    device,
    robot,
    right_arm_indices,
    right_ee_index,
):
    joint_state_msg = right_bot.dxl.joint_states

    joint_positions = np.array(
        joint_state_msg.position[:6],
        dtype=np.float32,
    )

    gripper_position = np.array(
        [joint_state_msg.position[6]],
        dtype=np.float32,
    )

    obs_state = np.concatenate(
        [joint_positions, gripper_position],
        axis=0,
    )

    # ---------------------------------------------------
    # Forward kinematics for end effector pose
    # ---------------------------------------------------

    q = np.zeros(
        robot.joints.num_actuated_joints,
        dtype=np.float32,
    )

    q[right_arm_indices] = joint_positions

    fk = robot.forward_kinematics(q)

    T_right = jaxlie.SE3(
        fk[right_ee_index]
    ).as_matrix()

    position = T_right[:3, 3]

    quat_xyzw = R.from_matrix(
        T_right[:3, :3]
    ).as_quat()

    ee_pose = np.concatenate([
        position,
        quat_xyzw,
    ]).astype(np.float32)

    observation = {
        "observation.state": (
            torch.from_numpy(obs_state)
            .unsqueeze(0)
            .to(device)
        ),

        "observation.ee_pose": (
            torch.from_numpy(ee_pose)
            .unsqueeze(0)
            .to(device)
        ),
    }

    for key, pipeline in pipelines.items():
        frame_bgr = get_color_frame(pipeline)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb, centroid, mask = extract_green_object_features(frame_rgb)
        # frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        frame = (
            torch.from_numpy(frame_rgb)
            .float()
            / 255.0
        )

        frame = (
            frame.permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        observation[key] = frame

        mask_tensor = (
            torch.from_numpy(mask_rgb)
            .float()
            .unsqueeze(0)
            .to(device)
        )

        if key == "observation.images.top_scene":
            observation["observation.object_mask"] = mask_tensor

            observation["observation.object_centroid"] = (
                torch.from_numpy(centroid)
                .float()
                .unsqueeze(0)
                .to(device)
            )
        

    return observation


def send_action(right_bot, action):
    action = action.detach().float().cpu().numpy().reshape(-1)
    if action.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")


    

    arm_cmd = action[:6]
    gripper_cmd = float(action[6])

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
    dataset_metadata = LeRobotDatasetMetadata(Path(dataset_root))
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if k == "action"}
    # input_features = {k: v for k, v in features.items() if k != "action"}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        # "observation.ee_pose",
    }

    input_features = {
        k: v
        for k, v in features.items()
        if (
            k != "action"
            and k not in excluded_features
        )
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
        chunk_size=10,
        n_action_steps=5,
    )
    policy = make_policy(cfg, ds_meta=dataset_metadata)

    policy_path = Path(policy_dir)
    try:
        loaded = policy.__class__.from_pretrained(policy_path)
        policy = loaded
    except Exception:
        ckpt = torch.load(policy_path / "checkpoint.pt", map_location=device)
        state_dict = ckpt.get("policy_state_dict", ckpt)
        policy.load_state_dict(state_dict)

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

    set_register(
        "puppet_right",
        "gripper",
        "Current_Limit",
        GRIPPER_CURRENT_LIMIT,
    )

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


def main():
    dataset_root = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260524_224911"
    )
    policy_dir = Path("outputs/act_rgb_mask_centroid_ee_vae_c10_a5")
    rollout_seconds = 10
    control_hz = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.init_node("act_rollout_test", anonymous=True)

    policy, _ = load_policy(policy_dir, dataset_root, device)

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="block_square",
        root=dataset_root,
    )

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

    set_register(
        "puppet_right",
        "gripper",
        "Current_Limit",
        GRIPPER_CURRENT_LIMIT,
    )

    right_bot.dxl.robot_set_operating_modes(
        "single",
        "gripper",
        "current_based_position",
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    pipelines = {key: setup_camera(serial) for key, serial in CAMERA_SERIALS.items()}

    try:
        reset_robot(right_bot)
        dt = 1.0 / control_hz
        num_steps = int(rollout_seconds * control_hz)

        print(f"Loaded policy from {policy_dir}")
        print(f"Dataset fps={dataset_metadata.fps}")
        print(f"Running rollout for {num_steps} steps at {control_hz} Hz")
        print("data set metadata states on action")

        print(dataset_metadata.stats["action"])

        robot, right_arm_indices, right_ee_index = build_robot_model()

        obs = make_observation(
            right_bot,
            pipelines,
            device,
            robot,
            right_arm_indices,
            right_ee_index,
        )

        for k, v in obs.items():
            if torch.is_tensor(v):
                print(k, v.shape)
            else:
                print(k, type(v))

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
                output = policy.select_action(obs)
                # print(output.squeeze().cpu().numpy())
                action_np = output.squeeze().cpu().numpy()

                print(
                    f"step={step} "
                    f"arm={np.round(action_np[:6], 3)} "
                    f"gripper={action_np[6]:.4f}"
                )
                send_action(right_bot, output)
                # if step > 10:
                #     output[0, 6] = -1.5
                #     send_action(right_bot, output)
                # else:
                #     send_action(right_bot, output)

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