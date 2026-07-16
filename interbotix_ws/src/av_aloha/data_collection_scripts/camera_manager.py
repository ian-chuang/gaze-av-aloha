import pyrealsense2 as rs
import depthai as dai
import numpy as np
import threading
import time
import cv2
from dataclasses import dataclass
from data_col_config import ARM_MODES

CAMERAS = [
    "left_wrist",
    "right_wrist",
    "oak_left",
    "oak_right",
    "top_scene",
    "low_scene",
]

CAMERA_SERIALS = {
    "left_wrist": "230322272239",
    "right_wrist": "230322270105",
    "top_scene": "230322270396",
    "low_scene": "230322271312",
}

@dataclass
class CameraConfig:
    top_active: bool = True
    low_active: bool = False

def get_active_cameras(arm_mode: str, camera_config: CameraConfig):
    cameras = []
    arms = ARM_MODES[arm_mode]

    if "left" in arms:
        cameras.append("left_wrist")

    if "right" in arms:
        cameras.append("right_wrist")

    if "middle" in arms:
        cameras.append("oak_left")
        cameras.append("oak_right")

    if camera_config.top_active:
        cameras.append("top_scene")

    if camera_config.low_active:
        cameras.append("low_scene")

    return cameras

def setup_oak_stereo():
    oak_pipeline = dai.Pipeline()

    cam_left = oak_pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = oak_pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = cam_left.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=60,
    )
    right_out = cam_right.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=60,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    oak_pipeline.start()
    time.sleep(0.5)
    return oak_pipeline, q_left, q_right

def setup_realsense_cameras(active_cameras, camera_shutdown, frame_lock, latest_frames, latest_timestamps):
    rs_pipelines = {}

    for name in active_cameras:
        if name.startswith("oak"):
            continue

        serial = CAMERA_SERIALS[name]

        rs_pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)

        profile = rs_pipeline.start(config)

        color_profile = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
        )

        rs_pipelines[name] = rs_pipeline

        threading.Thread(
            target=camera_worker,
            args=(name, rs_pipeline, camera_shutdown, frame_lock, latest_frames, latest_timestamps),
            daemon=True,
        ).start()

    return rs_pipelines

def setup_cameras(
    active_cameras,
    camera_shutdown,
    frame_lock,
    latest_frames,
    latest_timestamps,
):
    cameras = {}

    if any(cam.startswith("oak") for cam in active_cameras):
        cameras["oak"] = setup_oak_stereo()

    cameras["realsense"] = setup_realsense_cameras(
        active_cameras,
        camera_shutdown,
        frame_lock,
        latest_frames,
        latest_timestamps,
    )

    return cameras

def camera_worker(name, rs_pipeline, camera_shutdown, frame_lock, latest_frames, latest_timestamps):

    while not camera_shutdown.is_set():
        try:
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        except Exception as e:
            print(f"Warning: Waiting for camera frames failed: {e}")
            continue

        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        color_ts = color_frame.get_timestamp() * 1e-3

        with frame_lock:
            latest_frames[name] = color
            latest_timestamps[name] = color_ts

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