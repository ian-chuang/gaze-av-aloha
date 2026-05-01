import depthai as dai
import time
import cv2
import numpy as np
from webrtc_headset import WebRTCHeadset

# ---- Setup headset ----
headset = WebRTCHeadset()
headset.run_in_thread()

# ---- Setup pipeline ----
pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

left_out = cam_left.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888i,
    fps=25,
)

right_out = cam_right.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888i,
    fps=25,
)

q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()

time.sleep(0.5)

# ---- Main loop ----
while pipeline.isRunning():
    left_img = q_left.get().getCvFrame()
    right_img = q_right.get().getCvFrame()

    # left_img = np.ascontiguousarray(left_img)
    # right_img = np.ascontiguousarray(right_img)

    headset.send_images(left_img, right_img)