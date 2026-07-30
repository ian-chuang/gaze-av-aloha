import time
import depthai as dai
import numpy as np
import cv2

from gym_av_aloha.vr.headset import Headset

FPS = 25
WIDTH, HEIGHT = 640, 480

# --- Headset ---
headset = Headset()
headset.run_in_thread()

# --- DepthAI pipeline ---
pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
# FIX OV9782/OV9282: force CAM_B to COLOR because this device reports OV9282
# for an OV9782-class sensor, which leads to mono/grid artifacts.
cam_left.setSensorType(dai.CameraSensorType.COLOR)
left_out = cam_left.requestOutput(
    size=(WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.RGB888p,
    fps=FPS,
)

cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
# FIX OV9782/OV9282: force CAM_C to COLOR for the same misdetection reason.
cam_right.setSensorType(dai.CameraSensorType.COLOR)
right_out = cam_right.requestOutput(
    size=(WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.RGB888p,
    fps=FPS,
)

q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()

print("Streaming OAK cameras to headset...")

# --- Main loop ---
while True:
    start = time.time()

    left_msg = q_left.get()
    right_msg = q_right.get()

    if left_msg is None or right_msg is None:
        continue

    left_frame = left_msg.getCvFrame()
    right_frame = right_msg.getCvFrame()
    left_frame = cv2.cvtColor(left_msg.getCvFrame(), cv2.COLOR_BGR2RGB)
    right_frame = cv2.cvtColor(right_msg.getCvFrame(), cv2.COLOR_BGR2RGB)

    if left_frame is None or right_frame is None:
        continue

    # ensure correct format
    if left_frame.dtype != np.uint8:
        left_frame = left_frame.astype(np.uint8)
    if right_frame.dtype != np.uint8:
        right_frame = right_frame.astype(np.uint8)

    # --- SEND TO HEADSET ---
    headset.send_left_image(left_frame, 0)
    headset.send_right_image(right_frame, 0)

    # --- FPS control ---
    dt = time.time() - start
    time.sleep(max(0, 1.0 / FPS - dt))



# # Stream video from your OAK camera to the headset and overlay controller pose (relative to headset frame) as text.

# import depthai as dai
# import time
# import cv2
# import numpy as np
# from webrtc_headset import WebRTCHeadset
# from headset_control import HeadsetFullControl as HeadsetControl
# from headset_utils import HeadsetFeedback
# from transform_utils import (pose2mat)

# # ---- Setup headset ----
# headset = WebRTCHeadset()
# headset.run_in_thread()

# while not headset.data_channel_open:
#     print("Waiting for data channel...")
#     time.sleep(0.1)

# print("Headset connected.")
# headset_control = HeadsetControl()
# feedback = HeadsetFeedback()
# headset_control.reset()

# # ---- Setup pipeline ----
# pipeline = dai.Pipeline()

# cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
# cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# left_out = cam_left.requestOutput(
#     (640, 480),
#     type=dai.ImgFrame.Type.GRAY8,
#     fps=90,
# )

# right_out = cam_right.requestOutput(
#     (640, 480),
#     type=dai.ImgFrame.Type.GRAY8,
#     fps=90,
# )

# q_left = left_out.createOutputQueue()
# q_right = right_out.createOutputQueue()

# pipeline.start()

# time.sleep(0.5)

# # ---- Main loop ----
# while pipeline.isRunning():
#     left_img = q_left.get().getCvFrame()
#     right_img = q_right.get().getCvFrame()
#     print("Sending images to headset...")

#     headset.send_images(left_img, right_img)

#     # headset_data = headset.receive_data()

#     # if headset_data is None:
#     #     print("No headset data received, skipping this iteration.")
#     #     time.sleep(0.02)
#     #     continue

#     """ # --- Extract positions ---
#     h_pos = headset_data.h_pos
#     l_pos = headset_data.l_pos
#     r_pos = headset_data.r_pos

#     # --- Compute naive relative ---
#     l_rel = l_pos - h_pos
#     r_rel = r_pos - h_pos

#     # --- Build string ---
#     feedback.info = (
#         f"HEAD\n"
#         f"x: {h_pos[0]:.3f} y: {h_pos[1]:.3f} z: {h_pos[2]:.3f}\n\n"
#         f"LEFT\n"
#         f"x: {l_pos[0]:.3f} y: {l_pos[1]:.3f} z: {l_pos[2]:.3f}\n\n"
#         f"RIGHT\n"
#         f"x: {r_pos[0]:.3f} y: {r_pos[1]:.3f} z: {r_pos[2]:.3f}\n\n"
#         f"REL\n"
#         f"L-H: ({l_rel[0]:.3f}, {l_rel[1]:.3f}, {l_rel[2]:.3f})\n"
#         f"R-H: ({r_rel[0]:.3f}, {r_rel[1]:.3f}, {r_rel[2]:.3f})"
#     )

#     headset.send_feedback(feedback) """
