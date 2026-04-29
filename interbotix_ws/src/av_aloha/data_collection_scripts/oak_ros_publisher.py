# #!/usr/bin/env python3

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import depthai as dai
# import cv2

# def main():
#     rospy.init_node("oak_stereo_publisher")
#     pub_left = rospy.Publisher("/oak/left/img", Image, queue_size=10)
#     pub_right = rospy.Publisher("/oak/right/img". Image, queue_size=10)
#     bridge = CvBridge()

#     # ---- DepthAI pipeline ----
#     pipeline = dai.Pipeline()

#     cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
#     left_out = cam_left.requestFullResolutionOutput()

#     xout_left = 

#     cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
#     right_out = cam_right.requestFullResolutionOutput()

#     xout.setStreamName("/oak/left/img")
#     cam_left.preview.link(xout.input)

#     # ---- Start device ----
#     with dai.Device(pipeline) as device:
#         q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

#         rate = rospy.Rate(30)

#         while not rospy.is_shutdown():
#             in_frame = q.get()
#             frame = in_frame.getCvFrame()

#             msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
#             pub.publish(msg)

#             rate.sleep()

# if __name__ == "__main__":
#     main()