import numpy as np
import time
import threading
import cv2
from gym_av_aloha.vr.headset import WebRTCHeadset
import rospy

from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from functools import partial
import IPython
e = IPython.embed

class StereoImageRecorder:
    def __init__(self, cam_index, headset : WebRTCHeadset = None, auto_start=True):
        self.cap = None
        self.lock = threading.Lock()
        self.left_image = None
        self.right_image = None
        self.is_running = False
        self.thread = None
        self.headset = headset
        self.cam_index = cam_index
        self.count = 0
        self.count_lock = threading.Lock()
        if auto_start:
            self.start()

    def __del__(self):
        self.stop()

    def start(self):
        self.thread = threading.Thread(target=self.record_images)
        self.thread.start()

        print("Started Image")

        # try to see if we can get the first image timeout after 5 seconds
        start_time = time.time()
        while self.left_image is None or self.right_image is None:
            if time.time() - start_time > 5:
                raise Exception("Timeout, failed to get image from webcam")
            time.sleep(0.1)
    
    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()

    def set_count(self, count):
        with self.count_lock:
            self.count = count

    def record_images(self):
        # Open the webcam
        # Define the GStreamer pipeline with resolution set to 720x2560
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.cam_index} ! "
            "jpegdec ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw, width=2560, height=720 ! "
            "appsink"
        )

        # Create a VideoCapture object with the GStreamer pipeline
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        # self.cap = cv2.VideoCapture(self.cam_index)
        # if not self.cap.isOpened():
        #     raise Exception("Failed to open webcam")

        # # Set the webcam resolution
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # # set to mpjg
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            raise Exception("Failed to open webcam")

        self.is_running = True
        while self.is_running and not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                continue

            height, width, _ = frame.shape
            mid = width // 2

            # Split the frame into left and right halves
            left_image_ocv = frame[:, :mid]
            right_image_ocv = frame[:, mid:]

            # Convert to RGB
            left_image_ocv = cv2.cvtColor(left_image_ocv, cv2.COLOR_BGR2RGB)
            right_image_ocv = cv2.cvtColor(right_image_ocv, cv2.COLOR_BGR2RGB)

            # take off 50px from the left side of the left img
            left_image_ocv = left_image_ocv[:, 200:]
            right_image_ocv = right_image_ocv[:, 200:] # now is 720x1080

            # shave off both sides so that becomes 720x960
            left_image_ocv = left_image_ocv[:, 60:-60]
            right_image_ocv = right_image_ocv[:, 60:-60]
            left_image_ocv = cv2.resize(left_image_ocv, (640, 480))
            right_image_ocv = cv2.resize(right_image_ocv, (640, 480))
        
            # print(left_image_ocv.shape, right_image_ocv.shape)

            if self.headset is not None:
                with self.count_lock:
                    count = self.count

                self.headset.send_images((left_image_ocv, count), (right_image_ocv, count))

            with self.lock:
                self.left_image = left_image_ocv
                self.right_image = right_image_ocv

        self.cap.release()

    def get_images(self):

        with self.lock:
            left_image = self.left_image
            right_image = self.right_image

        if self.is_running and left_image is not None and right_image is not None:
            return left_image, right_image
        else:
            raise Exception("Webcam is not running or image is not available")
        




class ROSImageRecorder:
    def __init__(self, 
                 camera_names=['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
                 init_node=True, 
                 is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = camera_names
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            callback_func = partial(self.image_cb, cam_name)
            rospy.Subscriber(f"/{cam_name}/color/image_raw", Image, callback_func) 
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

            rospy.wait_for_message(f"/{cam_name}/color/image_raw", Image, timeout=1.0)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()