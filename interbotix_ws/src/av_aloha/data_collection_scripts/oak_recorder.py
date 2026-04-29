import depthai as dai
# import threading
import time


class OAKImageRecorder:
    def __init__(self, width=640, height=480, fps=25):
        self.width = width
        self.height = height
        self.fps = fps

        # --- Pipeline ---
        self.pipeline = dai.Pipeline()

        self.device = dai.Device()
        self.device.startPipeline(self.pipeline)

        cam_left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        self.left_out = cam_left.requestOutput(
            size=(width, height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=fps,
        )

        self.right_out = cam_right.requestOutput(
            size=(width, height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=fps,
        )

        self.q_left = self.left_out.createOutputQueue()
        self.q_right = self.right_out.createOutputQueue()

        self.left = None
        self.right = None

        time.sleep(0.5)

        # self.running = True
        # self.thread = threading.Thread(target=self._loop, daemon=True)
        # self.thread.start()

        # # wait for first frame
        # self._wait_for_first_frame()

    def _wait_for_first_frame(self):
        start = time.time()
        while self.left is None or self.right is None:
            if time.time() - start > 5:
                raise RuntimeError("OAK camera failed to start")
            time.sleep(0.05)

    def _loop(self):
        while self.running:
            try:
                left = self.q_left.get().getCvFrame()
                right = self.q_right.get().getCvFrame()

                self.left = left
                self.right = right

            except Exception as e:
                print(f"[OAK ERROR] {e}")
                time.sleep(0.01)

    def get_images(self):
        try:
            print("In get_images in oak recorder")
            return self.q_left.get().getCvFrame(), self.q_right.get().getCvFrame()
        except Exception as e:
            print(f"[OAK Error]: {e}")
            return None

        # if self.left is None or self.right is None:
        #     return None
        # return self.left.copy(), self.right.copy()

    # def stop(self):
    #     self.running = False
    #     if self.thread.is_alive():
    #         self.thread.join()
        # self.device.close()