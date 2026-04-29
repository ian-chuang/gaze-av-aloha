# from oak_recorder import OAKImageRecorder
# import time

# rec = OAKImageRecorder()

# while True:
#     imgs = rec.get_images()
#     if imgs is not None:
#         left, right = imgs
#         print(left.shape, right.shape)
#         break
#     time.sleep(0.1)

import cv2
import time
from oak_recorder import OAKImageRecorder

recorder = OAKImageRecorder()

print("Starting OAK...")
# IMPORTANT: only call this if your class requires it
if hasattr(recorder, "start"):
    recorder.start()

time.sleep(2)  # give camera time to warm up

print("Streaming... Press 'q' to quit.")

while True:
    print("In true")
    images = recorder.get_images()
    print("got images from recorder")

    if images is None:
        print("No frames yet...")
        time.sleep(0.1)
        continue

    left, right = images
    print("assigned left and right")

    # DEBUG
    print("Left mean:", left.mean(), "Right mean:", right.mean())

    # Show images
    cv2.imshow("OAK Left", left)
    cv2.imshow("OAK Right", right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recorder.stop()
cv2.destroyAllWindows()