import cv2
from collections import deque
from threading import Thread

# This video codec worked after some trial and error as long as the filename was ".avi"
# See https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

class VideoStream:
    """
    Define VideoStream class to handle streaming of video from webcam in separate processing thread
    """
    def __init__(self,inputdevice,resolution=(640,480),framerate=30):

        self.inputdevice = inputdevice

        self.framerate = framerate

        # Initialize the PiCamera and the camera image stream
        if self.inputdevice == '/dev/video0':
            self.stream = cv2.VideoCapture(0)

            # When reading from the camera, it seems to be bottlenecked on
            # other things like model inference and so the videostream thread is
            # always keeps the framebuffer full, so the framebuffer size can be
            # very small at the risk of other threads missing some frames.
            framebuffer_size = 5 * 5  # 5 FPS * 5 seconds

        else:
            self.stream = cv2.VideoCapture(self.inputdevice)

            # When reading from a file, it's OK to introduce more latency
            # in order to reduce the chance of:
            # 1) starving the other threads if the videostream thread is producing frames slower than
            # other threads are consuming frames
            # 2) overwriting frames if the videostream thread is producing faster than other
            # threads are consuming.
            framebuffer_size = 5 * 60  # 5 FPS * 60 seconds

        ret = self.stream.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        self.framebuffer = deque([], framebuffer_size)

        # Read first frame from the stream
        (self.grabbed, frame) = self.stream.read()

        self.framebuffer.appendleft(frame)

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, frame) = self.stream.read()

            self.framebuffer.appendleft(frame)

            if self.inputdevice != '/dev/video0':
                # Force it to 8 FPS if reading from file
                time.sleep(0.125)

    def read(self):
        # Return the most recent frame
        frame = self.framebuffer.pop()
        return frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
