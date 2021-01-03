######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import boto3
from botocore.exceptions import ClientError
import json
import datetime
import concurrent.futures


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

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
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


def future_callback_error_logger(future):
    try:
        result = future.result()
        print("Future result: {}".format(result))
    except Exception as e:
        print("Executor Exception: {}".format(e))

class EyePiDetectionEvent(object):
    """
    An event correlated with objects being detected in the video stream
    """
    def __init__(self, frame, detected_classes, detected_scores):
        self.frame = frame
        self.detected_classes = detected_classes
        self.detected_scores = detected_scores

class EyePiEventStream(object):
    """
    Handles overall EyePi functionality in terms of injesting event stream from
    camera and model
    """
    def __init__(self, labels):
        self.s3_client = boto3.client('s3')
        self.labels = labels
        self.min_detection_threshold = 0.72
        self.num_captured_frames = 0
        self.num_frames_per_video = 5  # at 1 FPS, this is 5s worth of video
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.state = 'IDLE'
        self.last_person_detected_confidence = float(0)

        # Threadpool executor to keep from blocking the main thread.
        # Keep max workers at 1 so it's easier to reason about concurrent access to data.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.bucket_name = "eyepi"

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def process_event(self, event):
        """

        After a person is detected with a high enough threshold (eg, 72%), it will
        enter the PERSON_DETECTED_CAPTURING_VIDEO state and:

        - Capture each video frame to a buffer in memory for the next 5 seconds
        - Trigger the state transition back to the idle state, which will do the following:
            - Kick off a separate thread so we don't block the VideoStream thread
            - Collect the frames from the buffer and write to a file
            - Upload the video file to s3 (alert_123.mp4)
            - Generate a pre-signed url for the video file
            - Upload an alert_123.txt file like "A person was detected with 72% accuracy: {presigned_url} to s3

        States:

            IDLE
            PERSON_DETECTED_CAPTURING_VIDEO

        """

        if self.state == 'IDLE':
            self.process_event_idle_state(event)
        elif self.state == 'PERSON_DETECTED_CAPTURING_VIDEO':
            self.process_event_capturing_state(event)
        else:
            raise Exception("Unknown state: {}".format(self.state))

    def person_detected(self, event):
        found_person = False
        # Loop over all scores, find the corresponding object name, and see if it's a person
        for i in range(len(event.detected_scores)):
            if ((event.detected_scores[i] > self.min_detection_threshold) and (event.detected_scores[i] <= 1.0)):
                object_name = self.labels[int(event.detected_classes[i])]
                if object_name.lower() == 'person':
                    found_person = True
                    self.last_person_detected_confidence = event.detected_scores[i]
                    break

        return found_person

    def process_event_idle_state(self, event):
        if self.person_detected(event) == True:
            self.transition_to_capturing_state(event)

    def process_event_capturing_state(self, event):

        self.num_captured_frames += 1
        self.writer.write(event.frame)  # TODO: this should happen in self.executor() so it doesn't block the main thread
        print("Captured frame {}/{}".format(self.num_captured_frames, self.num_frames_per_video))

        if self.num_captured_frames > self.num_frames_per_video:
            self.transition_to_idle_state(event)

    def transition_to_capturing_state(self, event):

        print("Person detected!!  Capturing video")

        self.state = 'PERSON_DETECTED_CAPTURING_VIDEO'
        self.num_captured_frames = 0

        self.latest_capture_file_name = "alert_{}.avi".format(datetime.datetime.utcnow().timestamp())
        self.latest_capture_file_path = "/tmp/{}".format(self.latest_capture_file_name)
        (h, w) = event.frame.shape[:2]

        self.writer = cv2.VideoWriter(
            self.latest_capture_file_path,
            self.fourcc,
            1, # 1 FPS -- should be tuned, but it worked on pyshell test
            (w, h),
            True,
        )

    def transition_to_idle_state(self, event):

        print("Finished capturing video, returning to IDLE state")

        self.state = 'IDLE'
        print("Finished capturing video: {}".format(self.latest_capture_file_path))

        # Kick off thread to save video and json with signed s3 url and push both to s3
        future = self.executor.submit(
            self.push_event_to_s3,
            event=event,
            filename=self.latest_capture_file_path,
            object_name=self.latest_capture_file_name,
        )
        future.add_done_callback(future_callback_error_logger)

        self.writer.release()  # TODO: should happen in self.executor() task
        self.num_captured_frames = 0

    def push_event_to_s3(self, event, filename, object_name):
        """
        - Push video to s3
        - Generate signed URL for video
        - Write an alert file that says "Person detected .. <link to video>"
        - Write alert file to s3
        """

        try:
            print("Uploading {} -> {}/{} .. ".format(filename, self.bucket_name, object_name))
            response = self.s3_client.upload_file(
                filename,
                self.bucket_name,
                object_name,
            )
            print("Finished uploading {} -> {}/{} .. ".format(filename, self.bucket_name, object_name))

            # Make the video capture file public
            # TODO: use signed URLs instead of making the file public
            self.s3_client.put_object_acl(ACL='public-read', Bucket=self.bucket_name, Key="%s" % (object_name))


            # Create and upload alert meta file
            public_url = f'https://{self.bucket_name}.s3.amazonaws.com/{object_name}'

            alert_meta = {
                "detected_object": "person",
                "detection_confidence": float(self.last_person_detected_confidence),
                "captured_video_url": public_url,
            }

            alert_meta_object_name = "{}.json".format(object_name)
            alert_meta_filepath = "/tmp/{}".format(alert_meta_object_name)
            f = open(alert_meta_filepath, "a")
            f.write(json.dumps(alert_meta))
            f.close()

            print("Uploading {} -> {}/{} .. ".format(alert_meta_filepath, self.bucket_name, alert_meta_object_name))

            response = self.s3_client.upload_file(
                alert_meta_filepath,
                self.bucket_name,
                alert_meta_object_name,
            )
            print("Finished uploading {} -> {}/{} .. ".format(alert_meta_filepath, self.bucket_name, alert_meta_object_name))

        except Exception as e:
            print("Exception writing {} to s3: {}".format(object_name, str(e)))
            raise e

    def last_alert_sent_minutes(self):
        # TODO
        return 0

    def write_alert_hash_to_s3(self, alert_hash):

        filename = "/tmp/alert_hash.json"
        bucket_name = "eyepi"
        object_name = "alert_{}.json".format(datetime.datetime.utcnow().timestamp())

        f = open(filename, "a")
        f.write(json.dumps(alert_hash))
        f.close()

        # Upload to s3
        print("writing alert to s3 bucket {} at {}".format(bucket_name, object_name))

        try:
            response = self.s3_client.upload_file(
                filename,
                bucket_name,
                object_name,
            )
        except ClientError as e:
            print("Exception writing alert hash to s3: {}. response: {}".format(str(e), str(response)))


    def possibly_trigger_alert(self, event):

        # if it's not a person, ignore
        if event.detected_object_name != 'person':
            print("not a person, ignoring")
            return

        # if we've sent an alert in the last X minutes, ignore
        if self.last_alert_sent_minutes() > 5:
            print("already sent alert, ignoring")
            return

        # Create a json file with the person detection score
        alert_hash = {
            "person": float(event.detected_object_score)
        }

        # Write to S3
        print("writing alert to s3")
        self.write_alert_hash_to_s3(alert_hash)
        print("done writing alert to s3")


def main(args):

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    eyePiEventStream = EyePiEventStream(labels)

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        print("read a frame")

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        print("Frame rate: {}".format(frame_rate_calc))

        # Process EyePiDetectionEvent
        eyePiEvent = EyePiDetectionEvent(
            frame=frame,
            detected_classes=classes,
            detected_scores=scores
        )
        eyePiEventStream.process_event(eyePiEvent)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    videostream.stop()
    eyePiEventStream.shutdown()



if __name__ == "__main__":

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()
    main(args)